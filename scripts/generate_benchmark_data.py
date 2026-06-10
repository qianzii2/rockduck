#!/usr/bin/env python3
"""
Generate synthetic router training data from TPC-H and TPC-C query templates.

This script derives query feature vectors, synthetic winner labels, and simulated
latency columns. It does not run the queries or collect measured execution times.

Usage:
    python scripts/generate_benchmark_data.py --output training_data.csv

Output CSV format:
    query_hash, op_type, selectivity, row_count, num_filters, num_columns,
    delta_simulated_ms, vortex_simulated_ms, winner

Where:
    - query_hash: SHA256 of the query filter string
    - op_type: one of scan, point_get, range_scan, filter, aggregate
    - selectivity: heuristic selectivity derived from the filter
    - delta_simulated_ms: simulated DeltaStore latency consistent with `winner`
    - vortex_simulated_ms: simulated Vortex latency consistent with `winner`
    - winner: synthetic oracle label ("delta" or "vortex")

TPC-H queries used for OLAP-flavored synthetic examples:
    Q1:  Full scan + aggregation
    Q3:  Large range scan with grouping
    Q5:  Multi-table join with aggregation
    Q6:  High-selectivity range scan
    Q7:  Large filter + aggregation
    Q10: Complex filter + ordering
    Q18: Large GROUP BY with HAVING

TPC-C transactions used for OLTP-flavored synthetic examples:
    New-Order: Point-get (by PK) + range scan (by customer)
    Payment: Point-get (by PK)
    Delivery: Range scan + point-get
    Stock-Level: Range scan (by warehouse_id)
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

# This generator produces synthetic labels and simulated latency columns for
# router experimentation. It does not execute the queries or measure real
# Delta/Vortex runtimes.

# TPC-H query templates (simplified filter strings).
# Each template maps to a filter string that can be parsed by filter_expr.rs.
TPCH_QUERIES = [
    # Q1: Full scan with aggregation — high selectivity → Vortex wins
    {
        "id": "Q1",
        "filter": "l_shipdate <= '1998-12-31'",
        "columns": ["l_shipdate", "l_extendedprice", "l_quantity", "l_discount"],
        "op_type": "aggregate",
    },
    # Q3: Large range scan with ORDER BY — medium-high selectivity → Vortex
    {
        "id": "Q3",
        "filter": "o_orderdate < '1998-03-01' AND o_shippriority > 0",
        "columns": ["o_orderdate", "o_shippriority", "o_totalprice"],
        "op_type": "range_scan",
    },
    # Q5: Join + aggregation — complex filter → Vortex
    {
        "id": "Q5",
        "filter": "n_regionkey == 1 AND o_orderdate >= '1998-01-01'",
        "columns": ["n_regionkey", "o_orderdate", "o_totalprice"],
        "op_type": "aggregate",
    },
    # Q6: High-selectivity range scan — Vortex wins
    {
        "id": "Q6",
        "filter": "l_shipdate >= '1998-01-01' AND l_discount BETWEEN 0.05 AND 0.07",
        "columns": ["l_shipdate", "l_discount", "l_extendedprice"],
        "op_type": "range_scan",
    },
    # Q7: Large filter — Vortex wins
    {
        "id": "Q7",
        "filter": "l_quantity < 25 AND l_extendedprice > 1000",
        "columns": ["l_quantity", "l_extendedprice"],
        "op_type": "filter",
    },
    # Q10: Complex filter + ordering — Vortex
    {
        "id": "Q10",
        "filter": "c_mktsegment == 'AUTOMOBILE' AND o_orderdate >= '1998-01-01'",
        "columns": ["c_mktsegment", "o_orderdate", "o_totalprice"],
        "op_type": "aggregate",
    },
    # Q18: Large GROUP BY — Vortex
    {
        "id": "Q18",
        "filter": "o_totalprice > 1000 AND l_quantity > 10",
        "columns": ["o_totalprice", "l_quantity", "c_name"],
        "op_type": "aggregate",
    },
]

# TPC-C transaction patterns for OLTP.
TPCC_TRANSACTIONS = [
    # New-Order: Point-get by PK — Delta wins
    {
        "id": "NO-1",
        "filter": "o_orderkey == 12345",
        "columns": ["o_orderkey", "o_custkey", "o_totalprice"],
        "op_type": "point_get",
    },
    {
        "id": "NO-2",
        "filter": "o_orderkey == 67890",
        "columns": ["o_orderkey", "o_custkey", "o_totalprice"],
        "op_type": "point_get",
    },
    # Payment: Point-get by PK — Delta wins
    {
        "id": "PAY-1",
        "filter": "c_custkey == 5000",
        "columns": ["c_custkey", "c_balance", "c_mktsegment"],
        "op_type": "point_get",
    },
    {
        "id": "PAY-2",
        "filter": "c_custkey == 9999",
        "columns": ["c_custkey", "c_balance", "c_mktsegment"],
        "op_type": "point_get",
    },
    # Stock-Level: Range scan by warehouse — Delta wins
    {
        "id": "SL-1",
        "filter": "w_warehouse == 3 AND s_quantity < 10",
        "columns": ["w_warehouse", "s_quantity"],
        "op_type": "range_scan",
    },
    {
        "id": "SL-2",
        "filter": "w_warehouse == 7 AND s_quantity < 20",
        "columns": ["w_warehouse", "s_quantity"],
        "op_type": "range_scan",
    },
]


def query_hash(filter_str: str) -> str:
    return hashlib.sha256(filter_str.encode()).hexdigest()[:16]


def selectivity_from_filter(filter_str: str) -> float:
    """Heuristic selectivity from filter string."""
    upper = filter_str.upper()
    predicate = _first_predicate(filter_str)
    predicate_upper = predicate.upper()

    if "BETWEEN" in predicate_upper:
        base = 0.1
    elif "==" in predicate:
        base = 0.001
    elif "LIKE" in predicate_upper:
        base = 0.05
    elif ">" in predicate or "<" in predicate or "=" in predicate:
        base = 0.25
    else:
        base = 0.1

    if "AND" in upper:
        return max(0.0001, base * 0.25)
    if "OR" in upper:
        return min(1.0, base * 0.5)
    return base


def _first_predicate(filter_str: str) -> str:
    upper = filter_str.upper()
    for token in (" AND ", " OR "):
        idx = upper.find(token)
        if idx != -1:
            return filter_str[:idx]
    return filter_str


def row_count_from_op_type(op_type: str) -> int:
    """Estimated row count based on TPC-H/C scale."""
    if op_type == "point_get":
        return 1
    if op_type == "range_scan":
        return 10_000
    if op_type == "filter":
        return 100_000
    if op_type == "aggregate":
        return 1_000_000
    return 100_000


def ground_truth_winner(op_type: str, selectivity: float) -> str:
    """Oracle routing decision based on the cost model."""
    if op_type == "point_get":
        return "delta"
    if selectivity > 0.5:
        return "vortex"
    if selectivity < 0.01:
        return "delta"
    return "delta" if selectivity < 0.05 else "vortex"


def generate_dataset(queries: list, n_variations: int = 10) -> list:
    """Generate dataset entries from query templates with variations."""
    rows = []
    rng = __import__("random").Random(42)

    for q in queries:
        for var in range(n_variations):
            filter_str = q["filter"]
            # Vary the constants to create different selectivities.
            if var > 0:
                filter_str = filter_str.replace("12345", str(10000 + var * 1000))
                filter_str = filter_str.replace("5000", str(5000 + var * 100))
                filter_str = filter_str.replace("67890", str(60000 + var * 1000))
                filter_str = filter_str.replace("9999", str(9999 + var * 100))

            sel = selectivity_from_filter(filter_str)
            rows.append({
                "query_hash": query_hash(filter_str),
                "op_type": q["op_type"],
                "selectivity": round(sel, 6),
                "row_count": row_count_from_op_type(q["op_type"]) * max(1, var),
                "num_filters": filter_str.count("AND") + filter_str.count("OR") + 1,
                "num_columns": len(q["columns"]),
                "winner": ground_truth_winner(q["op_type"], sel),
                "delta_simulated_ms": rng.uniform(0.1, 5.0) if ground_truth_winner(q["op_type"], sel) == "delta" else rng.uniform(5.0, 50.0),
                "vortex_simulated_ms": rng.uniform(0.5, 10.0) if ground_truth_winner(q["op_type"], sel) == "vortex" else rng.uniform(2.0, 20.0),
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic router training data")
    parser.add_argument("--output", type=str, default="training_data.csv",
                        help="Output CSV file path")
    parser.add_argument("--n-variations", type=int, default=10,
                        help="Number of variations per query template")
    parser.add_argument("--tpch-only", action="store_true")
    parser.add_argument("--tpcc-only", action="store_true")
    args = parser.parse_args()

    queries = []
    if not args.tpcc_only:
        queries.extend(TPCH_QUERIES)
    if not args.tpch_only:
        queries.extend(TPCC_TRANSACTIONS)

    print(f"Generating {args.n_variations} variations of {len(queries)} query templates...")
    rows = generate_dataset(queries, n_variations=args.n_variations)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["query_hash", "op_type", "selectivity", "row_count",
                  "num_filters", "num_columns", "winner",
                  "delta_simulated_ms", "vortex_simulated_ms"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {output_path}")
    delta_wins = sum(1 for r in rows if r["winner"] == "delta")
    vortex_wins = len(rows) - delta_wins
    print(f"  Delta wins: {delta_wins} ({100*delta_wins/len(rows):.1f}%)")
    print(f"  Vortex wins: {vortex_wins} ({100*vortex_wins/len(rows):.1f}%)")
    print(f"\nSynthetic dataset only — to train the model:")
    print(f"  python scripts/train_tree_cnn.py --samples {len(rows)}")


if __name__ == "__main__":
    main()
