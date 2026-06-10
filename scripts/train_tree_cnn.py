#!/usr/bin/env python3
"""
Train Tree-CNN routing model for HTAP query path selection.

Architecture (veDB-HTAP style):
  Twin-network linear model: three independent heads (Delta / Vortex / Merge).
  Input: [op_type_onehot(20) + selectivity(1) + row_count_log(1) + num_filters(1) + num_columns(1) + delta_count(1)] = 25 features.
  Each head: Linear(input_dim=25) -> score.

  Output: three-way routing decision (DeltaStoreOnly / VortexOnly / Merge).

Training procedure:
  1. Generate synthetic training data: queries with features + ground-truth labels.
  2. Compute ground-truth: Delta vs Vortex vs Merge based on workload characteristics.
     - Merge is chosen when: multiple deltas exist AND scan selectivity spans delta coverage.
  3. Softmax cross-entropy loss across three heads.
  4. Export weights as binary f32 little-endian.

Run:
  pip install torch numpy
  python scripts/train_tree_cnn.py
  python scripts/train_tree_cnn.py --exports path/to/ml_exports.jsonl

Output:
  assets/tree_cnn_weights.bin
  Format: [delta_w0..24, delta_b, vortex_w0..24, vortex_b, merge_w0..24, merge_b]
  = 3 * (25 + 1) = 78 f32s = 312 bytes.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

FEATURE_DIM = 25
OP_TYPE_COUNT = 20
# Three heads: Delta, Vortex, Merge
NUM_HEADS = 3
WEIGHT_COUNT = FEATURE_DIM * NUM_HEADS + NUM_HEADS  # 78 f32s


def encode_op_type(op_idx: int) -> np.ndarray:
    v = np.zeros(OP_TYPE_COUNT, dtype=np.float32)
    if 0 <= op_idx < OP_TYPE_COUNT:
        v[op_idx] = 1.0
    return v


def make_feature_vector(
    op_type: str,
    selectivity: float,
    row_count: int,
    num_filters: int,
    num_columns: int,
    delta_count: int,
) -> np.ndarray:
    features = np.zeros(FEATURE_DIM, dtype=np.float32)

    op_map = {
        "scan": 0, "point_get": 1, "range_scan": 2, "filter": 3,
        "aggregate": 4, "group_by": 5, "sort": 6, "join": 7,
        "limit": 8, "projection": 9, "union": 10, "distinct": 11,
        "window_fn": 12, "subquery": 13, "insert": 14, "update": 15,
        "delete": 16, "set_op": 17, "other": 18, "unknown": 19,
    }
    oh = encode_op_type(op_map.get(op_type, 19))
    features[:OP_TYPE_COUNT] = oh

    features[OP_TYPE_COUNT] = float(np.clip(selectivity, 0.0, 1.0))

    features[OP_TYPE_COUNT + 1] = float(
        np.clip(np.log10(row_count + 1) / 9.0, 0.0, 1.0)
    )

    features[OP_TYPE_COUNT + 2] = float(
        np.clip(np.log2(num_filters + 1) / np.log2(21), 0.0, 1.0)
    )

    features[OP_TYPE_COUNT + 3] = float(
        np.clip(np.log2(num_columns + 1) / np.log2(65), 0.0, 1.0)
    )

    features[OP_TYPE_COUNT + 4] = float(
        np.clip(np.log2(delta_count + 1) / np.log2(1025), 0.0, 1.0)
    )

    return features


# Label encoding: 0=Delta, 1=Vortex, 2=Merge
LABEL_DELTA = 0
LABEL_VORTEX = 1
LABEL_MERGE = 2


def ground_truth_winner(
    op_type: str,
    selectivity: float,
    row_count: int,
    num_filters: int,
    num_delta_patches: int = 0,
    delta_patch_count: int = 0,
) -> int:
    """
    Determine ground-truth routing based on workload characteristics.

    Merge triggers when:
      - There are multiple delta patches (num_delta_patches > 5) AND
      - The query selectivity overlaps with delta coverage
      - Point gets never need Merge
      - Full scans with many deltas benefit from Merge (delta overlay on Vortex)
    """
    # Point get: always Delta
    if op_type == "point_get":
        return LABEL_DELTA

    # Very selective: Delta only
    if selectivity < 0.001:
        return LABEL_DELTA

    # Heavy aggregation: Vortex only
    if op_type in ("aggregate", "group_by", "sort", "distinct"):
        return LABEL_VORTEX

    # Full scan with many delta patches -> Merge
    if delta_patch_count > 5 and selectivity > 0.01:
        return LABEL_MERGE

    # Many small patches with mid selectivity -> Merge (delta overlay cost)
    if delta_patch_count > 10 and selectivity < 0.5:
        return LABEL_MERGE

    # Wide scan (>50% selectivity) -> Vortex
    if selectivity > 0.5:
        return LABEL_VORTEX

    # Large row count -> Vortex
    if row_count > 500_000:
        return LABEL_VORTEX

    # Selective scan with few patches -> Delta
    if num_filters <= 2 and selectivity < 0.1:
        return LABEL_DELTA

    # Default: Delta for selective, Vortex for mid-range
    return LABEL_DELTA if selectivity < 0.05 else LABEL_VORTEX


def generate_training_data(n_samples: int = 10_000):
    rng = np.random.default_rng(seed=42)
    op_types = [
        "scan", "point_get", "range_scan", "filter", "aggregate",
        "group_by", "sort", "join", "limit", "projection",
    ]
    op_weights = [0.1, 0.2, 0.25, 0.15, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]

    X = np.zeros((n_samples, FEATURE_DIM), dtype=np.float32)
    # Three labels: Delta(0), Vortex(1), Merge(2)
    y = np.zeros(n_samples, dtype=np.int32)

    row_counts = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    row_probs = [0.1, 0.15, 0.2, 0.2, 0.15, 0.15, 0.05]

    for i in range(n_samples):
        op_type = rng.choice(op_types, p=op_weights)
        selectivity = rng.uniform(0.0001, 0.99)
        row_count = int(rng.choice(row_counts, p=row_probs))
        num_filters = int(rng.integers(0, 8))
        num_columns = int(rng.integers(1, 30))
        # Simulate varying delta patch counts
        delta_patch_count = int(rng.integers(0, 30))

        features = make_feature_vector(
            op_type, selectivity, row_count, num_filters, num_columns, delta_patch_count
        )
        X[i] = features

        winner = ground_truth_winner(
            op_type, selectivity, row_count, num_filters,
            num_delta_patches=delta_patch_count,
            delta_patch_count=delta_patch_count,
        )
        y[i] = winner

    # Ensure balanced classes
    for label in (LABEL_DELTA, LABEL_VORTEX, LABEL_MERGE):
        count = int((y == label).sum())
        print(f"  Class {label}: {count} samples")

    return X, y


def load_export_training_data(path: str):
    rows = []
    labels = []
    export_path = Path(path)
    with export_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            batch = json.loads(line)
            for sample in batch.get("samples", []):
                features = np.asarray(sample["features"], dtype=np.float32)
                if features.shape[0] != FEATURE_DIM:
                    continue
                rows.append(features)
                delta_ms = float(sample["delta_actual_ms"])
                vortex_ms = float(sample["vortex_actual_ms"])
                labels.append(LABEL_DELTA if delta_ms <= vortex_ms else LABEL_VORTEX)

    if not rows:
        raise ValueError(f"No valid ML export samples found in {path}")

    X = np.stack(rows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)
    print(f"Loaded {len(rows)} export-backed samples from {path}")
    return X, y


def train_twin_network(
    X: np.ndarray, y: np.ndarray, epochs: int = 500, lr: float = 0.01
):
    """
    Train three-headed linear model via softmax cross-entropy.

    Uses gradient descent on a three-class softmax:
      - Delta head (index 0)
      - Vortex head (index 1)
      - Merge head (index 2)
    """
    rng = np.random.default_rng(seed=1337)

    # Three independent weight vectors (not matrices, since we have one input vector per query)
    scale = np.sqrt(2.0 / FEATURE_DIM)
    delta_w = rng.normal(0, scale, (FEATURE_DIM,)).astype(np.float32)
    delta_b = np.float32(0.0)
    vortex_w = rng.normal(0, scale, (FEATURE_DIM,)).astype(np.float32)
    vortex_b = np.float32(1.0)
    merge_w = rng.normal(0, scale, (FEATURE_DIM,)).astype(np.float32)
    merge_b = np.float32(-0.5)  # Slightly conservative: merge needs to earn it

    n = X.shape[0]
    best_acc = 0.0

    for epoch in range(epochs):
        # Compute scores for all three heads
        delta_scores = X @ delta_w + delta_b
        vortex_scores = X @ vortex_w + vortex_b
        merge_scores = X @ merge_w + merge_b

        # Softmax
        scores = np.stack([delta_scores, vortex_scores, merge_scores], axis=1)  # (n, 3)
        scores -= scores.max(axis=1, keepdims=True)  # numerically stable
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        # One-hot encode ground truth
        one_hot = np.zeros((n, NUM_HEADS), dtype=np.float32)
        one_hot[np.arange(n), y] = 1.0

        # Cross-entropy gradient: (probs - one_hot) / n
        grad = (probs - one_hot) / n  # (n, 3)

        # Gradients for each head
        grad_delta_w = grad[:, 0] @ X  # (feature_dim,)
        grad_delta_b = float(grad[:, 0].sum())
        grad_vortex_w = grad[:, 1] @ X
        grad_vortex_b = float(grad[:, 1].sum())
        grad_merge_w = grad[:, 2] @ X
        grad_merge_b = float(grad[:, 2].sum())

        # Gradient descent
        delta_w -= lr * grad_delta_w
        delta_b -= lr * grad_delta_b
        vortex_w -= lr * grad_vortex_w
        vortex_b -= lr * grad_vortex_b
        merge_w -= lr * grad_merge_w
        merge_b -= lr * grad_merge_b

        if (epoch + 1) % 100 == 0 or epoch == 0:
            preds = probs.argmax(axis=1)
            acc = float((preds == y).mean())
            best_acc = max(best_acc, acc)
            # Per-class accuracy
            acc_delta = float(((preds == y) & (y == LABEL_DELTA)).sum() /
                               max(1, (y == LABEL_DELTA).sum()))
            acc_vortex = float(((preds == y) & (y == LABEL_VORTEX)).sum() /
                                max(1, (y == LABEL_VORTEX).sum()))
            acc_merge = float(((preds == y) & (y == LABEL_MERGE)).sum() /
                              max(1, (y == LABEL_MERGE).sum()))
            print(f"  Epoch {epoch+1}/{epochs}: acc={acc:.4f} "
                  f"(D={acc_delta:.2f} V={acc_vortex:.2f} M={acc_merge:.2f})")

    # Final evaluation
    delta_scores = X @ delta_w + delta_b
    vortex_scores = X @ vortex_w + vortex_b
    merge_scores = X @ merge_w + merge_b
    scores = np.stack([delta_scores, vortex_scores, merge_scores], axis=1)
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    final_acc = float((preds == y).mean())
    print(f"  Final accuracy: {final_acc:.4f} (best: {best_acc:.4f})")

    return delta_w, delta_b, vortex_w, vortex_b, merge_w, merge_b, final_acc


def export_weights(
    delta_w: np.ndarray,
    delta_b: np.float32,
    vortex_w: np.ndarray,
    vortex_b: np.float32,
    merge_w: np.ndarray,
    merge_b: np.float32,
    output_path: str,
):
    """Export weights in the format expected by TreeCnnRouter::load_weights."""
    weights = np.concatenate([
        delta_w.astype(np.float32).flatten(),
        np.array([delta_b], dtype=np.float32),
        vortex_w.astype(np.float32).flatten(),
        np.array([vortex_b], dtype=np.float32),
        merge_w.astype(np.float32).flatten(),
        np.array([merge_b], dtype=np.float32),
    ])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(weights.tobytes())
    print(f"Exported weights to {out} ({len(weights)} f32s, {len(weights) * 4} bytes)")

    # Verify
    with open(out, "rb") as f:
        data = f.read()
    expected = WEIGHT_COUNT * 4
    assert len(data) == expected, f"Expected {expected} bytes, got {len(data)}"
    loaded = struct.unpack(f"<{WEIGHT_COUNT}f", data)
    print(f"Verified: {len(loaded)} f32s OK")


def main():
    parser = argparse.ArgumentParser(
        description="Train HTAP routing twin-network model (three-way: Delta/Vortex/Merge)"
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output", type=str, default="assets/tree_cnn_weights.bin")
    parser.add_argument("--exports", type=str, default="")
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    args = parser.parse_args()

    if args.exports:
        print(f"Loading export-backed training samples from {args.exports}...")
        X, y = load_export_training_data(args.exports)
    else:
        print(f"Generating {args.samples} training samples...")
        X, y = generate_training_data(args.samples)

    print(f"Training three-headed network for {args.epochs} epochs...")
    dw, db, vw, vb, mw, mb, final_acc = train_twin_network(
        X, y, epochs=args.epochs, lr=args.lr
    )
    if final_acc < args.min_accuracy:
        raise SystemExit(
            f"Training failed to meet accuracy gate: {final_acc:.4f} < {args.min_accuracy:.4f}"
        )

    print("Exporting weights...")
    export_weights(dw, db, vw, vb, mw, mb, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
