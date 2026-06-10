# RockDuck Constitution

**Version**: 2.0
**Date**: 2026-06-07
**Status**: Draft — baseline reset for v2 ascension execution

This document is the supreme law for RockDuck. Every developer, every review, every
implementation must pass through these gates before landing.

---

## Preamble

RockDuck is an HTAP embedded database. It is built on three layers:

1. **Truth layer** — durability and consistency (WAL + MVCC + Checkpoint)
2. **Evidence layer** — metadata and adaptive routing (zone maps, selectivity, feedback)
3. **Maintenance layer** — storage optimization (compaction, rewrite, debt management)

The separation is not architectural convenience — it is the contract between layers.
**Evidence cannot contradict Truth. Maintenance cannot corrupt Truth.**

This constitution now distinguishes two views:

- **CurrentReality**: what the codebase verifiably does today.
- **AscensionTarget**: the constraints the codebase is moving toward, which are not yet
  automatically assumed to be complete.

No transition-state implementation should be ratified as if it were already the final form.

---

## Article I — Truth Plane

### Section 1.1: CurrentReality — Visibility Authority

The primary visibility authority is still expressed through the `VisFilter` trait (`src/mvcc/visibility.rs`).

Every visibility decision in the codebase is expected to flow through one of these currently verified surfaces:

| Surface | File | CurrentReality | Notes |
|---------|------|----------------|-------|
| ScanIterator | `src/read/scan.rs` | Verified main path | Calls `VisibilityManager::is_visible` via `TxnSnapshot::is_row_visible` |
| point_get | `src/read/point_get.rs` | Verified bypass | Calls `VisibilityManager::is_visible` for current reads |
| point_get_as_of | `src/read/point_get.rs` | Verified bypass | Calls `HistoricalVisibility::is_visible_at` via sanctioned historical projection |
| VTab filter | `src/query/vtab_quack.rs` | Verified bypass | Calls `TxnSnapshot::is_row_visible` directly |
| Compaction | `src/compaction/pdt_merge.rs` | Verified rewrite surface | Calls `SegmentOverlay::is_row_visible` via `compaction_overlay_filter` |

No other surface may make a visibility decision without being classified.

### Section 1.2: CurrentReality — Transition Exceptions

`HistoricalVisibility::is_visible_at` (`src/query/time_travel_impl.rs`) is a **transition-state exception** to Section 1.1. It uses a synthetic `commit_ts_map` (txn_id → txn_id) rather
than real wall-clock commit timestamps. This exception is tolerated only under these currently verified constraints:

1. The `committed_txns` set is filtered at construction to only include transactions with
   `commit_ts <= target_txn` (verified in `TimeTravelReader::new` and `TimeTravelScanner::new`).
2. The `get_as_of` fallback path in `point_get.rs` uses real `commit_ts` via
   `vis_mgr.get_commit_ts()` for delta cell filtering.
3. This exception is not the target architecture. Any future change that adds wall-clock commit timestamps must first replace or re-project this surface.

### Section 1.3: CurrentReality — Recovery Protocol

Crash recovery follows this authoritative sequence (see `RecoveryVerificationCard.recovery_order`):

```
Step 1: CheckpointManager::load_latest → CheckpointMvccState
Step 2: metadata::get_committed_txn(kv) → baseline committed_txn
Step 3: replay_wal_ops → RecoveryResult (authoritative for overlapping txn_ids)
Step 4: committed_txn = max(KV baseline, WAL result)
Step 5: if WAL higher → metadata::put_committed_txn (D8extra fix)
Step 6: commit_ts_map = KV baseline → WAL overlay (WAL wins for overlapping)
Step 7: active_txns = KV baseline → WAL overlay (WAL deduplication)
Step 8: recover_committed_history_with_config → VisibilityManager
Final: VisibilityManager is durable truth
```

### Section 1.4: Failure Taxonomy

Every failure mode falls into exactly one category:

| Category | Meaning | Action |
|----------|---------|--------|
| **Truth-safe** | Error contained, no corruption, correctness preserved | Skip affected txn, continue recovery |
| **Evidence-stale** | Some state may be partially applied, evidence potentially stale | Continue, flag as stale, may need re-verification |
| **Fail-stop** | Data loss risk, manual intervention required | Abort recovery, surface error to operator |

Unclassified errors default to **fail-stop**.

---

## Article II — Evidence Plane

### Section 2.1: Evidence Tier Classification

All metadata falls into three tiers with distinct freshness contracts:

| Tier | Name | Freshness | Examples |
|------|------|-----------|---------|
| T1 | Datasource of Truth | Durable, WAL-replayed | `CF_MVCC` committed/active txns, `CF_PK_IDX` |
| T2 | Inference Evidence | Updated on events | `CF_STAT` table_stats, `CF_ZONE` zone maps, `CF_SEG_META` |
| T3 | Cache / Shadow | Recomputed on demand | `CF_VERSIONS` time-travel index, `CF_BF` bloom filters, `CF_DELTA` delta patches |

Evidence from a lower tier must never be treated as authoritative for a higher tier.

### Section 2.2: CurrentReality — Routing Contract

`QueryRouter` and `route_table_segments` are the current verified surfaces for adaptive read routing.
`RouteDecision` is already consumed as an **execution-template selector** for the main scan path:
`DeltaStoreOnly`, `VortexOnly`, and `Merge` map to distinct read templates in `scan.rs`, while
segment routing and zone-map pruning still shape the candidate set before execution.

This is still a bounded current-reality statement, not a declaration that the final cooperative execution target is complete.

### Section 2.3: CurrentReality — Feedback Surface

| Surface | CurrentReality | Requirement |
|---------|----------------|------------|
| `record_feedback` (selectivity) | Active with routed-candidate denominator | Uses real rows returned over routed candidate rows (`candidate_rows` / `total_segment_rows`), not post-visibility work or full-table cardinality |
| `record_regret_feedback` | Implemented as proxy instrumentation | Uses historical avg as comparison, not concurrent dual-path |
| `record_shadow_timing` | Observation-only | Same-query shadow timing samples may be stored for analysis but must not mutate routing authority |
| `record_ml_feedback` | Stub / intentionally ephemeral | No callers; samples remain memory-only until durability and replay contract are ratified |

### Section 2.4: AscensionTarget — Evidence Contract

The target state is:

1. `RouteDecision` selects an execution template, not only a segment filter.
2. Feedback is recorded at execution completion using real runtime outcomes.
3. Lower-tier evidence never silently upgrades itself into truth semantics.

---

## Article III — Maintenance Plane

### Section 3.1: CurrentReality — Three-Layer Model

Maintenance signals are currently classified into three layers:

| Layer | Description | Commitment | Example |
|-------|-------------|-----------|---------|
| Debt signal | Directly represents storage/read debt | Semantic: these ARE the debt | `del_ratio`, `staleness_penalty`, `miss_penalty` |
| Heuristic factor | Approximation for compaction benefit | Heuristic: reasonable proxy, not debt itself | `size_score`, `age_score` |
| Tuning coefficient | Tunable knob for weight calibration | Arbitrary: hill-climbing target, not semantic debt | `del_coef`, `stale_coef`, `miss_coef` |

All three layers are currently mixed into a single `f64` score. This is acceptable as current scheduling reality but does NOT constitute the final typed maintenance model.

RockDuck currently also exposes an additive typed debt classification layer via `DebtFlags`.
`DebtFlags` does **not** replace the scalar priority score. Instead, it makes distinct debt
surfaces explicit for governance and future design, while current dispatch remains driven by
scalar/runtime gates rather than typed debt classification.

### Section 3.2: CurrentReality — Rewrite Action Inventory

| Action | Implementation | Physical Effect | Budget Gate |
|--------|--------------|----------------|------------|
| PDT merge | Production | Filter deleted rows, rewrite as new segment | `min_del_ratio` threshold |
| Small file merge | Production wrapper | Sequential PDT merges | No separate budget |
| Query-driven | Stub / conditionally wired transition path | AccessTracker-based sort | Not authoritative in default runtime |

PDT merge is the only production rewrite that changes physical layout today.

### Section 3.3: AscensionTarget — Maintenance Contract

The target state is:

1. rewrite / flush / metadata refresh share a common maintenance language and budget model;
2. checkpoint participates in coordination and evidence, but remains a truth-plane privileged operation;
3. typed debt becomes an explicit dispatch input, not only a governance annotation.

---

## Article IV — Governance

### Section 4.1: CurrentReality — Admission Gates

Any change touching routing, compaction, or data layout must pass these gates:

| Gate | Criterion | Blocking |
|------|----------|---------|
| **G1** | Sanctioned bypass classification (SANCTIONED or UNSANCTIONED documented) | UNSANCTIONED gaps block |
| **G2** | Truth package integrity (VisFilter delegation or classified transition exception) | Unclassified visibility surfaces block |
| **G3** | Recovery boundary (new state is recoverable or recovery protocol is extended) | Unrecoverable new state blocks |
| **G4** | Maintenance signal classification (debt signal / heuristic / new) | Unclassified signals block |
| **G5** | Feedback surface verification (stub has at least one caller) | Stub without callers block |
| **G6** | Sidecar classification (Core vs Optional) | Misclassified sidecars block |

### Section 4.2: CurrentReality — Sidecar Classification

| Sidecar | Classification | Requirement |
|---------|---------------|-------------|
| DuckDB VTab | Core | Must pass VisFilter contract; regressions block DB open |
| CDC | Optional | May degrade gracefully; failures must not propagate to core |
| Iceberg export | Optional (feature-gated) | Disabled by default; spec violations documented in lib.rs |
| ML routing | Optional | Stub until trained model exists |

### Section 4.3: CurrentReality — Governance Posture

#### Lightweight Governance Glossary

- **sanctioned bypass** — a path that intentionally does not follow the main surface, but is explicitly classified, bounded, and documented.
- **transition exception** — a temporary, explicitly visible semantic gap tolerated only with stated exit conditions and no hidden authority upgrade.
- **proxy instrumentation** — a metric derived from historical or indirect evidence that is useful for observation but not sufficient to control authority.
- **stub** — a declared but not truly wired feature surface; it may exist in code but has no production-closing caller path.
- **intentionally ephemeral** — a stub or sample surface that is explicitly kept memory-only and excluded from durable authority paths until purpose, replay semantics, and evaluation discipline are ratified.
- **conditionally wired** — a surface whose callback/executor path exists structurally but is not live in the default runtime or all declared deployments.
- **non-authoritative** — evidence, sidecar, or governance output that may inform humans or future design but must not override truth-plane or active decision authority.
- **observation-only** — instrumentation that may record or persist measurements, but is forbidden from mutating routing, maintenance, or recovery decisions.
- **sanctioned bypass classification** — the admission-gate act of marking a path as SANCTIONED or UNSANCTIONED before it can be relied on.

Current governance is an institutionalized **hint layer** (verification cards + tracing debug),
not yet a full interception layer.

### Section 4.4: AscensionTarget — Enforcement Posture

The target state is layered enforcement:

1. tests and CI enforce classified surfaces;
2. structural/static checks block new unclassified bypasses;
3. a small set of runtime invariants protect correctness-critical truth and recovery seams.

---

## Article V — Implementation Rhythm

### Section 5.1: CurrentReality — Baseline, Not Ratified End-State

The following items are current verified capabilities or partial infrastructure, not proof that the v2 ascension target is already complete:

| Capability | CurrentReality |
|------------|----------------|
| Truth plane baseline | WAL + checkpoint + MVCC recovery chain exists |
| Evidence plane baseline | Router, feedback state, and metadata evidence surfaces exist |
| Maintenance baseline | Adaptive scheduler, PDT merge, and flush infrastructure exist |
| Governance baseline | Verification cards and tracing-based hint layer exist |
| Regret / typed debt scaffolding | Additive support exists, but not yet the final closed-loop architecture |

### Section 5.2: Rhythm Rules

1. Never treat a current baseline as if it already satisfies the ascension target.
2. Gaps block, not defer. A classified transition exception is visible debt; an unclassified gap is a blocker.
3. Transition exceptions are time-limited and should be converged into projections or removed.
4. Stub features must remain visible stubs with explicit tracking and may not silently acquire authority.

---

## Article VI — Amendments

This constitution may be amended by:

1. A design document describing the change and its rationale
2. Confirmation that all affected packages pass the admission gates
3. A PR that updates this document and all affected code
4. Code review sign-off from two maintainers

No exception to Articles I through IV may be added without a ratified amendment.
