//! Compaction module for RockDuck
//!
//! Provides compaction functionality including:
//! - Adaptive compaction
//! - Non-blocking compaction
//! - PDT merge
//! - Query-driven compaction (RangeReduce)
//! - Hybrid Compaction Engine: Layer 1 (TRIAD) / Layer 2 (SILK) / Layer 3 (RangeReduce) / Layer 4 (EcoTune)
//!
//! # Hybrid Compaction Engine Layers
//!
//! - **Layer 1 (TRIAD, VLDB 2022)**: `WriteHeatTracker` — hot/cold key separation at WAL flush.
//! - **Layer 2 (SILK, ATC 2019)**: `CompactionIOScheduler` — I/O scheduling with foreground coordination.
//! - **Layer 3 (RangeReduce, ICDE 2026)**: `AccessTracker` — query-driven compaction via scan access tracking.
//! - **Layer 4 (EcoTune, SIGMOD 2025)**: `EcoTune` — dynamic compaction policy selection via DP.
//!
//! ## Level Classification Stability (CMP-06)
//!
//! Segment "levels" are determined by the scheduler's adaptive feedback. Over time, hot segments
//! may be compacted more frequently (and thus promoted), while cold segments may be demoted.
//! This is expected behavior — the hybrid engine continuously reclassifies segments based on
//! access patterns. However, if the `AdaptiveCompactionScheduler`'s scoring diverges from the
//! physical level (e.g., due to stale metrics after a crash), segments may be incorrectly
//! prioritized. The fix is ongoing: `CompactionIOScheduler` uses timestamp-based L1/L2 promotion
//! thresholds that are independent of the adaptive score. Monitoring `CompactionMetrics`
//! (del_ratio, read_amp, write_amp) for anomalies is recommended.
//!
//! # TODO[LLM-TUNE]: Layer 5 — LLM-assisted Compaction Weight Tuning
//!
//! Future direction: use on-device small language models (TinyLlama / Qwen2.5-0.5B)
//! for real-time parameter tuning of compaction policies.
//!
//! Input: `CompactionMetrics { del_ratio, read_amp, write_amp, p99_latency_ms }`
//!
//! Output: `CompactionWeights` incremental adjustment recommendations
//!
//! Reference: arXiv:2602.12669 (ICLR/LLM-on-device for LSM compaction)
//!
//! Prerequisites: Layer 4 output must provide a stable metrics collection system
//!
//! Risk: Inference latency must be < 10ms to meet real-time requirements

pub mod access_tracker; // RangeReduce: granule-level access tracking (ICDE 2026)
pub mod adaptive;
pub mod ecotune; // EcoTune: dynamic compaction policy selection (SIGMOD 2025)
pub mod io_scheduler; // SILK: I/O scheduling and rate limiting (ATC 2019)
pub mod nonblocking;
pub mod pdt_merge;
pub mod phase;
pub mod scheduler; // Priority queue + RangeReduce task selection

pub use access_tracker::*;
pub use adaptive::*;
pub use ecotune::*;
pub use io_scheduler::*;
pub use nonblocking::*;
pub use pdt_merge::*;
pub use phase::*;
pub use scheduler::*;
