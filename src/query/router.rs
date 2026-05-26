//! ReadPath 路由决策器
//!
//! 根据查询类型和数据状态选择最优读取路径：
//! - DeltaStoreOnly: 纯点查 → DeltaStore（最新数据）
//! - VortexOnly: 全表聚合 → Vortex（历史数据）
//! - Merge: 有点查有扫描 → DeltaStore overlay + Vortex

use std::collections::HashMap;
use crate::segment::delta_store::CellDelta;

/// 查询类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// 点查：获取单条记录
    PointGet,
    /// 范围扫描：获取一段记录
    RangeScan,
    /// 全表扫描：无范围限制的扫描
    FullScan,
    /// 聚合查询：COUNT/SUM/AVG 等
    Aggregate,
}

impl QueryType {
    /// 从字符串推断查询类型
    pub fn from_str(s: &str) -> Self {
        match s {
            "point_get" | "pointget" | "get" => QueryType::PointGet,
            "range_scan" | "rangescan" | "scan" | "range" => QueryType::RangeScan,
            "full_scan" | "fullscan" | "select" => QueryType::FullScan,
            "aggregate" | "count" | "sum" | "avg" | "min" | "max" => QueryType::Aggregate,
            _ => QueryType::RangeScan,
        }
    }
}

/// 读取路径
#[derive(Debug, Clone)]
pub enum ReadPath {
    /// 纯点查 → 仅读 DeltaStore
    DeltaStoreOnly {
        deltas: HashMap<String, HashMap<u64, CellDelta>>,
    },
    /// 全表聚合 → 仅读 Vortex
    VortexOnly,
    /// 有点查有扫描 → DeltaStore overlay + Vortex
    Merge {
        deltas: HashMap<String, HashMap<u64, CellDelta>>,
        vortex_row_count: u64,
    },
}

/// 路由决策参数
pub struct RouterParams {
    /// 查询类型
    pub query_type: QueryType,
    /// 是否有过滤器
    pub has_filter: bool,
    /// 过滤器选择率（0.0 - 1.0，来自 adaptive_lm 估算）
    pub filter_selectivity: f64,
    /// DeltaStore 是否非空
    pub has_updates: bool,
    /// DeltaStore 中的 deltas 数量
    pub delta_count: usize,
    /// 查询的行数范围（用于 RangeScan）
    pub row_range_size: Option<u64>,
}

impl Default for RouterParams {
    fn default() -> Self {
        Self {
            query_type: QueryType::RangeScan,
            has_filter: false,
            filter_selectivity: 1.0,
            has_updates: false,
            delta_count: 0,
            row_range_size: None,
        }
    }
}

/// 选择最优读取路径
pub fn choose_read_path(params: &RouterParams) -> ReadPath {
    let RouterParams {
        query_type,
        has_filter,
        filter_selectivity,
        has_updates,
        delta_count,
        row_range_size,
    } = *params;

    // 没有 updates，全部走 Vortex
    if !has_updates || delta_count == 0 {
        return ReadPath::VortexOnly;
    }

    // 点查：优先 DeltaStore
    if query_type == QueryType::PointGet {
        return ReadPath::DeltaStoreOnly {
            deltas: HashMap::new(), // caller should populate via delta_mgr.get_all_visible_deltas()
        };
    }

    // 全表扫描 + 无过滤 + 无 updates
    if query_type == QueryType::FullScan && !has_filter && !has_updates {
        return ReadPath::VortexOnly;
    }

    // 全表扫描 + 无过滤 + 有 updates → Merge
    if query_type == QueryType::FullScan && !has_filter && has_updates {
        return ReadPath::Merge {
            deltas: HashMap::new(),
            vortex_row_count: u64::MAX,
        };
    }

    // 聚合查询 + 高选择率（扫描大量数据）→ Vortex
    if query_type == QueryType::Aggregate && filter_selectivity > 0.1 {
        return ReadPath::VortexOnly;
    }

    // 聚合查询 + 低选择率（只查少量数据）→ DeltaStore
    if query_type == QueryType::Aggregate && filter_selectivity <= 0.1 {
        return ReadPath::DeltaStoreOnly {
            deltas: HashMap::new(),
        };
    }

    // 范围扫描：基于 selectivity 和 delta_count 决策
    if query_type == QueryType::RangeScan {
        // 高选择性（只查 1% 的数据）+ delta 少 → DeltaStore
        if filter_selectivity < 0.01 && delta_count < 100 {
            return ReadPath::DeltaStoreOnly {
                deltas: HashMap::new(),
            };
        }

        // 低选择性（扫描大量数据）+ delta 多 → Merge
        if filter_selectivity > 0.5 || delta_count > 1000 {
            let row_count = row_range_size.unwrap_or(u64::MAX);
            return ReadPath::Merge {
                deltas: HashMap::new(),
                vortex_row_count: row_count,
            };
        }
    }

    // 默认：Merge 路径（DeltaStore overlay + Vortex）
    ReadPath::Merge {
        deltas: HashMap::new(),
        vortex_row_count: row_range_size.unwrap_or(1_000_000),
    }
}

/// 估算过滤器选择率（启发式方法）
///
/// 当没有 adaptive_lm 时，使用简单的启发式估算：
/// - 有比较运算符（<, >, <=, >=）且范围窄 → 低选择性
/// - 有相等运算符（=, IN）→ 低/中等选择性
/// - 无过滤器 → 1.0（全部）
pub fn estimate_selectivity(filter: Option<&str>) -> f64 {
    match filter {
        None => 1.0,
        Some(f) => {
            let f_lower = f.to_lowercase();
            // 启发式：简单的范围估算
            if f_lower.contains(">=") || f_lower.contains("<=") || f_lower.contains(">") || f_lower.contains("<") {
                0.1 // 假设范围查询选择率约 10%
            } else if f_lower.contains("=") {
                0.01 // 假设等值查询选择率约 1%
            } else {
                0.5 // 保守估计
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_type_from_str() {
        assert_eq!(QueryType::from_str("point_get"), QueryType::PointGet);
        assert_eq!(QueryType::from_str("range_scan"), QueryType::RangeScan);
        assert_eq!(QueryType::from_str("full_scan"), QueryType::FullScan);
        assert_eq!(QueryType::from_str("aggregate"), QueryType::Aggregate);
        assert_eq!(QueryType::from_str("unknown"), QueryType::RangeScan);
    }

    #[test]
    fn test_choose_read_path_no_updates() {
        let params = RouterParams {
            query_type: QueryType::RangeScan,
            has_filter: true,
            filter_selectivity: 0.5,
            has_updates: false,
            delta_count: 0,
            row_range_size: None,
        };
        assert!(matches!(choose_read_path(&params), ReadPath::VortexOnly));
    }

    #[test]
    fn test_choose_read_path_point_get() {
        let params = RouterParams {
            query_type: QueryType::PointGet,
            has_filter: false,
            filter_selectivity: 1.0,
            has_updates: true,
            delta_count: 5,
            row_range_size: None,
        };
        assert!(matches!(choose_read_path(&params), ReadPath::DeltaStoreOnly { .. }));
    }

    #[test]
    fn test_choose_read_path_full_scan_with_updates() {
        let params = RouterParams {
            query_type: QueryType::FullScan,
            has_filter: false,
            filter_selectivity: 1.0,
            has_updates: true,
            delta_count: 100,
            row_range_size: None,
        };
        assert!(matches!(choose_read_path(&params), ReadPath::Merge { .. }));
    }

    #[test]
    fn test_choose_read_path_aggregate_high_selectivity() {
        let params = RouterParams {
            query_type: QueryType::Aggregate,
            has_filter: true,
            filter_selectivity: 0.5,
            has_updates: true,
            delta_count: 50,
            row_range_size: None,
        };
        assert!(matches!(choose_read_path(&params), ReadPath::VortexOnly));
    }

    #[test]
    fn test_choose_read_path_aggregate_low_selectivity() {
        let params = RouterParams {
            query_type: QueryType::Aggregate,
            has_filter: true,
            filter_selectivity: 0.05,
            has_updates: true,
            delta_count: 50,
            row_range_size: None,
        };
        assert!(matches!(choose_read_path(&params), ReadPath::DeltaStoreOnly { .. }));
    }

    #[test]
    fn test_choose_read_path_range_scan_low_selectivity_few_deltas() {
        let params = RouterParams {
            query_type: QueryType::RangeScan,
            has_filter: true,
            filter_selectivity: 0.005,
            has_updates: true,
            delta_count: 50,
            row_range_size: None,
        };
        assert!(matches!(choose_read_path(&params), ReadPath::DeltaStoreOnly { .. }));
    }

    #[test]
    fn test_choose_read_path_range_scan_high_selectivity() {
        let params = RouterParams {
            query_type: QueryType::RangeScan,
            has_filter: true,
            filter_selectivity: 0.8,
            has_updates: true,
            delta_count: 200,
            row_range_size: None,
        };
        assert!(matches!(choose_read_path(&params), ReadPath::Merge { .. }));
    }

    #[test]
    fn test_choose_read_path_range_scan_many_deltas() {
        let params = RouterParams {
            query_type: QueryType::RangeScan,
            has_filter: true,
            filter_selectivity: 0.3,
            has_updates: true,
            delta_count: 2000,
            row_range_size: None,
        };
        assert!(matches!(choose_read_path(&params), ReadPath::Merge { .. }));
    }

    #[test]
    fn test_estimate_selectivity_no_filter() {
        assert!((estimate_selectivity(None) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_selectivity_range() {
        let s = estimate_selectivity(Some("age > 30"));
        assert!((s - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_selectivity_eq() {
        let s = estimate_selectivity(Some("id = 5"));
        assert!((s - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_selectivity_no_op() {
        let s = estimate_selectivity(Some("status"));
        assert!((s - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_read_path_debug() {
        let rp = ReadPath::DeltaStoreOnly { deltas: HashMap::new() };
        let debug_str = format!("{:?}", rp);
        assert!(!debug_str.is_empty());

        let rp = ReadPath::VortexOnly;
        let debug_str = format!("{:?}", rp);
        assert!(!debug_str.is_empty());

        let rp = ReadPath::Merge { deltas: HashMap::new(), vortex_row_count: 1000 };
        let debug_str = format!("{:?}", rp);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_router_params_default() {
        let params = RouterParams::default();
        assert_eq!(params.query_type, QueryType::RangeScan);
        assert!(!params.has_filter);
        assert!((params.filter_selectivity - 1.0).abs() < f64::EPSILON);
        assert!(!params.has_updates);
        assert_eq!(params.delta_count, 0);
        assert!(params.row_range_size.is_none());
    }

    #[test]
    fn test_full_scan_no_filter_no_updates() {
        let params = RouterParams {
            query_type: QueryType::FullScan,
            has_filter: false,
            filter_selectivity: 1.0,
            has_updates: false,
            delta_count: 0,
            row_range_size: None,
        };
        assert!(matches!(choose_read_path(&params), ReadPath::VortexOnly));
    }

    // ============================================================
    // Merge path data-correctness: verify vortex_row_count propagation
    // ============================================================

    #[test]
    fn test_merge_path_propagates_vortex_row_count() {
        let params = RouterParams {
            query_type: QueryType::RangeScan,
            has_filter: true,
            filter_selectivity: 0.3,
            has_updates: true,
            delta_count: 2000,
            row_range_size: Some(500_000),
        };
        let path = choose_read_path(&params);
        match path {
            ReadPath::Merge { vortex_row_count, .. } => {
                assert_eq!(
                    vortex_row_count, 500_000,
                    "Merge must propagate row_range_size to vortex_row_count"
                );
            }
            other => panic!("Expected Merge, got {:?}", other),
        }
    }

    #[test]
    fn test_merge_path_default_row_count() {
        // Use params that fall through to the default Merge case (line 150):
        // RangeScan: filter_selectivity <= 0.5 AND delta_count <= 1000 → skip line 140
        let params = RouterParams {
            query_type: QueryType::RangeScan,
            has_filter: true,
            filter_selectivity: 0.3,
            has_updates: true,
            delta_count: 500,  // <= 1000 to skip the early Merge branch
            row_range_size: None,
        };
        let path = choose_read_path(&params);
        match path {
            ReadPath::Merge { vortex_row_count, .. } => {
                assert_eq!(
                    vortex_row_count, 1_000_000,
                    "Merge should default to 1_000_000 when row_range_size is None"
                );
            }
            other => panic!("Expected Merge, got {:?}", other),
        }
    }

    // ============================================================
    // VortexOnly and DeltaStoreOnly data-correctness
    // ============================================================

    #[test]
    fn test_vortex_only_excludes_updates() {
        let params = RouterParams {
            query_type: QueryType::FullScan,
            has_filter: false,
            filter_selectivity: 1.0,
            has_updates: false,
            delta_count: 0,
            row_range_size: None,
        };
        match choose_read_path(&params) {
            ReadPath::VortexOnly => {}
            other => panic!("Expected VortexOnly when has_updates=false, got {:?}", other),
        }

        // has_updates=true but delta_count=0 also routes to VortexOnly
        let params2 = RouterParams {
            query_type: QueryType::FullScan,
            has_filter: false,
            filter_selectivity: 1.0,
            has_updates: true,
            delta_count: 0,
            row_range_size: None,
        };
        match choose_read_path(&params2) {
            ReadPath::VortexOnly => {}
            other => panic!("Expected VortexOnly when delta_count=0, got {:?}", other),
        }
    }

    #[test]
    fn test_delta_store_only_returns_empty_deltas() {
        let params = RouterParams {
            query_type: QueryType::PointGet,
            has_filter: false,
            filter_selectivity: 1.0,
            has_updates: true,
            delta_count: 10,
            row_range_size: Some(1),
        };
        match choose_read_path(&params) {
            ReadPath::DeltaStoreOnly { deltas } => {
                assert!(
                    deltas.is_empty(),
                    "DeltaStoreOnly must return empty deltas map (caller populates it), got {} entries",
                    deltas.len()
                );
            }
            other => panic!("Expected DeltaStoreOnly for PointGet with updates, got {:?}", other),
        }
    }
}
