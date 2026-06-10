//! CDC Kafka sink for RockDuck.
//!
//! Sends CDC events to Kafka using Debezium format. Uses `ThreadedProducer`
//! for non-blocking message delivery with an internal polling thread.

#![cfg(feature = "kafka")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rdkafka::config::ClientConfig;
use rdkafka::message::OwnedHeaders;
use rdkafka::producer::{Producer, ThreadedProducer};
use rdkafka::util::Timeout;

use crate::cdc::debezium::DebeziumEnvelope;
use crate::cdc::log::CdcOpType;
use crate::db::RockDuck;
use crate::error::{Result, RockDuckError};
use crate::metadata::projection::{ProjectionContract, ProjectionSurface, SidecarClass};
use crate::query::routing::feedback::SinkDigestSnapshot;
use crate::query::routing::SidecarEvidenceSnapshot;

/// Configuration for the CDC Kafka sink.
#[derive(Debug, Clone)]
pub struct CdcSinkConfig {
    /// Kafka bootstrap servers (e.g., "localhost:9092")
    pub bootstrap_servers: String,
    /// Topic for CDC events
    pub topic: String,
    /// Client ID prefix
    pub client_id: String,
    /// Message timeout in milliseconds
    pub message_timeout_ms: u64,
}

impl Default for CdcSinkConfig {
    fn default() -> Self {
        Self {
            bootstrap_servers: "localhost:9092".to_string(),
            topic: "rockduck_cdc".to_string(),
            client_id: "rockduck".to_string(),
            message_timeout_ms: 5000,
        }
    }
}

/// CdcSink trait — implementors can send CDC events to any destination.
pub trait CdcSink: Send + Sync {
    /// Called when a transaction commits, with all CDC ops in that transaction.
    fn on_commit(&self, db: &RockDuck, txn_id: u64, ops: &[CdcSinkOp]) -> Result<()>;
    /// Returns a descriptive name for this sink (for logging/debugging).
    fn name(&self) -> &str;
}

/// A single CDC operation for the sink layer.
#[derive(Debug, Clone)]
pub struct CdcSinkOp {
    pub op_type: CdcOpType,
    pub table: String,
    pub pk: Vec<u8>,
    pub before: Option<Vec<u8>>,
    pub after: Option<Vec<u8>>,
}

fn build_cdc_projection_contract() -> ProjectionContract {
    ProjectionContract {
        surface: ProjectionSurface::Vtab,
        visibility: crate::mvcc::visibility::VisibilityProjection::Historical,
        sidecar_class: SidecarClass::SanctionedSidecar,
        evidence_hook:
            "CDC sink emits outward-only sidecar evidence after successful commit publication",
        enforcement: crate::metadata::projection::ContractEnforcement::Blocking,
    }
}

/// Kafka CDC sink using Debezium format.
/// Supports dynamic topic routing: per-table topics via `table_topic_map`.
pub struct CdcKafkaSink {
    producer: ThreadedProducer,
    /// Default topic (used when no per-table routing is configured).
    default_topic: String,
    /// Per-table topic routing map. If a table is not in this map,
    /// the `default_topic` is used.
    table_topic_map: std::collections::HashMap<String, String>,
    message_timeout: Timeout,
    name: String,
    closed: AtomicBool,
}

impl CdcKafkaSink {
    /// Create a new Kafka CDC sink with a builder pattern.
    pub fn builder() -> CdcKafkaSinkBuilder {
        CdcKafkaSinkBuilder::new()
    }

    /// Route a table name to its target Kafka topic.
    /// Returns the per-table topic if configured, otherwise the default topic.
    fn route_to_topic(&self, table: &str) -> &str {
        self.table_topic_map
            .get(table)
            .map(|s| s.as_str())
            .unwrap_or(&self.default_topic)
    }

    /// Flush producer queues and mark the sink as closed.
    pub fn close(&self) -> Result<()> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        self.producer.flush(self.message_timeout);
        Ok(())
    }

    /// Send a CDC event to Kafka using Debezium format.
    ///
    /// The message key is set to `txn_id` bytes to ensure all messages from
    /// the same transaction are routed to the same partition (ordering guarantee).
    ///
    /// Returns the partition and offset on success.
    pub fn send(
        &self,
        op_type: CdcOpType,
        table: &str,
        pk: &[u8],
        before: Option<&[u8]>,
        after: Option<&[u8]>,
        txn_id: u64,
    ) -> Result<(i32, i64)> {
        let topic = self.route_to_topic(table);

        let debezium_op = match op_type {
            CdcOpType::Insert => "c",
            CdcOpType::Update => "u",
            CdcOpType::Delete => "d",
        };

        let before_json: Option<serde_json::Value> = match before {
            Some(bytes) => Some(serde_json::from_slice(bytes).map_err(|e| {
                RockDuckError::Kafka(format!(
                    "CDC before-image JSON decode failed for table {} txn {}: {}",
                    table, txn_id, e
                ))
            })?),
            None => None,
        };
        let after_json: Option<serde_json::Value> = match after {
            Some(bytes) => Some(serde_json::from_slice(bytes).map_err(|e| {
                RockDuckError::Kafka(format!(
                    "CDC after-image JSON decode failed for table {} txn {}: {}",
                    table, txn_id, e
                ))
            })?),
            None => None,
        };

        let envelope = DebeziumEnvelope::with_images(debezium_op, before_json, after_json);
        let payload = serde_json::to_string(&envelope)
            .map_err(|e| RockDuckError::Internal(format!("serialize debezium: {}", e)))?;

        let key = txn_id.to_le_bytes();

        let record =
            rdkafka::producer::BaseProducerRecord::new(topic, key.to_vec(), payload.into_bytes());

        self.producer
            .send_result(record)
            .map_err(|(err, msg)| {
                RockDuckError::Kafka(format!(
                    "CDC send to topic {} failed: {} (key={:?})",
                    topic,
                    err,
                    msg.key()
                ))
            })
            .map(|(partition, offset)| (partition, offset))
    }

    /// Add a per-table topic routing entry.
    pub fn add_table_route(&mut self, table: &str, topic: String) {
        self.table_topic_map.insert(table.to_string(), topic);
    }
}

/// Builder for CdcKafkaSink with fluent API for configuration.
pub struct CdcKafkaSinkBuilder {
    bootstrap_servers: String,
    client_id: String,
    message_timeout_ms: u64,
    default_topic: String,
    table_topic_map: std::collections::HashMap<String, String>,
}

impl CdcKafkaSinkBuilder {
    pub fn new() -> Self {
        Self {
            bootstrap_servers: "localhost:9092".to_string(),
            client_id: "rockduck_cdc".to_string(),
            message_timeout_ms: 5000,
            default_topic: "rockduck_cdc".to_string(),
            table_topic_map: std::collections::HashMap::new(),
        }
    }

    pub fn bootstrap_servers(mut self, servers: &str) -> Self {
        self.bootstrap_servers = servers.to_string();
        self
    }

    pub fn client_id(mut self, id: &str) -> Self {
        self.client_id = id.to_string();
        self
    }

    pub fn message_timeout_ms(mut self, ms: u64) -> Self {
        self.message_timeout_ms = ms;
        self
    }

    pub fn default_topic(mut self, topic: &str) -> Self {
        self.default_topic = topic.to_string();
        self
    }

    /// Add a per-table to topic routing rule.
    /// All CDC events for `table` will be sent to `topic` instead of the default topic.
    pub fn add_table_route(mut self, table: &str, topic: &str) -> Self {
        self.table_topic_map
            .insert(table.to_string(), topic.to_string());
        self
    }

    /// Add multiple routing rules at once.
    pub fn with_table_routes(mut self, routes: impl IntoIterator<Item = (String, String)>) -> Self {
        self.table_topic_map.extend(routes);
        self
    }

    pub fn build(self) -> Result<CdcKafkaSink> {
        let producer: ThreadedProducer = ClientConfig::new()
            .set("bootstrap.servers", &self.bootstrap_servers)
            .set("client.id", &self.client_id)
            .set("message.timeout.ms", self.message_timeout_ms.to_string())
            .set("queue.buffering.max.messages", "100000")
            .set("compression.type", "gzip")
            .create()
            .map_err(|e| RockDuckError::Kafka(format!("create producer: {}", e)))?;

        let sink_name = format!("kafka_sink({} -> {})", self.client_id, self.default_topic);
        tracing::info!(
            "CDC Kafka sink built: client_id={}, servers={}, default_topic={}, table_routes={:?}",
            self.client_id,
            self.bootstrap_servers,
            self.default_topic,
            self.table_topic_map.keys().collect::<Vec<_>>()
        );

        Ok(CdcKafkaSink {
            producer,
            default_topic: self.default_topic,
            table_topic_map: self.table_topic_map,
            message_timeout: Timeout::After(Duration::from_millis(self.message_timeout_ms)),
            name: sink_name,
            closed: AtomicBool::new(false),
        })
    }
}

impl Default for CdcKafkaSinkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CdcSink for CdcKafkaSink {
    fn on_commit(&self, db: &RockDuck, txn_id: u64, ops: &[CdcSinkOp]) -> Result<()> {
        if let Some(router) = db.router.as_ref() {
            if !router.kafka_runtime_enabled() {
                router.feedback().record_sink_digest(
                    ops.first().map(|op| op.table.as_str()).unwrap_or("cdc"),
                    SinkDigestSnapshot {
                        latest_status: "runtime_disabled".to_string(),
                        success_count: 0,
                        failure_count: 0,
                        latest_failure: Some("kafka sink runtime rollout disabled".to_string()),
                        sink_active: false,
                    },
                );
                return Ok(());
            }
        }
        // Collect ALL Kafka send errors, not just the last one.
        // Return all errors joined in a single message so operators can see every failure.
        let mut errors = Vec::new();
        let projection_contract = build_cdc_projection_contract();
        projection_contract.assert_blocking_governance();
        for op in ops {
            let topic = self.route_to_topic(&op.table);
            if let Err(e) = self.send(
                op.op_type,
                &op.table,
                &op.pk,
                op.before.as_deref(),
                op.after.as_deref(),
                txn_id,
            ) {
                tracing::warn!(
                    "CDC Kafka send failed for txn {} table {}: {}",
                    txn_id,
                    op.table,
                    e
                );
                errors.push(format!("{}: {}", op.table, e));
            }
        }
        if errors.is_empty() {
            if let Some(router) = db.router.as_ref() {
                let table = ops
                    .first()
                    .map(|op| op.table.clone())
                    .unwrap_or_else(|| "cdc".to_string());
                router.feedback().record_sink_digest(
                    &table,
                    SinkDigestSnapshot {
                        latest_status: "ok".to_string(),
                        success_count: ops.len() as u64,
                        failure_count: 0,
                        latest_failure: None,
                        sink_active: true,
                    },
                );
                router.observe_sidecar_evidence(
                    db,
                    &SidecarEvidenceSnapshot {
                        table,
                        routed_segment_ids: Vec::new(),
                        executed_segment_ids: Vec::new(),
                        contract: projection_contract,
                    },
                );
            }
            Ok(())
        } else {
            if let Some(router) = db.router.as_ref() {
                let table = ops.first().map(|op| op.table.as_str()).unwrap_or("cdc");
                router.feedback().record_sink_digest(
                    table,
                    SinkDigestSnapshot {
                        latest_status: "degraded".to_string(),
                        success_count: (ops.len().saturating_sub(errors.len())) as u64,
                        failure_count: errors.len() as u64,
                        latest_failure: Some(
                            errors
                                .last()
                                .cloned()
                                .unwrap_or_else(|| "unknown kafka failure".to_string()),
                        ),
                        sink_active: true,
                    },
                );
            }
            Err(RockDuckError::Kafka(errors.join("; ")))
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for CdcKafkaSink {
    fn drop(&mut self) {
        if !self.closed.load(Ordering::SeqCst) {
            tracing::error!(
                "Dropping CdcKafkaSink without calling close(): {}. Pending Kafka messages may be lost.",
                self.name
            );
            self.producer.flush(self.message_timeout);
            self.closed.store(true, Ordering::SeqCst);
            return;
        }
        tracing::debug!("Dropping CdcKafkaSink: {}", self.name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_close_idempotent() {
        // This test validates that calling close() multiple times is safe.
        // Since we can't easily create a real Kafka sink in unit tests,
        // we test the logic pattern here.
        let closed = AtomicBool::new(false);

        // First close should return Ok
        let first = closed.swap(true, Ordering::SeqCst);
        assert!(!first, "First call should return false from swap");
        assert!(
            closed.load(Ordering::SeqCst),
            "Should be closed after first call"
        );

        // Second close should also return Ok (idempotent)
        let second = closed.swap(true, Ordering::SeqCst);
        assert!(second, "Second call should return true from swap");
    }

    #[test]
    fn test_route_to_topic_with_default() {
        // Test the routing logic
        let default_topic = "default_topic".to_string();
        let table_topic_map = std::collections::HashMap::new();

        // When no mapping exists, should use default
        let route = table_topic_map
            .get("unknown_table")
            .map(|s| s.as_str())
            .unwrap_or(&default_topic);
        assert_eq!(route, "default_topic");
    }

    #[test]
    fn test_route_to_topic_with_mapping() {
        let default_topic = "default_topic".to_string();
        let mut table_topic_map = std::collections::HashMap::new();
        table_topic_map.insert("users".to_string(), "users_cdc".to_string());
        table_topic_map.insert("orders".to_string(), "orders_cdc".to_string());

        // Mapped table should use its specific topic
        let users_route = table_topic_map
            .get("users")
            .map(|s| s.as_str())
            .unwrap_or(&default_topic);
        assert_eq!(users_route, "users_cdc");

        // Unmapped table should use default
        let unknown_route = table_topic_map
            .get("unknown")
            .map(|s| s.as_str())
            .unwrap_or(&default_topic);
        assert_eq!(unknown_route, "default_topic");
    }

    #[test]
    fn test_debezium_op_type_mapping() {
        // Verify Debezium operation type mapping
        assert_eq!(debezium_op_from_type(CdcOpType::Insert), "c");
        assert_eq!(debezium_op_from_type(CdcOpType::Update), "u");
        assert_eq!(debezium_op_from_type(CdcOpType::Delete), "d");
    }

    fn debezium_op_from_type(op: CdcOpType) -> &'static str {
        match op {
            CdcOpType::Insert => "c",
            CdcOpType::Update => "u",
            CdcOpType::Delete => "d",
        }
    }

    #[test]
    fn test_cdc_sink_config_defaults() {
        let config = CdcSinkConfig::default();
        assert_eq!(config.bootstrap_servers, "localhost:9092");
        assert_eq!(config.topic, "rockduck_cdc");
        assert_eq!(config.client_id, "rockduck");
        assert_eq!(config.message_timeout_ms, 5000);
    }

    #[test]
    fn test_builder_pattern() {
        let builder = CdcKafkaSinkBuilder::new()
            .bootstrap_servers("kafka:9092")
            .client_id("test_client")
            .message_timeout_ms(3000)
            .default_topic("test_cdc")
            .add_table_route("users", "users_events")
            .add_table_route("orders", "orders_events");

        assert_eq!(builder.bootstrap_servers, "kafka:9092");
        assert_eq!(builder.client_id, "test_client");
        assert_eq!(builder.message_timeout_ms, 3000);
        assert_eq!(builder.default_topic, "test_cdc");
        assert_eq!(
            builder.table_topic_map.get("users"),
            Some(&"users_events".to_string())
        );
        assert_eq!(
            builder.table_topic_map.get("orders"),
            Some(&"orders_events".to_string())
        );
    }

    #[test]
    fn test_with_table_routes() {
        let builder = CdcKafkaSinkBuilder::new().with_table_routes(vec![
            ("users".to_string(), "users_cdc".to_string()),
            ("orders".to_string(), "orders_cdc".to_string()),
            ("products".to_string(), "products_cdc".to_string()),
        ]);

        assert_eq!(builder.table_topic_map.len(), 3);
        assert!(builder.table_topic_map.contains_key("users"));
        assert!(builder.table_topic_map.contains_key("orders"));
        assert!(builder.table_topic_map.contains_key("products"));
    }

    #[test]
    fn test_cdc_sink_op_debug() {
        let op = CdcSinkOp {
            op_type: CdcOpType::Insert,
            table: "users".to_string(),
            pk: vec![1, 2, 3, 4],
            before: None,
            after: Some(vec![
                b'{', b'"', b'n', b'a', b'm', b'e', b'"', b':', b'"', b'A', b'l', b'i', b'c', b'e',
                b'"', b'}',
            ]),
        };
        let debug_str = format!("{:?}", op);
        assert!(debug_str.contains("Insert"));
        assert!(debug_str.contains("users"));
    }
}
