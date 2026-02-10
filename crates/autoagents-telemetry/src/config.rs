use autoagents_protocol::RuntimeID;
use std::collections::HashMap;

/// Top-level telemetry configuration applied when a tracer starts.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub service_name: String,
    pub service_version: Option<String>,
    pub environment: Option<String>,
    pub runtime_id: Option<RuntimeID>,
    pub exporter: ExporterConfig,
    pub span_batch: SpanBatchConfig,
    pub redaction: RedactionConfig,
    pub metrics_enabled: bool,
    pub install_tracing_subscriber: bool,
}

impl TelemetryConfig {
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            service_version: None,
            environment: None,
            runtime_id: None,
            exporter: ExporterConfig::default(),
            span_batch: SpanBatchConfig::default(),
            redaction: RedactionConfig::default(),
            metrics_enabled: true,
            install_tracing_subscriber: true,
        }
    }

    pub fn with_runtime_id(mut self, runtime_id: RuntimeID) -> Self {
        self.runtime_id = Some(runtime_id);
        self
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            service_name: "autoagents".into(),
            service_version: None,
            environment: None,
            runtime_id: None,
            exporter: ExporterConfig::default(),
            span_batch: SpanBatchConfig::default(),
            redaction: RedactionConfig::default(),
            metrics_enabled: true,
            install_tracing_subscriber: true,
        }
    }
}

/// Exporter wiring for spans/metrics.
#[derive(Debug, Clone, Default)]
pub struct ExporterConfig {
    pub otlp: Option<OtlpConfig>,
    pub stdout: bool,
}

/// Span batcher configuration to avoid per-span exports.
#[derive(Debug, Clone)]
pub struct SpanBatchConfig {
    pub max_queue_size: usize,
    pub max_export_batch_size: usize,
    pub scheduled_delay: std::time::Duration,
    pub max_export_timeout: std::time::Duration,
    pub max_concurrent_exports: usize,
}

impl Default for SpanBatchConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 2048,
            max_export_batch_size: 512,
            scheduled_delay: std::time::Duration::from_secs(5),
            max_export_timeout: std::time::Duration::from_secs(30),
            max_concurrent_exports: 1,
        }
    }
}

/// OTLP endpoint and transport details.
#[derive(Debug, Clone)]
pub struct OtlpConfig {
    pub endpoint: Option<String>,
    pub protocol: OtlpProtocol,
    pub headers: HashMap<String, String>,
    pub debug_http: bool,
}

impl OtlpConfig {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: Some(endpoint.into()),
            protocol: OtlpProtocol::HttpBinary,
            headers: HashMap::new(),
            debug_http: false,
        }
    }
}

impl Default for OtlpConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            protocol: OtlpProtocol::HttpBinary,
            headers: HashMap::new(),
            debug_http: false,
        }
    }
}

/// Supported OTLP wire protocols.
#[derive(Debug, Clone, Copy)]
pub enum OtlpProtocol {
    HttpBinary,
    HttpJson,
}

/// Redaction flags for telemetry payloads.
#[derive(Debug, Clone, Default)]
pub struct RedactionConfig {
    pub redact_task_inputs: bool,
    pub redact_task_outputs: bool,
    pub redact_tool_arguments: bool,
    pub redact_tool_results: bool,
}
