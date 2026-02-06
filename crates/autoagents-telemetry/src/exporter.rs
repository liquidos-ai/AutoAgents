use crate::config::{OtlpConfig, OtlpProtocol, TelemetryConfig};
use opentelemetry::KeyValue;
#[cfg(not(target_arch = "wasm32"))]
use opentelemetry_http::{Bytes, HttpClient, HttpError, Request, Response};
use opentelemetry_otlp::{
    MetricExporter, SpanExporter as OtlpSpanExporter, WithExportConfig, WithHttpConfig,
};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::metrics::Temporality;
use opentelemetry_sdk::trace::{SpanData, SpanExporter};
#[cfg(not(target_arch = "wasm32"))]
use reqwest::Client as ReqwestClient;
#[cfg(not(target_arch = "wasm32"))]
use std::fmt;
use std::time::Duration;

/// Fan-out exporter that forwards span batches to all configured backends.
#[derive(Debug)]
pub(crate) struct MultiSpanExporter {
    exporters: Vec<SpanExporterWrapper>,
}

impl MultiSpanExporter {
    pub(crate) fn new(exporters: Vec<SpanExporterWrapper>) -> Self {
        Self { exporters }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.exporters.is_empty()
    }
}

#[derive(Debug)]
pub(crate) enum SpanExporterWrapper {
    Otlp(Box<OtlpSpanExporter>),
    Stdout(StdoutSpanExporter),
}

impl SpanExporterWrapper {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        match self {
            SpanExporterWrapper::Otlp(exporter) => exporter.export(batch).await,
            SpanExporterWrapper::Stdout(exporter) => exporter.export(batch).await,
        }
    }

    fn force_flush(&mut self) -> OTelSdkResult {
        match self {
            SpanExporterWrapper::Otlp(exporter) => exporter.force_flush(),
            SpanExporterWrapper::Stdout(exporter) => exporter.force_flush(),
        }
    }

    fn shutdown_with_timeout(&mut self, timeout: Duration) -> OTelSdkResult {
        match self {
            SpanExporterWrapper::Otlp(exporter) => exporter.shutdown_with_timeout(timeout),
            SpanExporterWrapper::Stdout(exporter) => exporter.shutdown_with_timeout(timeout),
        }
    }

    fn set_resource(&mut self, resource: &Resource) {
        match self {
            SpanExporterWrapper::Otlp(exporter) => exporter.set_resource(resource),
            SpanExporterWrapper::Stdout(exporter) => exporter.set_resource(resource),
        }
    }
}

// Wrapper that logs OTLP HTTP responses for debugging.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct LoggingHttpClient {
    inner: ReqwestClient,
}

#[cfg(not(target_arch = "wasm32"))]
impl LoggingHttpClient {
    fn new() -> Self {
        Self {
            inner: ReqwestClient::new(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl fmt::Debug for LoggingHttpClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoggingHttpClient").finish()
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
impl HttpClient for LoggingHttpClient {
    async fn send_bytes(&self, request: Request<Bytes>) -> Result<Response<Bytes>, HttpError> {
        let request = request.try_into()?;
        let mut response = self.inner.execute(request).await?;
        let status = response.status();
        let headers = std::mem::take(response.headers_mut());
        let body = response.bytes().await?;

        if status.is_success() {
            tracing::debug!(
                target: "autoagents.telemetry.otlp",
                status = %status,
                body_len = body.len(),
                "OTLP export succeeded"
            );
        } else {
            let body_preview = String::from_utf8_lossy(&body);
            let preview = body_preview.chars().take(4096).collect::<String>();
            tracing::warn!(
                target: "autoagents.telemetry.otlp",
                status = %status,
                body = %preview,
                "OTLP export failed"
            );
            return Err(format!("OTLP export failed with status {status}: {preview}").into());
        }

        let mut http_response = Response::builder().status(status).body(body)?;
        *http_response.headers_mut() = headers;
        Ok(http_response)
    }
}

impl SpanExporter for MultiSpanExporter {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        let mut result = Ok(());
        for exporter in self.exporters.iter() {
            let batch = batch.clone();
            if let Err(err) = exporter.export(batch).await {
                result = Err(err);
            }
        }
        result
    }

    fn shutdown_with_timeout(&mut self, timeout: Duration) -> OTelSdkResult {
        let mut result = Ok(());
        for exporter in &mut self.exporters {
            if let Err(err) = exporter.shutdown_with_timeout(timeout) {
                result = Err(err);
            }
        }
        result
    }

    fn force_flush(&mut self) -> OTelSdkResult {
        let mut result = Ok(());
        for exporter in &mut self.exporters {
            if let Err(err) = exporter.force_flush() {
                result = Err(err);
            }
        }
        result
    }

    fn set_resource(&mut self, resource: &Resource) {
        for exporter in &mut self.exporters {
            exporter.set_resource(resource);
        }
    }
}

/// Span exporter that logs span data to the tracing subscriber.
#[derive(Debug, Default)]
pub(crate) struct StdoutSpanExporter;

impl SpanExporter for StdoutSpanExporter {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        for span in batch {
            tracing::info!(
                name = %span.name,
                trace_id = %span.span_context.trace_id(),
                span_id = %span.span_context.span_id(),
                start = ?span.start_time,
                end = ?span.end_time,
                attributes = ?span.attributes,
                status = ?span.status,
            );
        }
        Ok(())
    }
}

pub(crate) fn build_span_exporter(
    config: &TelemetryConfig,
) -> Result<MultiSpanExporter, opentelemetry_otlp::ExporterBuildError> {
    let mut exporters = Vec::new();

    if let Some(otlp) = &config.exporter.otlp {
        let exporter = build_otlp_span_exporter(otlp)?;
        exporters.push(SpanExporterWrapper::Otlp(Box::new(exporter)));
    }

    if config.exporter.stdout {
        exporters.push(SpanExporterWrapper::Stdout(StdoutSpanExporter));
    }

    Ok(MultiSpanExporter::new(exporters))
}

pub(crate) fn build_metric_exporter(
    config: &OtlpConfig,
) -> Result<MetricExporter, opentelemetry_otlp::ExporterBuildError> {
    let mut builder = MetricExporter::builder()
        .with_http()
        .with_temporality(Temporality::default());
    builder = apply_otlp_config(builder, config, "/v1/metrics");
    builder.build()
}

fn build_otlp_span_exporter(
    config: &OtlpConfig,
) -> Result<OtlpSpanExporter, opentelemetry_otlp::ExporterBuildError> {
    let mut builder = OtlpSpanExporter::builder().with_http();
    builder = apply_otlp_config(builder, config, "/v1/traces");
    builder.build()
}

fn apply_otlp_config<B>(builder: B, config: &OtlpConfig, signal_path: &str) -> B
where
    B: WithExportConfig + WithHttpConfig,
{
    let mut builder = builder.with_protocol(match config.protocol {
        OtlpProtocol::HttpBinary => opentelemetry_otlp::Protocol::HttpBinary,
        OtlpProtocol::HttpJson => opentelemetry_otlp::Protocol::HttpJson,
    });

    #[cfg(not(target_arch = "wasm32"))]
    {
        if config.debug_http {
            builder = builder.with_http_client(LoggingHttpClient::new());
        } else {
            let client = ReqwestClient::new();
            builder = builder.with_http_client(client);
        }
    }

    if let Some(endpoint) = resolve_signal_endpoint(config, signal_path) {
        builder = builder.with_endpoint(endpoint);
    }

    if !config.headers.is_empty() {
        builder = builder.with_headers(config.headers.clone());
    }

    builder
}

fn resolve_signal_endpoint(config: &OtlpConfig, signal_path: &str) -> Option<String> {
    config.endpoint.as_ref().map(|endpoint| {
        if endpoint.contains("/v1/") || endpoint.ends_with(signal_path) {
            endpoint.clone()
        } else {
            let trimmed = endpoint.trim_end_matches('/');
            if signal_path.starts_with('/') {
                format!("{trimmed}{signal_path}")
            } else {
                format!("{trimmed}/{signal_path}")
            }
        }
    })
}

pub(crate) fn resource_attributes(config: &TelemetryConfig) -> Vec<KeyValue> {
    let mut attributes = Vec::new();

    if let Some(version) = &config.service_version {
        attributes.push(KeyValue::new("service.version", version.clone()));
    }

    if let Some(environment) = &config.environment {
        attributes.push(KeyValue::new("deployment.environment", environment.clone()));
    }

    if let Some(runtime_id) = &config.runtime_id {
        attributes.push(KeyValue::new("runtime.id", runtime_id.to_string()));
    }

    attributes
}
