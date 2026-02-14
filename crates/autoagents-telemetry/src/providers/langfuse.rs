use crate::config::{ExporterConfig, OtlpConfig, OtlpProtocol, TelemetryConfig};
use crate::providers::{TelemetryAttributeProvider, TelemetryProvider};
use base64::{Engine as _, engine::general_purpose};
use opentelemetry::Value;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;

/// Langfuse deployment target for OTLP export.
#[derive(Debug, Clone)]
pub enum LangfuseRegion {
    Us,
    Eu,
    Custom(String),
}

impl LangfuseRegion {
    fn base_url(&self) -> String {
        match self {
            LangfuseRegion::Us => "https://us.cloud.langfuse.com".to_string(),
            LangfuseRegion::Eu => "https://cloud.langfuse.com".to_string(),
            LangfuseRegion::Custom(url) => url.clone(),
        }
    }
}

/// Langfuse-specific telemetry configuration builder.
#[derive(Debug, Clone)]
pub struct LangfuseTelemetry {
    public_key: String,
    secret_key: String,
    region: LangfuseRegion,
    stdout: bool,
    service_name: String,
    debug_http: bool,
    install_tracing_subscriber: bool,
}

impl LangfuseTelemetry {
    pub fn new(public_key: impl Into<String>, secret_key: impl Into<String>) -> Self {
        Self {
            public_key: public_key.into(),
            secret_key: secret_key.into(),
            region: LangfuseRegion::Us,
            stdout: false,
            service_name: "autoagents".to_string(),
            debug_http: false,
            install_tracing_subscriber: true,
        }
    }

    pub fn with_region(mut self, region: LangfuseRegion) -> Self {
        self.region = region;
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.region = LangfuseRegion::Custom(base_url.into());
        self
    }

    pub fn with_stdout(mut self, enabled: bool) -> Self {
        self.stdout = enabled;
        self
    }

    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    pub fn with_http_debug(mut self, enabled: bool) -> Self {
        self.debug_http = enabled;
        self
    }

    pub fn with_tracing_subscriber(mut self, enabled: bool) -> Self {
        self.install_tracing_subscriber = enabled;
        self
    }

    pub fn build(self) -> TelemetryConfig {
        let mut config = TelemetryConfig::new(self.service_name.clone());
        config.install_tracing_subscriber = self.install_tracing_subscriber;
        config.exporter = ExporterConfig {
            otlp: Some(self.otlp_config()),
            stdout: self.stdout,
        };
        config
    }

    fn otlp_config(&self) -> OtlpConfig {
        let mut otlp = OtlpConfig::new(format!(
            "{}/api/public/otel",
            self.region.base_url().trim_end_matches('/')
        ));
        otlp.protocol = OtlpProtocol::HttpBinary;
        otlp.headers = self.auth_headers();
        otlp.debug_http = self.debug_http;
        otlp
    }

    fn auth_headers(&self) -> HashMap<String, String> {
        let creds = format!("{}:{}", self.public_key, self.secret_key);
        let encoded = general_purpose::STANDARD.encode(creds.as_bytes());
        HashMap::from([("Authorization".to_string(), format!("Basic {}", encoded))])
    }
}

impl TelemetryProvider for LangfuseTelemetry {
    fn telemetry_config(&self) -> TelemetryConfig {
        self.clone().build()
    }

    fn attribute_provider(&self) -> Option<Arc<dyn TelemetryAttributeProvider>> {
        Some(Arc::new(LangfuseAttributeProvider))
    }
}

#[derive(Debug)]
struct LangfuseAttributeProvider;

impl TelemetryAttributeProvider for LangfuseAttributeProvider {
    fn task_started_attributes(
        &self,
        actor_name: &str,
        task_input: &str,
    ) -> Vec<(&'static str, Value)> {
        vec![
            ("langfuse.trace.name", Value::from(actor_name.to_string())),
            (
                "langfuse.trace.input",
                Value::from(normalize_langfuse_json(task_input)),
            ),
            ("langfuse.observation.type", Value::from("span")),
            (
                "langfuse.observation.input",
                Value::from(normalize_langfuse_json(task_input)),
            ),
        ]
    }

    fn task_completed_attributes(&self, task_output: &str) -> Vec<(&'static str, Value)> {
        let output = normalize_langfuse_json(task_output);
        vec![
            ("langfuse.trace.output", Value::from(output.clone())),
            ("langfuse.observation.output", Value::from(output)),
        ]
    }

    fn tool_started_attributes(
        &self,
        tool_name: &str,
        tool_args: &str,
    ) -> Vec<(&'static str, Value)> {
        vec![
            ("langfuse.observation.type", Value::from("tool")),
            (
                "langfuse.observation.name",
                Value::from(tool_name.to_string()),
            ),
            (
                "langfuse.observation.input",
                Value::from(normalize_langfuse_json(tool_args)),
            ),
        ]
    }

    fn tool_completed_attributes(
        &self,
        tool_name: &str,
        tool_output: &str,
    ) -> Vec<(&'static str, Value)> {
        vec![
            ("langfuse.observation.type", Value::from("tool")),
            (
                "langfuse.observation.name",
                Value::from(tool_name.to_string()),
            ),
            (
                "langfuse.observation.output",
                Value::from(normalize_langfuse_json(tool_output)),
            ),
        ]
    }

    fn tool_failed_attributes(&self, tool_name: &str, _error: &str) -> Vec<(&'static str, Value)> {
        vec![
            ("langfuse.observation.type", Value::from("tool")),
            (
                "langfuse.observation.name",
                Value::from(tool_name.to_string()),
            ),
        ]
    }
}

fn normalize_langfuse_json(value: &str) -> String {
    if serde_json::from_str::<JsonValue>(value).is_ok() {
        value.to_string()
    } else {
        serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_langfuse_region_base_url() {
        assert_eq!(
            LangfuseRegion::Us.base_url(),
            "https://us.cloud.langfuse.com"
        );
        assert_eq!(LangfuseRegion::Eu.base_url(), "https://cloud.langfuse.com");
        assert_eq!(
            LangfuseRegion::Custom("https://example.com".to_string()).base_url(),
            "https://example.com"
        );
    }

    #[test]
    fn test_langfuse_telemetry_build_headers() {
        let telemetry = LangfuseTelemetry::new("pub", "secret")
            .with_region(LangfuseRegion::Us)
            .with_service_name("svc");
        let config = telemetry.build();
        let otlp = config.exporter.otlp.unwrap();
        if let Some(endpoint) = otlp.endpoint {
            assert!(endpoint.contains("langfuse"));
        } else {
            panic!("expected otlp endpoint");
        }
        assert!(otlp.headers.contains_key("Authorization"));
        assert_eq!(config.service_name, "svc");
    }

    #[test]
    fn test_normalize_langfuse_json() {
        assert_eq!(normalize_langfuse_json("{\"a\":1}"), "{\"a\":1}");
        assert_eq!(normalize_langfuse_json("plain"), "\"plain\"");
    }

    #[test]
    fn test_langfuse_attribute_provider_keys() {
        let provider = LangfuseAttributeProvider;
        let attrs = provider.task_started_attributes("actor", "{\"a\":1}");
        assert!(attrs.iter().any(|(k, _)| *k == "langfuse.trace.name"));
        let attrs = provider.tool_failed_attributes("tool", "err");
        assert!(attrs.iter().any(|(k, _)| *k == "langfuse.observation.name"));
    }
}
