use crate::config::{ExporterConfig, OtlpConfig, OtlpProtocol, TelemetryConfig};
use crate::providers::TelemetryProvider;
use base64::{Engine as _, engine::general_purpose};
use std::collections::HashMap;

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
}
