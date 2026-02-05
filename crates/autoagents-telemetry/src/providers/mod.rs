use crate::TelemetryConfig;

pub trait TelemetryProvider {
    fn telemetry_config(&self) -> TelemetryConfig;
}

#[cfg(feature = "langfuse")]
pub mod langfuse;
