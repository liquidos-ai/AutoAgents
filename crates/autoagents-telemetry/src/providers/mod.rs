use std::fmt::Debug;
use std::sync::Arc;

use crate::TelemetryConfig;
use opentelemetry::Value;

/// Provides a telemetry configuration per tracer instance.
pub trait TelemetryProvider: Debug {
    fn telemetry_config(&self) -> TelemetryConfig;
    fn attribute_provider(&self) -> Option<Arc<dyn TelemetryAttributeProvider>> {
        None
    }
}

/// Provider-specific span attributes for downstream systems.
pub trait TelemetryAttributeProvider: Debug + Send + Sync {
    fn task_started_attributes(
        &self,
        actor_name: &str,
        task_input: &str,
    ) -> Vec<(&'static str, Value)>;
    fn task_completed_attributes(&self, task_output: &str) -> Vec<(&'static str, Value)>;
    fn tool_started_attributes(
        &self,
        tool_name: &str,
        tool_args: &str,
    ) -> Vec<(&'static str, Value)>;
    fn tool_completed_attributes(
        &self,
        tool_name: &str,
        tool_output: &str,
    ) -> Vec<(&'static str, Value)>;
    fn tool_failed_attributes(&self, tool_name: &str, error: &str) -> Vec<(&'static str, Value)>;
}

#[cfg(feature = "langfuse")]
pub mod langfuse;
