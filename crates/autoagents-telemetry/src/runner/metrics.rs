use opentelemetry::metrics::{Counter, Histogram, MeterProvider as _};
use opentelemetry_sdk::metrics::SdkMeterProvider;

/// Metric instruments used by the event mapper.
pub(crate) struct TelemetryMetrics {
    pub(crate) tasks_total: Counter<u64>,
    pub(crate) tool_calls_total: Counter<u64>,
    pub(crate) errors_total: Counter<u64>,
    pub(crate) task_duration: Histogram<f64>,
    pub(crate) turn_duration: Histogram<f64>,
    pub(crate) tool_duration: Histogram<f64>,
}

impl TelemetryMetrics {
    pub(crate) fn new(provider: &SdkMeterProvider) -> Self {
        let meter = provider.meter("autoagents.telemetry");

        Self {
            tasks_total: meter.u64_counter("autoagents.tasks.total").build(),
            tool_calls_total: meter.u64_counter("autoagents.tool_calls.total").build(),
            errors_total: meter.u64_counter("autoagents.errors.total").build(),
            task_duration: meter
                .f64_histogram("autoagents.task.duration.seconds")
                .with_unit("s")
                .build(),
            turn_duration: meter
                .f64_histogram("autoagents.turn.duration.seconds")
                .with_unit("s")
                .build(),
            tool_duration: meter
                .f64_histogram("autoagents.tool.duration.seconds")
                .with_unit("s")
                .build(),
        }
    }
}
