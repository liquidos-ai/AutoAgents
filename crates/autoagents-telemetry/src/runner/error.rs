use autoagents_core::environment::EnvironmentError;

/// Errors emitted by telemetry startup/shutdown and environment wiring.
#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    #[error("No telemetry exporter configured")]
    MissingExporter,
    #[error("Failed to build OTLP exporter: {0}")]
    ExporterBuild(#[from] opentelemetry_otlp::ExporterBuildError),
    #[error("Failed to access runtime events: {0}")]
    Environment(Box<EnvironmentError>),
    #[error("Telemetry already started")]
    AlreadyStarted,
    #[error("Telemetry event stream not available")]
    MissingEventStream,
}

impl From<EnvironmentError> for TelemetryError {
    fn from(err: EnvironmentError) -> Self {
        Self::Environment(Box::new(err))
    }
}
