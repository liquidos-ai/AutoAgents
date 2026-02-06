use autoagents_core::agent::{AgentDeriveT, AgentExecutor, AgentHooks, DirectAgentHandle};
use autoagents_core::environment::Environment;
use autoagents_core::utils::BoxEventStream;
use autoagents_protocol::{Event, RuntimeID};
use std::sync::Arc;

use crate::runner::start_telemetry;
use crate::{TelemetryConfig, TelemetryError, TelemetryHandle, TelemetryProvider};

/// Owns the telemetry lifecycle for a specific event stream.
pub struct Tracer {
    provider: Arc<dyn TelemetryProvider>,
    event_stream: Option<BoxEventStream<Event>>,
    runtime_id: Option<RuntimeID>,
    handle: Option<TelemetryHandle>,
    shutdown_grace: std::time::Duration,
}

impl Tracer {
    pub fn new(provider: Arc<dyn TelemetryProvider>, event_stream: BoxEventStream<Event>) -> Self {
        Self {
            provider,
            event_stream: Some(event_stream),
            runtime_id: None,
            handle: None,
            shutdown_grace: std::time::Duration::from_secs(10),
        }
    }

    pub fn from_direct<T>(
        provider: Arc<dyn TelemetryProvider>,
        handle: &mut DirectAgentHandle<T>,
    ) -> Self
    where
        T: AgentDeriveT + AgentExecutor + AgentHooks + Send + Sync + 'static,
    {
        let stream = handle.subscribe_events();
        Self::new(provider, stream)
    }

    pub async fn from_environment(
        provider: Arc<dyn TelemetryProvider>,
        env: &mut Environment,
        runtime_id: Option<RuntimeID>,
    ) -> Result<Self, TelemetryError> {
        let stream = env.subscribe_events(runtime_id).await?;
        Ok(Self {
            provider,
            event_stream: Some(stream),
            runtime_id,
            handle: None,
            shutdown_grace: std::time::Duration::from_secs(2),
        })
    }

    pub fn with_shutdown_grace(mut self, duration: std::time::Duration) -> Self {
        self.shutdown_grace = duration;
        self
    }

    /// Start exporting spans and metrics from the configured event stream.
    pub fn start(&mut self) -> Result<(), TelemetryError> {
        if self.handle.is_some() {
            return Err(TelemetryError::AlreadyStarted);
        }

        let event_stream = self
            .event_stream
            .take()
            .ok_or(TelemetryError::MissingEventStream)?;
        let config = self.provider_config();
        let attributes = self.provider.attribute_provider();
        let handle = start_telemetry(event_stream, config, attributes, self.shutdown_grace)?;
        self.handle = Some(handle);
        Ok(())
    }

    /// Flush and shut down exporters.
    pub async fn shutdown(&mut self) -> Result<(), TelemetryError> {
        if let Some(handle) = self.handle.take() {
            handle.shutdown().await;
        }
        Ok(())
    }

    fn provider_config(&self) -> TelemetryConfig {
        let mut config = self.provider.telemetry_config();
        if let Some(runtime_id) = self.runtime_id {
            config = config.with_runtime_id(runtime_id);
        }
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_core::utils::BoxEventStream;
    use autoagents_protocol::Event;
    use tokio::sync::mpsc;
    use tokio::time::{Duration, timeout};
    use tokio_stream::wrappers::ReceiverStream;

    #[derive(Debug, Clone)]
    struct TestProvider;

    impl TelemetryProvider for TestProvider {
        fn telemetry_config(&self) -> TelemetryConfig {
            let mut config = TelemetryConfig::new("autoagents-test");
            config.exporter.stdout = true;
            config.metrics_enabled = false;
            config.install_tracing_subscriber = false;
            config
        }
    }

    #[tokio::test]
    async fn tracer_start_rejects_double_start() {
        let (tx, rx) = mpsc::channel::<Event>(1);
        let stream: BoxEventStream<Event> = Box::pin(ReceiverStream::new(rx));
        let provider: Arc<dyn TelemetryProvider> = Arc::new(TestProvider);
        let mut tracer = Tracer::new(provider, stream);

        tracer.start().expect("start succeeds");
        let err = tracer.start().expect_err("double start fails");
        assert!(matches!(err, TelemetryError::AlreadyStarted));

        drop(tx);
        timeout(Duration::from_secs(2), tracer.shutdown())
            .await
            .expect("shutdown completes")
            .expect("shutdown succeeds");
    }

    #[tokio::test]
    async fn tracer_shutdown_is_idempotent() {
        let (tx, rx) = mpsc::channel::<Event>(1);
        let stream: BoxEventStream<Event> = Box::pin(ReceiverStream::new(rx));
        let provider: Arc<dyn TelemetryProvider> = Arc::new(TestProvider);
        let mut tracer = Tracer::new(provider, stream);

        tracer.start().expect("start succeeds");
        drop(tx);
        timeout(Duration::from_secs(2), tracer.shutdown())
            .await
            .expect("shutdown completes")
            .expect("shutdown succeeds");
        timeout(Duration::from_secs(2), tracer.shutdown())
            .await
            .expect("shutdown completes")
            .expect("shutdown is idempotent");
    }
}
