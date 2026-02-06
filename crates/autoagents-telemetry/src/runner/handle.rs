use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::trace::SdkTracerProvider;
use std::time::Duration;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::timeout;

/// Holds exporter state and the mapper task for graceful shutdown.
pub struct TelemetryHandle {
    pub(crate) task: Option<JoinHandle<()>>,
    pub(crate) tracer_provider: SdkTracerProvider,
    pub(crate) meter_provider: Option<SdkMeterProvider>,
    pub(crate) shutdown_tx: Option<watch::Sender<bool>>,
}

impl TelemetryHandle {
    pub async fn shutdown(mut self) {
        const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);
        const OTEL_TIMEOUT: Duration = Duration::from_secs(2);

        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(true);
        }

        if let Some(task) = self.task.take() {
            let mut task = task;
            if timeout(SHUTDOWN_TIMEOUT, &mut task).await.is_err() {
                // If the mapper doesn't exit in time, abort to avoid hanging shutdown.
                task.abort();
                let _ = task.await;
            }
        }

        let tracer_provider = self.tracer_provider;
        let _ = timeout(
            OTEL_TIMEOUT,
            tokio::task::spawn_blocking(move || {
                let _ = tracer_provider.force_flush();
                let _ = tracer_provider.shutdown();
            }),
        )
        .await;

        if let Some(meter_provider) = self.meter_provider {
            let _ = timeout(
                OTEL_TIMEOUT,
                tokio::task::spawn_blocking(move || {
                    let _ = meter_provider.force_flush();
                    let _ = meter_provider.shutdown();
                }),
            )
            .await;
        }
    }
}
