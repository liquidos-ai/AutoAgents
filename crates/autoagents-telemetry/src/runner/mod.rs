mod error;
mod handle;
mod mapper;
mod metrics;

use crate::config::TelemetryConfig;
use crate::exporter::{build_metric_exporter, build_span_exporter, resource_attributes};
use autoagents_core::utils::BoxEventStream;
use autoagents_protocol::Event;
use futures_util::StreamExt;
use opentelemetry::trace::TracerProvider;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::metrics::periodic_reader_with_async_runtime::PeriodicReader;
use opentelemetry_sdk::resource::Resource;
use opentelemetry_sdk::runtime;
use opentelemetry_sdk::trace::BatchConfigBuilder;
use opentelemetry_sdk::trace::span_processor_with_async_runtime::BatchSpanProcessor;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanExporter};
use tokio::sync::{mpsc, watch};
use tokio::time::{Duration, Instant};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::Layer;
use tracing_subscriber::prelude::*;

pub use error::TelemetryError;
pub use handle::TelemetryHandle;

use mapper::EventMapper;
use metrics::TelemetryMetrics;

async fn flush_telemetry(
    tracer_provider: &SdkTracerProvider,
    meter_provider: Option<&SdkMeterProvider>,
    timeout: Duration,
) {
    let tracer_provider = tracer_provider.clone();
    let meter_provider = meter_provider.cloned();
    let flush = tokio::task::spawn_blocking(move || {
        let _ = tracer_provider.force_flush();
        if let Some(provider) = meter_provider {
            let _ = provider.force_flush();
        }
    });
    let _ = tokio::time::timeout(timeout, flush).await;
}

// Build exporters, install tracing, and spawn the event mapper loop.
pub(crate) fn start_telemetry(
    event_stream: BoxEventStream<Event>,
    config: TelemetryConfig,
    attributes: Option<std::sync::Arc<dyn crate::providers::TelemetryAttributeProvider>>,
    shutdown_grace: Duration,
) -> Result<TelemetryHandle, TelemetryError> {
    let mut exporters = build_span_exporter(&config)?;
    if exporters.is_empty() {
        return Err(TelemetryError::MissingExporter);
    }

    let resource = Resource::builder()
        .with_service_name(config.service_name.clone())
        .with_attributes(resource_attributes(&config))
        .build();

    exporters.set_resource(&resource);

    let batch_processor = BatchSpanProcessor::builder(exporters, runtime::Tokio)
        .with_batch_config(
            BatchConfigBuilder::default()
                .with_max_queue_size(config.span_batch.max_queue_size)
                .with_max_export_batch_size(config.span_batch.max_export_batch_size)
                .with_scheduled_delay(config.span_batch.scheduled_delay)
                .with_max_export_timeout(config.span_batch.max_export_timeout)
                .with_max_concurrent_exports(config.span_batch.max_concurrent_exports)
                .build(),
        )
        .build();
    let tracer_provider = SdkTracerProvider::builder()
        .with_resource(resource.clone())
        .with_span_processor(batch_processor)
        .build();
    let tracer = tracer_provider.tracer("autoagents.telemetry");

    if config.install_tracing_subscriber {
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let subscriber = tracing_subscriber::Registry::default()
            .with(otel_layer.with_filter(filter.clone()))
            .with(tracing_subscriber::fmt::layer().with_filter(filter));
        if subscriber.try_init().is_err() {
            eprintln!(
                "[autoagents-telemetry] tracing subscriber already set; OTLP layer not installed"
            );
        }
    }

    let meter_provider = if config.metrics_enabled {
        let mut builder = SdkMeterProvider::builder().with_resource(resource);
        if let Some(otlp) = &config.exporter.otlp {
            let metric_exporter = build_metric_exporter(otlp)?;
            let reader = PeriodicReader::builder(metric_exporter, runtime::Tokio).build();
            builder = builder.with_reader(reader);
        }
        Some(builder.build())
    } else {
        None
    };

    let metrics = meter_provider.as_ref().map(TelemetryMetrics::new);

    let redaction = config.redaction.clone();
    let runtime_id = config.runtime_id;

    let tracer_provider_for_task = tracer_provider.clone();
    let meter_provider_for_task = meter_provider.clone();
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    let (flush_tx, mut flush_rx) = mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        let mut mapper =
            EventMapper::new(metrics, redaction, runtime_id, attributes, Some(flush_tx));
        let mut event_stream = event_stream;
        let mut shutdown_requested = false;
        let mut shutdown_deadline: Option<Instant> = None;
        let mut last_event_at = Instant::now();
        let idle_window = Duration::from_millis(50);
        loop {
            match shutdown_deadline {
                Some(deadline) => {
                    tokio::select! {
                        Some(_) = flush_rx.recv() => {
                            flush_telemetry(
                                &tracer_provider_for_task,
                                meter_provider_for_task.as_ref(),
                                shutdown_grace,
                            ).await;
                        }
                        _ = shutdown_rx.changed() => {
                            shutdown_requested = true;
                            if shutdown_deadline.is_none() {
                                shutdown_deadline = Some(Instant::now() + shutdown_grace);
                            }
                        }
                        event = event_stream.next() => {
                            match event {
                                Some(event) => {
                                    last_event_at = Instant::now();
                                    mapper.handle_event(event);
                                }
                                None => break,
                            }
                        }
                        _ = tokio::time::sleep_until(deadline) => {
                            break;
                        }
                    }
                }
                None => {
                    tokio::select! {
                        Some(_) = flush_rx.recv() => {
                            flush_telemetry(
                                &tracer_provider_for_task,
                                meter_provider_for_task.as_ref(),
                                shutdown_grace,
                            ).await;
                        }
                        _ = shutdown_rx.changed() => {
                            shutdown_requested = true;
                            shutdown_deadline = Some(Instant::now() + shutdown_grace);
                        }
                        event = event_stream.next() => {
                            match event {
                                Some(event) => {
                                    last_event_at = Instant::now();
                                    mapper.handle_event(event);
                                }
                                None => break,
                            }
                        }
                    }
                }
            }

            if shutdown_requested
                && !mapper.has_open_tasks()
                && last_event_at.elapsed() >= idle_window
            {
                break;
            }
        }
        mapper.flush();
        flush_telemetry(
            &tracer_provider_for_task,
            meter_provider_for_task.as_ref(),
            shutdown_grace,
        )
        .await;
    });

    Ok(TelemetryHandle {
        task: Some(task),
        tracer_provider,
        meter_provider,
        shutdown_tx: Some(shutdown_tx),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_core::utils::BoxEventStream;
    use autoagents_protocol::{ActorID, Event, SubmissionId};
    use tokio::sync::mpsc;
    use tokio::time::{Duration, timeout};
    use tokio_stream::wrappers::ReceiverStream;

    fn test_config() -> TelemetryConfig {
        let mut config = TelemetryConfig::new("autoagents-test");
        config.exporter.stdout = true;
        config.metrics_enabled = false;
        config.install_tracing_subscriber = false;
        config
    }

    #[tokio::test]
    async fn start_telemetry_errors_without_exporter() {
        let (_tx, rx) = mpsc::channel::<Event>(1);
        let stream: BoxEventStream<Event> = Box::pin(ReceiverStream::new(rx));
        let mut config = TelemetryConfig::new("autoagents-test");
        config.metrics_enabled = false;
        config.install_tracing_subscriber = false;

        let err = match start_telemetry(stream, config, None, Duration::from_secs(2)) {
            Ok(_) => panic!("missing exporter"),
            Err(err) => err,
        };
        assert!(matches!(err, TelemetryError::MissingExporter));
    }

    #[tokio::test]
    async fn shutdown_completes_without_hanging() {
        let (tx, rx) = mpsc::channel::<Event>(4);
        let stream: BoxEventStream<Event> = Box::pin(ReceiverStream::new(rx));
        let handle = start_telemetry(stream, test_config(), None, Duration::from_secs(2))
            .expect("telemetry starts");

        let _ = tx
            .send(Event::TaskStarted {
                sub_id: SubmissionId::new_v4(),
                actor_id: ActorID::new_v4(),
                actor_name: "tester".to_string(),
                task_description: "test".to_string(),
            })
            .await;

        drop(tx);
        timeout(Duration::from_secs(2), handle.shutdown())
            .await
            .expect("shutdown completes");
    }
}
