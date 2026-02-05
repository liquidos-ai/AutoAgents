use crate::config::{RedactionConfig, TelemetryConfig};
use crate::exporter::{build_metric_exporter, build_span_exporter, resource_attributes};
use autoagents_core::agent::{AgentDeriveT, AgentExecutor, AgentHooks, DirectAgentHandle};
use autoagents_core::environment::{Environment, EnvironmentError};
use autoagents_core::protocol::{ActorID, Event, RuntimeID, SubmissionId};
use autoagents_core::utils::BoxEventStream;
use futures_util::StreamExt;
use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Histogram, MeterProvider as _};
use opentelemetry::trace::{Status, TracerProvider};
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::metrics::periodic_reader_with_async_runtime::PeriodicReader;
use opentelemetry_sdk::resource::Resource;
use opentelemetry_sdk::runtime;
use opentelemetry_sdk::trace::span_processor_with_async_runtime::BatchSpanProcessor;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanExporter};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Instant;
use tokio::task::JoinHandle;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::Layer;
use tracing_subscriber::prelude::*;

#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    #[error("No telemetry exporter configured")]
    MissingExporter,
    #[error("Failed to build OTLP exporter: {0}")]
    ExporterBuild(#[from] opentelemetry_otlp::ExporterBuildError),
    #[error("Failed to access runtime events: {0}")]
    Environment(Box<EnvironmentError>),
}

impl From<EnvironmentError> for TelemetryError {
    fn from(err: EnvironmentError) -> Self {
        Self::Environment(Box::new(err))
    }
}

pub struct TelemetryHandle {
    task: Option<JoinHandle<()>>,
    tracer_provider: SdkTracerProvider,
    meter_provider: Option<SdkMeterProvider>,
}

impl TelemetryHandle {
    pub async fn shutdown(mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
            let _ = task.await;
        }

        let _ = self.tracer_provider.force_flush();
        let _ = self.tracer_provider.shutdown();

        if let Some(meter_provider) = &self.meter_provider {
            let _ = meter_provider.force_flush();
            let _ = meter_provider.shutdown();
        }
    }
}

pub fn attach_to_stream(
    event_stream: BoxEventStream<Event>,
    config: TelemetryConfig,
) -> Result<TelemetryHandle, TelemetryError> {
    attach_to_stream_inner(event_stream, config)
}

pub async fn attach_to_environment(
    env: &mut Environment,
    runtime_id: Option<RuntimeID>,
    config: TelemetryConfig,
) -> Result<TelemetryHandle, TelemetryError> {
    let event_stream = env.subscribe_events(runtime_id).await?;
    let config = if let Some(runtime_id) = runtime_id {
        config.with_runtime_id(runtime_id)
    } else {
        config
    };
    attach_to_stream_inner(event_stream, config)
}

pub fn attach_to_direct<T>(
    handle: &mut DirectAgentHandle<T>,
    config: TelemetryConfig,
) -> Result<TelemetryHandle, TelemetryError>
where
    T: AgentDeriveT + AgentExecutor + AgentHooks + Send + Sync + 'static,
{
    let stream = handle.subscribe_events();
    attach_to_stream_inner(stream, config)
}

fn attach_to_stream_inner(
    mut event_stream: BoxEventStream<Event>,
    config: TelemetryConfig,
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

    let batch_processor = BatchSpanProcessor::builder(exporters, runtime::Tokio).build();
    let tracer_provider = SdkTracerProvider::builder()
        .with_resource(resource.clone())
        .with_span_processor(batch_processor)
        .build();
    opentelemetry::global::set_tracer_provider(tracer_provider.clone());
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

    let task = tokio::spawn(async move {
        let mut mapper = EventMapper::new(metrics, redaction, runtime_id);
        while let Some(event) = event_stream.next().await {
            mapper.handle_event(event);
        }
        mapper.flush();
    });

    Ok(TelemetryHandle {
        task: Some(task),
        tracer_provider,
        meter_provider,
    })
}

struct TelemetryMetrics {
    tasks_total: Counter<u64>,
    tool_calls_total: Counter<u64>,
    errors_total: Counter<u64>,
    task_duration: Histogram<f64>,
    turn_duration: Histogram<f64>,
    tool_duration: Histogram<f64>,
}

impl TelemetryMetrics {
    fn new(provider: &SdkMeterProvider) -> Self {
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

#[derive(Clone, Copy, Debug)]
struct TaskKey {
    sub_id: SubmissionId,
    actor_id: ActorID,
}

impl TaskKey {
    fn new(sub_id: SubmissionId, actor_id: ActorID) -> Self {
        Self { sub_id, actor_id }
    }
}

impl Hash for TaskKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sub_id.hash(state);
        self.actor_id.hash(state);
    }
}

impl PartialEq for TaskKey {
    fn eq(&self, other: &Self) -> bool {
        self.sub_id == other.sub_id && self.actor_id == other.actor_id
    }
}

impl Eq for TaskKey {}

#[derive(Clone, Copy, Debug)]
struct TurnKey {
    sub_id: SubmissionId,
    actor_id: ActorID,
    turn_number: usize,
}

impl TurnKey {
    fn new(sub_id: SubmissionId, actor_id: ActorID, turn_number: usize) -> Self {
        Self {
            sub_id,
            actor_id,
            turn_number,
        }
    }
}

impl Hash for TurnKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sub_id.hash(state);
        self.actor_id.hash(state);
        self.turn_number.hash(state);
    }
}

impl PartialEq for TurnKey {
    fn eq(&self, other: &Self) -> bool {
        self.sub_id == other.sub_id
            && self.actor_id == other.actor_id
            && self.turn_number == other.turn_number
    }
}

impl Eq for TurnKey {}

#[derive(Clone, Debug)]
struct ToolKey {
    sub_id: SubmissionId,
    actor_id: ActorID,
    tool_call_id: String,
}

impl ToolKey {
    fn new(sub_id: SubmissionId, actor_id: ActorID, tool_call_id: String) -> Self {
        Self {
            sub_id,
            actor_id,
            tool_call_id,
        }
    }
}

impl Hash for ToolKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sub_id.hash(state);
        self.actor_id.hash(state);
        self.tool_call_id.hash(state);
    }
}

impl PartialEq for ToolKey {
    fn eq(&self, other: &Self) -> bool {
        self.sub_id == other.sub_id
            && self.actor_id == other.actor_id
            && self.tool_call_id == other.tool_call_id
    }
}

impl Eq for ToolKey {}

struct TelemetryState {
    task_spans: HashMap<TaskKey, tracing::Span>,
    task_start: HashMap<TaskKey, Instant>,
    turn_spans: HashMap<TurnKey, tracing::Span>,
    turn_start: HashMap<TurnKey, Instant>,
    tool_spans: HashMap<ToolKey, tracing::Span>,
    tool_start: HashMap<ToolKey, Instant>,
}

impl TelemetryState {
    fn new() -> Self {
        Self {
            task_spans: HashMap::new(),
            task_start: HashMap::new(),
            turn_spans: HashMap::new(),
            turn_start: HashMap::new(),
            tool_spans: HashMap::new(),
            tool_start: HashMap::new(),
        }
    }
}

struct EventMapper {
    metrics: Option<TelemetryMetrics>,
    redaction: RedactionConfig,
    runtime_id: Option<RuntimeID>,
    state: TelemetryState,
}

impl EventMapper {
    fn new(
        metrics: Option<TelemetryMetrics>,
        redaction: RedactionConfig,
        runtime_id: Option<RuntimeID>,
    ) -> Self {
        Self {
            metrics,
            redaction,
            runtime_id,
            state: TelemetryState::new(),
        }
    }

    fn handle_event(&mut self, event: Event) {
        match event {
            Event::TaskStarted {
                sub_id,
                actor_id,
                actor_name,
                task_description,
            } => self.on_task_started(sub_id, actor_id, actor_name, task_description),
            Event::TaskComplete {
                sub_id,
                actor_id,
                actor_name,
                result,
            } => self.on_task_complete(sub_id, actor_id, actor_name, result),
            Event::TaskError {
                sub_id,
                actor_id,
                error,
            } => self.on_task_error(sub_id, actor_id, error),
            Event::TurnStarted {
                sub_id,
                actor_id,
                turn_number,
                max_turns,
            } => self.on_turn_started(sub_id, actor_id, turn_number, max_turns),
            Event::TurnCompleted {
                sub_id,
                actor_id,
                turn_number,
                final_turn,
            } => self.on_turn_completed(sub_id, actor_id, turn_number, final_turn),
            Event::ToolCallRequested {
                sub_id,
                actor_id,
                id,
                tool_name,
                arguments,
            } => self.on_tool_requested(sub_id, actor_id, id, tool_name, arguments),
            Event::ToolCallCompleted {
                sub_id,
                actor_id,
                id,
                tool_name,
                result,
            } => self.on_tool_completed(sub_id, actor_id, id, tool_name, result.to_string()),
            Event::ToolCallFailed {
                sub_id,
                actor_id,
                id,
                tool_name,
                error,
            } => self.on_tool_failed(sub_id, actor_id, id, tool_name, error),
            _ => {}
        }
    }

    fn flush(&mut self) {
        for (_, span) in self.state.tool_spans.drain() {
            drop(span);
        }
        for (_, span) in self.state.turn_spans.drain() {
            drop(span);
        }
        for (_, span) in self.state.task_spans.drain() {
            drop(span);
        }
    }

    fn on_task_started(
        &mut self,
        sub_id: SubmissionId,
        actor_id: ActorID,
        actor_name: String,
        task_description: String,
    ) {
        let span = tracing::info_span!(
            "autoagents.task",
            submission_id = %sub_id,
            actor_id = %actor_id,
            actor_name = %actor_name,
        );
        if let Some(runtime_id) = &self.runtime_id {
            let runtime_id = runtime_id.to_string();
            span.record("runtime_id", runtime_id.as_str());
        }
        let description =
            self.redact_value(task_description.clone(), self.redaction.redact_task_inputs);
        let description_json = self.normalize_langfuse_json(description.clone());
        span.set_attribute("task.description", description.clone());
        span.set_attribute("langfuse.trace.name", actor_name.clone());
        span.set_attribute(
            "langfuse.trace.input",
            self.normalize_langfuse_json(
                self.redact_value(task_description, self.redaction.redact_task_inputs),
            ),
        );
        span.set_attribute("langfuse.observation.type", "span");
        span.set_attribute("langfuse.observation.input", description_json);

        let key = TaskKey::new(sub_id, actor_id);
        self.state.task_spans.insert(key, span);
        self.state
            .task_start
            .insert(TaskKey::new(sub_id, actor_id), Instant::now());
    }

    fn on_task_complete(
        &mut self,
        sub_id: SubmissionId,
        actor_id: ActorID,
        actor_name: String,
        result: String,
    ) {
        let key = TaskKey::new(sub_id, actor_id);
        if let Some(span) = self.state.task_spans.remove(&key) {
            span.set_attribute("actor_name", actor_name);
            let redacted = self.redact_value(result, self.redaction.redact_task_outputs);
            let redacted_json = self.normalize_langfuse_json(redacted.clone());
            span.set_attribute("task.result", redacted.clone());
            span.set_attribute("langfuse.trace.output", redacted_json.clone());
            span.set_attribute("langfuse.observation.output", redacted_json);
            span.set_status(Status::Ok);
            drop(span);
        }

        if let Some(start) = self.state.task_start.remove(&key)
            && let Some(metrics) = &self.metrics
        {
            metrics.task_duration.record(
                start.elapsed().as_secs_f64(),
                &self.task_metric_attributes(sub_id, actor_id),
            );
            metrics.tasks_total.add(
                1,
                &self.task_metric_attributes_with_status(sub_id, actor_id, "completed"),
            );
        }
    }

    fn on_task_error(&mut self, sub_id: SubmissionId, actor_id: ActorID, error: String) {
        let key = TaskKey::new(sub_id, actor_id);
        if let Some(span) = self.state.task_spans.remove(&key) {
            span.set_status(Status::error(error.clone()));
            span.set_attribute("error.message", error.clone());
            drop(span);
        }

        if let Some(start) = self.state.task_start.remove(&key)
            && let Some(metrics) = &self.metrics
        {
            metrics.task_duration.record(
                start.elapsed().as_secs_f64(),
                &self.task_metric_attributes(sub_id, actor_id),
            );
            metrics.tasks_total.add(
                1,
                &self.task_metric_attributes_with_status(sub_id, actor_id, "error"),
            );
            metrics
                .errors_total
                .add(1, &[KeyValue::new("error.kind", "task")]);
        }
    }

    fn on_turn_started(
        &mut self,
        sub_id: SubmissionId,
        actor_id: ActorID,
        turn_number: usize,
        max_turns: usize,
    ) {
        let key = TurnKey::new(sub_id, actor_id, turn_number);
        let span = if let Some(parent) = self.state.task_spans.get(&TaskKey::new(sub_id, actor_id))
        {
            tracing::info_span!(
                parent: parent,
                "autoagents.turn",
                submission_id = %sub_id,
                actor_id = %actor_id,
                turn_number = turn_number as i64,
                turn_max = max_turns as i64
            )
        } else {
            tracing::info_span!(
                "autoagents.turn",
                submission_id = %sub_id,
                actor_id = %actor_id,
                turn_number = turn_number as i64,
                turn_max = max_turns as i64
            )
        };

        self.state.turn_spans.insert(key, span);
        self.state
            .turn_start
            .insert(TurnKey::new(sub_id, actor_id, turn_number), Instant::now());
    }

    fn on_turn_completed(
        &mut self,
        sub_id: SubmissionId,
        actor_id: ActorID,
        turn_number: usize,
        final_turn: bool,
    ) {
        let key = TurnKey::new(sub_id, actor_id, turn_number);
        if let Some(span) = self.state.turn_spans.remove(&key) {
            span.set_attribute("turn.final", final_turn);
            span.set_status(Status::Ok);
            drop(span);
        }

        if let Some(start) = self.state.turn_start.remove(&key)
            && let Some(metrics) = &self.metrics
        {
            metrics.turn_duration.record(
                start.elapsed().as_secs_f64(),
                &self.task_metric_attributes(sub_id, actor_id),
            );
        }
    }

    fn on_tool_requested(
        &mut self,
        sub_id: SubmissionId,
        actor_id: ActorID,
        id: String,
        tool_name: String,
        arguments: String,
    ) {
        let key = ToolKey::new(sub_id, actor_id, id.clone());
        let span = if let Some(parent) = self.state.task_spans.get(&TaskKey::new(sub_id, actor_id))
        {
            tracing::info_span!(
                parent: parent,
                "autoagents.tool_call",
                submission_id = %sub_id,
                actor_id = %actor_id,
                tool_call_id = %id,
                tool_name = %tool_name
            )
        } else {
            tracing::info_span!(
                "autoagents.tool_call",
                submission_id = %sub_id,
                actor_id = %actor_id,
                tool_call_id = %id,
                tool_name = %tool_name
            )
        };

        let redacted_args = self.redact_value(arguments, self.redaction.redact_tool_arguments);
        let redacted_args_json = self.normalize_langfuse_json(redacted_args.clone());
        span.set_attribute("tool.arguments", redacted_args.clone());
        span.set_attribute("langfuse.observation.type", "tool");
        span.set_attribute("langfuse.observation.name", tool_name.clone());
        span.set_attribute("langfuse.observation.input", redacted_args_json);

        self.state.tool_spans.insert(key.clone(), span);
        self.state.tool_start.insert(key, Instant::now());
    }

    fn on_tool_completed(
        &mut self,
        sub_id: SubmissionId,
        actor_id: ActorID,
        id: String,
        tool_name: String,
        result: String,
    ) {
        let key = ToolKey::new(sub_id, actor_id, id.clone());
        if let Some(span) = self.state.tool_spans.remove(&key) {
            span.set_attribute("tool.name", tool_name.clone());
            let redacted = self.redact_value(result, self.redaction.redact_tool_results);
            let redacted_json = self.normalize_langfuse_json(redacted.clone());
            span.set_attribute("tool.result", redacted.clone());
            span.set_attribute("langfuse.observation.type", "tool");
            span.set_attribute("langfuse.observation.name", tool_name.clone());
            span.set_attribute("langfuse.observation.output", redacted_json);
            span.set_status(Status::Ok);
            drop(span);
        }

        if let Some(start) = self.state.tool_start.remove(&key)
            && let Some(metrics) = &self.metrics
        {
            metrics.tool_duration.record(
                start.elapsed().as_secs_f64(),
                &self.tool_metric_attributes(sub_id, actor_id, &tool_name),
            );
            metrics.tool_calls_total.add(
                1,
                &self.tool_metric_attributes(sub_id, actor_id, &tool_name),
            );
        }
    }

    fn on_tool_failed(
        &mut self,
        sub_id: SubmissionId,
        actor_id: ActorID,
        id: String,
        tool_name: String,
        error: String,
    ) {
        let key = ToolKey::new(sub_id, actor_id, id.clone());
        if let Some(span) = self.state.tool_spans.remove(&key) {
            span.set_attribute("tool.name", tool_name.clone());
            span.set_status(Status::error(error.clone()));
            span.set_attribute("error.message", error);
            span.set_attribute("langfuse.observation.type", "tool");
            span.set_attribute("langfuse.observation.name", tool_name.clone());
            drop(span);
        }

        if let Some(start) = self.state.tool_start.remove(&key)
            && let Some(metrics) = &self.metrics
        {
            metrics.tool_duration.record(
                start.elapsed().as_secs_f64(),
                &self.tool_metric_attributes(sub_id, actor_id, &tool_name),
            );
            metrics.tool_calls_total.add(
                1,
                &self.tool_metric_attributes(sub_id, actor_id, &tool_name),
            );
            metrics
                .errors_total
                .add(1, &[KeyValue::new("error.kind", "tool")]);
        }
    }

    fn redact_value(&self, value: String, enabled: bool) -> String {
        if enabled {
            "[REDACTED]".to_string()
        } else {
            value
        }
    }

    fn normalize_langfuse_json(&self, value: String) -> String {
        if serde_json::from_str::<JsonValue>(&value).is_ok() {
            value
        } else {
            serde_json::to_string(&value).unwrap_or(value)
        }
    }

    fn task_metric_attributes(&self, sub_id: SubmissionId, actor_id: ActorID) -> Vec<KeyValue> {
        let mut attrs = vec![
            KeyValue::new("submission_id", sub_id.to_string()),
            KeyValue::new("actor_id", actor_id.to_string()),
        ];
        if let Some(runtime_id) = &self.runtime_id {
            attrs.push(KeyValue::new("runtime_id", runtime_id.to_string()));
        }
        attrs
    }

    fn task_metric_attributes_with_status(
        &self,
        sub_id: SubmissionId,
        actor_id: ActorID,
        status: &'static str,
    ) -> Vec<KeyValue> {
        let mut attrs = self.task_metric_attributes(sub_id, actor_id);
        attrs.push(KeyValue::new("status", status));
        attrs
    }

    fn tool_metric_attributes(
        &self,
        sub_id: SubmissionId,
        actor_id: ActorID,
        tool_name: &str,
    ) -> Vec<KeyValue> {
        let mut attrs = self.task_metric_attributes(sub_id, actor_id);
        attrs.push(KeyValue::new("tool.name", tool_name.to_string()));
        attrs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry_sdk::trace::InMemorySpanExporterBuilder;
    use serde_json::json;

    #[test]
    fn maps_events_to_spans() {
        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        opentelemetry::global::set_tracer_provider(provider);
        let otel_layer = tracing_opentelemetry::layer()
            .with_tracer(opentelemetry::global::tracer("autoagents.telemetry.test"));
        let _ = tracing_subscriber::Registry::default()
            .with(otel_layer)
            .try_init();

        let mut mapper = EventMapper::new(None, RedactionConfig::default(), None);

        let sub_id = SubmissionId::new_v4();
        let actor_id = ActorID::new_v4();

        mapper.handle_event(Event::TaskStarted {
            sub_id,
            actor_id,
            actor_name: "test-agent".to_string(),
            task_description: "test task".to_string(),
        });
        mapper.handle_event(Event::TurnStarted {
            sub_id,
            actor_id,
            turn_number: 0,
            max_turns: 3,
        });
        mapper.handle_event(Event::ToolCallRequested {
            sub_id,
            actor_id,
            id: "call_1".to_string(),
            tool_name: "lookup".to_string(),
            arguments: "{\"q\":\"value\"}".to_string(),
        });
        mapper.handle_event(Event::ToolCallCompleted {
            sub_id,
            actor_id,
            id: "call_1".to_string(),
            tool_name: "lookup".to_string(),
            result: json!({"ok": true}),
        });
        mapper.handle_event(Event::TurnCompleted {
            sub_id,
            actor_id,
            turn_number: 0,
            final_turn: true,
        });
        mapper.handle_event(Event::TaskComplete {
            sub_id,
            actor_id,
            actor_name: "test-agent".to_string(),
            result: "{\"value\": 1}".to_string(),
        });

        mapper.flush();

        let spans = exporter.get_finished_spans().expect("spans available");
        let names: Vec<_> = spans.iter().map(|span| span.name.as_ref()).collect();

        assert!(names.contains(&"autoagents.task"));
        assert!(names.contains(&"autoagents.turn"));
        assert!(names.contains(&"autoagents.tool_call"));
    }
}
