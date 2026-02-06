use crate::config::RedactionConfig;
use crate::providers::TelemetryAttributeProvider;
use crate::runner::metrics::TelemetryMetrics;
use autoagents_protocol::{ActorID, Event, RuntimeID, SubmissionId};
use opentelemetry::KeyValue;
use opentelemetry::Value;
use opentelemetry::trace::Status;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;
use tracing_opentelemetry::OpenTelemetrySpanExt;

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

/// Translates protocol events into spans and metrics.
pub(crate) struct EventMapper {
    metrics: Option<TelemetryMetrics>,
    redaction: RedactionConfig,
    runtime_id: Option<RuntimeID>,
    attributes: Option<Arc<dyn TelemetryAttributeProvider>>,
    flush_tx: Option<tokio::sync::mpsc::UnboundedSender<()>>,
    state: TelemetryState,
}

impl EventMapper {
    pub(crate) fn new(
        metrics: Option<TelemetryMetrics>,
        redaction: RedactionConfig,
        runtime_id: Option<RuntimeID>,
        attributes: Option<Arc<dyn TelemetryAttributeProvider>>,
        flush_tx: Option<tokio::sync::mpsc::UnboundedSender<()>>,
    ) -> Self {
        Self {
            metrics,
            redaction,
            runtime_id,
            attributes,
            flush_tx,
            state: TelemetryState::new(),
        }
    }

    pub(crate) fn handle_event(&mut self, event: Event) {
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

    pub(crate) fn flush(&mut self) {
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
        span.set_attribute("task.description", description.clone());
        self.apply_attributes(
            &span,
            self.attributes
                .as_ref()
                .map(|provider| provider.task_started_attributes(&actor_name, &description)),
        );

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
            span.set_attribute("task.result", redacted.clone());
            self.apply_attributes(
                &span,
                self.attributes
                    .as_ref()
                    .map(|provider| provider.task_completed_attributes(&redacted)),
            );
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

        if let Some(tx) = &self.flush_tx {
            let _ = tx.send(());
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

        if let Some(tx) = &self.flush_tx {
            let _ = tx.send(());
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
        span.set_attribute("tool.arguments", redacted_args.clone());
        self.apply_attributes(
            &span,
            self.attributes
                .as_ref()
                .map(|provider| provider.tool_started_attributes(&tool_name, &redacted_args)),
        );

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
            span.set_attribute("tool.result", redacted.clone());
            self.apply_attributes(
                &span,
                self.attributes
                    .as_ref()
                    .map(|provider| provider.tool_completed_attributes(&tool_name, &redacted)),
            );
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
            span.set_attribute("error.message", error.clone());
            self.apply_attributes(
                &span,
                self.attributes
                    .as_ref()
                    .map(|provider| provider.tool_failed_attributes(&tool_name, &error)),
            );
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

    fn apply_attributes(
        &self,
        span: &tracing::Span,
        attributes: Option<Vec<(&'static str, Value)>>,
    ) {
        if let Some(attributes) = attributes {
            for (key, value) in attributes {
                span.set_attribute(key, value);
            }
        }
    }

    pub(crate) fn has_open_tasks(&self) -> bool {
        !self.state.task_spans.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::trace::TracerProvider;
    use opentelemetry_sdk::trace::InMemorySpanExporterBuilder;
    use opentelemetry_sdk::trace::SpanData;
    use serde_json::json;
    use tracing_subscriber::prelude::*;

    fn find_span<'a>(spans: &'a [SpanData], name: &str) -> &'a SpanData {
        spans
            .iter()
            .find(|span| span.name.as_ref() == name)
            .unwrap_or_else(|| panic!("missing span: {name}"))
    }

    fn attr_value(span: &SpanData, key: &str) -> Option<Value> {
        span.attributes
            .iter()
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| kv.value.clone())
    }

    #[test]
    fn maps_events_to_spans() {
        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer = provider.tracer("autoagents.telemetry.test");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        let subscriber = tracing_subscriber::Registry::default().with(otel_layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let mut mapper = EventMapper::new(None, RedactionConfig::default(), None, None, None);

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
        let task_span = find_span(&spans, "autoagents.task");
        let turn_span = find_span(&spans, "autoagents.turn");
        let tool_span = find_span(&spans, "autoagents.tool_call");

        let task_id = task_span.span_context.span_id();
        assert_eq!(turn_span.parent_span_id, task_id);
        assert_eq!(tool_span.parent_span_id, task_id);
    }

    #[derive(Debug)]
    struct TestAttributes;

    impl TelemetryAttributeProvider for TestAttributes {
        fn task_started_attributes(
            &self,
            actor_name: &str,
            _task_input: &str,
        ) -> Vec<(&'static str, Value)> {
            vec![("provider.task.actor", Value::from(actor_name.to_string()))]
        }

        fn task_completed_attributes(&self, task_output: &str) -> Vec<(&'static str, Value)> {
            vec![("provider.task.output", Value::from(task_output.to_string()))]
        }

        fn tool_started_attributes(
            &self,
            tool_name: &str,
            _tool_args: &str,
        ) -> Vec<(&'static str, Value)> {
            vec![("provider.tool.name", Value::from(tool_name.to_string()))]
        }

        fn tool_completed_attributes(
            &self,
            tool_name: &str,
            tool_output: &str,
        ) -> Vec<(&'static str, Value)> {
            vec![
                ("provider.tool.name", Value::from(tool_name.to_string())),
                ("provider.tool.output", Value::from(tool_output.to_string())),
            ]
        }

        fn tool_failed_attributes(
            &self,
            tool_name: &str,
            error: &str,
        ) -> Vec<(&'static str, Value)> {
            vec![
                ("provider.tool.name", Value::from(tool_name.to_string())),
                ("provider.tool.error", Value::from(error.to_string())),
            ]
        }
    }

    #[test]
    fn provider_attributes_are_applied() {
        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer = provider.tracer("autoagents.telemetry.test.provider");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        let subscriber = tracing_subscriber::Registry::default().with(otel_layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let attributes: Option<Arc<dyn TelemetryAttributeProvider>> =
            Some(Arc::new(TestAttributes));
        let mut mapper = EventMapper::new(None, RedactionConfig::default(), None, attributes, None);

        let sub_id = SubmissionId::new_v4();
        let actor_id = ActorID::new_v4();

        mapper.handle_event(Event::TaskStarted {
            sub_id,
            actor_id,
            actor_name: "provider-test".to_string(),
            task_description: "task".to_string(),
        });
        mapper.handle_event(Event::ToolCallRequested {
            sub_id,
            actor_id,
            id: "call_1".to_string(),
            tool_name: "lookup".to_string(),
            arguments: "{\"q\":\"value\"}".to_string(),
        });
        mapper.handle_event(Event::ToolCallFailed {
            sub_id,
            actor_id,
            id: "call_1".to_string(),
            tool_name: "lookup".to_string(),
            error: "oops".to_string(),
        });
        mapper.handle_event(Event::TaskComplete {
            sub_id,
            actor_id,
            actor_name: "provider-test".to_string(),
            result: "done".to_string(),
        });

        mapper.flush();

        let spans = exporter.get_finished_spans().expect("spans available");
        let task_span = find_span(&spans, "autoagents.task");
        let tool_span = find_span(&spans, "autoagents.tool_call");

        assert!(attr_value(task_span, "provider.task.actor").is_some());
        assert!(attr_value(tool_span, "provider.tool.name").is_some());
    }

    #[test]
    fn redaction_is_applied() {
        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer = provider.tracer("autoagents.telemetry.test.redaction");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        let subscriber = tracing_subscriber::Registry::default().with(otel_layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let redaction = RedactionConfig {
            redact_task_inputs: true,
            redact_tool_arguments: true,
            ..Default::default()
        };
        let mut mapper = EventMapper::new(None, redaction, None, None, None);

        let sub_id = SubmissionId::new_v4();
        let actor_id = ActorID::new_v4();

        mapper.handle_event(Event::TaskStarted {
            sub_id,
            actor_id,
            actor_name: "test-agent".to_string(),
            task_description: "secret task".to_string(),
        });
        mapper.handle_event(Event::ToolCallRequested {
            sub_id,
            actor_id,
            id: "call_1".to_string(),
            tool_name: "lookup".to_string(),
            arguments: "{\"q\":\"secret\"}".to_string(),
        });

        mapper.flush();

        let spans = exporter.get_finished_spans().expect("spans available");
        let task_span = find_span(&spans, "autoagents.task");
        let tool_span = find_span(&spans, "autoagents.tool_call");

        assert_eq!(
            attr_value(task_span, "task.description"),
            Some(Value::from("[REDACTED]"))
        );
        assert_eq!(
            attr_value(tool_span, "tool.arguments"),
            Some(Value::from("[REDACTED]"))
        );
    }

    #[test]
    fn tool_failure_sets_error_status() {
        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer = provider.tracer("autoagents.telemetry.test.error");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        let subscriber = tracing_subscriber::Registry::default().with(otel_layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let mut mapper = EventMapper::new(None, RedactionConfig::default(), None, None, None);

        let sub_id = SubmissionId::new_v4();
        let actor_id = ActorID::new_v4();

        mapper.handle_event(Event::ToolCallRequested {
            sub_id,
            actor_id,
            id: "call_1".to_string(),
            tool_name: "lookup".to_string(),
            arguments: "{\"q\":\"value\"}".to_string(),
        });
        mapper.handle_event(Event::ToolCallFailed {
            sub_id,
            actor_id,
            id: "call_1".to_string(),
            tool_name: "lookup".to_string(),
            error: "boom".to_string(),
        });

        mapper.flush();

        let spans = exporter.get_finished_spans().expect("spans available");
        let tool_span = find_span(&spans, "autoagents.tool_call");

        assert!(matches!(tool_span.status, Status::Error { .. }));
        assert_eq!(
            attr_value(tool_span, "error.message"),
            Some(Value::from("boom"))
        );
    }
}
