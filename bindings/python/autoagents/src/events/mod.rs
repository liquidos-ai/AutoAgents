use crate::convert::json_value_to_py;
use autoagents_core::utils::BoxEventStream;
use autoagents_protocol::Event;
use futures::FutureExt;
use futures::StreamExt;
use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::prelude::*;
use pyo3::types::PyList;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::{BroadcastStream, errors::BroadcastStreamRecvError};

fn event_to_payload(event: Event) -> PyResult<Value> {
    match event {
        Event::PublishMessage { .. } => {
            unreachable!("PublishMessage must be filtered before event_to_payload")
        }
        Event::NewTask { actor_id, task } => Ok(json!({
            "kind": "new_task",
            "actor_id": actor_id.to_string(),
            "prompt": task.prompt,
            "system_prompt": task.system_prompt,
        })),
        Event::TaskStarted {
            sub_id,
            actor_id,
            actor_name,
            task_description,
        } => Ok(task_payload(
            "task_started",
            sub_id,
            actor_id,
            json!({
                "actor_name": actor_name,
                "task_description": task_description,
            }),
        )),
        Event::TaskComplete {
            sub_id,
            actor_id,
            actor_name,
            result,
        } => Ok(task_payload(
            "task_complete",
            sub_id,
            actor_id,
            json!({
                "actor_name": actor_name,
                "result": result,
            }),
        )),
        Event::TaskError {
            sub_id,
            actor_id,
            error,
        } => Ok(task_payload(
            "task_error",
            sub_id,
            actor_id,
            json!({
                "error": error,
            }),
        )),
        Event::SendMessage { message, actor_id } => Ok(json!({
            "kind": "send_message",
            "actor_id": actor_id.to_string(),
            "message": message,
        })),
        Event::ToolCallRequested {
            sub_id,
            actor_id,
            id,
            tool_name,
            arguments,
        } => Ok(tool_payload(
            "tool_call_requested",
            sub_id,
            actor_id,
            id,
            tool_name,
            json!({ "arguments": arguments }),
        )),
        Event::ToolCallCompleted {
            sub_id,
            actor_id,
            id,
            tool_name,
            result,
        } => Ok(tool_payload(
            "tool_call_completed",
            sub_id,
            actor_id,
            id,
            tool_name,
            json!({ "result": result }),
        )),
        Event::ToolCallFailed {
            sub_id,
            actor_id,
            id,
            tool_name,
            error,
        } => Ok(tool_payload(
            "tool_call_failed",
            sub_id,
            actor_id,
            id,
            tool_name,
            json!({ "error": error }),
        )),
        Event::TurnStarted {
            sub_id,
            actor_id,
            turn_number,
            max_turns,
        } => Ok(task_payload(
            "turn_started",
            sub_id,
            actor_id,
            json!({
                "turn_number": turn_number,
                "max_turns": max_turns,
            }),
        )),
        Event::TurnCompleted {
            sub_id,
            actor_id,
            turn_number,
            final_turn,
        } => Ok(task_payload(
            "turn_completed",
            sub_id,
            actor_id,
            json!({
                "turn_number": turn_number,
                "final_turn": final_turn,
            }),
        )),
        Event::StreamChunk { sub_id, chunk } => stream_chunk_payload(sub_id, chunk),
        Event::StreamToolCall { sub_id, tool_call } => Ok(json!({
            "kind": "stream_tool_call",
            "sub_id": sub_id.to_string(),
            "tool_call": tool_call,
        })),
        Event::StreamComplete { sub_id } => Ok(json!({
            "kind": "stream_complete",
            "sub_id": sub_id.to_string(),
        })),
    }
}

fn task_payload(kind: &str, sub_id: impl ToString, actor_id: impl ToString, extra: Value) -> Value {
    let mut payload = json!({
        "kind": kind,
        "sub_id": sub_id.to_string(),
        "actor_id": actor_id.to_string(),
    });
    merge_payload(&mut payload, extra);
    payload
}

fn tool_payload(
    kind: &str,
    sub_id: impl ToString,
    actor_id: impl ToString,
    id: String,
    tool_name: String,
    extra: Value,
) -> Value {
    let mut payload = task_payload(
        kind,
        sub_id,
        actor_id,
        json!({
            "id": id,
            "tool_name": tool_name,
        }),
    );
    merge_payload(&mut payload, extra);
    payload
}

fn stream_chunk_payload(sub_id: impl ToString, chunk: impl serde::Serialize) -> PyResult<Value> {
    let chunk = serde_json::to_value(chunk)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(json!({
        "kind": "stream_chunk",
        "sub_id": sub_id.to_string(),
        "chunk": chunk,
    }))
}

fn merge_payload(payload: &mut Value, extra: Value) {
    if let (Some(base), Some(extra)) = (payload.as_object_mut(), extra.as_object()) {
        base.extend(extra.clone());
    }
}

pub(crate) fn event_to_py(py: Python<'_>, event: Event) -> PyResult<Py<PyAny>> {
    let payload = event_to_payload(event)?;
    json_value_to_py(py, &payload)
}

pub(crate) fn events_to_py_list(py: Python<'_>, events: Vec<Event>) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for event in events {
        if matches!(event, Event::PublishMessage { .. }) {
            continue;
        }
        list.append(event_to_py(py, event)?)?;
    }
    Ok(list.into_any().unbind())
}

pub(crate) struct PySharedEventStream {
    tx: broadcast::Sender<Event>,
    flush_tx: mpsc::UnboundedSender<oneshot::Sender<()>>,
    _task: JoinHandle<()>,
}

impl PySharedEventStream {
    pub(crate) fn new(mut stream: BoxEventStream<Event>) -> Self {
        let (tx, _) = broadcast::channel(128);
        let (flush_tx, mut flush_rx) = mpsc::unbounded_channel::<oneshot::Sender<()>>();
        let tx_clone = tx.clone();
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    maybe_event = stream.next() => match maybe_event {
                        Some(event) => {
                            let _ = tx_clone.send(event);
                        }
                        None => {
                            while let Some(ack) = flush_rx.recv().await {
                                let _ = ack.send(());
                            }
                            return;
                        }
                    },
                    maybe_ack = flush_rx.recv() => match maybe_ack {
                        Some(ack) => {
                            let stream_ended = drain_ready_events(&mut stream, &tx_clone);
                            let _ = ack.send(());
                            if stream_ended {
                                return;
                            }
                        }
                        None => {
                            // Keep forwarding events even if no more explicit flush waiters exist.
                            while let Some(event) = stream.next().await {
                                let _ = tx_clone.send(event);
                            }
                            return;
                        }
                    }
                }
            }
        });

        Self {
            tx,
            flush_tx,
            _task: task,
        }
    }

    pub(crate) fn subscribe_receiver(&self) -> broadcast::Receiver<Event> {
        self.tx.subscribe()
    }

    pub(crate) async fn flush(&self) {
        let (ack_tx, ack_rx) = oneshot::channel();
        if self.flush_tx.send(ack_tx).is_err() {
            return;
        }
        let _ = ack_rx.await;
    }

    pub(crate) fn subscribe(&self) -> BoxEventStream<Event> {
        let rx = self.subscribe_receiver();
        let stream = BroadcastStream::new(rx)
            .filter_map(|item: Result<Event, BroadcastStreamRecvError>| async move { item.ok() });
        Box::pin(stream)
    }

    pub(crate) fn subscribe_py(&self) -> PyEventStream {
        PyEventStream {
            rx: Arc::new(tokio::sync::Mutex::new(self.subscribe())),
        }
    }
}

fn drain_ready_events(stream: &mut BoxEventStream<Event>, tx: &broadcast::Sender<Event>) -> bool {
    loop {
        match stream.next().now_or_never() {
            Some(Some(event)) => {
                let _ = tx.send(event);
            }
            Some(None) => return true,
            None => return false,
        }
    }
}

#[pyclass(name = "EventStream")]
pub struct PyEventStream {
    pub rx: Arc<tokio::sync::Mutex<BoxEventStream<Event>>>,
}

#[pymethods]
impl PyEventStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rx = Arc::clone(&self.rx);

        crate::async_bridge::future_into_py(py, async move {
            loop {
                let next = {
                    let mut guard = rx.lock().await;
                    guard.next().await
                };

                match next {
                    Some(Event::PublishMessage { .. }) => continue,
                    Some(event) => return Python::attach(|py| event_to_py(py, event)),
                    None => return Err(PyStopAsyncIteration::new_err("stream ended")),
                }
            }
        })
    }
}
