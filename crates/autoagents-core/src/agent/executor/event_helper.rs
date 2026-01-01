use crate::protocol::{ActorID, Event, SubmissionId};
use autoagents_llm::chat::StreamChunk;
use serde_json::Value;

#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc;

#[cfg(target_arch = "wasm32")]
use futures::channel::mpsc;

#[cfg(target_arch = "wasm32")]
use futures::SinkExt;

/// Helper for managing event emissions
pub struct EventHelper;

impl EventHelper {
    /// Send an event if sender is available
    pub async fn send(tx: &Option<mpsc::Sender<Event>>, event: Event) {
        if let Some(tx) = tx {
            #[cfg(not(target_arch = "wasm32"))]
            //TODO: WASM Targets currently does not support event handling
            let _ = tx.send(event).await;
        }
    }

    /// Send task started event
    pub async fn send_task_started(
        tx: &Option<mpsc::Sender<Event>>,
        sub_id: SubmissionId,
        actor_id: ActorID,
        actor_name: String,
        task_description: String,
    ) {
        Self::send(
            tx,
            Event::TaskStarted {
                sub_id,
                actor_id,
                actor_name,
                task_description,
            },
        )
        .await;
    }

    /// Send task started event
    pub async fn send_task_completed(
        tx: &Option<mpsc::Sender<Event>>,
        sub_id: SubmissionId,
        actor_id: ActorID,
        actor_name: String,
        result: String,
    ) {
        Self::send(
            tx,
            Event::TaskComplete {
                sub_id,
                result,
                actor_id,
                actor_name,
            },
        )
        .await;
    }

    /// Send turn started event
    pub async fn send_turn_started(
        tx: &Option<mpsc::Sender<Event>>,
        turn_number: usize,
        max_turns: usize,
    ) {
        Self::send(
            tx,
            Event::TurnStarted {
                turn_number,
                max_turns,
            },
        )
        .await;
    }

    /// Send turn completed event
    pub async fn send_turn_completed(
        tx: &Option<mpsc::Sender<Event>>,
        turn_number: usize,
        final_turn: bool,
    ) {
        Self::send(
            tx,
            Event::TurnCompleted {
                turn_number,
                final_turn,
            },
        )
        .await;
    }

    /// Send stream chunk event
    pub async fn send_stream_chunk(
        tx: &Option<mpsc::Sender<Event>>,
        sub_id: SubmissionId,
        chunk: StreamChunk,
    ) {
        Self::send(tx, Event::StreamChunk { sub_id, chunk }).await;
    }

    /// Send stream tool call event
    pub async fn send_stream_tool_call(
        tx: &Option<mpsc::Sender<Event>>,
        sub_id: SubmissionId,
        tool_call: Value,
    ) {
        Self::send(tx, Event::StreamToolCall { sub_id, tool_call }).await;
    }

    /// Send stream complete event
    pub async fn send_stream_complete(tx: &Option<mpsc::Sender<Event>>, sub_id: SubmissionId) {
        Self::send(tx, Event::StreamComplete { sub_id }).await;
    }
}
