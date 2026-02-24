use autoagents_llm::chat::StreamChunk as LlmStreamChunk;
use autoagents_protocol::StreamChunk;
use autoagents_protocol::{ActorID, Event, SubmissionId};
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

    /// Send task completed event with a JSON value result
    pub async fn send_task_completed_value(
        tx: &Option<mpsc::Sender<Event>>,
        sub_id: SubmissionId,
        actor_id: ActorID,
        actor_name: String,
        result: &Value,
    ) -> Result<(), serde_json::Error> {
        let result = serde_json::to_string_pretty(result)?;
        Self::send_task_completed(tx, sub_id, actor_id, actor_name, result).await;
        Ok(())
    }

    /// Send task error event
    pub async fn send_task_error(
        tx: &Option<mpsc::Sender<Event>>,
        sub_id: SubmissionId,
        actor_id: ActorID,
        error: String,
    ) {
        Self::send(
            tx,
            Event::TaskError {
                sub_id,
                actor_id,
                error,
            },
        )
        .await;
    }

    /// Send turn started event
    pub async fn send_turn_started(
        tx: &Option<mpsc::Sender<Event>>,
        sub_id: SubmissionId,
        actor_id: ActorID,
        turn_number: usize,
        max_turns: usize,
    ) {
        Self::send(
            tx,
            Event::TurnStarted {
                sub_id,
                actor_id,
                turn_number,
                max_turns,
            },
        )
        .await;
    }

    /// Send turn completed event
    pub async fn send_turn_completed(
        tx: &Option<mpsc::Sender<Event>>,
        sub_id: SubmissionId,
        actor_id: ActorID,
        turn_number: usize,
        final_turn: bool,
    ) {
        Self::send(
            tx,
            Event::TurnCompleted {
                sub_id,
                actor_id,
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
        chunk: LlmStreamChunk,
    ) {
        let chunk: StreamChunk = chunk.into();
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

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_llm::chat::StreamChunk as LlmStreamChunk;
    use autoagents_protocol::StreamChunk as ProtocolStreamChunk;

    #[tokio::test]
    async fn stream_chunk_is_converted_to_protocol() {
        let (tx, mut rx) = mpsc::channel::<Event>(1);
        let tx = Some(tx);
        let sub_id = SubmissionId::new_v4();
        let chunk = LlmStreamChunk::Text("hello".to_string());

        let expected: ProtocolStreamChunk = chunk.clone().into();
        EventHelper::send_stream_chunk(&tx, sub_id, chunk.clone()).await;

        let event = rx.recv().await.expect("event");
        match event {
            Event::StreamChunk { sub_id: id, chunk } => {
                assert_eq!(id, sub_id);
                let expected_json = serde_json::to_string(&expected).unwrap();
                let actual_json = serde_json::to_string(&chunk).unwrap();
                assert_eq!(actual_json, expected_json);
            }
            _ => panic!("unexpected event"),
        }
    }
}
