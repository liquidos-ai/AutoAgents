use autoagents_core::utils::BoxEventStream;
use autoagents_protocol::Event;
use futures_util::StreamExt;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::{BroadcastStream, errors::BroadcastStreamRecvError};

/// Broadcasts a single event stream to multiple subscribers.
pub struct EventFanout {
    tx: broadcast::Sender<Event>,
    _task: JoinHandle<()>,
}

impl EventFanout {
    /// Spawn a background task that forwards events into a broadcast channel.
    pub fn new(mut event_stream: BoxEventStream<Event>, buffer: usize) -> Self {
        let (tx, _) = broadcast::channel(buffer);
        let tx_clone = tx.clone();
        let task = tokio::spawn(async move {
            while let Some(event) = event_stream.next().await {
                let _ = tx_clone.send(event);
            }
        });

        Self { tx, _task: task }
    }

    /// Create a new stream receiver over the broadcast channel.
    pub fn subscribe(&self) -> BoxEventStream<Event> {
        let rx = self.tx.subscribe();
        let stream = BroadcastStream::new(rx)
            .filter_map(|item: Result<Event, BroadcastStreamRecvError>| async move { item.ok() });
        Box::pin(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use tokio_stream::iter;

    #[tokio::test]
    async fn test_event_fanout_forwards_events() {
        let event = Event::TaskStarted {
            sub_id: autoagents_protocol::SubmissionId::new_v4(),
            actor_id: autoagents_protocol::ActorID::new_v4(),
            actor_name: "agent".to_string(),
            task_description: "task".to_string(),
        };
        let stream = Box::pin(iter(vec![event.clone()]));
        let fanout = EventFanout::new(stream, 8);
        let mut rx = fanout.subscribe();

        let received = rx.next().await.expect("event");
        match received {
            Event::TaskStarted { actor_name, .. } => {
                assert_eq!(actor_name, "agent");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
}
