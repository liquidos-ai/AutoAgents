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
