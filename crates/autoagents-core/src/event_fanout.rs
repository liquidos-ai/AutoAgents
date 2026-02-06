#[cfg(not(target_arch = "wasm32"))]
use crate::utils::BoxEventStream;
#[cfg(not(target_arch = "wasm32"))]
use autoagents_protocol::Event;
#[cfg(not(target_arch = "wasm32"))]
use futures_util::StreamExt;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::broadcast;
#[cfg(not(target_arch = "wasm32"))]
use tokio::task::JoinHandle;
#[cfg(not(target_arch = "wasm32"))]
use tokio_stream::wrappers::{BroadcastStream, errors::BroadcastStreamRecvError};

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct EventFanout {
    tx: broadcast::Sender<Event>,
    _task: JoinHandle<()>,
}

#[cfg(not(target_arch = "wasm32"))]
impl EventFanout {
    pub(crate) fn new(mut event_stream: BoxEventStream<Event>, buffer: usize) -> Self {
        let (tx, _) = broadcast::channel(buffer);
        let tx_clone = tx.clone();
        let task = tokio::spawn(async move {
            while let Some(event) = event_stream.next().await {
                let _ = tx_clone.send(event);
            }
        });

        Self { tx, _task: task }
    }

    pub(crate) fn subscribe(&self) -> BoxEventStream<Event> {
        let rx = self.tx.subscribe();
        let stream = BroadcastStream::new(rx)
            .filter_map(|item: Result<Event, BroadcastStreamRecvError>| async move { item.ok() });
        Box::pin(stream)
    }
}
