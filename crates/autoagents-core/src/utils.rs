use futures_core::Stream;
#[cfg(target_arch = "wasm32")]
use futures_util::stream::StreamExt;
use std::pin::Pin;

// -----------------------------
// Channel aliases
// -----------------------------
#[cfg(not(target_arch = "wasm32"))]
pub use tokio::sync::mpsc::{Receiver, Sender, channel};

#[cfg(target_arch = "wasm32")]
pub use futures::channel::mpsc::{Receiver, Sender, channel};

// -----------------------------
// Unified boxed stream type
// -----------------------------
#[cfg(not(target_arch = "wasm32"))]
pub type BoxEventStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Sync>>;

#[cfg(target_arch = "wasm32")]
pub type BoxEventStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;

// -----------------------------
// Conversion helpers
// -----------------------------
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn receiver_into_stream<T: 'static + Send>(rx: Receiver<T>) -> BoxEventStream<T> {
    use tokio_stream::wrappers::ReceiverStream;
    Box::pin(ReceiverStream::new(rx))
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn receiver_into_stream<T: 'static + Send>(rx: Receiver<T>) -> BoxEventStream<T> {
    rx.boxed()
}

// Platform-specific spawn functions
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn spawn_future<F>(fut: F) -> tokio::task::JoinHandle<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    tokio::spawn(fut)
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn spawn_future<F>(fut: F)
where
    F: std::future::Future<Output = ()> + 'static,
{
    wasm_bindgen_futures::spawn_local(fut)
}
