use futures::Stream;
use std::pin::Pin;
use tokio::sync::mpsc::Receiver;

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
