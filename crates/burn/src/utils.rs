use futures_core::Stream;
#[cfg(target_arch = "wasm32")]
use futures_util::stream::StreamExt;
use std::pin::Pin;

// -----------------------------
// Channel aliases
// -----------------------------
#[cfg(not(target_arch = "wasm32"))]
pub use tokio::sync::mpsc::Receiver;

#[cfg(target_arch = "wasm32")]
pub use futures::channel::mpsc::{channel, Receiver, Sender};

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

// Platform-specific imports

#[cfg(target_arch = "wasm32")]
use futures::channel::mpsc::unbounded as unbounded_channel;

#[cfg(target_arch = "wasm32")]
pub type CustomMutex<T> = futures::lock::Mutex<T>;
#[cfg(not(target_arch = "wasm32"))]
pub type CustomMutex<T> = tokio::sync::Mutex<T>;

// Platform-specific spawn_blocking functions
#[cfg(not(target_arch = "wasm32"))]
pub async fn spawn_blocking<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| format!("Task join error: {}", e))
}

#[cfg(target_arch = "wasm32")]
pub async fn spawn_blocking<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce() -> R + 'static,
    R: 'static,
{
    // In WASM, we can't spawn blocking tasks to separate threads
    // so we just execute the function directly
    Ok(f())
}
