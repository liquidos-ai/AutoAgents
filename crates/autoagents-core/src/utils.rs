use futures_core::Stream;
#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
use std::future::Future;
use std::pin::Pin;
#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
use std::task::{Context, Poll};

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
#[cfg(not(all(target_arch = "wasm32", target_os = "wasi")))]
pub type BoxRuntimeStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;

#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
pub type BoxRuntimeStream<T> = Pin<Box<dyn Stream<Item = T>>>;

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
    Box::pin(rx)
}

#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
struct WasiDrivenStream<T> {
    producer: Pin<Box<dyn Future<Output = ()>>>,
    receiver: BoxRuntimeStream<T>,
    producer_done: bool,
}

#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
impl<T> Stream for WasiDrivenStream<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if !self.producer_done && self.producer.as_mut().poll(cx).is_ready() {
            self.producer_done = true;
        }

        match self.receiver.as_mut().poll_next(cx) {
            Poll::Ready(item) => Poll::Ready(item),
            Poll::Pending if self.producer_done => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
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

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
pub(crate) fn spawn_future<F>(fut: F)
where
    F: std::future::Future<Output = ()> + 'static,
{
    wasm_bindgen_futures::spawn_local(fut)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn stream_from_producer<T, F>(rx: Receiver<T>, producer: F) -> BoxRuntimeStream<T>
where
    T: 'static + Send,
    F: std::future::Future<Output = ()> + Send + 'static,
{
    use tokio_stream::wrappers::ReceiverStream;

    spawn_future(producer);
    Box::pin(ReceiverStream::new(rx))
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
pub(crate) fn stream_from_producer<T, F>(rx: Receiver<T>, producer: F) -> BoxRuntimeStream<T>
where
    T: 'static + Send,
    F: std::future::Future<Output = ()> + 'static,
{
    spawn_future(producer);
    Box::pin(rx)
}

#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
pub(crate) fn stream_from_producer<T, F>(rx: Receiver<T>, producer: F) -> BoxRuntimeStream<T>
where
    T: 'static + Send,
    F: std::future::Future<Output = ()> + 'static,
{
    Box::pin(WasiDrivenStream {
        producer: Box::pin(producer),
        receiver: Box::pin(rx),
        producer_done: false,
    })
}

#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
pub fn block_on_local_executor<F>(future: F) -> F::Output
where
    F: std::future::Future,
{
    // Keep the public WASI entrypoint stable for downstream crates while the
    // streaming internals run cooperatively instead of on a detached executor.
    futures::executor::block_on(future)
}
