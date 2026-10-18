use futures_core::Stream;
use std::future::Future;
use std::pin::Pin;
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

// The consumer owns the producer, including any nested agent/provider streams.
// Dropping the stream synchronously drops its nested producer futures.
// Executors that offload blocking work must propagate cancellation to that work.
// The producer future advances only while the stream is polled, on all runtimes.
struct ProducerStream<T, F> {
    producer: Option<Pin<Box<F>>>,
    receiver: BoxRuntimeStream<T>,
}

impl<T, F: Future<Output = ()>> Stream for ProducerStream<T, F> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(producer) = self.producer.as_mut()
            && producer.as_mut().poll(cx).is_ready()
        {
            self.producer = None;
        }

        match self.receiver.as_mut().poll_next(cx) {
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            // Drain buffered items first, and wait for producer cleanup even if
            // it closed its sender before completing.
            Poll::Ready(None) if self.producer.is_none() => Poll::Ready(None),
            _ => Poll::Pending,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn stream_from_producer<T, F>(rx: Receiver<T>, producer: F) -> BoxRuntimeStream<T>
where
    T: 'static + Send,
    F: std::future::Future<Output = ()> + Send + 'static,
{
    use tokio_stream::wrappers::ReceiverStream;

    Box::pin(ProducerStream {
        producer: Some(Box::pin(producer)),
        receiver: Box::pin(ReceiverStream::new(rx)),
    })
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
pub(crate) fn stream_from_producer<T, F>(rx: Receiver<T>, producer: F) -> BoxRuntimeStream<T>
where
    T: 'static + Send,
    F: std::future::Future<Output = ()> + 'static,
{
    // Browser futures are thread-affine. SendWrapper preserves the existing
    // Send stream interface while checking that polling/drop stay on that thread.
    Box::pin(send_wrapper::SendWrapper::new(ProducerStream {
        producer: Some(Box::pin(producer)),
        receiver: Box::pin(rx),
    }))
}

#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
pub(crate) fn stream_from_producer<T, F>(rx: Receiver<T>, producer: F) -> BoxRuntimeStream<T>
where
    T: 'static + Send,
    F: std::future::Future<Output = ()> + 'static,
{
    Box::pin(ProducerStream {
        producer: Some(Box::pin(producer)),
        receiver: Box::pin(rx),
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    #[tokio::test]
    async fn receiver_into_stream_yields_messages_in_order() {
        let (tx, rx) = channel(2);
        tx.send(1).await.expect("send first message");
        tx.send(2).await.expect("send second message");
        drop(tx);

        let values: Vec<_> = receiver_into_stream(rx).collect().await;
        assert_eq!(values, vec![1, 2]);
    }

    #[tokio::test]
    async fn stream_from_producer_drains_values_across_backpressure() {
        let (tx, rx) = channel(1);
        let stream = stream_from_producer(rx, async move {
            for value in [1, 2, 3] {
                tx.send(value).await.expect("send produced value");
            }
        });

        let values: Vec<_> = stream.collect().await;
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn completed_producer_drains_buffers_across_cooperative_yields() {
        // Tokio may yield Pending with buffered values when the current task's
        // cooperative budget runs out. Producer completion is not receiver EOF.
        for count in [129, 1_025, 1_100] {
            let (tx, rx) = channel(100);
            let inner = stream_from_producer(rx, async move {
                for value in 0..count {
                    tx.send(value).await.unwrap();
                }
            });
            let (tx, rx) = channel(100);
            let outer = stream_from_producer(rx, async move {
                futures_util::pin_mut!(inner);
                while let Some(value) = inner.next().await {
                    tx.send(value).await.unwrap();
                }
            });
            assert_eq!(
                outer.collect::<Vec<_>>().await,
                (0..count).collect::<Vec<_>>()
            );
        }
    }

    #[tokio::test]
    async fn dropping_unpolled_stream_drops_producer_immediately() {
        let (tx, rx) = channel::<()>(1);
        let (guard, mut dropped) = tokio::sync::oneshot::channel::<()>();
        let stream = stream_from_producer(rx, async move {
            let _guard = guard;
            let _tx = tx;
            std::future::pending::<()>().await;
        });
        drop(stream);
        assert_eq!(
            dropped.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Closed)
        );
    }

    #[tokio::test]
    async fn dropping_nested_streams_synchronously_cancels_pending_work() {
        let (tx, rx) = channel::<()>(1);
        let (guard, mut dropped) = tokio::sync::oneshot::channel::<()>();
        let inner = stream_from_producer(rx, async move {
            let _guard = guard;
            let _tx = tx;
            std::future::pending::<()>().await;
        });
        let (tx, rx) = channel::<()>(1);
        let mut outer = stream_from_producer(rx, async move {
            let _tx = tx;
            inner.collect::<Vec<_>>().await;
        });
        assert!(futures::poll!(outer.next()).is_pending());
        drop(outer);
        assert_eq!(
            dropped.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Closed)
        );
    }

    #[tokio::test]
    async fn stream_waits_for_producer_cleanup_after_sender_closes() {
        let (tx, rx) = channel::<()>(1);
        let (release, gate) = tokio::sync::oneshot::channel::<()>();
        let (finished, mut completion) = tokio::sync::oneshot::channel();
        let mut stream = stream_from_producer(rx, async move {
            drop(tx);
            gate.await.unwrap();
            finished.send(()).unwrap();
        });
        assert!(futures::poll!(stream.next()).is_pending());
        release.send(()).unwrap();
        assert_eq!(stream.next().await, None);
        assert_eq!(completion.try_recv(), Ok(()));
    }
}
