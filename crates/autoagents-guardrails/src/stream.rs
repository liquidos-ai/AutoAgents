use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use autoagents_llm::{
    chat::{StreamChoice, StreamChunk, StreamDelta, StreamResponse, Usage},
    error::LLMError,
};
use futures::Stream;

use crate::{
    engine::{GuardrailsEngine, RequestScope},
    guard::{CompletionGuardOutput, GuardContext, GuardedOutput},
};

enum StreamFinalize {
    Noop,
    Emit(String),
}

type FinalizeFuture = Pin<Box<dyn Future<Output = Result<StreamFinalize, LLMError>> + Send>>;

fn finalize_future(
    engine: Arc<GuardrailsEngine>,
    context: GuardContext,
    text: String,
    usage: Option<Usage>,
) -> FinalizeFuture {
    Box::pin(async move {
        let original = text.clone();
        let mut output = GuardedOutput::Completion(CompletionGuardOutput { text, usage });
        engine.evaluate_output(&mut output, &context).await?;

        match output {
            GuardedOutput::Completion(completion) => {
                if completion.text == original {
                    Ok(StreamFinalize::Noop)
                } else {
                    Ok(StreamFinalize::Emit(completion.text))
                }
            }
            GuardedOutput::Chat(_) => Err(LLMError::ProviderError(
                "unexpected chat output in stream finalization".to_string(),
            )),
        }
    })
}

pub(crate) struct TextGuardedStream {
    inner: Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>,
    engine: Arc<GuardrailsEngine>,
    context: GuardContext,
    aggregate_text: String,
    usage: Option<Usage>,
    finalize: Option<FinalizeFuture>,
    done: bool,
    /// Ends the request (releasing per-request guard state) when the stream
    /// is dropped, whether or not it was fully consumed.
    _scope: RequestScope,
}

impl TextGuardedStream {
    pub(crate) fn new(
        inner: Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>,
        engine: Arc<GuardrailsEngine>,
        context: GuardContext,
        scope: RequestScope,
    ) -> Self {
        Self {
            inner,
            engine,
            context,
            aggregate_text: String::default(),
            usage: None,
            finalize: None,
            done: false,
            _scope: scope,
        }
    }
}

impl Stream for TextGuardedStream {
    type Item = Result<String, LLMError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.done {
            return Poll::Ready(None);
        }

        if let Some(fut) = this.finalize.as_mut() {
            return match fut.as_mut().poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(result) => {
                    this.finalize = None;
                    this.done = true;
                    match result {
                        Ok(StreamFinalize::Noop) => Poll::Ready(None),
                        Ok(StreamFinalize::Emit(text)) => Poll::Ready(Some(Ok(text))),
                        Err(err) => Poll::Ready(Some(Err(err))),
                    }
                }
            };
        }

        match this.inner.as_mut().poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(Ok(chunk))) => {
                this.aggregate_text.push_str(&chunk);
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(err))) => {
                this.done = true;
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(None) => {
                this.finalize = Some(finalize_future(
                    this.engine.clone(),
                    this.context.clone(),
                    std::mem::take(&mut this.aggregate_text),
                    this.usage.take(),
                ));
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

pub(crate) struct StructGuardedStream {
    inner: Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
    engine: Arc<GuardrailsEngine>,
    context: GuardContext,
    aggregate_text: String,
    usage: Option<Usage>,
    finalize: Option<FinalizeFuture>,
    done: bool,
    /// Ends the request (releasing per-request guard state) when the stream
    /// is dropped, whether or not it was fully consumed.
    _scope: RequestScope,
}

impl StructGuardedStream {
    pub(crate) fn new(
        inner: Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
        engine: Arc<GuardrailsEngine>,
        context: GuardContext,
        scope: RequestScope,
    ) -> Self {
        Self {
            inner,
            engine,
            context,
            aggregate_text: String::default(),
            usage: None,
            finalize: None,
            done: false,
            _scope: scope,
        }
    }
}

impl Stream for StructGuardedStream {
    type Item = Result<StreamResponse, LLMError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.done {
            return Poll::Ready(None);
        }

        if let Some(fut) = this.finalize.as_mut() {
            return match fut.as_mut().poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(result) => {
                    this.finalize = None;
                    this.done = true;
                    match result {
                        Ok(StreamFinalize::Noop) => Poll::Ready(None),
                        Ok(StreamFinalize::Emit(text)) => Poll::Ready(Some(Ok(StreamResponse {
                            choices: vec![StreamChoice {
                                delta: StreamDelta {
                                    content: Some(text),
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                            }],
                            usage: None,
                        }))),
                        Err(err) => Poll::Ready(Some(Err(err))),
                    }
                }
            };
        }

        match this.inner.as_mut().poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(Ok(chunk))) => {
                for choice in &chunk.choices {
                    if let Some(content) = &choice.delta.content {
                        this.aggregate_text.push_str(content);
                    }
                }
                if let Some(usage) = &chunk.usage {
                    this.usage = Some(usage.clone());
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(err))) => {
                this.done = true;
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(None) => {
                this.finalize = Some(finalize_future(
                    this.engine.clone(),
                    this.context.clone(),
                    std::mem::take(&mut this.aggregate_text),
                    this.usage.take(),
                ));
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

pub(crate) struct ToolGuardedStream {
    inner: Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>,
    engine: Arc<GuardrailsEngine>,
    context: GuardContext,
    aggregate_text: String,
    usage: Option<Usage>,
    finalize: Option<FinalizeFuture>,
    done: bool,
    /// Ends the request (releasing per-request guard state) when the stream
    /// is dropped, whether or not it was fully consumed.
    _scope: RequestScope,
}

impl ToolGuardedStream {
    pub(crate) fn new(
        inner: Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>,
        engine: Arc<GuardrailsEngine>,
        context: GuardContext,
        scope: RequestScope,
    ) -> Self {
        Self {
            inner,
            engine,
            context,
            aggregate_text: String::default(),
            usage: None,
            finalize: None,
            done: false,
            _scope: scope,
        }
    }
}

impl Stream for ToolGuardedStream {
    type Item = Result<StreamChunk, LLMError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.done {
            return Poll::Ready(None);
        }

        if let Some(fut) = this.finalize.as_mut() {
            return match fut.as_mut().poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(result) => {
                    this.finalize = None;
                    this.done = true;
                    match result {
                        Ok(StreamFinalize::Noop) => Poll::Ready(None),
                        Ok(StreamFinalize::Emit(text)) => {
                            Poll::Ready(Some(Ok(StreamChunk::Text(text))))
                        }
                        Err(err) => Poll::Ready(Some(Err(err))),
                    }
                }
            };
        }

        match this.inner.as_mut().poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(Ok(chunk))) => {
                match &chunk {
                    StreamChunk::Text(content) => this.aggregate_text.push_str(content),
                    StreamChunk::Usage(usage) => this.usage = Some(usage.clone()),
                    _ => {}
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(err))) => {
                this.done = true;
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(None) => {
                this.finalize = Some(finalize_future(
                    this.engine.clone(),
                    this.context.clone(),
                    std::mem::take(&mut this.aggregate_text),
                    this.usage.take(),
                ));
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}
