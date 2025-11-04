use crate::agent::hooks::HookOutcome;
use crate::agent::task::Task;
use crate::agent::{AgentDeriveT, AgentExecutor, AgentHooks, Context, EventHelper, ExecutorConfig};
use crate::tool::{ToolCallResult, ToolT};
use async_trait::async_trait;
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType};
use autoagents_llm::ToolCall;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;

/// Output of the Basic executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicAgentOutput {
    pub response: String,
    pub done: bool,
}

impl From<BasicAgentOutput> for Value {
    fn from(output: BasicAgentOutput) -> Self {
        serde_json::to_value(output).unwrap_or(Value::Null)
    }
}
impl From<BasicAgentOutput> for String {
    fn from(output: BasicAgentOutput) -> Self {
        output.response
    }
}

/// Error type for Basic executor
#[derive(Debug, thiserror::Error)]
pub enum BasicExecutorError {
    #[error("LLM error: {0}")]
    LLMError(String),

    #[error("Other error: {0}")]
    Other(String),
}

/// Wrapper type for Basic executor
#[derive(Debug)]
pub struct BasicAgent<T: AgentDeriveT> {
    inner: Arc<T>,
}

impl<T: AgentDeriveT> Clone for BasicAgent<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: AgentDeriveT> BasicAgent<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl<T: AgentDeriveT> Deref for BasicAgent<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Implement AgentDeriveT for the wrapper by delegating to the inner type
#[async_trait]
impl<T: AgentDeriveT> AgentDeriveT for BasicAgent<T> {
    type Output = <T as AgentDeriveT>::Output;

    fn description(&self) -> &'static str {
        self.inner.description()
    }

    fn output_schema(&self) -> Option<Value> {
        self.inner.output_schema()
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        self.inner.tools()
    }
}

#[async_trait]
impl<T> AgentHooks for BasicAgent<T>
where
    T: AgentDeriveT + AgentHooks + Send + Sync + 'static,
{
    async fn on_agent_create(&self) {
        self.inner.on_agent_create().await
    }

    async fn on_run_start(&self, task: &Task, ctx: &Context) -> HookOutcome {
        self.inner.on_run_start(task, ctx).await
    }

    async fn on_run_complete(&self, task: &Task, result: &Self::Output, ctx: &Context) {
        self.inner.on_run_complete(task, result, ctx).await
    }

    async fn on_turn_start(&self, turn_index: usize, ctx: &Context) {
        self.inner.on_turn_start(turn_index, ctx).await
    }

    async fn on_turn_complete(&self, turn_index: usize, ctx: &Context) {
        self.inner.on_turn_complete(turn_index, ctx).await
    }

    async fn on_tool_call(&self, tool_call: &ToolCall, ctx: &Context) -> HookOutcome {
        self.inner.on_tool_call(tool_call, ctx).await
    }

    async fn on_tool_start(&self, tool_call: &ToolCall, ctx: &Context) {
        self.inner.on_tool_start(tool_call, ctx).await
    }

    async fn on_tool_result(&self, tool_call: &ToolCall, result: &ToolCallResult, ctx: &Context) {
        self.inner.on_tool_result(tool_call, result, ctx).await
    }

    async fn on_tool_error(&self, tool_call: &ToolCall, err: Value, ctx: &Context) {
        self.inner.on_tool_error(tool_call, err, ctx).await
    }

    async fn on_llm_token_usage(&self, usage: &autoagents_llm::chat::Usage, ctx: &Context) {
        self.inner.on_llm_token_usage(usage, ctx).await
    }

    async fn on_agent_shutdown(&self) {
        self.inner.on_agent_shutdown().await
    }
}

/// Implementation of AgentExecutor for the BasicExecutorWrapper
#[async_trait]
impl<T: AgentDeriveT + AgentHooks> AgentExecutor for BasicAgent<T> {
    type Output = BasicAgentOutput;
    type Error = BasicExecutorError;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig { max_turns: 1 }
    }

    async fn execute(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        let tx_event = context.tx().ok();
        EventHelper::send_task_started(
            &tx_event,
            task.submission_id,
            context.config().id,
            task.prompt.clone(),
            context.config().name.clone(),
        )
        .await;

        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            message_type: MessageType::Text,
            content: context.config().description.clone(),
        }];

        let chat_msg = if let Some((mime, image_data)) = &task.image {
            // Task has an image, create an Image message
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Image((*mime, image_data.clone())),
                content: task.prompt.clone(),
            }
        } else {
            // Text-only task
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: task.prompt.clone(),
            }
        };
        messages.push(chat_msg);
        let response = context
            .llm()
            .chat(&messages, None, context.config().output_schema.clone())
            .await
            .map_err(|e| BasicExecutorError::LLMError(e.to_string()))?;

        // Record token usage if available
        if let Some(usage) = response.usage() {
            let state_arc = context.state();
            let mut state = state_arc.lock().await;
            state.record_usage(usage.clone());
            drop(state); // Release lock before calling hook
            
            // Call the hook for token usage tracking
            self.on_llm_token_usage(&usage, &context).await;
        }

        let response_text = response.text().unwrap_or_default();
        Ok(BasicAgentOutput {
            response: response_text,
            done: true,
        })
    }

    async fn execute_stream(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Self::Output, Self::Error>> + Send>>, Self::Error>
    {
        use futures::StreamExt;

        let tx_event = context.tx().ok();
        EventHelper::send_task_started(
            &tx_event,
            task.submission_id,
            context.config().id,
            task.prompt.clone(),
            context.config().name.clone(),
        )
        .await;

        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            message_type: MessageType::Text,
            content: context.config().description.clone(),
        }];

        let chat_msg = if let Some((mime, image_data)) = &task.image {
            // Task has an image, create an Image message
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Image((*mime, image_data.clone())),
                content: task.prompt.clone(),
            }
        } else {
            // Text-only task
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: task.prompt.clone(),
            }
        };
        messages.push(chat_msg);

        let stream = context
            .llm()
            .chat_stream_struct(&messages, None, context.config().output_schema.clone())
            .await
            .map_err(|e| BasicExecutorError::LLMError(e.to_string()))?;

        let context_clone = Arc::clone(&context);
        let self_clone = self.clone();

        let mapped_stream = stream.map(move |chunk_result| {
            let context = Arc::clone(&context_clone);
            let self_ref = self_clone.clone();
            
            match chunk_result {
                Ok(chunk) => {
                    // Check for token usage in the chunk (typically in the final chunk)
                    if let Some(usage) = &chunk.usage {
                        let usage_clone = usage.clone();
                        let ctx = Arc::clone(&context);
                        let agent = self_ref.clone();
                        
                        // Spawn task to record usage without blocking the stream
                        tokio::spawn(async move {
                            let state_arc = ctx.state();
                            let mut state = state_arc.lock().await;
                            state.record_usage(usage_clone.clone());
                            drop(state);
                            
                            agent.on_llm_token_usage(&usage_clone, &ctx).await;
                        });
                    }

                    let content = chunk
                        .choices
                        .first()
                        .and_then(|choice| choice.delta.content.as_ref())
                        .map_or("", |v| v)
                        .to_string();

                    Ok(BasicAgentOutput {
                        response: content,
                        done: false,
                    })
                }
                Err(e) => Err(BasicExecutorError::LLMError(e.to_string())),
            }
        });

        Ok(Box::pin(mapped_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentDeriveT;
    use crate::tests::agent::MockAgentImpl;
    use autoagents_test_utils::llm::MockLLMProvider;
    use std::sync::Arc;

    #[test]
    fn test_basic_agent_creation() {
        let mock_agent = MockAgentImpl::new("test_agent", "Test agent description");
        let basic_agent = BasicAgent::new(mock_agent);

        assert_eq!(basic_agent.name(), "test_agent");
        assert_eq!(basic_agent.description(), "Test agent description");
    }

    #[test]
    fn test_basic_agent_clone() {
        let mock_agent = MockAgentImpl::new("test_agent", "Test agent description");
        let basic_agent = BasicAgent::new(mock_agent);
        let cloned_agent = basic_agent.clone();

        assert_eq!(cloned_agent.name(), "test_agent");
        assert_eq!(cloned_agent.description(), "Test agent description");
    }

    #[test]
    fn test_basic_agent_output_conversions() {
        let output = BasicAgentOutput {
            response: "Test response".to_string(),
            done: true,
        };

        // Test conversion to Value
        let value: Value = output.clone().into();
        assert!(value.is_object());

        // Test conversion to String
        let string: String = output.into();
        assert_eq!(string, "Test response");
    }

    #[tokio::test]
    async fn test_basic_agent_execute() {
        use crate::agent::task::Task;
        use crate::agent::{AgentConfig, Context};
        use crate::protocol::ActorID;

        let mock_agent = MockAgentImpl::new("test_agent", "Test agent description");
        let basic_agent = BasicAgent::new(mock_agent);

        let llm = Arc::new(MockLLMProvider {});
        let config = AgentConfig {
            id: ActorID::new_v4(),
            name: "test_agent".to_string(),
            description: "Test agent description".to_string(),
            output_schema: None,
        };

        let context = Context::new(llm, None).with_config(config);

        let context_arc = Arc::new(context);
        let task = Task::new("Test task");
        let result = basic_agent.execute(&task, context_arc).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.response, "Mock response");
        assert!(output.done);
    }

    #[test]
    fn test_executor_config() {
        let mock_agent = MockAgentImpl::new("test_agent", "Test agent description");
        let basic_agent = BasicAgent::new(mock_agent);

        let config = basic_agent.config();
        assert_eq!(config.max_turns, 1);
    }
}
