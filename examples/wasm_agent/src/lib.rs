#![allow(dead_code, unused_variables, unused_imports)]

use crate::phi::Model;
use crate::phi_provider::PhiProvider;
use async_trait::async_trait;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{
    ReActAgentOutput, ReActExecutor, ReActExecutorError,
};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentExecutor};
use autoagents::core::agent::{
    AgentConfig, AgentOutputT, AgentProtocol, Context, ExecutorConfig, RunnableAgent,
    RunnableAgentImpl,
};
use autoagents::core::error::Error;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::chat::ChatProvider;
use autoagents::llm::chat::{ChatMessage, ChatRole, MessageType, StructuredOutputFormat};
use autoagents::llm::LLMProvider;
use autoagents_derive::{agent, tool, AgentOutput, ToolInput};
use futures::channel::mpsc::Receiver;
use futures::{Stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

mod phi;
pub mod phi_provider;
pub mod streaming_interface;
pub use streaming_interface::TokenStreamer;

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => ($crate::log(&format_args!($($t)*).to_string()))
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
#[wasm_bindgen]
pub struct AdditionArgs {
    #[input(description = "Left Operand for addition")]
    left: i64,
    #[input(description = "Right Operand for addition")]
    right: i64,
}

#[tool(
    name = "Addition",
    description = "Use this tool to Add two numbers",
    input = AdditionArgs,
)]
#[wasm_bindgen]
struct Addition {}

impl ToolRuntime for Addition {
    fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let typed_args: AdditionArgs = serde_json::from_value(args)?;
        let result = typed_args.left + typed_args.right;
        console_log!("Tool Call: {}", result);
        Ok(result.into())
    }
}

/// Math agent output with Value and Explanation
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
#[wasm_bindgen]
pub struct MathAgentOutput {
    #[output(description = "The addition result")]
    value: i64,
    #[output(description = "Explanation of the logic")]
    explanation: String,
    #[output(description = "If user asks other than math questions, use this to answer them.")]
    generic: Option<String>,
}

#[agent(
    name = "math_agent",
    description = "You are a Math agent",
    tools = [Addition],
    output = MathAgentOutput
)]
#[derive(Default, Clone)]
#[wasm_bindgen]
pub struct MathAgent {}

// impl ReActExecutor for MathAgent {}

#[async_trait]
impl AgentExecutor for MathAgent {
    type Output = String;
    type Error = Error;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig { max_turns: 10 }
    }

    async fn execute(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        console_log!("Running Execution");
        let prompt = task.prompt.clone();
        let llm = context.llm().clone();
        let mut stream = llm
            .chat_stream(
                &vec![ChatMessage {
                    role: ChatRole::System,
                    message_type: MessageType::Text,
                    content: prompt,
                }],
                None,
                None,
            )
            .await?;
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    console_log!("Chunk: {:?}", chunk);
                }
                _ => {}
            }
        }
        Ok("Hello world!".into())
    }

    async fn execute_stream(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Self::Output, Self::Error>> + Send>>, Self::Error>
    {
        console_log!("Running Execution Stream");
        let prompt = task.prompt.clone();
        let llm = context.llm().clone();
        let stream = llm
            .chat_stream_struct(
                &vec![ChatMessage {
                    role: ChatRole::System,
                    message_type: MessageType::Text,
                    content: prompt,
                }],
                None,
                None,
            )
            .await?;

        let output_stream = stream.map(|chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    // Extract the actual text content from StreamResponse
                    let content = chunk
                        .choices
                        .into_iter()
                        .filter_map(|choice| choice.delta.content)
                        .collect::<Vec<String>>()
                        .join("");

                    console_log!("Token: {}", content);
                    Ok(content)
                }
                Err(e) => Err(Error::from(e)),
            }
        });

        Ok(Box::pin(output_stream))
    }
}
