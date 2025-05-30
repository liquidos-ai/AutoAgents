use async_trait::async_trait;
use autoagents_llm::{
    llm::{ChatCompletionResponse, ChatMessage, ChatRole, LLM},
    tool::Tool,
};
use futures::stream::BoxStream;
use serde::Serialize;
use serde_json::Value;
use std::{
    error::Error,
    ops::{Deref, DerefMut},
};
use uuid::Uuid;
pub mod error;
use error::AgentError;
pub mod types;

pub struct Agent<'a, T: AgentDeriveT + AgentT, L: LLM + 'static> {
    pub id: Uuid,
    pub inner: T,
    pub llm: &'a mut L,
}

impl<'a, T: AgentDeriveT + AgentT, L: LLM> Agent<'a, T, L> {
    pub fn new(inner: T, mut llm: &'a mut L) -> Result<Self, AgentError> {
        let tools = inner.tools();
        llm = Self::register_tools(llm, tools)?;
        Ok(Self {
            id: Uuid::new_v4(),
            inner,
            llm,
        })
    }

    fn register_tools(llm: &'a mut L, tools: Vec<Box<dyn Tool>>) -> Result<&'a mut L, AgentError> {
        if !tools.is_empty() && !llm.supports_tools() {
            return Err(AgentError::ModelToolNotSupported(llm.model_name()));
        }
        for tool in tools {
            llm.register_tool(tool);
        }
        Ok(llm)
    }

    pub fn llm(&self) -> &L {
        self.llm
    }

    async fn run_tools(&mut self, message: ChatCompletionResponse) -> Option<Value> {
        match message.message.role {
            ChatRole::Assistant => {
                let tools_call = message.message.tool_calls;
                if let Some(tool) = tools_call.into_iter().next() {
                    let tool_func = &tool.function;
                    let tool_result = self
                        .llm
                        .call_tool(&tool_func.name, tool_func.arguments.clone());
                    return tool_result;
                }
            }
            _ => {
                return None;
            }
        }
        None
    }

    pub async fn run(&mut self, prompt: &str) -> Result<Value, T::Err> {
        let response = self.inner.call::<L>(self.llm, prompt).await.unwrap();
        let val = self.run_tools(response).await.unwrap();
        Ok(val)
    }
}

impl<T: AgentDeriveT + AgentT, L: LLM> Deref for Agent<'_, T, L> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: AgentDeriveT + AgentT, L: LLM> DerefMut for Agent<'_, T, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[async_trait]
pub trait AgentDeriveT: Send + Sync {
    fn description(&self) -> &'static str;
    fn name(&self) -> &'static str;
    fn tools(&self) -> Vec<Box<dyn Tool>>;
}

#[async_trait]
pub trait AgentT: Send + Sync + AgentDeriveT {
    type Err: Error;
    type Output: Serialize;
    async fn call<T: LLM>(
        &self,
        llm: &mut T,
        prompt: &str,
    ) -> Result<ChatCompletionResponse, T::Error>;

    async fn chat_completion_stream<T: LLM>(
        &self,
        llm: &mut T,
        messages: Vec<ChatMessage>,
    ) -> BoxStream<'static, Result<ChatCompletionResponse, T::Error>> {
        llm.chat_completion_stream(messages, None).await
    }

    async fn chat_completion<T: LLM>(
        &self,
        llm: &mut T,
        mut messages: Vec<ChatMessage>,
    ) -> Result<ChatCompletionResponse, T::Error> {
        let agent_description = self.description();
        messages.insert(
            0,
            ChatMessage {
                role: ChatRole::System,
                content: agent_description.into(),
            },
        );
        llm.chat_completion(messages, None).await
    }
}

mod test {
    #![allow(unused_imports)]
    use super::*;
    use autoagents_derive::agent;
    use autoagents_llm::{error::LLMError, providers::ollama::Ollama};
    use serde::{Deserialize, Serialize};
    use serde_json::Error;

    #[tokio::test]
    async fn test_agent() {
        #[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
        struct AgentOutput {}

        #[agent(name = "basic_agent_1", description = "hello")]
        struct TestAgent {}

        #[async_trait]
        impl AgentT for TestAgent {
            type Err = Error;
            type Output = AgentOutput;

            async fn call<T: LLM>(
                &self,
                _llm: &mut T,
                _prompt: &str,
            ) -> Result<ChatCompletionResponse, T::Error> {
                Ok(serde_json::from_str("{}").unwrap())
            }
        }

        let mut llm = Ollama::new();
        let mut agent = Agent::new(TestAgent {}, &mut llm).unwrap();
        assert_eq!("basic_agent_1", agent.name());
        assert_eq!("hello", agent.description());
        let _val: AgentOutput = serde_json::from_str("{}").unwrap();
        let _agent_out = agent.run("Hello").await.unwrap();
        // assert_eq!(agent_out, val);
    }
}
