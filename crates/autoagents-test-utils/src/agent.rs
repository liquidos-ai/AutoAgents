use autoagents::async_trait;
use autoagents::core::agent::AgentHooks;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentDeriveT, AgentExecutor, AgentOutputT, Context, ExecutorConfig};
use autoagents::core::tool::ToolT;
use autoagents_derive::AgentOutput;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum TestError {
    #[error("Test error: {0}")]
    TestError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, AgentOutput)]
pub struct TestAgentOutput {
    #[output(description = "The result")]
    pub result: String,
}

impl From<TestAgentOutput> for Value {
    fn from(output: TestAgentOutput) -> Self {
        serde_json::to_value(output).unwrap_or(Value::Null)
    }
}

#[derive(Debug)]
pub struct MockAgentImpl {
    pub name: String,
    pub description: String,
    pub should_fail: bool,
}

impl MockAgentImpl {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            should_fail: false,
        }
    }

    pub fn with_failure(mut self, should_fail: bool) -> Self {
        self.should_fail = should_fail;
        self
    }
}

#[async_trait]
impl AgentDeriveT for MockAgentImpl {
    type Output = TestAgentOutput;

    fn description(&self) -> &'static str {
        Box::leak(self.description.clone().into_boxed_str())
    }

    fn output_schema(&self) -> Option<Value> {
        Some(TestAgentOutput::structured_output_format())
    }

    fn name(&self) -> &'static str {
        Box::leak(self.name.clone().into_boxed_str())
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        vec![]
    }
}

#[async_trait]
impl AgentExecutor for MockAgentImpl {
    type Output = TestAgentOutput;
    type Error = TestError;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig::default()
    }

    async fn execute(
        &self,
        task: &Task,
        _context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        if self.should_fail {
            return Err(TestError::TestError("Mock execution failed".to_string()));
        }

        Ok(TestAgentOutput {
            result: format!("Processed: {}", task.prompt),
        })
    }
    async fn execute_stream(
        &self,
        _task: &Task,
        _context: Arc<Context>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<Self::Output, Self::Error>> + Send>>,
        Self::Error,
    > {
        unimplemented!()
    }
}

impl AgentHooks for MockAgentImpl {}

// Test tool for agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestToolArgs {
    pub input: String,
}

#[derive(Debug)]
pub struct MockTool {
    pub name: String,
    pub description: String,
}

impl MockTool {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::MockLLMProvider;
    use autoagents::core::agent::Context;
    use futures::executor::block_on;
    use std::sync::Arc;

    #[test]
    fn test_mock_agent_metadata() {
        let agent = MockAgentImpl::new("agent", "desc");
        assert_eq!(agent.name(), "agent");
        assert_eq!(agent.description(), "desc");
        assert!(agent.output_schema().is_some());
    }

    #[test]
    fn test_mock_agent_execute_success() {
        let agent = MockAgentImpl::new("agent", "desc");
        let task = Task::new("hello");
        let ctx = Arc::new(Context::new(Arc::new(MockLLMProvider), None));
        let output = block_on(agent.execute(&task, ctx)).unwrap();
        assert_eq!(output.result, "Processed: hello");
    }

    #[test]
    fn test_mock_agent_execute_failure() {
        let agent = MockAgentImpl::new("agent", "desc").with_failure(true);
        let task = Task::new("hello");
        let ctx = Arc::new(Context::new(Arc::new(MockLLMProvider), None));
        let err = block_on(agent.execute(&task, ctx)).unwrap_err();
        assert!(err.to_string().contains("Mock execution failed"));
    }
}
