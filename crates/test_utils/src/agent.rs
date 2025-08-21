use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub enum TestError {
    #[error("Test error: {0}")]
    TestError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestAgentOutput {
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
