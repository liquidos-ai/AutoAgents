use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AgentResultError {
    #[error("No output available in result")]
    NoOutput,

    #[error("Failed to deserialize executor output: {0}")]
    DeserializationError(#[from] serde_json::Error),

    #[error("Agent output extraction error: {0}")]
    AgentOutputError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunResult {
    /// Indicates if the run was successful
    pub success: bool,

    /// The actual output of the agent, if any
    pub output: Option<Value>,

    /// Optional error message if failed
    pub error_message: Option<String>,

    /// Additional metadata, e.g. timestamps, logs, etc.
    pub metadata: Option<Value>,
}

impl AgentRunResult {
    /// Helper to create a success result
    pub fn success(output: Value) -> Self {
        Self {
            success: true,
            output: Some(output),
            error_message: None,
            metadata: None,
        }
    }

    /// Helper to create a failure result
    pub fn failure(error_message: String) -> Self {
        Self {
            success: false,
            output: None,
            error_message: Some(error_message),
            metadata: None,
        }
    }

    /// Extract the executor output from the AgentRunResult
    /// This deserializes the stored Value back to the executor's output type
    pub fn extract_executor_output<T>(&self) -> Result<T, AgentResultError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let output = self.output.as_ref().ok_or(AgentResultError::NoOutput)?;

        let executor_output: T = serde_json::from_value(output.clone())?;

        Ok(executor_output)
    }

    /// Extract the agent output from the result, given an executor output type
    /// This is a generic method that works with any executor output type that has
    /// a way to extract the agent output from it
    pub fn extract_agent_output<ExecutorOutput, AgentOutput, E>(
        &self,
        extractor: impl Fn(ExecutorOutput) -> Result<AgentOutput, E>,
    ) -> Result<AgentOutput, AgentResultError>
    where
        ExecutorOutput: for<'de> Deserialize<'de>,
        AgentOutput: for<'de> Deserialize<'de>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let executor_output = self.extract_executor_output::<ExecutorOutput>()?;
        extractor(executor_output).map_err(|e| AgentResultError::AgentOutputError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestExecutorOutput {
        response: String,
        metadata: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestAgentOutput {
        value: i32,
        message: String,
    }

    #[test]
    fn test_extract_executor_output_success() {
        let executor_output = TestExecutorOutput {
            response: "test response".to_string(),
            metadata: Some("test metadata".to_string()),
        };

        let result =
            AgentRunResult::success(serde_json::to_value(executor_output.clone()).unwrap());

        let extracted: TestExecutorOutput = result.extract_executor_output().unwrap();
        assert_eq!(extracted, executor_output);
    }

    #[test]
    fn test_extract_executor_output_no_output() {
        let result = AgentRunResult::failure("Test error".to_string());

        let extracted: Result<TestExecutorOutput, _> = result.extract_executor_output();
        assert!(extracted.is_err());
        assert!(matches!(extracted.unwrap_err(), AgentResultError::NoOutput));
    }

    #[test]
    fn test_extract_agent_output_success() {
        let agent_output = TestAgentOutput {
            value: 42,
            message: "Hello, world!".to_string(),
        };

        let executor_output = TestExecutorOutput {
            response: serde_json::to_string(&agent_output).unwrap(),
            metadata: None,
        };

        let result = AgentRunResult::success(serde_json::to_value(executor_output).unwrap());

        let extracted: TestAgentOutput = result
            .extract_agent_output(|exec_output: TestExecutorOutput| {
                serde_json::from_str(&exec_output.response).map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Failed to parse agent response: {e}"),
                    )
                })
            })
            .unwrap();

        assert_eq!(extracted, agent_output);
    }

    #[test]
    fn test_extract_agent_output_extractor_error() {
        let executor_output = TestExecutorOutput {
            response: "invalid json".to_string(),
            metadata: None,
        };

        let result = AgentRunResult::success(serde_json::to_value(executor_output).unwrap());

        let extracted: Result<TestAgentOutput, _> =
            result.extract_agent_output(|exec_output: TestExecutorOutput| {
                serde_json::from_str(&exec_output.response).map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Failed to parse agent response: {e}"),
                    )
                })
            });

        assert!(extracted.is_err());
        assert!(matches!(
            extracted.unwrap_err(),
            AgentResultError::AgentOutputError(_)
        ));
    }
}
