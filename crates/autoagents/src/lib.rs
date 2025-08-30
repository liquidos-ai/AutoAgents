// Re-export for convenience
pub use async_trait::async_trait;

pub use autoagents_core::{self as core, error as core_error};
pub use autoagents_llm::{self as llm, error as llm_error};

#[inline]
/// Initialize logging using env_logger if the "logging" feature is enabled.
/// This is a no-op if the feature is not enabled.
pub fn init_logging() {
    #[cfg(feature = "logging")]
    {
        let _ = env_logger::try_init();
    }
}

/*

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_core::{
        runtime::SingleThreadedRuntime,
        tool::{ToolCallResult, ToolT},
    };
    use autoagents_test_utils::llm::MockLLMProvider;
    use std::sync::Arc;

    #[test]
    fn test_init_logging_no_panic() {
        // Test that init_logging doesn't panic when called
        init_logging();
        init_logging(); // Should be safe to call multiple times
    }

    #[test]
    fn test_llm_module_available() {
        // Test that we can access llm module
        let request = llm::completion::CompletionRequest::new("test prompt");
        assert_eq!(request.prompt, "test prompt");
    }

    #[test]
    fn test_core_error_available() {
        // Test that core error types are accessible
        // let error = core_error::Error::SessionError(core::session::SessionError::EmptyTask);
        // assert!(error.to_string().contains("Task is None"));
    }

    #[test]
    fn test_llm_error_available() {
        // Test that llm error types are accessible
        let error = llm_error::LLMError::AuthError("test error".to_string());
        assert_eq!(error.to_string(), "Auth Error: test error");
    }

    #[test]
    fn test_chat_message_builder() {
        // Test chat message builder from llm module
        let message = llm::chat::ChatMessage::user()
            .content("Hello world")
            .build();

        assert_eq!(message.role, llm::chat::ChatRole::User);
        assert_eq!(message.content, "Hello world");
    }

    #[tokio::test]
    async fn test_agent_builder_available() {
        // Test that agent builder types are accessible
        use crate::core::agent::prebuilt::react::ReActExecutor;
        use crate::core::agent::AgentBuilder;

        let runtime = SingleThreadedRuntime::new(None);
        // Mock agent for testing
        #[derive(Debug)]
        struct MockAgent;

        impl core::agent::AgentDeriveT for MockAgent {
            type Output = String;

            fn name(&self) -> &'static str {
                "MockAgent"
            }

            fn description(&self) -> &'static str {
                "A mock agent for testing"
            }

            fn tools(&self) -> Vec<Box<dyn ToolT>> {
                vec![]
            }

            fn output_schema(&self) -> Option<serde_json::Value> {
                None
            }
        }

        impl ReActExecutor for MockAgent {}

        let llm: Arc<MockLLMProvider> = Arc::new(MockLLMProvider {});
        let builder = AgentBuilder::new(MockAgent)
            .with_llm(llm)
            .runtime(runtime)
            .build()
            .await
            .unwrap();
        assert_eq!(builder.description(), "A mock agent for testing");
    }

    #[test]
    fn test_memory_types_available() {
        // Test that memory types are accessible
        let memory = crate::core::memory::SlidingWindowMemory::new(5);
        assert_eq!(memory.window_size(), 5);
    }

    #[test]
    fn test_tool_types_available() {
        // Test that tool types are accessible
        let tool_result = ToolCallResult {
            tool_name: "test_tool".to_string(),
            success: true,
            arguments: serde_json::json!({"param": "value"}),
            result: serde_json::json!({"output": "result"}),
        };

        assert_eq!(tool_result.tool_name, "test_tool");
        assert!(tool_result.success);
    }

    #[test]
    fn test_environment_creation() {
        // Test that environment can be created
        use crate::core::environment::EnvironmentConfig;

        let config = EnvironmentConfig {
            working_dir: std::path::PathBuf::from("/tmp"),
        };

        assert_eq!(config.working_dir, std::path::PathBuf::from("/tmp"));
    }

    #[tokio::test]
    async fn test_async_functionality() {
        // Test async functionality is available
        use crate::core::memory::MemoryProvider;
        use llm::chat::{ChatMessage, ChatRole, MessageType};

        let mut memory = crate::core::memory::SlidingWindowMemory::new(3);
        let message = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "Test message".to_string(),
        };

        let result = memory.remember(&message).await;
        assert!(result.is_ok());
        assert_eq!(memory.size(), 1);
    }

    #[test]
    fn test_builder_patterns() {
        // Test that builder patterns are working
        use llm::completion::CompletionRequest;

        let request = CompletionRequest::builder("Test prompt")
            .max_tokens(100)
            .temperature(0.7)
            .build();

        assert_eq!(request.prompt, "Test prompt");
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.7));
    }

    #[test]
    fn test_structured_output_format() {
        // Test structured output format
        use llm::chat::StructuredOutputFormat;
        use serde_json::json;

        let format = StructuredOutputFormat {
            name: "TestSchema".to_string(),
            description: Some("A test schema".to_string()),
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            })),
            strict: Some(true),
        };

        assert_eq!(format.name, "TestSchema");
        assert!(format.strict.unwrap());
    }

    #[test]
    fn test_tool_choice_serialization() {
        // Test tool choice serialization
        use llm::chat::ToolChoice;

        let choice = ToolChoice::Auto;
        let serialized = serde_json::to_string(&choice).unwrap();
        assert_eq!(serialized, "\"auto\"");

        let choice = ToolChoice::Tool("my_function".to_string());
        let serialized = serde_json::to_string(&choice).unwrap();
        assert!(serialized.contains("my_function"));
    }

    #[test]
    fn test_error_conversion() {
        // Test error conversion between modules
        use crate::core::error::Error;
        use llm::error::LLMError;

        let llm_error = LLMError::AuthError("Auth failed".to_string());
        let core_error: Error = llm_error.into();

        assert!(core_error.to_string().contains("Auth failed"));
    }
}

*/
