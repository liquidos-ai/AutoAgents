use autoagents_llm::chat::{FunctionTool, Tool};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;
use std::sync::Arc;
mod runtime;
use async_trait::async_trait;
pub use runtime::ToolRuntime;

#[cfg(feature = "wasmtime")]
pub use runtime::{WasmRuntime, WasmRuntimeError};

/// Result emitted after executing a single tool call, including the tool name,
/// success flag, original arguments, and the tool's structured result payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub tool_name: String,
    pub success: bool,
    pub arguments: Value,
    pub result: Value,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolCallError {
    #[error("Runtime Error {0}")]
    RuntimeError(#[from] Box<dyn std::error::Error + Sync + Send>),

    #[error("Serde Error {0}")]
    SerdeError(#[from] serde_json::Error),
}

pub trait ToolT: Send + Sync + Debug + ToolRuntime {
    /// The name of the tool.
    fn name(&self) -> &str;
    /// A description explaining the tool’s purpose.
    fn description(&self) -> &str;
    /// Return a description of the expected arguments.
    fn args_schema(&self) -> Value;
}

/// Marker trait for input types used by `#[derive(ToolInput)]` macros.
pub trait ToolInputT {
    fn io_schema() -> &'static str;
}

/// A wrapper that allows Arc<dyn ToolT> to be used as Box<dyn ToolT>
/// This is useful for sharing tools across multiple agents without cloning
/// Wrapper around `Arc<dyn ToolT>` that presents a `Box<dyn ToolT>`-like API.
/// Useful when sharing tool instances across multiple agents without cloning.
#[derive(Debug)]
pub struct SharedTool {
    inner: Arc<dyn ToolT>,
}

impl SharedTool {
    /// Create a new SharedTool from an Arc<dyn ToolT>
    pub fn new(tool: Arc<dyn ToolT>) -> Self {
        Self { inner: tool }
    }
}

#[async_trait]
impl ToolRuntime for SharedTool {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        self.inner.execute(args).await
    }
}

impl ToolT for SharedTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn args_schema(&self) -> Value {
        self.inner.args_schema()
    }
}

/// Helper function to convert Vec<Arc<dyn ToolT>> to Vec<Box<dyn ToolT>>
/// This is useful when implementing AgentDeriveT::tools() with shared tools
/// Convert a vector of `Arc<dyn ToolT>` into boxed trait objects for use in
/// agent definitions.
pub fn shared_tools_to_boxes(tools: &[Arc<dyn ToolT>]) -> Vec<Box<dyn ToolT>> {
    tools
        .iter()
        .map(|t| Box::new(SharedTool::new(Arc::clone(t))) as Box<dyn ToolT>)
        .collect()
}

/// Convert a ToolT trait object to an LLM Tool
#[allow(clippy::borrowed_box)]
pub fn to_llm_tool(tool: &Box<dyn ToolT>) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: FunctionTool {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: tool.args_schema(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_llm::chat::Tool;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Debug, Serialize, Deserialize)]
    struct TestInput {
        name: String,
        value: i32,
    }

    impl ToolInputT for TestInput {
        fn io_schema() -> &'static str {
            r#"{"type":"object","properties":{"name":{"type":"string"},"value":{"type":"integer"}},"required":["name","value"]}"#
        }
    }

    #[derive(Debug)]
    struct MockTool {
        name: &'static str,
        description: &'static str,
        should_fail: bool,
    }

    impl MockTool {
        fn new(name: &'static str, description: &'static str) -> Self {
            Self {
                name,
                description,
                should_fail: false,
            }
        }

        fn with_failure(name: &'static str, description: &'static str) -> Self {
            Self {
                name,
                description,
                should_fail: true,
            }
        }
    }

    impl ToolT for MockTool {
        fn name(&self) -> &'static str {
            self.name
        }

        fn description(&self) -> &'static str {
            self.description
        }

        fn args_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "integer"}
                },
                "required": ["name", "value"]
            })
        }
    }

    #[async_trait]
    impl ToolRuntime for MockTool {
        async fn execute(
            &self,
            args: serde_json::Value,
        ) -> Result<serde_json::Value, ToolCallError> {
            if self.should_fail {
                return Err(ToolCallError::RuntimeError(
                    "Mock tool failure".to_string().into(),
                ));
            }

            let input: TestInput = serde_json::from_value(args)?;
            Ok(json!({
                "processed_name": input.name,
                "doubled_value": input.value * 2
            }))
        }
    }

    #[test]
    fn test_tool_call_error_runtime_error() {
        let error = ToolCallError::RuntimeError("Runtime error".to_string().into());
        assert_eq!(error.to_string(), "Runtime Error Runtime error");
    }

    #[test]
    fn test_tool_call_error_serde_error() {
        let json_error = serde_json::from_str::<Value>("invalid json").unwrap_err();
        let error = ToolCallError::SerdeError(json_error);
        assert!(error.to_string().contains("Serde Error"));
    }

    #[test]
    fn test_tool_call_error_debug() {
        let error = ToolCallError::RuntimeError("Debug test".to_string().into());
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("RuntimeError"));
    }

    #[test]
    fn test_tool_call_error_from_serde() {
        let json_error = serde_json::from_str::<Value>("invalid json").unwrap_err();
        let error: ToolCallError = json_error.into();
        assert!(matches!(error, ToolCallError::SerdeError(_)));
    }

    #[test]
    fn test_tool_call_error_from_box_error() {
        let box_error: Box<dyn std::error::Error + Send + Sync> = "Test error".into();
        let error: ToolCallError = box_error.into();
        assert!(matches!(error, ToolCallError::RuntimeError(_)));
    }

    #[test]
    fn test_mock_tool_creation() {
        let tool = MockTool::new("test_tool", "A test tool");
        assert_eq!(tool.name(), "test_tool");
        assert_eq!(tool.description(), "A test tool");
        assert!(!tool.should_fail);
    }

    #[test]
    fn test_mock_tool_with_failure() {
        let tool = MockTool::with_failure("failing_tool", "A failing tool");
        assert_eq!(tool.name(), "failing_tool");
        assert_eq!(tool.description(), "A failing tool");
        assert!(tool.should_fail);
    }

    #[test]
    fn test_mock_tool_args_schema() {
        let tool = MockTool::new("schema_tool", "Schema test");
        let schema = tool.args_schema();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"].is_object());
        assert!(schema["properties"]["name"].is_object());
        assert!(schema["properties"]["value"].is_object());
        assert_eq!(schema["properties"]["name"]["type"], "string");
        assert_eq!(schema["properties"]["value"]["type"], "integer");
    }

    #[tokio::test]
    async fn test_mock_tool_run_success() {
        let tool = MockTool::new("success_tool", "Success test");
        let input = json!({
            "name": "test",
            "value": 42
        });

        let result = tool.execute(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output["processed_name"], "test");
        assert_eq!(output["doubled_value"], 84);
    }

    #[tokio::test]
    async fn test_mock_tool_run_failure() {
        let tool = MockTool::with_failure("failure_tool", "Failure test");
        let input = json!({
            "name": "test",
            "value": 42
        });

        let result = tool.execute(input).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Mock tool failure"));
    }

    #[tokio::test]
    async fn test_mock_tool_run_invalid_input() {
        let tool = MockTool::new("invalid_input_tool", "Invalid input test");
        let input = json!({
            "invalid_field": "test"
        });

        let result = tool.execute(input).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolCallError::SerdeError(_)));
    }

    #[tokio::test]
    async fn test_mock_tool_run_with_extra_fields() {
        let tool = MockTool::new("extra_fields_tool", "Extra fields test");
        let input = json!({
            "name": "test",
            "value": 42,
            "extra_field": "ignored"
        });

        let result = tool.execute(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output["processed_name"], "test");
        assert_eq!(output["doubled_value"], 84);
    }

    #[test]
    fn test_mock_tool_debug() {
        let tool = MockTool::new("debug_tool", "Debug test");
        let debug_str = format!("{tool:?}");
        assert!(debug_str.contains("MockTool"));
        assert!(debug_str.contains("debug_tool"));
    }

    #[test]
    fn test_tool_input_trait() {
        let schema = TestInput::io_schema();
        assert!(schema.contains("object"));
        assert!(schema.contains("name"));
        assert!(schema.contains("value"));
        assert!(schema.contains("string"));
        assert!(schema.contains("integer"));
    }

    #[test]
    fn test_test_input_serialization() {
        let input = TestInput {
            name: "test".to_string(),
            value: 42,
        };
        let serialized = serde_json::to_string(&input).unwrap();
        assert!(serialized.contains("test"));
        assert!(serialized.contains("42"));
    }

    #[test]
    fn test_test_input_deserialization() {
        let json = r#"{"name":"test","value":42}"#;
        let input: TestInput = serde_json::from_str(json).unwrap();
        assert_eq!(input.name, "test");
        assert_eq!(input.value, 42);
    }

    #[test]
    fn test_test_input_debug() {
        let input = TestInput {
            name: "debug".to_string(),
            value: 123,
        };
        let debug_str = format!("{input:?}");
        assert!(debug_str.contains("TestInput"));
        assert!(debug_str.contains("debug"));
        assert!(debug_str.contains("123"));
    }

    #[test]
    fn test_boxed_tool_to_tool_conversion() {
        let mock_tool = MockTool::new("convert_tool", "Conversion test");
        let boxed_tool: Box<dyn ToolT> = Box::new(mock_tool);

        let tool: Tool = to_llm_tool(&boxed_tool);
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "convert_tool");
        assert_eq!(tool.function.description, "Conversion test");
        assert_eq!(tool.function.parameters["type"], "object");
    }

    #[test]
    fn test_tool_conversion_preserves_schema() {
        let mock_tool = MockTool::new("schema_tool", "Schema preservation test");
        let boxed_tool: Box<dyn ToolT> = Box::new(mock_tool);

        let tool: Tool = to_llm_tool(&boxed_tool);
        let schema = &tool.function.parameters;

        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["name"]["type"], "string");
        assert_eq!(schema["properties"]["value"]["type"], "integer");
        assert_eq!(schema["required"][0], "name");
        assert_eq!(schema["required"][1], "value");
    }

    #[test]
    fn test_tool_trait_object_usage() {
        let tools: Vec<Box<dyn ToolT>> = vec![
            Box::new(MockTool::new("tool1", "First tool")),
            Box::new(MockTool::new("tool2", "Second tool")),
            Box::new(MockTool::with_failure("tool3", "Third tool")),
        ];

        for tool in &tools {
            assert!(!tool.name().is_empty());
            assert!(!tool.description().is_empty());
            assert!(tool.args_schema().is_object());
        }
    }

    #[tokio::test]
    async fn test_tool_run_with_different_inputs() {
        let tool = MockTool::new("varied_input_tool", "Varied input test");

        let inputs = vec![
            json!({"name": "test1", "value": 1}),
            json!({"name": "test2", "value": -5}),
            json!({"name": "", "value": 0}),
            json!({"name": "long_name_test", "value": 999999}),
        ];

        for input in inputs {
            let result = tool.execute(input.clone()).await;
            assert!(result.is_ok());

            let output = result.unwrap();
            assert_eq!(output["processed_name"], input["name"]);
            assert_eq!(
                output["doubled_value"],
                input["value"].as_i64().unwrap() * 2
            );
        }
    }

    #[test]
    fn test_tool_error_chaining() {
        let json_error = serde_json::from_str::<Value>("invalid").unwrap_err();
        let tool_error = ToolCallError::SerdeError(json_error);

        // Test error source chain
        use std::error::Error;
        assert!(tool_error.source().is_some());
    }

    #[test]
    fn test_tool_with_empty_name() {
        let tool = MockTool::new("", "Empty name test");
        assert_eq!(tool.name(), "");
        assert_eq!(tool.description(), "Empty name test");
    }

    #[test]
    fn test_tool_with_empty_description() {
        let tool = MockTool::new("empty_desc", "");
        assert_eq!(tool.name(), "empty_desc");
        assert_eq!(tool.description(), "");
    }

    #[test]
    fn test_tool_schema_complex() {
        let tool = MockTool::new("complex_tool", "Complex schema test");
        let schema = tool.args_schema();

        // Verify schema structure
        assert!(schema.is_object());
        assert!(schema["properties"].is_object());
        assert!(schema["required"].is_array());
        assert_eq!(schema["required"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_multiple_tool_instances() {
        let tool1 = MockTool::new("tool1", "First instance");
        let tool2 = MockTool::new("tool2", "Second instance");

        assert_ne!(tool1.name(), tool2.name());
        assert_ne!(tool1.description(), tool2.description());

        // Both should have the same schema structure
        assert_eq!(tool1.args_schema(), tool2.args_schema());
    }

    #[test]
    fn test_tool_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockTool>();
    }

    #[test]
    fn test_tool_trait_object_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn ToolT>>();
    }

    #[test]
    fn test_tool_call_result_creation() {
        let result = ToolCallResult {
            tool_name: "test_tool".to_string(),
            success: true,
            arguments: json!({"param": "value"}),
            result: json!({"output": "success"}),
        };

        assert_eq!(result.tool_name, "test_tool");
        assert!(result.success);
        assert_eq!(result.arguments, json!({"param": "value"}));
        assert_eq!(result.result, json!({"output": "success"}));
    }

    #[test]
    fn test_tool_call_result_serialization() {
        let result = ToolCallResult {
            tool_name: "serialize_tool".to_string(),
            success: false,
            arguments: json!({"input": "test"}),
            result: json!({"error": "failed"}),
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: ToolCallResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.tool_name, "serialize_tool");
        assert!(!deserialized.success);
        assert_eq!(deserialized.arguments, json!({"input": "test"}));
        assert_eq!(deserialized.result, json!({"error": "failed"}));
    }

    #[test]
    fn test_tool_call_result_clone() {
        let result = ToolCallResult {
            tool_name: "clone_tool".to_string(),
            success: true,
            arguments: json!({"data": [1, 2, 3]}),
            result: json!({"processed": [2, 4, 6]}),
        };

        let cloned = result.clone();
        assert_eq!(result.tool_name, cloned.tool_name);
        assert_eq!(result.success, cloned.success);
        assert_eq!(result.arguments, cloned.arguments);
        assert_eq!(result.result, cloned.result);
    }

    #[test]
    fn test_tool_call_result_debug() {
        let result = ToolCallResult {
            tool_name: "debug_tool".to_string(),
            success: true,
            arguments: json!({}),
            result: json!(null),
        };

        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("ToolCallResult"));
        assert!(debug_str.contains("debug_tool"));
    }

    #[test]
    fn test_tool_call_result_with_null_values() {
        let result = ToolCallResult {
            tool_name: "null_tool".to_string(),
            success: false,
            arguments: json!(null),
            result: json!(null),
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: ToolCallResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.tool_name, "null_tool");
        assert!(!deserialized.success);
        assert_eq!(deserialized.arguments, json!(null));
        assert_eq!(deserialized.result, json!(null));
    }

    #[test]
    fn test_tool_call_result_with_complex_data() {
        let complex_args = json!({
            "nested": {
                "array": [1, 2, {"key": "value"}],
                "string": "test",
                "number": 42.5
            }
        });

        let complex_result = json!({
            "status": "completed",
            "data": {
                "items": ["a", "b", "c"],
                "count": 3
            }
        });

        let result = ToolCallResult {
            tool_name: "complex_tool".to_string(),
            success: true,
            arguments: complex_args.clone(),
            result: complex_result.clone(),
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: ToolCallResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.arguments, complex_args);
        assert_eq!(deserialized.result, complex_result);
    }

    #[test]
    fn test_tool_call_result_empty_tool_name() {
        let result = ToolCallResult {
            tool_name: String::new(),
            success: true,
            arguments: json!({}),
            result: json!({}),
        };

        assert!(result.tool_name.is_empty());
        assert!(result.success);
    }

    #[test]
    fn test_tool_call_result_large_data() {
        let large_string = "x".repeat(10000);
        let result = ToolCallResult {
            tool_name: "large_tool".to_string(),
            success: true,
            arguments: json!({"large_param": large_string}),
            result: json!({"processed": true}),
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: ToolCallResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.tool_name, "large_tool");
        assert!(deserialized.success);
        assert!(
            deserialized.arguments["large_param"]
                .as_str()
                .unwrap()
                .len()
                == 10000
        );
    }

    #[test]
    fn test_tool_call_result_equality() {
        let result1 = ToolCallResult {
            tool_name: "equal_tool".to_string(),
            success: true,
            arguments: json!({"param": "value"}),
            result: json!({"output": "result"}),
        };

        let result2 = ToolCallResult {
            tool_name: "equal_tool".to_string(),
            success: true,
            arguments: json!({"param": "value"}),
            result: json!({"output": "result"}),
        };

        let result3 = ToolCallResult {
            tool_name: "different_tool".to_string(),
            success: true,
            arguments: json!({"param": "value"}),
            result: json!({"output": "result"}),
        };

        // Test equality through serialization since ToolCallResult doesn't implement PartialEq
        let serialized1 = serde_json::to_string(&result1).unwrap();
        let serialized2 = serde_json::to_string(&result2).unwrap();
        let serialized3 = serde_json::to_string(&result3).unwrap();

        assert_eq!(serialized1, serialized2);
        assert_ne!(serialized1, serialized3);
    }

    #[test]
    fn test_tool_call_result_with_unicode() {
        let result = ToolCallResult {
            tool_name: "unicode_tool".to_string(),
            success: true,
            arguments: json!({"message": "Hello 世界! 🌍"}),
            result: json!({"response": "Processed: Hello 世界! 🌍"}),
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: ToolCallResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.arguments["message"], "Hello 世界! 🌍");
        assert_eq!(deserialized.result["response"], "Processed: Hello 世界! 🌍");
    }

    #[test]
    fn test_tool_call_result_with_arrays() {
        let result = ToolCallResult {
            tool_name: "array_tool".to_string(),
            success: true,
            arguments: json!({"numbers": [1, 2, 3, 4, 5]}),
            result: json!({"sum": 15, "count": 5}),
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: ToolCallResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.arguments["numbers"], json!([1, 2, 3, 4, 5]));
        assert_eq!(deserialized.result["sum"], 15);
        assert_eq!(deserialized.result["count"], 5);
    }

    #[test]
    fn test_tool_call_result_boolean_values() {
        let result = ToolCallResult {
            tool_name: "bool_tool".to_string(),
            success: false,
            arguments: json!({"enabled": true, "debug": false}),
            result: json!({"valid": false, "error": true}),
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: ToolCallResult = serde_json::from_str(&serialized).unwrap();

        assert!(!deserialized.success);
        assert_eq!(deserialized.arguments["enabled"], true);
        assert_eq!(deserialized.arguments["debug"], false);
        assert_eq!(deserialized.result["valid"], false);
        assert_eq!(deserialized.result["error"], true);
    }
}
