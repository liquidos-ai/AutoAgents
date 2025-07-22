use serde_json::Value;
use std::fmt::Debug;

use crate::chat::{FunctionTool, Tool};

#[derive(Debug, thiserror::Error)]
pub enum ToolCallError {
    #[error("Runtime Error {0}")]
    RuntimeError(#[from] Box<dyn std::error::Error + Sync + Send>),

    #[error("Serde Error {0}")]
    SerdeError(#[from] serde_json::Error),
}

pub trait ToolT: Send + Sync + Debug {
    /// The name of the tool.
    fn name(&self) -> &'static str;
    /// A description explaining the tool’s purpose.
    fn description(&self) -> &'static str;
    /// Return a description of the expected arguments.
    fn args_schema(&self) -> Value;
    /// Run the tool with the given arguments (in JSON) and return the result (in JSON).
    fn run(&self, args: Value) -> Result<Value, ToolCallError>;
}

pub trait ToolInputT {
    fn io_schema() -> &'static str;
}

impl From<&Box<dyn ToolT>> for Tool {
    fn from(tool: &Box<dyn ToolT>) -> Self {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.args_schema(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::Tool;
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

        fn run(&self, args: Value) -> Result<Value, ToolCallError> {
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

    #[test]
    fn test_mock_tool_run_success() {
        let tool = MockTool::new("success_tool", "Success test");
        let input = json!({
            "name": "test",
            "value": 42
        });

        let result = tool.run(input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output["processed_name"], "test");
        assert_eq!(output["doubled_value"], 84);
    }

    #[test]
    fn test_mock_tool_run_failure() {
        let tool = MockTool::with_failure("failure_tool", "Failure test");
        let input = json!({
            "name": "test",
            "value": 42
        });

        let result = tool.run(input);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Mock tool failure"));
    }

    #[test]
    fn test_mock_tool_run_invalid_input() {
        let tool = MockTool::new("invalid_input_tool", "Invalid input test");
        let input = json!({
            "invalid_field": "test"
        });

        let result = tool.run(input);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolCallError::SerdeError(_)));
    }

    #[test]
    fn test_mock_tool_run_with_extra_fields() {
        let tool = MockTool::new("extra_fields_tool", "Extra fields test");
        let input = json!({
            "name": "test",
            "value": 42,
            "extra_field": "ignored"
        });

        let result = tool.run(input);
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

        let tool: Tool = (&boxed_tool).into();
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "convert_tool");
        assert_eq!(tool.function.description, "Conversion test");
        assert_eq!(tool.function.parameters["type"], "object");
    }

    #[test]
    fn test_tool_conversion_preserves_schema() {
        let mock_tool = MockTool::new("schema_tool", "Schema preservation test");
        let boxed_tool: Box<dyn ToolT> = Box::new(mock_tool);

        let tool: Tool = (&boxed_tool).into();
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

    #[test]
    fn test_tool_run_with_different_inputs() {
        let tool = MockTool::new("varied_input_tool", "Varied input test");

        let inputs = vec![
            json!({"name": "test1", "value": 1}),
            json!({"name": "test2", "value": -5}),
            json!({"name": "", "value": 0}),
            json!({"name": "long_name_test", "value": 999999}),
        ];

        for input in inputs {
            let result = tool.run(input.clone());
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
}
