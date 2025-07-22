use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub tool_name: String,
    pub success: bool,
    pub arguments: Value,
    pub result: Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
