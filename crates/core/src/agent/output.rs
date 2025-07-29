use serde::{de::DeserializeOwned, Serialize};

/// Trait for agent output types that can generate structured output schemas
pub trait AgentOutputT: Serialize + DeserializeOwned + Send + Sync {
    /// Get the JSON schema string for this output type
    fn output_schema() -> &'static str;

    /// Get the structured output format as a JSON value
    fn structured_output_format() -> serde_json::Value;
}

/// Implementation of AgentOutputT for String type (when no output is specified)
impl AgentOutputT for String {
    fn output_schema() -> &'static str {
        "{}"
    }

    fn structured_output_format() -> serde_json::Value {
        serde_json::Value::Null
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestOutput {
        message: String,
        count: u32,
    }

    impl AgentOutputT for TestOutput {
        fn output_schema() -> &'static str {
            r#"{"type":"object","properties":{"message":{"type":"string"},"count":{"type":"integer"}},"required":["message","count"]}"#
        }

        fn structured_output_format() -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "count": {"type": "integer"}
                },
                "required": ["message", "count"]
            })
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct SimpleOutput {
        value: String,
    }

    impl AgentOutputT for SimpleOutput {
        fn output_schema() -> &'static str {
            r#"{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}"#
        }

        fn structured_output_format() -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {
                    "value": {"type": "string"}
                },
                "required": ["value"]
            })
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct ComplexOutput {
        name: String,
        age: u32,
        active: bool,
        tags: Vec<String>,
    }

    impl AgentOutputT for ComplexOutput {
        fn output_schema() -> &'static str {
            r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"active":{"type":"boolean"},"tags":{"type":"array","items":{"type":"string"}}},"required":["name","age","active","tags"]}"#
        }

        fn structured_output_format() -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "active": {"type": "boolean"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "age", "active", "tags"]
            })
        }
    }

    #[test]
    fn test_agent_output_trait_basic() {
        // Test that the trait is defined correctly
        fn assert_agent_output<T: AgentOutputT>() {}
        assert_agent_output::<String>();
        assert_agent_output::<TestOutput>();
    }

    #[test]
    fn test_string_output_schema() {
        let schema = String::output_schema();
        assert_eq!(schema, "{}");
    }

    #[test]
    fn test_string_structured_output_format() {
        let format = String::structured_output_format();
        assert_eq!(format, serde_json::Value::Null);
    }

    #[test]
    fn test_test_output_schema() {
        let schema = TestOutput::output_schema();
        assert!(schema.contains("object"));
        assert!(schema.contains("message"));
        assert!(schema.contains("count"));
        assert!(schema.contains("string"));
        assert!(schema.contains("integer"));
        assert!(schema.contains("required"));
    }

    #[test]
    fn test_test_output_structured_format() {
        let format = TestOutput::structured_output_format();
        assert_eq!(format["type"], "object");
        assert_eq!(format["properties"]["message"]["type"], "string");
        assert_eq!(format["properties"]["count"]["type"], "integer");
        assert_eq!(format["required"][0], "message");
        assert_eq!(format["required"][1], "count");
    }

    #[test]
    fn test_test_output_serialization() {
        let output = TestOutput {
            message: "Hello World".to_string(),
            count: 42,
        };
        let serialized = serde_json::to_string(&output).unwrap();
        assert!(serialized.contains("Hello World"));
        assert!(serialized.contains("42"));
    }

    #[test]
    fn test_test_output_deserialization() {
        let json = r#"{"message":"Test Message","count":123}"#;
        let output: TestOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.message, "Test Message");
        assert_eq!(output.count, 123);
    }

    #[test]
    fn test_test_output_clone() {
        let output = TestOutput {
            message: "Clone Test".to_string(),
            count: 999,
        };
        let cloned = output.clone();
        assert_eq!(output, cloned);
    }

    #[test]
    fn test_test_output_debug() {
        let output = TestOutput {
            message: "Debug Test".to_string(),
            count: 456,
        };
        let debug_str = format!("{output:?}");
        assert!(debug_str.contains("TestOutput"));
        assert!(debug_str.contains("Debug Test"));
        assert!(debug_str.contains("456"));
    }

    #[test]
    fn test_simple_output_schema() {
        let schema = SimpleOutput::output_schema();
        assert!(schema.contains("object"));
        assert!(schema.contains("value"));
        assert!(schema.contains("string"));
    }

    #[test]
    fn test_simple_output_structured_format() {
        let format = SimpleOutput::structured_output_format();
        assert_eq!(format["type"], "object");
        assert_eq!(format["properties"]["value"]["type"], "string");
        assert_eq!(format["required"][0], "value");
    }

    #[test]
    fn test_simple_output_serialization() {
        let output = SimpleOutput {
            value: "simple value".to_string(),
        };
        let serialized = serde_json::to_string(&output).unwrap();
        assert!(serialized.contains("simple value"));
    }

    #[test]
    fn test_simple_output_deserialization() {
        let json = r#"{"value":"deserialized value"}"#;
        let output: SimpleOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.value, "deserialized value");
    }

    #[test]
    fn test_complex_output_schema() {
        let schema = ComplexOutput::output_schema();
        assert!(schema.contains("object"));
        assert!(schema.contains("name"));
        assert!(schema.contains("age"));
        assert!(schema.contains("active"));
        assert!(schema.contains("tags"));
        assert!(schema.contains("array"));
        assert!(schema.contains("boolean"));
    }

    #[test]
    fn test_complex_output_structured_format() {
        let format = ComplexOutput::structured_output_format();
        assert_eq!(format["type"], "object");
        assert_eq!(format["properties"]["name"]["type"], "string");
        assert_eq!(format["properties"]["age"]["type"], "integer");
        assert_eq!(format["properties"]["active"]["type"], "boolean");
        assert_eq!(format["properties"]["tags"]["type"], "array");
        assert_eq!(format["properties"]["tags"]["items"]["type"], "string");
    }

    #[test]
    fn test_complex_output_serialization() {
        let output = ComplexOutput {
            name: "Test User".to_string(),
            age: 25,
            active: true,
            tags: vec!["tag1".to_string(), "tag2".to_string()],
        };
        let serialized = serde_json::to_string(&output).unwrap();
        assert!(serialized.contains("Test User"));
        assert!(serialized.contains("25"));
        assert!(serialized.contains("true"));
        assert!(serialized.contains("tag1"));
        assert!(serialized.contains("tag2"));
    }

    #[test]
    fn test_complex_output_deserialization() {
        let json = r#"{"name":"Complex User","age":30,"active":false,"tags":["work","personal"]}"#;
        let output: ComplexOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.name, "Complex User");
        assert_eq!(output.age, 30);
        assert!(!output.active);
        assert_eq!(output.tags, vec!["work", "personal"]);
    }

    #[test]
    fn test_complex_output_with_empty_tags() {
        let output = ComplexOutput {
            name: "Empty Tags".to_string(),
            age: 35,
            active: true,
            tags: vec![],
        };
        let serialized = serde_json::to_string(&output).unwrap();
        let deserialized: ComplexOutput = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.tags.len(), 0);
    }

    #[test]
    fn test_output_with_special_characters() {
        let output = TestOutput {
            message: "Special chars: !@#$%^&*()".to_string(),
            count: 0,
        };
        let serialized = serde_json::to_string(&output).unwrap();
        let deserialized: TestOutput = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.message, "Special chars: !@#$%^&*()");
    }

    #[test]
    fn test_output_with_unicode() {
        let output = TestOutput {
            message: "Unicode: 你好世界 🌍".to_string(),
            count: 42,
        };
        let serialized = serde_json::to_string(&output).unwrap();
        let deserialized: TestOutput = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.message, "Unicode: 你好世界 🌍");
    }

    #[test]
    fn test_output_with_newlines() {
        let output = TestOutput {
            message: "Multi\nLine\nMessage".to_string(),
            count: 3,
        };
        let serialized = serde_json::to_string(&output).unwrap();
        let deserialized: TestOutput = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.message, "Multi\nLine\nMessage");
    }

    #[test]
    fn test_output_with_large_count() {
        let output = TestOutput {
            message: "Large count".to_string(),
            count: u32::MAX,
        };
        let serialized = serde_json::to_string(&output).unwrap();
        let deserialized: TestOutput = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.count, u32::MAX);
    }

    #[test]
    fn test_output_with_zero_count() {
        let output = TestOutput {
            message: "Zero count".to_string(),
            count: 0,
        };
        let serialized = serde_json::to_string(&output).unwrap();
        let deserialized: TestOutput = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.count, 0);
    }

    #[test]
    fn test_output_with_empty_string() {
        let output = TestOutput {
            message: "".to_string(),
            count: 100,
        };
        let serialized = serde_json::to_string(&output).unwrap();
        let deserialized: TestOutput = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.message, "");
        assert_eq!(deserialized.count, 100);
    }

    #[test]
    fn test_output_equality() {
        let output1 = TestOutput {
            message: "Equal test".to_string(),
            count: 50,
        };
        let output2 = TestOutput {
            message: "Equal test".to_string(),
            count: 50,
        };
        let output3 = TestOutput {
            message: "Different test".to_string(),
            count: 50,
        };
        assert_eq!(output1, output2);
        assert_ne!(output1, output3);
    }

    #[test]
    fn test_multiple_output_types() {
        let _test_output = TestOutput {
            message: "test".to_string(),
            count: 1,
        };
        let _simple_output = SimpleOutput {
            value: "simple".to_string(),
        };
        let _complex_output = ComplexOutput {
            name: "complex".to_string(),
            age: 25,
            active: true,
            tags: vec!["tag".to_string()],
        };

        // Test that all implement the trait
        assert_ne!(TestOutput::output_schema(), SimpleOutput::output_schema());
        assert_ne!(
            SimpleOutput::output_schema(),
            ComplexOutput::output_schema()
        );
        assert_ne!(TestOutput::output_schema(), ComplexOutput::output_schema());
    }

    #[test]
    fn test_schema_json_validity() {
        // Test that schemas are valid JSON
        let schemas = vec![
            String::output_schema(),
            TestOutput::output_schema(),
            SimpleOutput::output_schema(),
            ComplexOutput::output_schema(),
        ];

        for schema in schemas {
            if schema != "{}" {
                let _: serde_json::Value = serde_json::from_str(schema).unwrap();
            }
        }
    }

    #[test]
    fn test_structured_format_json_validity() {
        // Test that structured formats are valid JSON
        let formats = vec![
            String::structured_output_format(),
            TestOutput::structured_output_format(),
            SimpleOutput::structured_output_format(),
            ComplexOutput::structured_output_format(),
        ];

        for format in formats {
            if format != serde_json::Value::Null {
                assert!(format.is_object() || format.is_null());
            }
        }
    }

    #[test]
    fn test_send_sync_traits() {
        // Test that our types implement Send and Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TestOutput>();
        assert_send_sync::<SimpleOutput>();
        assert_send_sync::<ComplexOutput>();
    }

    #[test]
    fn test_round_trip_serialization() {
        let outputs = vec![
            TestOutput {
                message: "Round trip test".to_string(),
                count: 789,
            },
            TestOutput {
                message: "Another test".to_string(),
                count: 0,
            },
        ];

        for output in outputs {
            let serialized = serde_json::to_string(&output).unwrap();
            let deserialized: TestOutput = serde_json::from_str(&serialized).unwrap();
            assert_eq!(output, deserialized);
        }
    }
}
