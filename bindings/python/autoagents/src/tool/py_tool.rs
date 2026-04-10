use crate::convert::{json_value_to_py, py_any_to_json_value};
use async_trait::async_trait;
use autoagents_core::tool::{ToolCallError, ToolRuntime, ToolT};
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use serde_json::Value;
use std::fmt;

/// Helper: convert a `PyErr` into a `ToolCallError`.
fn py_to_tool_err(e: PyErr) -> ToolCallError {
    #[derive(Debug)]
    struct Msg(String);
    impl fmt::Display for Msg {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.0)
        }
    }
    impl std::error::Error for Msg {}
    ToolCallError::RuntimeError(Box::new(Msg(e.to_string())))
}

fn validation_err(message: impl Into<String>) -> ToolCallError {
    #[derive(Debug)]
    struct Msg(String);
    impl fmt::Display for Msg {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.0)
        }
    }
    impl std::error::Error for Msg {}
    ToolCallError::RuntimeError(Box::new(Msg(message.into())))
}

fn schema_object(schema: &Value) -> Result<&serde_json::Map<String, Value>, ToolCallError> {
    schema
        .as_object()
        .ok_or_else(|| validation_err("tool schema must be a JSON object"))
}

fn path_label(path: &str) -> &str {
    if path.is_empty() { "$" } else { path }
}

fn validate_type(path: &str, expected: &str, value: &Value) -> Result<(), ToolCallError> {
    let valid = match expected {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "string" => value.is_string(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        other => {
            return Err(validation_err(format!(
                "unsupported schema type '{other}' at {}",
                path_label(path)
            )));
        }
    };

    if valid {
        Ok(())
    } else {
        Err(validation_err(format!(
            "expected {expected} at {}, got {}",
            path_label(path),
            value
        )))
    }
}

fn validate_against_schema(path: &str, schema: &Value, value: &Value) -> Result<(), ToolCallError> {
    let object = schema_object(schema)?;
    validate_enum_constraint(path, object, value)?;
    validate_any_of_constraint(path, object, value)?;
    validate_type_constraint(path, object, value)?;
    validate_array_items(path, object, value)?;
    validate_object_properties(path, object, value)?;
    Ok(())
}

fn validate_enum_constraint(
    path: &str,
    object: &serde_json::Map<String, Value>,
    value: &Value,
) -> Result<(), ToolCallError> {
    if let Some(enum_values) = object.get("enum").and_then(Value::as_array) {
        if enum_values.iter().any(|candidate| candidate == value) {
            return Ok(());
        }

        return Err(validation_err(format!(
            "value at {} is not one of the allowed enum variants",
            path_label(path)
        )));
    }

    Ok(())
}

fn validate_any_of_constraint(
    path: &str,
    object: &serde_json::Map<String, Value>,
    value: &Value,
) -> Result<(), ToolCallError> {
    let Some(any_of) = object.get("anyOf").and_then(Value::as_array) else {
        return Ok(());
    };

    let mut last_error = None;
    for variant in any_of {
        match validate_against_schema(path, variant, value) {
            Ok(()) => return Ok(()),
            Err(error) => last_error = Some(error),
        }
    }

    Err(last_error.unwrap_or_else(|| {
        validation_err(format!(
            "value at {} did not satisfy anyOf",
            path_label(path)
        ))
    }))
}

fn validate_type_constraint(
    path: &str,
    object: &serde_json::Map<String, Value>,
    value: &Value,
) -> Result<(), ToolCallError> {
    let Some(raw_type) = object.get("type") else {
        return Ok(());
    };

    match raw_type {
        Value::String(expected) => validate_type(path, expected, value),
        Value::Array(types) => validate_one_of_types(path, types, value),
        _ => Err(validation_err(format!(
            "schema type for {} must be a string or string list",
            path_label(path)
        ))),
    }
}

fn validate_one_of_types(path: &str, types: &[Value], value: &Value) -> Result<(), ToolCallError> {
    for candidate in types {
        if let Some(expected) = candidate.as_str()
            && validate_type(path, expected, value).is_ok()
        {
            return Ok(());
        }
    }

    Err(validation_err(format!(
        "value at {} did not match any allowed schema types",
        path_label(path)
    )))
}

fn validate_array_items(
    path: &str,
    object: &serde_json::Map<String, Value>,
    value: &Value,
) -> Result<(), ToolCallError> {
    let Some(items_schema) = object.get("items") else {
        return Ok(());
    };

    let items = value
        .as_array()
        .ok_or_else(|| validation_err(format!("expected array at {}", path_label(path))))?;
    for (index, item) in items.iter().enumerate() {
        let item_path = format!("{}[{index}]", path_label(path));
        validate_against_schema(&item_path, items_schema, item)?;
    }
    Ok(())
}

fn validate_object_properties(
    path: &str,
    object: &serde_json::Map<String, Value>,
    value: &Value,
) -> Result<(), ToolCallError> {
    let Some(properties) = object.get("properties").and_then(Value::as_object) else {
        return Ok(());
    };

    let object_value = value
        .as_object()
        .ok_or_else(|| validation_err(format!("expected object at {}", path_label(path))))?;

    validate_required_fields(path, object, object_value)?;

    for (key, item) in object_value {
        if let Some(field_schema) = properties.get(key) {
            let field_path = field_path(path, key);
            validate_against_schema(&field_path, field_schema, item)?;
            continue;
        }

        validate_additional_property(path, object.get("additionalProperties"), key, item)?;
    }

    Ok(())
}

fn validate_required_fields(
    path: &str,
    object: &serde_json::Map<String, Value>,
    object_value: &serde_json::Map<String, Value>,
) -> Result<(), ToolCallError> {
    let required = object
        .get("required")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .collect::<std::collections::HashSet<_>>()
        })
        .unwrap_or_default();

    for key in required {
        if !object_value.contains_key(key) {
            let field_path = field_path(path, key);
            return Err(validation_err(format!(
                "missing required field {}",
                path_label(&field_path)
            )));
        }
    }

    Ok(())
}

fn validate_additional_property(
    path: &str,
    additional_properties: Option<&Value>,
    key: &str,
    item: &Value,
) -> Result<(), ToolCallError> {
    let field_path = field_path(path, key);
    match additional_properties {
        Some(Value::Bool(true)) => Ok(()),
        Some(Value::Bool(false)) | None => Err(validation_err(format!(
            "unexpected field {}",
            path_label(&field_path)
        ))),
        Some(extra_schema) => validate_against_schema(&field_path, extra_schema, item),
    }
}

fn field_path(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_string()
    } else {
        format!("{path}.{key}")
    }
}

fn validate_tool_args(schema: &Value, args: &Value) -> Result<(), ToolCallError> {
    if !schema.is_object() {
        return Err(validation_err("tool schema must be a JSON object"));
    }
    if !args.is_object() {
        return Err(validation_err("tool arguments must be a JSON object"));
    }
    validate_against_schema("", schema, args)
}

/// A Python callable wrapped as a Rust `ToolT + ToolRuntime`.
///
/// Sync tools execute on Tokio's blocking pool. Async tools resolve through the
/// shared PyO3 async bridge using the task locals captured when the tool was
/// created, when available.
#[pyclass(name = "Tool", skip_from_py_object)]
pub struct PyTool {
    name: String,
    description: String,
    schema: Value,
    output_schema: Option<Value>,
    callable: Py<PyAny>,
    task_locals: Option<TaskLocals>,
}

impl Clone for PyTool {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            name: self.name.clone(),
            description: self.description.clone(),
            schema: self.schema.clone(),
            output_schema: self.output_schema.clone(),
            callable: self.callable.clone_ref(py),
            task_locals: self.task_locals.clone(),
        })
    }
}

#[pymethods]
impl PyTool {
    #[new]
    #[pyo3(signature = (name, description, schema_json, callable, output_schema_json=None))]
    pub fn new(
        name: String,
        description: String,
        schema_json: String,
        callable: Py<PyAny>,
        output_schema_json: Option<String>,
    ) -> PyResult<Self> {
        let schema: Value = serde_json::from_str(&schema_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if !matches!(schema, Value::Object(_)) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tool schema must be a JSON object",
            ));
        }
        let output_schema = match output_schema_json {
            Some(output_schema_json) => {
                let output_schema: Value = serde_json::from_str(&output_schema_json)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                if !matches!(output_schema, Value::Object(_)) {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "tool output schema must be a JSON object",
                    ));
                }
                Some(output_schema)
            }
            None => None,
        };
        let task_locals = Python::attach(|py| -> PyResult<Option<TaskLocals>> {
            let bound = callable.bind(py);
            if !bound.is_none() && !bound.is_callable() {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "tool callable must be callable",
                ));
            }
            Ok(pyo3_async_runtimes::tokio::get_current_locals(py).ok())
        })?;
        Ok(Self {
            name,
            description,
            schema,
            output_schema,
            callable,
            task_locals,
        })
    }

    fn __repr__(&self) -> String {
        format!("Tool(name='{}')", self.name)
    }
}

impl fmt::Debug for PyTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PyTool").field("name", &self.name).finish()
    }
}

impl ToolT for PyTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn args_schema(&self) -> Value {
        self.schema.clone()
    }

    fn output_schema(&self) -> Option<Value> {
        self.output_schema.clone()
    }
}

#[async_trait]
impl ToolRuntime for PyTool {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        validate_tool_args(&self.schema, &args)?;

        let callable = Python::attach(|py| self.callable.clone_ref(py));
        let args_for_call = args.clone();
        let (result_obj, is_awaitable) = tokio::task::spawn_blocking(move || {
            Python::attach(|py: Python<'_>| {
                let callable = callable.bind(py);
                if callable.is_none() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "tool callable is not configured",
                    ));
                }

                let py_args = json_value_to_py(py, &args_for_call)?;
                let result = callable.call1((py_args.bind(py),))?;
                let is_awaitable = crate::async_bridge::is_awaitable(&result)?;
                Ok((result.unbind(), is_awaitable))
            })
        })
        .await
        .map_err(|error| validation_err(format!("tool task join failed: {error}")))?
        .map_err(py_to_tool_err)?;

        let resolved = crate::async_bridge::resolve_maybe_awaitable(
            result_obj,
            is_awaitable,
            self.task_locals.as_ref(),
        )
        .await
        .map_err(py_to_tool_err)?;

        Python::attach(|py| py_any_to_json_value(resolved.bind(py)).map_err(py_to_tool_err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_core::tool::{ToolRuntime, ToolT};
    use pyo3::types::PyModule;
    use serde_json::json;
    use std::ffi::CString;

    fn init_runtime_bridge() {
        Python::initialize();
        let runtime = crate::runtime::get_runtime().expect("shared runtime should initialize");
        let _ = pyo3_async_runtimes::tokio::init_with_runtime(runtime);
    }

    fn tool_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
        PyModule::from_code(
            py,
            &CString::new(
                "def sync_sum(payload):\n\
                 \treturn {\"sum\": payload[\"left\"] + payload[\"right\"], \"kind\": \"sync\"}\n\
                 \n\
                 async def async_echo(payload):\n\
                 \treturn {\"echo\": payload[\"value\"], \"kind\": \"async\"}\n\
                 \n\
                 def passthrough(payload):\n\
                 \treturn payload\n\
                 \n\
                 def failing_tool(payload):\n\
                 \traise ValueError(f\"boom:{payload['value']}\")\n\
                 \n\
                 non_callable = 123\n",
            )
            .expect("python module source should be valid CString"),
            &CString::new("autoagents_py/tests/py_tool.py")
                .expect("filename should be a valid CString"),
            &CString::new("autoagents_py_tool_tests")
                .expect("module name should be a valid CString"),
        )
    }

    #[test]
    fn validate_tool_args_accepts_valid_payload() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "mode": {"enum": ["fast", "slow"]}
            },
            "required": ["a"],
            "additionalProperties": false,
        });

        let args = json!({
            "a": 3,
            "mode": "fast"
        });

        assert!(validate_tool_args(&schema, &args).is_ok());
    }

    #[test]
    fn validate_tool_args_rejects_unknown_fields() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "integer"}
            },
            "required": ["a"],
            "additionalProperties": false,
        });

        let error = validate_tool_args(&schema, &json!({"a": 1, "extra": true}))
            .expect_err("unexpected field should fail");
        assert!(error.to_string().contains("unexpected field"));
    }

    #[test]
    fn validate_tool_args_rejects_type_mismatches() {
        let schema = json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "names": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["count", "names"],
            "additionalProperties": false,
        });

        let error = validate_tool_args(
            &schema,
            &json!({
                "count": "two",
                "names": ["ok"]
            }),
        )
        .expect_err("type mismatch should fail");
        assert!(error.to_string().contains("expected integer"));
    }

    #[test]
    fn validate_tool_args_covers_any_of_and_nested_additional_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"}
                    ]
                }
            },
            "required": ["value"],
            "additionalProperties": {
                "type": "integer"
            }
        });

        assert!(
            validate_tool_args(
                &schema,
                &json!({
                    "value": "ok",
                    "count": 2
                }),
            )
            .is_ok()
        );
        assert!(
            validate_tool_args(
                &schema,
                &json!({
                    "value": true
                }),
            )
            .is_err()
        );
        assert!(
            validate_tool_args(
                &schema,
                &json!({
                    "value": 1,
                    "count": "bad"
                }),
            )
            .expect_err("additional property schema should validate")
            .to_string()
            .contains("expected integer")
        );
    }

    #[test]
    fn py_tool_new_validates_schema_output_schema_and_callable() {
        init_runtime_bridge();
        Python::attach(|py| {
            let module = tool_module(py).expect("test module should load");
            let sync_callable = module
                .getattr("sync_sum")
                .expect("sync callable should exist")
                .unbind();
            let non_callable = module
                .getattr("non_callable")
                .expect("non-callable value should exist")
                .unbind();

            let invalid_schema = PyTool::new(
                "bad".to_string(),
                "bad".to_string(),
                "{".to_string(),
                sync_callable.clone_ref(py),
                None,
            )
            .expect_err("invalid schema json should fail");
            assert!(invalid_schema.is_instance_of::<pyo3::exceptions::PyValueError>(py));

            let non_object_schema = PyTool::new(
                "bad".to_string(),
                "bad".to_string(),
                "[]".to_string(),
                sync_callable.clone_ref(py),
                None,
            )
            .expect_err("schema must be an object");
            assert!(
                non_object_schema
                    .to_string()
                    .contains("tool schema must be a JSON object")
            );

            let non_object_output_schema = PyTool::new(
                "bad".to_string(),
                "bad".to_string(),
                r#"{"type":"object"}"#.to_string(),
                sync_callable.clone_ref(py),
                Some("[]".to_string()),
            )
            .expect_err("output schema must be an object");
            assert!(
                non_object_output_schema
                    .to_string()
                    .contains("tool output schema must be a JSON object")
            );

            let non_callable_error = PyTool::new(
                "bad".to_string(),
                "bad".to_string(),
                r#"{"type":"object"}"#.to_string(),
                non_callable,
                None,
            )
            .expect_err("non-callable should fail");
            assert!(
                non_callable_error
                    .to_string()
                    .contains("tool callable must be callable")
            );

            let tool = PyTool::new(
                "sum".to_string(),
                "Adds values".to_string(),
                r#"{"type":"object","properties":{"left":{"type":"integer"},"right":{"type":"integer"}},"required":["left","right"],"additionalProperties":false}"#.to_string(),
                sync_callable,
                Some(r#"{"type":"object","properties":{"sum":{"type":"integer"}},"required":["sum"]}"#.to_string()),
            )
            .expect("valid tool should construct");
            assert_eq!(tool.name(), "sum");
            assert_eq!(tool.description(), "Adds values");
            assert_eq!(tool.__repr__(), "Tool(name='sum')");
            assert!(tool.output_schema().is_some());
        });
    }

    #[test]
    fn py_tool_execute_supports_sync_and_async_callables() {
        init_runtime_bridge();
        Python::attach(|py| -> PyResult<()> {
            let event_loop = py.import("asyncio")?.call_method0("new_event_loop")?;
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                let (sync_tool, async_tool) = Python::attach(|py| -> PyResult<_> {
                    let module = tool_module(py)?;
                    let sync_callable = module.getattr("sync_sum")?.unbind();
                    let async_callable = module.getattr("async_echo")?.unbind();

                    Ok((
                        PyTool::new(
                            "sum".to_string(),
                            "Adds values".to_string(),
                            r#"{"type":"object","properties":{"left":{"type":"integer"},"right":{"type":"integer"}},"required":["left","right"],"additionalProperties":false}"#.to_string(),
                            sync_callable,
                            None,
                        )?,
                        PyTool::new(
                            "echo".to_string(),
                            "Echoes values".to_string(),
                            r#"{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false}"#.to_string(),
                            async_callable,
                            None,
                        )?,
                    ))
                })?;

                let sync_result = sync_tool
                    .execute(json!({"left": 20, "right": 22}))
                    .await
                    .expect("sync tool should succeed");
                assert_eq!(sync_result, json!({"sum": 42, "kind": "sync"}));

                let async_result = async_tool
                    .execute(json!({"value": 7}))
                    .await
                    .expect("async tool should succeed");
                assert_eq!(async_result, json!({"echo": 7, "kind": "async"}));
                Ok(())
            })
        })
        .expect("temporary Python event loop should run");
    }

    #[test]
    fn py_tool_execute_reports_validation_and_python_runtime_errors() {
        init_runtime_bridge();
        Python::attach(|py| -> PyResult<()> {
            let event_loop = py.import("asyncio")?.call_method0("new_event_loop")?;
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                let (passthrough_tool, failing_tool, none_tool) =
                    Python::attach(|py| -> PyResult<_> {
                        let module = tool_module(py)?;
                        let passthrough = module.getattr("passthrough")?.unbind();
                        let failing = module.getattr("failing_tool")?.unbind();
                        let none_value = py.None();

                        Ok((
                            PyTool::new(
                                "passthrough".to_string(),
                                "Returns the payload".to_string(),
                                r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":{"type":"integer"}}"#.to_string(),
                                passthrough,
                                None,
                            )?,
                            PyTool::new(
                                "failing".to_string(),
                                "Raises an exception".to_string(),
                                r#"{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false}"#.to_string(),
                                failing,
                                None,
                            )?,
                            PyTool::new(
                                "missing".to_string(),
                                "Missing callable".to_string(),
                                r#"{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false}"#.to_string(),
                                none_value,
                                None,
                            )?,
                        ))
                    })?;

                let validation_error = passthrough_tool
                    .execute(json!({"name": "alice", "count": "bad"}))
                    .await
                    .expect_err("invalid additional property type should fail");
                assert!(validation_error.to_string().contains("expected integer"));

                let python_error = failing_tool
                    .execute(json!({"value": 7}))
                    .await
                    .expect_err("python exception should surface");
                assert!(python_error.to_string().contains("boom:7"));

                let missing_callable = none_tool
                    .execute(json!({"value": 1}))
                    .await
                    .expect_err("missing callable should fail");
                assert!(missing_callable.to_string().contains("not configured"));
                Ok(())
            })
        })
        .expect("temporary Python event loop should run");
    }
}
