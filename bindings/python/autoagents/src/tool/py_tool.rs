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

    if let Some(enum_values) = object.get("enum").and_then(Value::as_array)
        && !enum_values.iter().any(|candidate| candidate == value)
    {
        return Err(validation_err(format!(
            "value at {} is not one of the allowed enum variants",
            path_label(path)
        )));
    }

    if let Some(any_of) = object.get("anyOf").and_then(Value::as_array) {
        let mut last_error = None;
        for variant in any_of {
            match validate_against_schema(path, variant, value) {
                Ok(()) => return Ok(()),
                Err(error) => last_error = Some(error),
            }
        }
        return Err(last_error.unwrap_or_else(|| {
            validation_err(format!(
                "value at {} did not satisfy anyOf",
                path_label(path)
            ))
        }));
    }

    if let Some(raw_type) = object.get("type") {
        match raw_type {
            Value::String(expected) => validate_type(path, expected, value)?,
            Value::Array(types) => {
                let mut matched = false;
                for candidate in types {
                    if let Some(expected) = candidate.as_str()
                        && validate_type(path, expected, value).is_ok()
                    {
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    return Err(validation_err(format!(
                        "value at {} did not match any allowed schema types",
                        path_label(path)
                    )));
                }
            }
            _ => {
                return Err(validation_err(format!(
                    "schema type for {} must be a string or string list",
                    path_label(path)
                )));
            }
        }
    }

    if let Some(items_schema) = object.get("items") {
        let items = value
            .as_array()
            .ok_or_else(|| validation_err(format!("expected array at {}", path_label(path))))?;
        for (index, item) in items.iter().enumerate() {
            let item_path = format!("{}[{index}]", path_label(path));
            validate_against_schema(&item_path, items_schema, item)?;
        }
    }

    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        let object_value = value
            .as_object()
            .ok_or_else(|| validation_err(format!("expected object at {}", path_label(path))))?;

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
                let field_path = if path.is_empty() {
                    key.to_string()
                } else {
                    format!("{path}.{key}")
                };
                return Err(validation_err(format!(
                    "missing required field {}",
                    path_label(&field_path)
                )));
            }
        }

        for (key, item) in object_value {
            if let Some(field_schema) = properties.get(key) {
                let field_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                validate_against_schema(&field_path, field_schema, item)?;
                continue;
            }

            match object.get("additionalProperties") {
                Some(Value::Bool(false)) | None => {
                    let field_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    return Err(validation_err(format!(
                        "unexpected field {}",
                        path_label(&field_path)
                    )));
                }
                Some(Value::Bool(true)) => {}
                Some(extra_schema) => {
                    let field_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    validate_against_schema(&field_path, extra_schema, item)?;
                }
            }
        }
    }

    Ok(())
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
    callable: Py<PyAny>,
    task_locals: Option<TaskLocals>,
}

impl Clone for PyTool {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            name: self.name.clone(),
            description: self.description.clone(),
            schema: self.schema.clone(),
            callable: self.callable.clone_ref(py),
            task_locals: self.task_locals.clone(),
        })
    }
}

#[pymethods]
impl PyTool {
    #[new]
    pub fn new(
        name: String,
        description: String,
        schema_json: String,
        callable: Py<PyAny>,
    ) -> PyResult<Self> {
        let schema: Value = serde_json::from_str(&schema_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if !matches!(schema, Value::Object(_)) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tool schema must be a JSON object",
            ));
        }
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
    use super::validate_tool_args;
    use serde_json::json;

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
}
