use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde_json::{Map, Number, Value};

pub fn json_value_to_py(py: Python<'_>, val: &Value) -> PyResult<Py<PyAny>> {
    match val {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Err(PyTypeError::new_err("unsupported JSON number"))
            }
        }
        Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_value_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Object(items) => {
            let dict = PyDict::new(py);
            for (key, value) in items {
                dict.set_item(key, json_value_to_py(py, value)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

pub fn py_any_to_json_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }

    if let Some(json) = extract_json_scalar(value)? {
        return Ok(json);
    }

    if let Some(json) = extract_json_sequence(value)? {
        return Ok(json);
    }

    if let Some(json) = extract_json_mapping(value)? {
        return Ok(json);
    }

    if let Some(dumped) = dump_custom_object(value)? {
        return py_any_to_json_value(&dumped);
    }

    Err(non_serializable_type_error(value))
}

fn extract_json_scalar(value: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    if let Ok(v) = value.extract::<bool>() {
        return Ok(Some(Value::Bool(v)));
    }

    if let Ok(v) = value.extract::<i64>() {
        return Ok(Some(Value::Number(Number::from(v))));
    }

    if let Ok(v) = value.extract::<u64>() {
        return Ok(Some(Value::Number(Number::from(v))));
    }

    if let Ok(v) = value.extract::<f64>() {
        let number = Number::from_f64(v)
            .ok_or_else(|| PyTypeError::new_err("cannot convert NaN or infinity to JSON"))?;
        return Ok(Some(Value::Number(number)));
    }

    if let Ok(v) = value.extract::<String>() {
        return Ok(Some(Value::String(v)));
    }

    Ok(None)
}

fn extract_json_sequence(value: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    if let Ok(list) = value.cast::<PyList>() {
        return Ok(Some(Value::Array(iterable_to_json_array(list.iter())?)));
    }

    if let Ok(tuple) = value.cast::<PyTuple>() {
        return Ok(Some(Value::Array(iterable_to_json_array(tuple.iter())?)));
    }

    Ok(None)
}

fn extract_json_mapping(value: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    let Ok(dict) = value.cast::<PyDict>() else {
        return Ok(None);
    };

    let mut items = Map::with_capacity(dict.len());
    for (key, item) in dict.iter() {
        let key = key
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("JSON object keys must be strings"))?;
        items.insert(key, py_any_to_json_value(&item)?);
    }
    Ok(Some(Value::Object(items)))
}

fn iterable_to_json_array<'py, I>(items: I) -> PyResult<Vec<Value>>
where
    I: IntoIterator<Item = Bound<'py, PyAny>>,
{
    items
        .into_iter()
        .map(|item| py_any_to_json_value(&item))
        .collect()
}

fn dump_custom_object<'py>(value: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if value.hasattr("model_dump")? {
        return value.call_method0("model_dump").map(Some);
    }

    if value.hasattr("dict")? {
        return value.call_method0("dict").map(Some);
    }

    if value.hasattr("__dataclass_fields__")? {
        let dataclasses = value.py().import("dataclasses")?;
        return dataclasses.call_method1("asdict", (value,)).map(Some);
    }

    Ok(None)
}

fn non_serializable_type_error(value: &Bound<'_, PyAny>) -> PyErr {
    let type_name = value
        .get_type()
        .name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "<unknown>".to_string());
    PyTypeError::new_err(format!(
        "value of type '{type_name}' is not JSON serializable by the AutoAgents binding"
    ))
}
