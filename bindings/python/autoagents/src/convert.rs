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

    if let Ok(v) = value.extract::<bool>() {
        return Ok(Value::Bool(v));
    }

    if let Ok(v) = value.extract::<i64>() {
        return Ok(Value::Number(Number::from(v)));
    }

    if let Ok(v) = value.extract::<u64>() {
        return Ok(Value::Number(Number::from(v)));
    }

    if let Ok(v) = value.extract::<f64>() {
        return Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| PyTypeError::new_err("cannot convert NaN or infinity to JSON"));
    }

    if let Ok(v) = value.extract::<String>() {
        return Ok(Value::String(v));
    }

    if let Ok(list) = value.cast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_any_to_json_value(&item)?);
        }
        return Ok(Value::Array(items));
    }

    if let Ok(tuple) = value.cast::<PyTuple>() {
        let mut items = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            items.push(py_any_to_json_value(&item)?);
        }
        return Ok(Value::Array(items));
    }

    if let Ok(dict) = value.cast::<PyDict>() {
        let mut items = Map::with_capacity(dict.len());
        for (key, item) in dict.iter() {
            let key = key
                .extract::<String>()
                .map_err(|_| PyTypeError::new_err("JSON object keys must be strings"))?;
            items.insert(key, py_any_to_json_value(&item)?);
        }
        return Ok(Value::Object(items));
    }

    if value.hasattr("model_dump")? {
        let dumped = value.call_method0("model_dump")?;
        return py_any_to_json_value(&dumped);
    }

    if value.hasattr("dict")? {
        let dumped = value.call_method0("dict")?;
        return py_any_to_json_value(&dumped);
    }

    if value.hasattr("__dataclass_fields__")? {
        let py = value.py();
        let dataclasses = py.import("dataclasses")?;
        let dumped = dataclasses.call_method1("asdict", (value,))?;
        return py_any_to_json_value(&dumped);
    }

    Err(PyTypeError::new_err(format!(
        "value of type '{}' is not JSON serializable by the AutoAgents binding",
        value.get_type().name()?
    )))
}
