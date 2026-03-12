use autoagents_llamacpp_py_core::register_module;
use pyo3::prelude::*;

#[pymodule]
fn _autoagents_llamacpp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_module(m)
}
