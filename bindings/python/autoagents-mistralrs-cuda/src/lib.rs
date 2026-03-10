use autoagents_mistral_rs_py_core::register_module;
use pyo3::prelude::*;

#[pymodule]
fn _autoagents_mistral_rs_cuda(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_module(m)
}
