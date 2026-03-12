use pyo3::prelude::*;

/// Python handle for a pub/sub topic name.
///
/// Used when wiring agents to a multi-agent `Runtime`.
#[pyclass(name = "Topic", skip_from_py_object)]
#[derive(Clone)]
pub struct PyTopic {
    pub name: String,
}

#[pymethods]
impl PyTopic {
    #[new]
    pub fn new(name: String) -> Self {
        Self { name }
    }

    fn __repr__(&self) -> String {
        format!("Topic('{}')", self.name)
    }

    #[getter]
    pub fn name(&self) -> &str {
        &self.name
    }
}
