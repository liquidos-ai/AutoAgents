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

#[cfg(test)]
mod tests {
    use super::PyTopic;

    #[test]
    fn topic_exposes_name_and_repr() {
        let topic = PyTopic::new("planner".to_string());
        assert_eq!(topic.name(), "planner");
        assert_eq!(topic.__repr__(), "Topic('planner')");
    }
}
