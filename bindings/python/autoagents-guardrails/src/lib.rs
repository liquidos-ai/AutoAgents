use autoagents_guardrails::guards::{
    PromptInjectionGuard as CorePromptInjectionGuard,
    RegexPiiRedactionGuard as CoreRegexPiiRedactionGuard, ToxicityGuard as CoreToxicityGuard,
};
use autoagents_guardrails::{EnforcementPolicy, Guardrails as CoreGuardrails};
use autoagents_py::llm::builder::{canonicalize_llm_provider, extract_llm_provider};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass(name = "PromptInjectionGuard", skip_from_py_object)]
#[derive(Clone, Default)]
struct PyPromptInjectionGuard {
    inner: Arc<CorePromptInjectionGuard>,
}

#[pymethods]
impl PyPromptInjectionGuard {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(CorePromptInjectionGuard::default()),
        }
    }

    fn __repr__(&self) -> &str {
        "PromptInjectionGuard()"
    }
}

#[pyclass(name = "RegexPiiRedactionGuard", skip_from_py_object)]
#[derive(Clone, Default)]
struct PyRegexPiiRedactionGuard {
    inner: Arc<CoreRegexPiiRedactionGuard>,
}

#[pymethods]
impl PyRegexPiiRedactionGuard {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(CoreRegexPiiRedactionGuard::default()),
        }
    }

    fn __repr__(&self) -> &str {
        "RegexPiiRedactionGuard()"
    }
}

#[pyclass(name = "ToxicityGuard", skip_from_py_object)]
#[derive(Clone, Default)]
struct PyToxicityGuard {
    inner: Arc<CoreToxicityGuard>,
}

#[pymethods]
impl PyToxicityGuard {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(CoreToxicityGuard::default()),
        }
    }

    fn __repr__(&self) -> &str {
        "ToxicityGuard()"
    }
}

#[pyclass(name = "Guardrails", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyGuardrails {
    inner: CoreGuardrails,
}

#[pymethods]
impl PyGuardrails {
    fn __repr__(&self) -> &str {
        "Guardrails(<configured>)"
    }

    fn build(&self, provider: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let provider = extract_llm_provider(provider)
            .map_err(|_| PyRuntimeError::new_err("build() expects an AutoAgents LLMProvider"))?;

        Python::attach(|py| canonicalize_llm_provider(py, self.inner.wrap(provider)))
    }

    fn wrap(&self, provider: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.build(provider)
    }
}

#[pyclass(name = "GuardrailsBuilder")]
#[derive(Default)]
struct PyGuardrailsBuilder {
    input_guards: Vec<InputGuardKind>,
    output_guards: Vec<OutputGuardKind>,
    policy: EnforcementPolicy,
}

#[derive(Clone)]
enum InputGuardKind {
    PromptInjection(Arc<CorePromptInjectionGuard>),
    RegexPiiRedaction(Arc<CoreRegexPiiRedactionGuard>),
}

#[derive(Clone)]
enum OutputGuardKind {
    Toxicity(Arc<CoreToxicityGuard>),
}

#[pymethods]
impl PyGuardrailsBuilder {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn input_guard<'a>(
        mut slf: PyRefMut<'a, Self>,
        guard: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.input_guards.push(extract_input_guard(guard)?);
        Ok(slf)
    }

    fn output_guard<'a>(
        mut slf: PyRefMut<'a, Self>,
        guard: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.output_guards.push(extract_output_guard(guard)?);
        Ok(slf)
    }

    fn enforcement_policy<'a>(
        mut slf: PyRefMut<'a, Self>,
        policy: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.policy = parse_policy(policy)?;
        Ok(slf)
    }

    fn build(&self) -> PyGuardrails {
        let mut builder = CoreGuardrails::builder().enforcement_policy(self.policy);

        for guard in &self.input_guards {
            builder = match guard {
                InputGuardKind::PromptInjection(guard) => builder.input_guard_arc(guard.clone()),
                InputGuardKind::RegexPiiRedaction(guard) => builder.input_guard_arc(guard.clone()),
            };
        }

        for guard in &self.output_guards {
            builder = match guard {
                OutputGuardKind::Toxicity(guard) => builder.output_guard_arc(guard.clone()),
            };
        }

        PyGuardrails {
            inner: builder.build(),
        }
    }
}

fn parse_policy(policy: &Bound<'_, PyAny>) -> PyResult<EnforcementPolicy> {
    let value = if let Ok(text) = policy.extract::<String>() {
        text
    } else if let Ok(text) = policy
        .getattr("value")
        .and_then(|value| value.extract::<String>())
    {
        text
    } else {
        return Err(PyRuntimeError::new_err(
            "enforcement_policy() expects 'block', 'sanitize', or 'audit'",
        ));
    };

    parse_policy_value(&value)
}

fn parse_policy_value(value: &str) -> PyResult<EnforcementPolicy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "block" => Ok(EnforcementPolicy::Block),
        "sanitize" => Ok(EnforcementPolicy::Sanitize),
        "audit" => Ok(EnforcementPolicy::Audit),
        other => Err(PyRuntimeError::new_err(format!(
            "invalid enforcement policy '{other}', expected one of: block, sanitize, audit"
        ))),
    }
}

fn extract_input_guard(guard: &Bound<'_, PyAny>) -> PyResult<InputGuardKind> {
    if let Ok(guard) = guard.extract::<PyRef<'_, PyPromptInjectionGuard>>() {
        return Ok(InputGuardKind::PromptInjection(guard.inner.clone()));
    }
    if let Ok(guard) = guard.extract::<PyRef<'_, PyRegexPiiRedactionGuard>>() {
        return Ok(InputGuardKind::RegexPiiRedaction(guard.inner.clone()));
    }
    Err(PyRuntimeError::new_err(
        "input_guard() expects PromptInjectionGuard or RegexPiiRedactionGuard",
    ))
}

fn extract_output_guard(guard: &Bound<'_, PyAny>) -> PyResult<OutputGuardKind> {
    if let Ok(guard) = guard.extract::<PyRef<'_, PyToxicityGuard>>() {
        return Ok(OutputGuardKind::Toxicity(guard.inner.clone()));
    }
    Err(PyRuntimeError::new_err(
        "output_guard() expects ToxicityGuard",
    ))
}

#[pymodule]
fn _autoagents_guardrails(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPromptInjectionGuard>()?;
    m.add_class::<PyRegexPiiRedactionGuard>()?;
    m.add_class::<PyToxicityGuard>()?;
    m.add_class::<PyGuardrails>()?;
    m.add_class::<PyGuardrailsBuilder>()?;
    Ok(())
}
