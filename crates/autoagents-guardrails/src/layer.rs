use std::sync::Arc;

use autoagents_llm::{LLMProvider, pipeline::LLMLayer};

use crate::{engine::GuardrailsEngine, provider::GuardedProvider};

/// `LLMLayer` adapter for `Guardrails`.
#[derive(Clone)]
pub struct GuardrailsLayer {
    engine: Arc<GuardrailsEngine>,
}

impl GuardrailsLayer {
    pub(crate) fn new(engine: Arc<GuardrailsEngine>) -> Self {
        Self { engine }
    }
}

impl LLMLayer for GuardrailsLayer {
    fn build(self: Box<Self>, next: Arc<dyn LLMProvider>) -> Arc<dyn LLMProvider> {
        Arc::new(GuardedProvider::new(next, self.engine.clone()))
    }
}
