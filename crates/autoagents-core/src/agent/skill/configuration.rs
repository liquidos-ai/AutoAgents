use crate::agent::skill::{SkillPolicy, SkillRegistry, SkillSession};
use std::fmt;
use std::sync::Arc;

#[derive(Clone)]
pub struct SkillConfiguration {
    registry: Arc<SkillRegistry>,
    session: Arc<SkillSession>,
    policy: Arc<dyn SkillPolicy>,
    max_resource_bytes: usize,
}

impl fmt::Debug for SkillConfiguration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SkillConfiguration")
            .field("registry", &self.registry)
            .field("session", &self.session)
            .field("policy", &self.policy)
            .field("max_resource_bytes", &self.max_resource_bytes)
            .finish()
    }
}

impl SkillConfiguration {
    pub fn new(
        registry: Arc<SkillRegistry>,
        session: Arc<SkillSession>,
        policy: Arc<dyn SkillPolicy>,
    ) -> Self {
        Self {
            registry,
            session,
            policy,
            max_resource_bytes: 1024 * 1024,
        }
    }

    pub fn with_max_resource_bytes(mut self, max_resource_bytes: usize) -> Self {
        self.max_resource_bytes = max_resource_bytes;
        self
    }

    pub fn registry(&self) -> Arc<SkillRegistry> {
        Arc::clone(&self.registry)
    }

    pub(crate) fn registry_ref(&self) -> &SkillRegistry {
        &self.registry
    }

    pub fn session(&self) -> Arc<SkillSession> {
        Arc::clone(&self.session)
    }

    pub(crate) fn session_ref(&self) -> &SkillSession {
        &self.session
    }

    pub fn policy(&self) -> Arc<dyn SkillPolicy> {
        Arc::clone(&self.policy)
    }

    pub(crate) fn policy_ref(&self) -> &dyn SkillPolicy {
        self.policy.as_ref()
    }

    pub fn max_resource_bytes(&self) -> usize {
        self.max_resource_bytes
    }
}
