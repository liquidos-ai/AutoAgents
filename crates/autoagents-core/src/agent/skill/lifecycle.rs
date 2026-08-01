use async_trait::async_trait;
use autoagents_protocol::SkillEvent;
use std::fmt::Debug;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SkillLifecycleDecision {
    Continue,
    Abort,
}

#[async_trait]
pub trait SkillLifecycle: Debug + Send + Sync {
    async fn on_catalog_changed(&self, _event: &SkillEvent) {}

    async fn on_activation_requested(&self, _event: &SkillEvent) -> SkillLifecycleDecision {
        SkillLifecycleDecision::Continue
    }

    async fn on_activated(&self, _event: &SkillEvent) {}

    async fn on_deactivated(&self, _event: &SkillEvent) {}

    async fn on_resource_access_requested(&self, _event: &SkillEvent) -> SkillLifecycleDecision {
        SkillLifecycleDecision::Continue
    }

    async fn on_resource_accessed(&self, _event: &SkillEvent) {}

    async fn on_operation_failed(&self, _event: &SkillEvent) {}
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SilentSkillLifecycle;

#[async_trait]
impl SkillLifecycle for SilentSkillLifecycle {}
