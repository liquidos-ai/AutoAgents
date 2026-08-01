mod configuration;
mod directory;
mod error;
mod lifecycle;
mod model;
mod policy;
mod prompt;
mod registry;
#[cfg(not(target_arch = "wasm32"))]
mod resource;
pub(crate) mod runtime;
mod session;
mod source;
mod tool_selector;

pub use configuration::SkillConfiguration;
pub use directory::{DirectorySkillSource, SkillDiscoveryLimits, SkillRefreshStrategy};
pub use error::SkillError;
pub use lifecycle::{SilentSkillLifecycle, SkillLifecycle, SkillLifecycleDecision};
pub use model::{Skill, SkillMetadata, SkillRevision};
pub use policy::{SkillActivationRequest, SkillPolicy, SkillResourceRequest, TrustedSkillPolicy};
pub use registry::{SkillDiagnostic, SkillRefreshReport, SkillRegistry, SkillSnapshot};
pub(crate) use runtime::SkillRuntimeIdentity;
pub use runtime::{
    SkillDeactivationReason, SkillEvent, SkillEventKind, SkillOperation, SkillRuntime,
};
pub use session::{SkillSession, SkillSessionReset};
pub use source::{SkillSource, SkillSourceId, SkillSourceSnapshot};
pub use tool_selector::SkillToolSelector;

#[cfg(test)]
mod tests;
