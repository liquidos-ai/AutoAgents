use crate::agent::skill::{Skill, SkillError};
use async_trait::async_trait;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SkillSourceId(Arc<str>);

impl SkillSourceId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(Arc::from(value.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Default)]
pub struct SkillSourceSnapshot {
    pub(crate) skills: Vec<Arc<Skill>>,
    pub(crate) diagnostics: Vec<String>,
}

impl SkillSourceSnapshot {
    pub fn new(skills: Vec<Arc<Skill>>, diagnostics: Vec<String>) -> Self {
        Self {
            skills,
            diagnostics,
        }
    }

    pub fn skills(&self) -> &[Arc<Skill>] {
        &self.skills
    }

    pub fn diagnostics(&self) -> &[String] {
        &self.diagnostics
    }
}

#[async_trait]
pub trait SkillSource: Debug + Send + Sync {
    fn id(&self) -> &SkillSourceId;

    fn precedence(&self) -> u16;

    fn invalidate(&self) {}

    async fn discover(&self) -> Result<SkillSourceSnapshot, SkillError>;
}
