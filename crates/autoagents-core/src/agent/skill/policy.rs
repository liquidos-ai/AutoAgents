use crate::agent::skill::{Skill, SkillError};
use async_trait::async_trait;
use autoagents_protocol::{ActorID, SkillSessionId, SubmissionId};
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct SkillActivationRequest {
    actor_id: ActorID,
    submission_id: SubmissionId,
    session_id: SkillSessionId,
    skill: Arc<Skill>,
}

impl SkillActivationRequest {
    pub(crate) fn new(
        actor_id: ActorID,
        submission_id: SubmissionId,
        session_id: SkillSessionId,
        skill: Arc<Skill>,
    ) -> Self {
        Self {
            actor_id,
            submission_id,
            session_id,
            skill,
        }
    }

    pub fn actor_id(&self) -> ActorID {
        self.actor_id
    }

    pub fn submission_id(&self) -> SubmissionId {
        self.submission_id
    }

    pub fn session_id(&self) -> SkillSessionId {
        self.session_id
    }

    pub fn skill(&self) -> &Skill {
        &self.skill
    }
}

#[derive(Clone, Debug)]
pub struct SkillResourceRequest {
    actor_id: ActorID,
    submission_id: SubmissionId,
    session_id: SkillSessionId,
    skill: Arc<Skill>,
    path: PathBuf,
}

impl SkillResourceRequest {
    pub(crate) fn new(
        actor_id: ActorID,
        submission_id: SubmissionId,
        session_id: SkillSessionId,
        skill: Arc<Skill>,
        path: PathBuf,
    ) -> Self {
        Self {
            actor_id,
            submission_id,
            session_id,
            skill,
            path,
        }
    }

    pub fn actor_id(&self) -> ActorID {
        self.actor_id
    }

    pub fn submission_id(&self) -> SubmissionId {
        self.submission_id
    }

    pub fn session_id(&self) -> SkillSessionId {
        self.session_id
    }

    pub fn skill(&self) -> &Skill {
        &self.skill
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[async_trait]
pub trait SkillPolicy: Debug + Send + Sync {
    async fn authorize_activation(
        &self,
        request: &SkillActivationRequest,
    ) -> Result<(), SkillError>;

    async fn authorize_resource_read(
        &self,
        request: &SkillResourceRequest,
    ) -> Result<(), SkillError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TrustedSkillPolicy;

#[async_trait]
impl SkillPolicy for TrustedSkillPolicy {
    async fn authorize_activation(
        &self,
        _request: &SkillActivationRequest,
    ) -> Result<(), SkillError> {
        Ok(())
    }

    async fn authorize_resource_read(
        &self,
        _request: &SkillResourceRequest,
    ) -> Result<(), SkillError> {
        Ok(())
    }
}
