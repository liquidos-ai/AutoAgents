use crate::agent::skill::{SkillRevision, SkillSnapshot};
use autoagents_protocol::SkillSessionId;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::RwLock;
use uuid::Uuid;

#[derive(Debug)]
struct SkillSessionState {
    id: SkillSessionId,
    active: BTreeMap<String, SkillRevision>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SkillSessionReconciliation {
    updated: Vec<String>,
    removed: Vec<String>,
}

impl SkillSessionReconciliation {
    pub(crate) fn updated(&self) -> &[String] {
        &self.updated
    }

    pub(crate) fn removed(&self) -> &[String] {
        &self.removed
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SkillSessionReset {
    previous_id: SkillSessionId,
    current_id: SkillSessionId,
    deactivated: Vec<String>,
}

impl SkillSessionReset {
    pub fn previous_id(&self) -> SkillSessionId {
        self.previous_id
    }

    pub fn current_id(&self) -> SkillSessionId {
        self.current_id
    }

    pub fn deactivated(&self) -> &[String] {
        &self.deactivated
    }
}

#[derive(Debug)]
pub struct SkillSession {
    state: RwLock<SkillSessionState>,
}

impl Default for SkillSession {
    fn default() -> Self {
        Self {
            state: RwLock::new(SkillSessionState {
                id: Uuid::new_v4(),
                active: BTreeMap::new(),
            }),
        }
    }
}

impl SkillSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn id(&self) -> SkillSessionId {
        self.state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .id
    }

    pub fn active(&self) -> BTreeSet<String> {
        self.state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .active
            .keys()
            .cloned()
            .collect()
    }

    pub fn is_active(&self, name: &str) -> bool {
        self.state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .active
            .contains_key(name)
    }

    pub fn reset(&self) -> SkillSessionReset {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous_id = state.id;
        let deactivated = state.active.keys().cloned().collect();
        state.id = Uuid::new_v4();
        state.active.clear();
        SkillSessionReset {
            previous_id,
            current_id: state.id,
            deactivated,
        }
    }

    pub(crate) fn activate(&self, name: String, revision: SkillRevision) -> bool {
        self.state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .active
            .insert(name, revision)
            .is_none_or(|active_revision| active_revision != revision)
    }

    pub(crate) fn deactivate(&self, name: &str) -> bool {
        self.state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .active
            .remove(name)
            .is_some()
    }

    pub(crate) fn reconcile(&self, available: &SkillSnapshot) -> SkillSessionReconciliation {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let removed = state
            .active
            .keys()
            .filter(|name| available.get_ref(name).is_none())
            .cloned()
            .collect::<Vec<_>>();
        let updated = state
            .active
            .iter()
            .filter(|(name, revision)| {
                available
                    .get_ref(name)
                    .is_some_and(|skill| skill.revision() != **revision)
            })
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        state.active.retain(|name, revision| {
            available
                .get_ref(name)
                .is_some_and(|skill| skill.revision() == *revision)
        });
        SkillSessionReconciliation { updated, removed }
    }
}
