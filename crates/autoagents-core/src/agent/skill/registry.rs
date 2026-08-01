use crate::agent::skill::{Skill, SkillSource, SkillSourceId, SkillSourceSnapshot};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

#[cfg(target_arch = "wasm32")]
use futures::lock::Mutex as RefreshMutex;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::Mutex as RefreshMutex;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SkillDiagnostic {
    source: SkillSourceId,
    message: String,
}

impl SkillDiagnostic {
    pub fn source(&self) -> &SkillSourceId {
        &self.source
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Clone, Debug, Default)]
pub struct SkillRefreshReport {
    generation: u64,
    discovered: usize,
    added: Vec<String>,
    updated: Vec<String>,
    removed: Vec<String>,
    diagnostics: Vec<SkillDiagnostic>,
}

impl SkillRefreshReport {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn discovered(&self) -> usize {
        self.discovered
    }

    pub fn added(&self) -> &[String] {
        &self.added
    }

    pub fn updated(&self) -> &[String] {
        &self.updated
    }

    pub fn removed(&self) -> &[String] {
        &self.removed
    }

    pub fn diagnostics(&self) -> &[SkillDiagnostic] {
        &self.diagnostics
    }

    pub fn changed(&self) -> bool {
        !(self.added.is_empty() && self.updated.is_empty() && self.removed.is_empty())
    }

    pub(crate) fn log_diagnostics(&self) {
        for diagnostic in &self.diagnostics {
            log::warn!(
                "skill source '{}' reported: {}",
                diagnostic.source.as_str(),
                diagnostic.message
            );
        }
    }

    fn unchanged(snapshot: &SkillSnapshot) -> Self {
        Self {
            generation: snapshot.generation(),
            discovered: snapshot.len(),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SkillSnapshot {
    generation: u64,
    skills: BTreeMap<String, Arc<Skill>>,
}

impl SkillSnapshot {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<Arc<Skill>> {
        self.skills.get(name).cloned()
    }

    pub(crate) fn get_ref(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name).map(Arc::as_ref)
    }

    pub fn skills(&self) -> impl Iterator<Item = &Arc<Skill>> {
        self.skills.values()
    }
}

#[derive(Debug)]
struct StoredSourceSnapshot {
    precedence: u16,
    snapshot: SkillSourceSnapshot,
}

#[derive(Debug, Default)]
struct SkillRegistryState {
    source_snapshots: BTreeMap<SkillSourceId, StoredSourceSnapshot>,
    merged: Arc<SkillSnapshot>,
    diagnostics: BTreeSet<(SkillSourceId, String)>,
}

impl SkillRegistryState {
    fn merged_skills(&self) -> BTreeMap<String, Arc<Skill>> {
        let mut sources = self.source_snapshots.iter().collect::<Vec<_>>();
        sources.sort_by(|(left_id, left), (right_id, right)| {
            left.precedence
                .cmp(&right.precedence)
                .then_with(|| left_id.cmp(right_id))
        });

        let mut skills = BTreeMap::new();
        for (_, source) in sources {
            for skill in &source.snapshot.skills {
                skills.insert(skill.metadata().name().to_string(), Arc::clone(skill));
            }
        }
        skills
    }

    fn replace_merged(&mut self, skills: BTreeMap<String, Arc<Skill>>) -> SkillRefreshReport {
        let previous = &self.merged.skills;
        let added = skills
            .keys()
            .filter(|name| !previous.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>();
        let removed = previous
            .keys()
            .filter(|name| !skills.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>();
        let updated = skills
            .iter()
            .filter(|(name, skill)| previous.get(*name).is_some_and(|old| old != *skill))
            .map(|(name, _)| name)
            .cloned()
            .collect::<Vec<_>>();
        let changed = !(added.is_empty() && updated.is_empty() && removed.is_empty());
        let generation = self.merged.generation + u64::from(changed);
        let discovered = skills.len();
        if changed {
            self.merged = Arc::new(SkillSnapshot { generation, skills });
        }

        SkillRefreshReport {
            generation,
            discovered,
            added,
            updated,
            removed,
            diagnostics: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct SkillRegistry {
    sources: Vec<Arc<dyn SkillSource>>,
    state: RwLock<SkillRegistryState>,
    refresh: RefreshMutex<()>,
    refresh_epoch: AtomicU64,
}

impl SkillRegistry {
    pub fn new(sources: Vec<Arc<dyn SkillSource>>) -> Self {
        Self {
            sources,
            state: RwLock::new(SkillRegistryState::default()),
            refresh: RefreshMutex::new(()),
            refresh_epoch: AtomicU64::new(0),
        }
    }

    pub fn from_directory(
        path: impl Into<std::path::PathBuf>,
        refresh_strategy: crate::agent::skill::SkillRefreshStrategy,
    ) -> Self {
        Self::new(vec![Arc::new(
            crate::agent::skill::DirectorySkillSource::new(path, refresh_strategy),
        )])
    }

    pub async fn refresh(&self) -> SkillRefreshReport {
        let observed_epoch = self.refresh_epoch.load(Ordering::Acquire);
        let _refresh_guard = self.refresh.lock().await;
        if self.refresh_epoch.load(Ordering::Acquire) != observed_epoch {
            return SkillRefreshReport::unchanged(&self.snapshot());
        }
        self.refresh_sources().await
    }

    pub async fn refresh_now(&self) -> SkillRefreshReport {
        let _refresh_guard = self.refresh.lock().await;
        for source in &self.sources {
            source.invalidate();
        }
        self.refresh_sources().await
    }

    async fn refresh_sources(&self) -> SkillRefreshReport {
        let discoveries = futures::future::join_all(self.sources.iter().map(|source| async move {
            (
                source.id().clone(),
                source.precedence(),
                source.discover().await,
            )
        }))
        .await;

        let mut state = self
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut diagnostics = Vec::new();
        for (source_id, precedence, discovery) in discoveries {
            match discovery {
                Ok(snapshot) => {
                    diagnostics.extend(snapshot.diagnostics.iter().cloned().map(|message| {
                        SkillDiagnostic {
                            source: source_id.clone(),
                            message,
                        }
                    }));
                    state.source_snapshots.insert(
                        source_id,
                        StoredSourceSnapshot {
                            precedence,
                            snapshot,
                        },
                    );
                }
                Err(error) => diagnostics.push(SkillDiagnostic {
                    source: source_id,
                    message: error.to_string(),
                }),
            }
        }

        let current_diagnostics = diagnostics
            .into_iter()
            .map(|diagnostic| {
                (
                    (diagnostic.source.clone(), diagnostic.message.clone()),
                    diagnostic,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let new_diagnostics = current_diagnostics
            .iter()
            .filter(|(key, _)| !state.diagnostics.contains(*key))
            .map(|(_, diagnostic)| diagnostic.clone())
            .collect::<Vec<_>>();
        state.diagnostics = current_diagnostics.into_keys().collect();

        let skills = state.merged_skills();
        let mut report = state.replace_merged(skills);
        report.diagnostics = new_diagnostics;
        self.refresh_epoch.fetch_add(1, Ordering::Release);
        report
    }

    pub fn snapshot(&self) -> Arc<SkillSnapshot> {
        self.state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .merged
            .clone()
    }

    pub fn get(&self, name: &str) -> Option<Arc<Skill>> {
        self.snapshot().get(name)
    }
}
