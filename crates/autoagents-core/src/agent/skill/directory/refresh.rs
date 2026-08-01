use super::SkillRefreshStrategy;
use super::cache::DirectorySkillCache;
use crate::agent::skill::SkillError;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

#[cfg(feature = "skill-watch")]
use notify::{Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
#[cfg(feature = "skill-watch")]
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DirectoryDiscovery {
    Skip,
    Metadata,
    VerifyContents,
}

pub(super) struct DirectoryRefreshState {
    forced: AtomicBool,
    last_discovery: Mutex<Option<Instant>>,
    #[cfg(feature = "skill-watch")]
    dirty: Arc<AtomicBool>,
    #[cfg(feature = "skill-watch")]
    last_change: Arc<Mutex<Option<Instant>>>,
    #[cfg(feature = "skill-watch")]
    watcher: Mutex<Option<RecommendedWatcher>>,
}

impl std::fmt::Debug for DirectoryRefreshState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("DirectoryRefreshState");
        debug.field("forced", &self.forced.load(Ordering::Acquire));
        #[cfg(feature = "skill-watch")]
        debug
            .field("dirty", &self.dirty.load(Ordering::Acquire))
            .field(
                "watcher",
                &self
                    .watcher
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .as_ref()
                    .map(|_| "active"),
            );
        debug.finish_non_exhaustive()
    }
}

impl DirectoryRefreshState {
    pub(super) fn new() -> Self {
        Self {
            forced: AtomicBool::new(true),
            last_discovery: Mutex::new(None),
            #[cfg(feature = "skill-watch")]
            dirty: Arc::new(AtomicBool::new(true)),
            #[cfg(feature = "skill-watch")]
            last_change: Arc::new(Mutex::new(None)),
            #[cfg(feature = "skill-watch")]
            watcher: Mutex::new(None),
        }
    }

    pub(super) fn invalidate(&self) {
        self.forced.store(true, Ordering::Release);
        #[cfg(feature = "skill-watch")]
        self.dirty.store(true, Ordering::Release);
    }

    pub(super) fn discovery(
        &self,
        root: &Path,
        containment_root: Option<&Path>,
        cache: &RwLock<DirectorySkillCache>,
        strategy: SkillRefreshStrategy,
    ) -> Result<DirectoryDiscovery, SkillError> {
        #[cfg(not(feature = "skill-watch"))]
        let _ = (root, containment_root, cache);

        match strategy {
            SkillRefreshStrategy::Manual => Ok(if self.take_forced() {
                DirectoryDiscovery::VerifyContents
            } else {
                DirectoryDiscovery::Skip
            }),
            SkillRefreshStrategy::Poll { interval } => Ok(if self.take_forced() {
                DirectoryDiscovery::VerifyContents
            } else if self.fallback_elapsed(interval) {
                DirectoryDiscovery::Metadata
            } else {
                DirectoryDiscovery::Skip
            }),
            SkillRefreshStrategy::Watch {
                debounce,
                fallback_interval,
            } => {
                #[cfg(feature = "skill-watch")]
                {
                    self.ensure_watcher(root, containment_root)?;
                    if self.take_forced() {
                        self.dirty.store(false, Ordering::Release);
                        return Ok(DirectoryDiscovery::VerifyContents);
                    }
                    Ok(self.watched_discovery(root, cache, debounce, fallback_interval))
                }
                #[cfg(not(feature = "skill-watch"))]
                {
                    let _ = (debounce, fallback_interval);
                    Err(SkillError::WatchUnavailable)
                }
            }
        }
    }

    pub(super) fn record_discovery(&self) {
        *self
            .last_discovery
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Instant::now());
    }

    fn take_forced(&self) -> bool {
        self.forced.swap(false, Ordering::AcqRel)
    }

    fn fallback_elapsed(&self, interval: Duration) -> bool {
        self.last_discovery
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_none_or(|last_discovery| last_discovery.elapsed() >= interval)
    }

    #[cfg(feature = "skill-watch")]
    fn watched_discovery(
        &self,
        root: &Path,
        cache: &RwLock<DirectorySkillCache>,
        debounce: Duration,
        fallback_interval: Duration,
    ) -> DirectoryDiscovery {
        let documents_changed = cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .documents_changed(root);
        let watcher_settled = self
            .last_change
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_none_or(|change| change.elapsed() >= debounce);
        let observed_dirty = self.dirty.swap(false, Ordering::AcqRel);
        if observed_dirty && (documents_changed || watcher_settled) {
            return DirectoryDiscovery::VerifyContents;
        }
        if observed_dirty {
            self.dirty.store(true, Ordering::Release);
        }
        if documents_changed {
            return DirectoryDiscovery::VerifyContents;
        }
        if self.fallback_elapsed(fallback_interval) {
            DirectoryDiscovery::Metadata
        } else {
            DirectoryDiscovery::Skip
        }
    }

    #[cfg(feature = "skill-watch")]
    fn ensure_watcher(
        &self,
        root: &Path,
        containment_root: Option<&Path>,
    ) -> Result<(), SkillError> {
        let mut watcher = self
            .watcher
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if watcher.is_some() {
            return Ok(());
        }
        if let Some(containment_root) = containment_root {
            let canonical_root = match std::fs::canonicalize(root) {
                Ok(canonical_root) => canonical_root,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(error) => {
                    return Err(SkillError::source_unavailable(
                        root.display().to_string(),
                        format!("cannot resolve watched skill root: {error}"),
                    ));
                }
            };
            let canonical_containment =
                std::fs::canonicalize(containment_root).map_err(|error| {
                    SkillError::source_unavailable(
                        containment_root.display().to_string(),
                        format!("cannot resolve watched skill containment root: {error}"),
                    )
                })?;
            if !canonical_root.starts_with(&canonical_containment) {
                return Err(SkillError::source_unavailable(
                    root.display().to_string(),
                    format!(
                        "resolved watched source '{}' escapes containment root '{}'",
                        canonical_root.display(),
                        canonical_containment.display()
                    ),
                ));
            }
        }
        let dirty = Arc::clone(&self.dirty);
        let last_change = Arc::clone(&self.last_change);
        let created = RecommendedWatcher::new(
            move |result: notify::Result<notify::Event>| match result {
                Ok(event) if !matches!(event.kind, EventKind::Access(_)) => {
                    dirty.store(true, Ordering::Release);
                    *last_change
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Instant::now());
                }
                Err(_) => {
                    dirty.store(true, Ordering::Release);
                    *last_change
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Instant::now());
                }
                Ok(_) => {}
            },
            Config::default(),
        )
        .and_then(|mut created| {
            created.watch(root, RecursiveMode::Recursive)?;
            Ok(created)
        });
        if let Ok(created) = created {
            *watcher = Some(created);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn debug_output_reports_refresh_state() {
        let state = DirectoryRefreshState::new();
        let debug = format!("{state:?}");

        assert!(debug.contains("DirectoryRefreshState"));
        assert!(debug.contains("forced"));
        #[cfg(feature = "skill-watch")]
        {
            assert!(debug.contains("dirty"));
            assert!(debug.contains("watcher"));
        }
    }

    #[test]
    fn poll_skips_until_its_fallback_interval_elapses() {
        let state = DirectoryRefreshState::new();
        let cache = RwLock::new(DirectorySkillCache::default());
        let root = Path::new("unused");
        let strategy = SkillRefreshStrategy::Poll {
            interval: Duration::from_secs(60),
        };

        assert_eq!(
            state
                .discovery(root, None, &cache, strategy)
                .expect("forced discovery"),
            DirectoryDiscovery::VerifyContents
        );
        state.record_discovery();
        assert_eq!(
            state
                .discovery(root, None, &cache, strategy)
                .expect("poll discovery"),
            DirectoryDiscovery::Skip
        );
    }

    #[cfg(feature = "skill-watch")]
    #[test]
    fn watched_discovery_covers_dirty_document_and_fallback_transitions() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let root = temporary.path().join("skills");
        fs::create_dir_all(&root).expect("skills root");
        let cache = RwLock::new(DirectorySkillCache::default());
        cache
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .capture_documents(&root);
        let state = DirectoryRefreshState::new();

        assert_eq!(
            state.watched_discovery(&root, &cache, Duration::ZERO, Duration::from_secs(60),),
            DirectoryDiscovery::VerifyContents
        );

        state.record_discovery();
        assert_eq!(
            state.watched_discovery(&root, &cache, Duration::ZERO, Duration::from_secs(60),),
            DirectoryDiscovery::Skip
        );

        state.dirty.store(true, Ordering::Release);
        *state
            .last_change
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Instant::now());
        assert_eq!(
            state.watched_discovery(
                &root,
                &cache,
                Duration::from_secs(60),
                Duration::from_secs(60),
            ),
            DirectoryDiscovery::Skip
        );
        assert!(state.dirty.load(Ordering::Acquire));

        state.dirty.store(false, Ordering::Release);
        let skill_directory = root.join("new-skill");
        fs::create_dir_all(&skill_directory).expect("skill directory");
        fs::write(skill_directory.join("SKILL.md"), "new document").expect("skill document");
        assert_eq!(
            state.watched_discovery(&root, &cache, Duration::ZERO, Duration::from_secs(60),),
            DirectoryDiscovery::VerifyContents
        );

        cache
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .capture_documents(&root);
        state.dirty.store(false, Ordering::Release);
        *state
            .last_discovery
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        assert_eq!(
            state.watched_discovery(&root, &cache, Duration::ZERO, Duration::ZERO,),
            DirectoryDiscovery::Metadata
        );
    }

    #[cfg(feature = "skill-watch")]
    #[test]
    fn watcher_setup_handles_missing_roots_and_containment() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let root = temporary.path().join("skills");
        let state = DirectoryRefreshState::new();

        state
            .ensure_watcher(&root, Some(temporary.path()))
            .expect("a missing watched root is allowed until it appears");

        fs::create_dir_all(&root).expect("skills root");
        let missing_containment = temporary.path().join("missing-containment");
        let error = state
            .ensure_watcher(&root, Some(&missing_containment))
            .expect_err("missing containment should fail");
        assert!(
            error
                .to_string()
                .contains("cannot resolve watched skill containment root")
        );

        state
            .ensure_watcher(&root, Some(temporary.path()))
            .expect("watcher should be created");
        assert!(
            format!("{state:?}").contains("active"),
            "debug output should report an active watcher"
        );
    }
}
