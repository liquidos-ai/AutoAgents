use crate::agent::skill::{SkillError, SkillSource, SkillSourceId, SkillSourceSnapshot};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
mod cache;
#[cfg(not(target_arch = "wasm32"))]
mod refresh;
#[cfg(not(target_arch = "wasm32"))]
mod scanner;

#[cfg(not(target_arch = "wasm32"))]
use cache::DirectorySkillCache;
#[cfg(not(target_arch = "wasm32"))]
use refresh::{DirectoryDiscovery, DirectoryRefreshState};
#[cfg(not(target_arch = "wasm32"))]
use scanner::DirectorySkillScanner;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, RwLock};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SkillRefreshStrategy {
    Manual,
    Poll {
        interval: Duration,
    },
    Watch {
        debounce: Duration,
        fallback_interval: Duration,
    },
}

#[derive(Clone, Debug)]
pub struct SkillDiscoveryLimits {
    pub max_skills: usize,
    pub max_skill_file_bytes: usize,
    pub max_total_skill_file_bytes: usize,
    pub max_resources_per_skill: usize,
    pub max_resource_file_bytes: usize,
    pub max_total_resource_bytes: usize,
    pub max_resource_depth: usize,
}

impl Default for SkillDiscoveryLimits {
    fn default() -> Self {
        Self {
            max_skills: 256,
            max_skill_file_bytes: 256 * 1024,
            max_total_skill_file_bytes: 4 * 1024 * 1024,
            max_resources_per_skill: 512,
            max_resource_file_bytes: 1024 * 1024,
            max_total_resource_bytes: 16 * 1024 * 1024,
            max_resource_depth: 6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DirectorySkillSource {
    id: SkillSourceId,
    root: PathBuf,
    refresh_strategy: SkillRefreshStrategy,
    containment_root: Option<PathBuf>,
    precedence: u16,
    limits: SkillDiscoveryLimits,
    #[cfg(not(target_arch = "wasm32"))]
    cache: Arc<RwLock<DirectorySkillCache>>,
    #[cfg(not(target_arch = "wasm32"))]
    refresh: Arc<DirectoryRefreshState>,
}

impl DirectorySkillSource {
    pub fn new(root: impl Into<PathBuf>, refresh_strategy: SkillRefreshStrategy) -> Self {
        let root = root.into();
        Self {
            id: SkillSourceId::new(format!("directory:{}", root.display())),
            root,
            refresh_strategy,
            containment_root: None,
            precedence: 0,
            limits: SkillDiscoveryLimits::default(),
            #[cfg(not(target_arch = "wasm32"))]
            cache: Arc::new(RwLock::new(DirectorySkillCache::default())),
            #[cfg(not(target_arch = "wasm32"))]
            refresh: Arc::new(DirectoryRefreshState::new()),
        }
    }

    pub fn with_precedence(mut self, precedence: u16) -> Self {
        self.precedence = precedence;
        self
    }

    pub fn with_containment_root(mut self, containment_root: impl Into<PathBuf>) -> Self {
        self.containment_root = Some(containment_root.into());
        self
    }

    pub fn with_limits(mut self, limits: SkillDiscoveryLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn refresh_strategy(&self) -> SkillRefreshStrategy {
        self.refresh_strategy
    }
}

#[async_trait]
impl SkillSource for DirectorySkillSource {
    fn id(&self) -> &SkillSourceId {
        &self.id
    }

    fn precedence(&self) -> u16 {
        self.precedence
    }

    fn invalidate(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        self.refresh.invalidate();
    }

    async fn discover(&self) -> Result<SkillSourceSnapshot, SkillError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let discovery = self.refresh.discovery(
                &self.root,
                self.containment_root.as_deref(),
                &self.cache,
                self.refresh_strategy,
            )?;
            if discovery == DirectoryDiscovery::Skip {
                return Ok(self
                    .cache
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .snapshot());
            }
            let scanner = DirectorySkillScanner::new(
                self.root.clone(),
                self.containment_root.clone(),
                self.limits.clone(),
                Arc::clone(&self.cache),
                discovery == DirectoryDiscovery::VerifyContents,
            );
            let result = tokio::task::spawn_blocking(move || scanner.scan())
                .await
                .map_err(|error| SkillError::SourceWorker(error.to_string()))?;
            if result.is_ok() {
                self.refresh.record_discovery();
            } else {
                self.refresh.invalidate();
            }
            result
        }

        #[cfg(target_arch = "wasm32")]
        {
            Err(SkillError::source_unavailable(
                self.id.as_str(),
                "filesystem skill discovery is unavailable on wasm32",
            ))
        }
    }
}
