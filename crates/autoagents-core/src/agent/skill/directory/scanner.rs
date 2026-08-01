use super::SkillDiscoveryLimits;
use super::cache::{
    CachedDirectorySkill, DirectorySkillCache, SkillDirectoryFingerprint, SkillFileFingerprint,
};
use crate::agent::skill::resource::{SkillResourceBoundary, SkillResourceDirectory};
use crate::agent::skill::{Skill, SkillError, SkillSourceSnapshot};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use walkdir::WalkDir;

#[derive(Debug)]
pub(super) struct DirectorySkillScanner {
    root: PathBuf,
    containment_root: Option<PathBuf>,
    limits: SkillDiscoveryLimits,
    cache: Arc<RwLock<DirectorySkillCache>>,
    verify_contents: bool,
}

impl DirectorySkillScanner {
    pub(super) fn new(
        root: PathBuf,
        containment_root: Option<PathBuf>,
        limits: SkillDiscoveryLimits,
        cache: Arc<RwLock<DirectorySkillCache>>,
        verify_contents: bool,
    ) -> Self {
        Self {
            root,
            containment_root,
            limits,
            cache,
            verify_contents,
        }
    }

    pub(super) fn scan(&self) -> Result<SkillSourceSnapshot, SkillError> {
        let canonical_root = match fs::canonicalize(&self.root) {
            Ok(root) => root,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                *self
                    .cache
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    DirectorySkillCache::default();
                return Ok(SkillSourceSnapshot::default());
            }
            Err(error) => {
                return Err(SkillError::source_unavailable(
                    self.root.display().to_string(),
                    error.to_string(),
                ));
            }
        };
        let boundary_path = if let Some(containment_root) = &self.containment_root {
            let canonical_containment = fs::canonicalize(containment_root).map_err(|error| {
                SkillError::source_unavailable(
                    containment_root.display().to_string(),
                    format!("cannot resolve containment root: {error}"),
                )
            })?;
            if !canonical_root.starts_with(&canonical_containment) {
                return Err(SkillError::source_unavailable(
                    self.root.display().to_string(),
                    format!(
                        "resolved source '{}' escapes containment root '{}'",
                        canonical_root.display(),
                        canonical_containment.display()
                    ),
                ));
            }
            canonical_containment
        } else {
            canonical_root.clone()
        };
        let resource_boundary = SkillResourceBoundary::open(boundary_path).map_err(|error| {
            SkillError::source_unavailable(
                self.root.display().to_string(),
                format!("cannot open resource boundary: {error}"),
            )
        })?;
        let entries = fs::read_dir(&self.root).map_err(|error| {
            SkillError::source_unavailable(self.root.display().to_string(), error.to_string())
        })?;

        let mut directories = entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let file_type = entry.file_type().ok()?;
                (file_type.is_dir() && !file_type.is_symlink()).then_some(entry.path())
            })
            .collect::<Vec<_>>();
        directories.sort();

        let mut snapshot = SkillSourceSnapshot::default();
        let mut refreshed_cache = DirectorySkillCache::default();
        #[cfg(feature = "skill-watch")]
        refreshed_cache.capture_documents(&self.root);
        if directories.len() > self.limits.max_skills {
            snapshot.diagnostics.push(format!(
                "skill directory '{}' contains {} entries; only the first {} are scanned",
                self.root.display(),
                directories.len(),
                self.limits.max_skills
            ));
            directories.truncate(self.limits.max_skills);
        }

        let mut total_skill_file_bytes = 0usize;
        for directory in directories {
            match self.load_skill(&canonical_root, &resource_boundary, &directory) {
                Ok(Some(loaded)) => {
                    if total_skill_file_bytes.saturating_add(loaded.document_bytes)
                        > self.limits.max_total_skill_file_bytes
                    {
                        snapshot.diagnostics.push(format!(
                            "skill document '{}' exceeds the {} byte aggregate skill document limit",
                            directory.join("SKILL.md").display(),
                            self.limits.max_total_skill_file_bytes
                        ));
                        continue;
                    }
                    total_skill_file_bytes =
                        total_skill_file_bytes.saturating_add(loaded.document_bytes);
                    refreshed_cache.insert(
                        loaded.skill.directory().to_path_buf(),
                        CachedDirectorySkill::new(loaded.fingerprint, Arc::clone(&loaded.skill)),
                    );
                    snapshot.skills.push(loaded.skill);
                }
                Ok(None) => {}
                Err(error) => {
                    if let Some((path, cached)) = self.last_valid_skill(&canonical_root, &directory)
                    {
                        snapshot.skills.push(Arc::clone(cached.skill()));
                        refreshed_cache.insert(path, cached);
                        snapshot.diagnostics.push(format!(
                            "{error}; retaining the last valid skill until the update becomes valid"
                        ));
                    } else {
                        snapshot.diagnostics.push(error.to_string());
                    }
                }
            }
        }
        *self
            .cache
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = refreshed_cache;
        Ok(snapshot)
    }

    fn last_valid_skill(
        &self,
        canonical_root: &Path,
        directory: &Path,
    ) -> Option<(PathBuf, CachedDirectorySkill)> {
        let canonical_directory = fs::canonicalize(directory).ok()?;
        if !canonical_directory.starts_with(canonical_root) {
            return None;
        }
        let metadata = fs::symlink_metadata(canonical_directory.join("SKILL.md")).ok()?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return None;
        }
        let cached = self
            .cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cached_skill(&canonical_directory)
            .cloned()?;
        Some((canonical_directory, cached))
    }

    fn load_skill(
        &self,
        canonical_root: &Path,
        resource_boundary: &SkillResourceBoundary,
        directory: &Path,
    ) -> Result<Option<LoadedDirectorySkill>, SkillError> {
        let canonical_directory = fs::canonicalize(directory).map_err(|error| {
            SkillError::invalid_document(directory, format!("cannot resolve directory: {error}"))
        })?;
        if !canonical_directory.starts_with(canonical_root) {
            return Err(SkillError::invalid_document(
                directory,
                "skill directory escapes the configured source root",
            ));
        }
        let resource_directory =
            resource_boundary
                .open_skill(&canonical_directory)
                .map_err(|error| {
                    SkillError::invalid_document(
                        directory,
                        format!("cannot open confined skill directory: {error}"),
                    )
                })?;

        let skill_file = canonical_directory.join("SKILL.md");
        let metadata = match fs::symlink_metadata(&skill_file) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(SkillError::invalid_document(
                    &skill_file,
                    "SKILL.md must not be a symbolic link",
                ));
            }
            Ok(metadata) if metadata.is_file() => metadata,
            Ok(_) => return Ok(None),
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(SkillError::invalid_document(
                    &skill_file,
                    format!("cannot inspect file: {error}"),
                ));
            }
        };
        let canonical_skill_file = fs::canonicalize(&skill_file).map_err(|error| {
            SkillError::invalid_document(&skill_file, format!("cannot resolve file: {error}"))
        })?;
        if !canonical_skill_file.starts_with(&canonical_directory) {
            return Err(SkillError::invalid_document(
                &skill_file,
                "SKILL.md escapes its skill directory",
            ));
        }
        if metadata.len() > self.limits.max_skill_file_bytes as u64 {
            return Err(SkillError::invalid_document(
                &skill_file,
                format!(
                    "file exceeds the {} byte limit",
                    self.limits.max_skill_file_bytes
                ),
            ));
        }
        let document_bytes = usize::try_from(metadata.len()).unwrap_or(usize::MAX);
        let document_fingerprint =
            SkillFileFingerprint::from_metadata(PathBuf::from("SKILL.md"), &metadata);
        let resource_manifest =
            self.discover_resource_manifest(&canonical_directory, &resource_directory)?;
        let metadata_fingerprint = SkillDirectoryFingerprint::new(
            document_fingerprint.clone(),
            resource_manifest.fingerprints,
        );
        let cached = if self.verify_contents {
            None
        } else {
            self.cache
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .cached_skill(&canonical_directory)
                .filter(|cached| cached.matches_metadata(&metadata_fingerprint))
                .cloned()
        };
        if let Some(cached) = cached {
            return Ok(Some(LoadedDirectorySkill {
                fingerprint: cached.fingerprint().clone(),
                skill: Arc::clone(cached.skill()),
                document_bytes,
            }));
        }

        let content = fs::read_to_string(&canonical_skill_file).map_err(|error| {
            SkillError::invalid_document(&skill_file, format!("cannot read file: {error}"))
        })?;
        if content.len() > self.limits.max_skill_file_bytes {
            return Err(SkillError::invalid_document(
                &skill_file,
                format!(
                    "file exceeds the {} byte limit",
                    self.limits.max_skill_file_bytes
                ),
            ));
        }
        let discovered_resources = self.load_resource_contents(
            &canonical_directory,
            &resource_directory,
            resource_manifest.paths,
        )?;
        let fingerprint = SkillDirectoryFingerprint::new(
            document_fingerprint.with_content(&content),
            discovered_resources.fingerprints,
        );
        if let Some(cached) = self
            .cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cached_skill(&canonical_directory)
            .filter(|cached| cached.matches_content(&fingerprint))
        {
            return Ok(Some(LoadedDirectorySkill {
                fingerprint,
                skill: Arc::clone(cached.skill()),
                document_bytes,
            }));
        }

        let skill = Skill::from_document(
            canonical_skill_file,
            canonical_directory,
            &content,
            discovered_resources.paths,
        )?
        .with_resource_digests(&discovered_resources.digests)
        .with_resource_directory(resource_directory);
        Ok(Some(LoadedDirectorySkill {
            fingerprint,
            skill: Arc::new(skill),
            document_bytes,
        }))
    }

    fn discover_resource_manifest(
        &self,
        skill_directory: &Path,
        resource_directory: &SkillResourceDirectory,
    ) -> Result<SkillResourceManifest, SkillError> {
        let mut resources = Vec::new();
        let mut fingerprints = Vec::new();
        let mut total_resource_bytes = 0usize;
        for entry in WalkDir::new(skill_directory)
            .follow_links(false)
            .max_depth(self.limits.max_resource_depth)
            .into_iter()
        {
            let entry = entry.map_err(|error| {
                SkillError::invalid_document(
                    skill_directory,
                    format!("cannot enumerate resources: {error}"),
                )
            })?;
            if !entry.file_type().is_file() || entry.file_name() == "SKILL.md" {
                continue;
            }
            if resources.len() >= self.limits.max_resources_per_skill {
                break;
            }
            let relative = entry
                .path()
                .strip_prefix(skill_directory)
                .map_err(|error| {
                    SkillError::invalid_document(
                        entry.path(),
                        format!("cannot resolve resource path: {error}"),
                    )
                })?
                .to_path_buf();
            let file = resource_directory.open_file(&relative).map_err(|error| {
                SkillError::invalid_document(
                    entry.path(),
                    format!("cannot open confined resource: {error}"),
                )
            })?;
            let metadata = file.metadata().map_err(|error| {
                SkillError::invalid_document(
                    entry.path(),
                    format!("cannot inspect resource: {error}"),
                )
            })?;
            if !metadata.is_file() {
                return Err(SkillError::invalid_document(
                    entry.path(),
                    "resource must be a regular file",
                ));
            }
            let resource_bytes = usize::try_from(metadata.len()).unwrap_or(usize::MAX);
            if resource_bytes > self.limits.max_resource_file_bytes {
                return Err(SkillError::invalid_document(
                    entry.path(),
                    format!(
                        "resource exceeds the {} byte limit",
                        self.limits.max_resource_file_bytes
                    ),
                ));
            }
            if total_resource_bytes.saturating_add(resource_bytes)
                > self.limits.max_total_resource_bytes
            {
                return Err(SkillError::invalid_document(
                    skill_directory,
                    format!(
                        "resources exceed the {} byte aggregate limit",
                        self.limits.max_total_resource_bytes
                    ),
                ));
            }
            total_resource_bytes = total_resource_bytes.saturating_add(resource_bytes);
            fingerprints.push(SkillFileFingerprint::from_values(
                relative.clone(),
                metadata.len(),
                metadata.modified().ok().map(|time| time.into_std()),
                metadata.created().ok().map(|time| time.into_std()),
            ));
            resources.push(relative);
        }
        resources.sort();
        fingerprints.sort_by(|left, right| left.path().cmp(right.path()));
        Ok(SkillResourceManifest {
            paths: resources,
            fingerprints,
        })
    }

    fn load_resource_contents(
        &self,
        skill_directory: &Path,
        resource_directory: &SkillResourceDirectory,
        paths: Vec<PathBuf>,
    ) -> Result<DiscoveredSkillResources, SkillError> {
        let mut fingerprints = Vec::with_capacity(paths.len());
        let mut digests = Vec::with_capacity(paths.len());
        let mut total_resource_bytes = 0usize;
        for relative in &paths {
            let resource_path = skill_directory.join(relative);
            let file = resource_directory.open_file(relative).map_err(|error| {
                SkillError::invalid_document(
                    &resource_path,
                    format!("cannot open confined resource: {error}"),
                )
            })?;
            let metadata = file.metadata().map_err(|error| {
                SkillError::invalid_document(
                    &resource_path,
                    format!("cannot inspect resource: {error}"),
                )
            })?;
            let resource_bytes = usize::try_from(metadata.len()).unwrap_or(usize::MAX);
            if resource_bytes > self.limits.max_resource_file_bytes {
                return Err(SkillError::invalid_document(
                    &resource_path,
                    format!(
                        "resource exceeds the {} byte limit",
                        self.limits.max_resource_file_bytes
                    ),
                ));
            }
            let mut bytes = Vec::with_capacity(resource_bytes);
            file.take(self.limits.max_resource_file_bytes.saturating_add(1) as u64)
                .read_to_end(&mut bytes)
                .map_err(|error| {
                    SkillError::invalid_document(
                        &resource_path,
                        format!("cannot read resource: {error}"),
                    )
                })?;
            if bytes.len() > self.limits.max_resource_file_bytes {
                return Err(SkillError::invalid_document(
                    &resource_path,
                    format!(
                        "resource exceeds the {} byte limit",
                        self.limits.max_resource_file_bytes
                    ),
                ));
            }
            if total_resource_bytes.saturating_add(bytes.len())
                > self.limits.max_total_resource_bytes
            {
                return Err(SkillError::invalid_document(
                    skill_directory,
                    format!(
                        "resources exceed the {} byte aggregate limit",
                        self.limits.max_total_resource_bytes
                    ),
                ));
            }
            total_resource_bytes = total_resource_bytes.saturating_add(bytes.len());
            let digest: [u8; 32] = Sha256::digest(&bytes).into();
            fingerprints.push(
                SkillFileFingerprint::from_values(
                    relative.clone(),
                    metadata.len(),
                    metadata.modified().ok().map(|time| time.into_std()),
                    metadata.created().ok().map(|time| time.into_std()),
                )
                .with_digest(digest),
            );
            digests.push((relative.clone(), digest));
        }
        digests.sort_by(|left, right| left.0.cmp(&right.0));
        Ok(DiscoveredSkillResources {
            paths,
            fingerprints,
            digests,
        })
    }
}

#[derive(Debug)]
struct LoadedDirectorySkill {
    fingerprint: SkillDirectoryFingerprint,
    skill: Arc<Skill>,
    document_bytes: usize,
}

#[derive(Debug)]
struct DiscoveredSkillResources {
    paths: Vec<PathBuf>,
    fingerprints: Vec<SkillFileFingerprint>,
    digests: Vec<(PathBuf, [u8; 32])>,
}

#[derive(Debug)]
struct SkillResourceManifest {
    paths: Vec<PathBuf>,
    fingerprints: Vec<SkillFileFingerprint>,
}
