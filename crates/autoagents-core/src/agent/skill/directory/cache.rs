use crate::agent::skill::{Skill, SkillSourceSnapshot};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

#[derive(Debug, Default)]
pub(super) struct DirectorySkillCache {
    skills: BTreeMap<PathBuf, CachedDirectorySkill>,
    #[cfg(feature = "skill-watch")]
    documents: DirectorySkillDocumentSnapshot,
}

impl DirectorySkillCache {
    pub(super) fn snapshot(&self) -> SkillSourceSnapshot {
        SkillSourceSnapshot::new(
            self.skills
                .values()
                .map(|cached| Arc::clone(cached.skill()))
                .collect(),
            Vec::new(),
        )
    }

    pub(super) fn cached_skill(&self, directory: &Path) -> Option<&CachedDirectorySkill> {
        self.skills.get(directory)
    }

    pub(super) fn insert(&mut self, directory: PathBuf, skill: CachedDirectorySkill) {
        self.skills.insert(directory, skill);
    }

    #[cfg(feature = "skill-watch")]
    pub(super) fn capture_documents(&mut self, root: &Path) {
        self.documents = DirectorySkillDocumentSnapshot::capture(root);
    }

    #[cfg(feature = "skill-watch")]
    pub(super) fn documents_changed(&self, root: &Path) -> bool {
        self.documents != DirectorySkillDocumentSnapshot::capture(root)
    }
}

#[derive(Clone, Debug)]
pub(super) struct CachedDirectorySkill {
    fingerprint: SkillDirectoryFingerprint,
    skill: Arc<Skill>,
}

impl CachedDirectorySkill {
    pub(super) fn new(fingerprint: SkillDirectoryFingerprint, skill: Arc<Skill>) -> Self {
        Self { fingerprint, skill }
    }

    pub(super) fn matches_metadata(&self, fingerprint: &SkillDirectoryFingerprint) -> bool {
        self.fingerprint.matches_metadata(fingerprint)
    }

    pub(super) fn matches_content(&self, fingerprint: &SkillDirectoryFingerprint) -> bool {
        self.fingerprint.matches_content(fingerprint)
    }

    pub(super) fn fingerprint(&self) -> &SkillDirectoryFingerprint {
        &self.fingerprint
    }

    pub(super) fn skill(&self) -> &Arc<Skill> {
        &self.skill
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SkillDirectoryFingerprint {
    document: SkillFileFingerprint,
    resources: Vec<SkillFileFingerprint>,
}

impl SkillDirectoryFingerprint {
    pub(super) fn new(
        document: SkillFileFingerprint,
        resources: Vec<SkillFileFingerprint>,
    ) -> Self {
        Self {
            document,
            resources,
        }
    }

    fn matches_metadata(&self, other: &Self) -> bool {
        self.document.matches_metadata(&other.document)
            && self.resources.len() == other.resources.len()
            && self
                .resources
                .iter()
                .zip(&other.resources)
                .all(|(left, right)| left.matches_metadata(right))
    }

    fn matches_content(&self, other: &Self) -> bool {
        self.document.matches_content(&other.document)
            && self.resources.len() == other.resources.len()
            && self
                .resources
                .iter()
                .zip(&other.resources)
                .all(|(left, right)| left.matches_content(right))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SkillFileFingerprint {
    path: PathBuf,
    length: u64,
    modified: Option<SystemTime>,
    created: Option<SystemTime>,
    content_hash: Option<[u8; 32]>,
}

impl SkillFileFingerprint {
    pub(super) fn from_metadata(path: PathBuf, metadata: &fs::Metadata) -> Self {
        Self::from_values(
            path,
            metadata.len(),
            metadata.modified().ok(),
            metadata.created().ok(),
        )
    }

    pub(super) fn from_values(
        path: PathBuf,
        length: u64,
        modified: Option<SystemTime>,
        created: Option<SystemTime>,
    ) -> Self {
        Self {
            path,
            length,
            modified,
            created,
            content_hash: None,
        }
    }

    pub(super) fn with_content(mut self, content: &str) -> Self {
        self.content_hash = Some(Sha256::digest(content.as_bytes()).into());
        self
    }

    pub(super) fn with_digest(mut self, digest: [u8; 32]) -> Self {
        self.content_hash = Some(digest);
        self
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    fn matches_metadata(&self, other: &Self) -> bool {
        self.path == other.path
            && self.length == other.length
            && self.modified == other.modified
            && self.created == other.created
    }

    fn matches_content(&self, other: &Self) -> bool {
        self.path == other.path
            && self.length == other.length
            && self.content_hash.is_some()
            && self.content_hash == other.content_hash
    }
}

#[cfg(feature = "skill-watch")]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct DirectorySkillDocumentSnapshot {
    documents: BTreeMap<PathBuf, SkillFileFingerprint>,
}

#[cfg(feature = "skill-watch")]
impl DirectorySkillDocumentSnapshot {
    fn capture(root: &Path) -> Self {
        let Ok(entries) = fs::read_dir(root) else {
            return Self::default();
        };
        let mut documents = BTreeMap::new();
        for entry in entries.filter_map(Result::ok) {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_dir() || file_type.is_symlink() {
                continue;
            }
            let path = entry.path().join("SKILL.md");
            let Ok(metadata) = fs::symlink_metadata(&path) else {
                continue;
            };
            documents.insert(
                entry.path(),
                SkillFileFingerprint::from_metadata(PathBuf::from("SKILL.md"), &metadata),
            );
        }
        Self { documents }
    }
}
