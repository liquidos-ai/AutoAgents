use crate::agent::skill::{SkillError, SkillToolSelector};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Component;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use crate::agent::skill::resource::SkillResourceDirectory;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SkillRevision([u8; 32]);

impl SkillRevision {
    fn from_document(content: &str) -> Self {
        Self(Sha256::digest(content.as_bytes()).into())
    }

    fn with_resources(self, resources: &[PathBuf]) -> Self {
        let mut digest = Sha256::new();
        digest.update(self.0);
        for resource in resources {
            let bytes = resource.as_os_str().as_encoded_bytes();
            digest.update(bytes.len().to_le_bytes());
            digest.update(bytes);
        }
        Self(digest.finalize().into())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn with_resource_digests(self, resources: &[(PathBuf, [u8; 32])]) -> Self {
        let mut digest = Sha256::new();
        digest.update(self.0);
        for (resource, resource_digest) in resources {
            let bytes = resource.as_os_str().as_encoded_bytes();
            digest.update(bytes.len().to_le_bytes());
            digest.update(bytes);
            digest.update(resource_digest);
        }
        Self(digest.finalize().into())
    }
}

impl std::fmt::Display for SkillRevision {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SkillMetadata {
    name: String,
    description: String,
    license: Option<String>,
    compatibility: Option<String>,
    metadata: BTreeMap<String, String>,
    allowed_tools: Arc<[SkillToolSelector]>,
}

impl SkillMetadata {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn license(&self) -> Option<&str> {
        self.license.as_deref()
    }

    pub fn compatibility(&self) -> Option<&str> {
        self.compatibility.as_deref()
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn allowed_tools(&self) -> &[SkillToolSelector] {
        &self.allowed_tools
    }

    fn validate(&self, directory_name: &str, path: &Path) -> Result<(), SkillError> {
        let name_length = self.name.chars().count();
        if !(1..=64).contains(&name_length) {
            return Err(SkillError::invalid_document(
                path,
                "name must contain between 1 and 64 characters",
            ));
        }
        if self.name.starts_with('-')
            || self.name.ends_with('-')
            || self.name.contains("--")
            || !self
                .name
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        {
            return Err(SkillError::invalid_document(
                path,
                "name must use lowercase ASCII letters, digits, and single hyphens",
            ));
        }
        if self.name != directory_name {
            return Err(SkillError::invalid_document(
                path,
                format!(
                    "name '{}' must match parent directory '{directory_name}'",
                    self.name
                ),
            ));
        }

        let description_length = self.description.chars().count();
        if self.description.trim().is_empty() || description_length > 1024 {
            return Err(SkillError::invalid_document(
                path,
                "description must contain between 1 and 1024 characters",
            ));
        }
        if self
            .compatibility
            .as_ref()
            .is_some_and(|value| value.trim().is_empty() || value.chars().count() > 500)
        {
            return Err(SkillError::invalid_document(
                path,
                "compatibility must contain between 1 and 500 characters",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Skill {
    metadata: SkillMetadata,
    instructions: Arc<str>,
    directory: PathBuf,
    skill_file: PathBuf,
    resources: Arc<[PathBuf]>,
    resource_digests: Arc<BTreeMap<PathBuf, [u8; 32]>>,
    revision: SkillRevision,
    #[cfg(not(target_arch = "wasm32"))]
    resource_directory: Option<SkillResourceDirectory>,
}

impl Skill {
    pub fn from_document(
        skill_file: impl Into<PathBuf>,
        directory: impl Into<PathBuf>,
        content: &str,
        resources: Vec<PathBuf>,
    ) -> Result<Self, SkillError> {
        SkillDocument::parse(skill_file.into(), directory.into(), content)?
            .with_resources(resources)
    }

    pub fn metadata(&self) -> &SkillMetadata {
        &self.metadata
    }

    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn skill_file(&self) -> &Path {
        &self.skill_file
    }

    pub fn resources(&self) -> &[PathBuf] {
        &self.resources
    }

    pub fn revision(&self) -> SkillRevision {
        self.revision
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn with_resource_digests(mut self, resources: &[(PathBuf, [u8; 32])]) -> Self {
        self.revision = self.revision.with_resource_digests(resources);
        self.resource_digests = Arc::new(resources.iter().cloned().collect());
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn resource_digest(&self, path: &Path) -> Option<[u8; 32]> {
        self.resource_digests.get(path).copied()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn with_resource_directory(
        mut self,
        resource_directory: SkillResourceDirectory,
    ) -> Self {
        self.resource_directory = Some(resource_directory);
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn resource_directory(&self) -> Option<SkillResourceDirectory> {
        self.resource_directory.clone()
    }

    fn with_resources(mut self, mut resources: Vec<PathBuf>) -> Result<Self, SkillError> {
        for resource in &resources {
            if resource.as_os_str().is_empty()
                || resource.is_absolute()
                || resource.to_str().is_none()
                || resource
                    .components()
                    .any(|component| !matches!(component, Component::Normal(_)))
            {
                return Err(SkillError::invalid_document(
                    &self.skill_file,
                    format!("resource path '{}' is invalid", resource.display()),
                ));
            }
        }
        resources.sort();
        resources.dedup();
        self.revision = self.revision.with_resources(&resources);
        self.resources = resources.into();
        Ok(self)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SkillFrontmatter {
    name: String,
    description: String,
    license: Option<String>,
    compatibility: Option<String>,
    #[serde(default)]
    metadata: BTreeMap<String, String>,
    #[serde(rename = "allowed-tools")]
    allowed_tools: Option<String>,
}

pub(crate) struct SkillDocument;

impl SkillDocument {
    pub(crate) fn parse(
        skill_file: PathBuf,
        directory: PathBuf,
        content: &str,
    ) -> Result<Skill, SkillError> {
        let (frontmatter, instructions) = Self::split(content, &skill_file)?;
        let parsed: SkillFrontmatter = serde_yaml_ng::from_str(frontmatter).map_err(|error| {
            SkillError::invalid_document(&skill_file, format!("invalid YAML frontmatter: {error}"))
        })?;
        let metadata = SkillMetadata {
            name: parsed.name,
            description: parsed.description,
            license: parsed.license,
            compatibility: parsed.compatibility,
            metadata: parsed.metadata,
            allowed_tools: parsed
                .allowed_tools
                .as_deref()
                .unwrap_or_default()
                .split_whitespace()
                .map(SkillToolSelector::parse)
                .collect::<Result<Vec<_>, _>>()?
                .into(),
        };
        let directory_name = directory
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                SkillError::invalid_document(&skill_file, "parent directory is not valid UTF-8")
            })?;
        metadata.validate(directory_name, &skill_file)?;
        if instructions.trim().is_empty() {
            return Err(SkillError::invalid_document(
                &skill_file,
                "instructions must not be empty",
            ));
        }

        Ok(Skill {
            metadata,
            instructions: Arc::from(instructions.trim()),
            directory,
            skill_file,
            resources: Arc::from([]),
            resource_digests: Arc::new(BTreeMap::new()),
            revision: SkillRevision::from_document(content),
            #[cfg(not(target_arch = "wasm32"))]
            resource_directory: None,
        })
    }

    fn split<'a>(content: &'a str, path: &Path) -> Result<(&'a str, &'a str), SkillError> {
        let normalized = content.strip_prefix('\u{feff}').unwrap_or(content);
        let mut lines = normalized.split_inclusive('\n');
        let opening = lines
            .next()
            .unwrap_or_default()
            .trim_end_matches(['\r', '\n']);
        if opening != "---" {
            return Err(SkillError::invalid_document(
                path,
                "SKILL.md must start with a YAML frontmatter delimiter",
            ));
        }

        let frontmatter_start =
            opening.len() + (normalized[opening.len()..].starts_with("\r\n") as usize) + 1;
        let mut offset = frontmatter_start;
        for line in lines {
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed == "---" {
                let frontmatter = &normalized[frontmatter_start..offset];
                let body_start = offset + line.len();
                return Ok((frontmatter, &normalized[body_start..]));
            }
            offset += line.len();
        }

        Err(SkillError::invalid_document(
            path,
            "YAML frontmatter is missing its closing delimiter",
        ))
    }
}
