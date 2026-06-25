use std::path::Path;

use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolRuntime, ToolT},
};
use autoagents_derive::{ToolInput, tool};
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::fs;

use super::{BaseFileTool, FilesystemSandbox, sandbox_error};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct ListDirArgs {
    #[input(description = "Path of the directory to list")]
    directory_path: String,
    #[serde(default)]
    #[input(
        description = "Whether to include hidden files and directories (names starting with '.')"
    )]
    include_hidden: bool,
}

#[tool(
    name = "list_dir",
    description = "List contents of a directory",
    input = ListDirArgs,
)]
pub struct ListDir {
    sandbox: FilesystemSandbox,
}

impl ListDir {
    pub fn new(root: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            sandbox: FilesystemSandbox::new(root)?,
        })
    }
    pub fn with_sandbox(sandbox: FilesystemSandbox) -> Self {
        Self { sandbox }
    }
}

impl BaseFileTool for ListDir {
    fn sandbox(&self) -> &FilesystemSandbox {
        &self.sandbox
    }
}

#[async_trait]
impl ToolRuntime for ListDir
where
    Self: BaseFileTool,
{
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let ListDirArgs {
            directory_path,
            include_hidden,
        } = serde_json::from_value(args)?;

        debug!("List Directory Executing: Directory: {}", directory_path);

        let dir_path = self
            .sandbox()
            .resolve_relative(&directory_path)
            .map_err(sandbox_error)?;

        if !dir_path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("Directory does not exist: {}", dir_path.display()).into(),
            ));
        }

        if !dir_path.is_dir() {
            return Err(ToolCallError::RuntimeError(
                format!("Path is not a directory: {}", dir_path.display()).into(),
            ));
        }

        let dir_path = self
            .sandbox()
            .ensure_resolved(&dir_path)
            .map_err(sandbox_error)?;

        let mut entries = Vec::new();

        let mut read_dir = fs::read_dir(&dir_path)
            .await
            .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

        while let Some(entry) = read_dir
            .next_entry()
            .await
            .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?
        {
            let path = entry.path();
            let validated_path = self
                .sandbox()
                .validate_walk_entry(&path)
                .map_err(sandbox_error)?;

            let file_name = entry.file_name().to_string_lossy().to_string();

            if !include_hidden && file_name.starts_with('.') {
                continue;
            }

            let metadata = entry
                .metadata()
                .await
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

            let is_dir = metadata.is_dir();

            entries.push(json!({
                "name": file_name,
                "path": validated_path.display().to_string(),
                "is_dir": is_dir,
                "size": if !is_dir { metadata.len() } else { 0 },
            }));
        }

        Ok(json!({
            "success": true,
            "directory": dir_path.display().to_string(),
            "count": entries.len(),
            "entries": entries
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_list_dir_simple() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        std::fs::write(temp_dir.path().join("file1.txt"), "content1")
            .expect("Failed to create file1");
        std::fs::write(temp_dir.path().join("file2.rs"), "content2")
            .expect("Failed to create file2");
        std::fs::create_dir_all(temp_dir.path().join("subdir")).expect("Failed to create subdir");

        let list_dir = ListDir::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "directory_path": "."
        });

        let result = list_dir
            .execute(args)
            .await
            .expect("Failed to list directory");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());

        let entries = result.get("entries").and_then(|v| v.as_array()).unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[tokio::test]
    async fn test_list_nonexistent_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        let list_dir = ListDir::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "directory_path": "nonexistent"
        });

        let result = list_dir.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_dir_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        std::fs::write(temp_dir.path().join("test.txt"), "content")
            .expect("Failed to create test file");

        let list_dir = ListDir::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "directory_path": "."
        });

        let result = list_dir
            .execute(args)
            .await
            .expect("Failed to list directory");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());

        let entries = result.get("entries").and_then(|v| v.as_array()).unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[tokio::test]
    async fn test_list_dir_include_hidden() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        std::fs::write(temp_dir.path().join("visible.txt"), "content")
            .expect("Failed to create visible file");
        std::fs::write(temp_dir.path().join(".hidden"), "secret")
            .expect("Failed to create hidden file");

        let list_dir = ListDir::new(temp_dir.path()).expect("sandbox");

        let hidden_excluded = list_dir
            .execute(json!({ "directory_path": "." }))
            .await
            .expect("list should succeed");
        let excluded = hidden_excluded
            .get("entries")
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(excluded.len(), 1);
        assert_eq!(
            excluded[0].get("name").and_then(|v| v.as_str()),
            Some("visible.txt")
        );

        let hidden_included = list_dir
            .execute(json!({ "directory_path": ".", "include_hidden": true }))
            .await
            .expect("list with hidden should succeed");
        let included = hidden_included
            .get("entries")
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(included.len(), 2);
    }
}
