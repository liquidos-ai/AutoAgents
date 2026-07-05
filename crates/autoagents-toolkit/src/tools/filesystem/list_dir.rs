use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolRuntime, ToolT},
};
use autoagents_derive::{ToolInput, tool};
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::fs;

use super::{BaseFileTool, default_root_dir};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct ListDirArgs {
    #[input(description = "Path of the directory to list")]
    directory_path: String,
}

#[tool(
    name = "list_dir",
    description = "List contents of a directory",
    input = ListDirArgs,
)]
pub struct ListDir {
    root_dir: Option<String>,
}

impl Default for ListDir {
    fn default() -> Self {
        Self {
            root_dir: Some(default_root_dir()),
        }
    }
}

impl ListDir {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_unrestricted() -> Self {
        Self { root_dir: None }
    }

    pub fn new_with_root_dir(root_dir: String) -> Self {
        Self {
            root_dir: Some(root_dir),
        }
    }
}

impl BaseFileTool for ListDir {
    fn root_dir(&self) -> Option<String> {
        self.root_dir.clone()
    }
}

#[async_trait]
impl ToolRuntime for ListDir
where
    Self: BaseFileTool,
{
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let ListDirArgs { directory_path } = serde_json::from_value(args)?;

        debug!("List Directory Executing: Directory: {}", directory_path);

        let dir_path = self
            .resolve_path(&directory_path)
            .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
        let _recursive = false; // Simplified to non-recursive
        let include_hidden = false;
        let filter_extension: Option<String> = None;

        // Validate directory exists
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

        let mut entries = Vec::new();

        // Simplified to non-recursive listing only
        {
            let mut read_dir = fs::read_dir(&dir_path)
                .await
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

            while let Some(entry) = read_dir
                .next_entry()
                .await
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?
            {
                let path = entry.path();
                let file_name = entry.file_name().to_string_lossy().to_string();

                // Skip hidden files if requested
                if !include_hidden && file_name.starts_with('.') {
                    continue;
                }

                let file_type = entry
                    .file_type()
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

                let metadata = fs::symlink_metadata(&path)
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

                let is_dir = file_type.is_dir();

                // Apply extension filter if provided
                if let Some(ref ext_filter) = filter_extension
                    && !is_dir
                {
                    if let Some(ext) = path.extension() {
                        if ext.to_string_lossy() != *ext_filter {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }

                entries.push(json!({
                    "name": file_name,
                    "path": self.output_path(&path),
                    "is_dir": is_dir,
                    "size": if !is_dir { metadata.len() } else { 0 },
                }));
            }
        }

        Ok(json!({
            "success": true,
            "directory": self.output_path(&dir_path),
            "count": entries.len(),
            "entries": entries
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::filesystem::ReadFile;
    // std::io::Write import removed - not used in simplified tests
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_list_dir_simple() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        // Create some test files and directories
        std::fs::write(temp_dir.path().join("file1.txt"), "content1")
            .expect("Failed to create file1");
        std::fs::write(temp_dir.path().join("file2.rs"), "content2")
            .expect("Failed to create file2");
        std::fs::create_dir_all(temp_dir.path().join("subdir")).expect("Failed to create subdir");

        let list_dir = ListDir::new_unrestricted();
        let args = json!({
            "directory_path": temp_dir.path().display().to_string()
        });

        let result = list_dir
            .execute(args)
            .await
            .expect("Failed to list directory");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());

        let entries = result.get("entries").and_then(|v| v.as_array()).unwrap();
        assert_eq!(entries.len(), 3);
    }

    // Filter test removed - feature not implemented in simplified version

    // Recursive test removed - feature not implemented in simplified version

    // Hidden files test removed - feature not implemented in simplified version

    #[tokio::test]
    async fn test_list_nonexistent_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let nonexistent = temp_dir.path().join("nonexistent");

        let list_dir = ListDir::new_unrestricted();
        let args = json!({
            "directory_path": nonexistent.display().to_string()
        });

        let result = list_dir.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_dir_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let root_dir = temp_dir.path().to_str().unwrap().to_string();

        std::fs::write(temp_dir.path().join("test.txt"), "content")
            .expect("Failed to create test file");

        let list_dir = ListDir::new_with_root_dir(root_dir);
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
        assert_eq!(result.get("directory").and_then(|v| v.as_str()), Some("."));
        assert_eq!(
            entries[0].get("path").and_then(|v| v.as_str()),
            Some("test.txt")
        );
    }

    #[tokio::test]
    async fn test_list_dir_rooted_output_path_can_feed_read_file() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let root_dir = temp_dir.path().to_str().unwrap().to_string();

        std::fs::write(temp_dir.path().join("test.txt"), "content")
            .expect("Failed to create test file");

        let list_dir = ListDir::new_with_root_dir(root_dir.clone());
        let result = list_dir
            .execute(json!({
                "directory_path": "."
            }))
            .await
            .expect("Failed to list directory");

        let path = result["entries"][0]["path"]
            .as_str()
            .expect("entry path should be a string");

        let read_file = ReadFile::new_with_root_dir(root_dir);
        let result = read_file
            .execute(json!({
                "file_path": path
            }))
            .await
            .expect("listed path should be readable");

        assert_eq!(
            result.get("content").and_then(|v| v.as_str()),
            Some("content")
        );
    }

    #[tokio::test]
    async fn test_list_dir_rejects_absolute_path_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let root_dir = temp_dir.path().to_str().unwrap().to_string();

        let list_dir = ListDir::new_with_root_dir(root_dir);
        let result = list_dir
            .execute(json!({
                "directory_path": temp_dir.path().display().to_string()
            }))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_dir_rejects_traversal_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let root_dir = temp_dir.path().join("root");
        let outside_dir = temp_dir.path().join("outside");
        std::fs::create_dir_all(&root_dir).expect("Failed to create root");
        std::fs::create_dir_all(&outside_dir).expect("Failed to create outside");

        let list_dir = ListDir::new_with_root_dir(root_dir.to_string_lossy().to_string());
        let result = list_dir
            .execute(json!({
                "directory_path": "../outside"
            }))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_list_dir_rejects_symlink_escape_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let root_dir = temp_dir.path().join("root");
        let outside_dir = temp_dir.path().join("outside");
        std::fs::create_dir_all(&root_dir).expect("Failed to create root");
        std::fs::create_dir_all(&outside_dir).expect("Failed to create outside");
        std::os::unix::fs::symlink(&outside_dir, root_dir.join("outside_link"))
            .expect("Failed to create symlink");

        let list_dir = ListDir::new_with_root_dir(root_dir.to_string_lossy().to_string());
        let result = list_dir
            .execute(json!({
                "directory_path": "outside_link"
            }))
            .await;

        assert!(result.is_err());
    }
}
