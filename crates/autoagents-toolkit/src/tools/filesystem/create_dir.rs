use std::path::Path;

use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolRuntime},
};
use autoagents_derive::{ToolInput, tool};
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::fs;

use autoagents::prelude::ToolT;

use super::{BaseFileTool, FilesystemSandbox, sandbox_error};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct CreateDirArgs {
    #[input(description = "Path of the directory to create")]
    pub directory_path: String,

    #[serde(default)]
    #[input(description = "Create parent directories if they don't exist")]
    pub recursive: bool,
}
#[tool(
    name = "create_dir",
    description = "Create a directory in the filesystem. Idempotent: succeeds if the directory already exists.",
    input = CreateDirArgs,
)]
pub struct CreateDir {
    sandbox: FilesystemSandbox,
}

impl CreateDir {
    pub fn new(root: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            sandbox: FilesystemSandbox::new(root)?,
        })
    }
    pub fn with_sandbox(sandbox: FilesystemSandbox) -> Self {
        Self { sandbox }
    }
}

impl BaseFileTool for CreateDir {
    fn sandbox(&self) -> &FilesystemSandbox {
        &self.sandbox
    }
}

#[async_trait]
impl ToolRuntime for CreateDir
where
    Self: BaseFileTool,
{
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let CreateDirArgs {
            directory_path,
            recursive,
        } = serde_json::from_value(args)?;

        debug!("CreateDir Executing: directory_path={}", directory_path);

        let path = self
            .sandbox()
            .resolve_relative(&directory_path)
            .map_err(sandbox_error)?;
        let path = self
            .sandbox()
            .ensure_resolved(&path)
            .map_err(sandbox_error)?;

        use std::io::ErrorKind;

        let existed_before = fs::metadata(&path).await.is_ok();

        let (mut already_existed, mut created) = (false, false);

        if recursive {
            fs::create_dir_all(&path)
                .await
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

            already_existed = existed_before;
            created = !existed_before;
        } else {
            match fs::create_dir(&path).await {
                Ok(_) => {
                    created = true;
                }
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    already_existed = true;
                }
                Err(e) => return Err(ToolCallError::RuntimeError(Box::new(e))),
            }
        }

        let path = self
            .sandbox()
            .ensure_resolved(&path)
            .map_err(sandbox_error)?;

        let message = if already_existed {
            "Directory already existed"
        } else if created {
            "Directory created successfully"
        } else {
            "Directory ensured"
        };

        Ok(json!({
            "success": true,
            "path": path.display().to_string(),
            "already_existed": already_existed,
            "created": created,
            "recursive": recursive,
            "message": message
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_create_dir_creates_new_directory() {
        let tmp = tempdir().expect("tempdir");

        let tool = CreateDir::new(tmp.path()).expect("sandbox");

        let args = json!({
            "directory_path": "test",
            "recursive": false
        });

        let result = tool.execute(args).await.expect("execute");
        assert_eq!(result.get("success").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            result.get("already_existed").and_then(|v| v.as_bool()),
            Some(false)
        );
        assert_eq!(result.get("created").and_then(|v| v.as_bool()), Some(true));

        let expected_path = tmp.path().join("test");
        assert!(expected_path.exists());
        assert!(expected_path.is_dir());
    }

    #[tokio::test]
    async fn test_create_dir_idempotent_when_already_exists() {
        let tmp = tempdir().expect("tempdir");

        let initial = tmp.path().join("test");
        std::fs::create_dir_all(&initial).expect("create initial dir");

        let tool = CreateDir::new(tmp.path()).expect("sandbox");

        let args = json!({
            "directory_path": "test",
            "recursive": false
        });

        let result = tool.execute(args).await.expect("execute");
        assert_eq!(result.get("success").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            result.get("already_existed").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(result.get("created").and_then(|v| v.as_bool()), Some(false));
        assert!(initial.exists());
        assert!(initial.is_dir());
    }

    #[tokio::test]
    async fn test_create_dir_recursive_creates_nested_directories() {
        let tmp = tempdir().expect("tempdir");

        let tool = CreateDir::new(tmp.path()).expect("sandbox");

        let args = json!({
            "directory_path": "a/b/c",
            "recursive": true
        });

        let result = tool.execute(args).await.expect("execute");
        assert_eq!(result.get("success").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            result.get("already_existed").and_then(|v| v.as_bool()),
            Some(false)
        );
        assert_eq!(result.get("created").and_then(|v| v.as_bool()), Some(true));

        let expected = tmp.path().join("a").join("b").join("c");
        assert!(expected.exists());
        assert!(expected.is_dir());
    }

    #[tokio::test]
    async fn test_create_dir_recursive_default_field_is_false_when_omitted() {
        let tmp = tempdir().expect("tempdir");

        let tool = CreateDir::new(tmp.path()).expect("sandbox");

        let args = json!({
            "directory_path": "test"
        });

        let result = tool.execute(args).await.expect("execute");
        assert_eq!(result.get("success").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            result.get("recursive").and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[tokio::test]
    async fn test_create_dir_rejects_path_outside_root() {
        let tmp = tempdir().expect("tempdir");

        let tool = CreateDir::new(tmp.path()).expect("sandbox");

        let args = json!({
            "directory_path": "../outside",
            "recursive": true
        });

        let err = tool.execute(args).await.expect_err("should fail");
        match err {
            ToolCallError::RuntimeError(_) => {}
            other => panic!("expected RuntimeError, got: {other:?}"),
        }
    }
}
