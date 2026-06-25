mod copy_file;
mod create_dir;
mod delete_file;
mod list_dir;
mod move_file;
mod read_file;
mod sandbox;
mod search_file;
mod write_file;

pub use copy_file::CopyFile;
pub use create_dir::CreateDir;
pub use delete_file::DeleteFile;
pub use list_dir::ListDir;
pub use move_file::MoveFile;
pub use read_file::ReadFile;
pub use sandbox::FilesystemSandbox;
pub use search_file::SearchFile;
pub use write_file::WriteFile;

use std::path::PathBuf;

use autoagents::core::tool::ToolCallError;

pub(crate) trait BaseFileTool {
    fn sandbox(&self) -> &FilesystemSandbox;
}

pub(crate) fn sandbox_error(error: std::io::Error) -> ToolCallError {
    ToolCallError::RuntimeError(Box::new(error))
}

/// Resolve a relative path, create parent directories when needed, then re-validate
/// the full path to close TOCTOU gaps from symlink races in parent chains.
pub(crate) async fn prepare_mutation_path(
    sandbox: &FilesystemSandbox,
    user_path: &str,
) -> Result<PathBuf, ToolCallError> {
    let path = sandbox.resolve_relative(user_path).map_err(sandbox_error)?;

    if let Some(parent) = path.parent()
        && parent != sandbox.root()
    {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
    }

    sandbox.ensure_resolved(&path).map_err(sandbox_error)
}
