use std::path::PathBuf;

use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolRuntime, ToolT},
};
use autoagents_derive::{ToolInput, tool};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::runtime_error;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct CheckVerifiedArgs {
    #[input(description = "Optional note on why you are checking; not used by the tool")]
    note: Option<String>,
}

#[tool(
    name = "check_verified",
    description = "Check whether the workspace, exactly as it is now, has a signed passing test receipt. Any edit after the last passing run makes it unverified. Call this before reporting that the work is complete.",
    input = CheckVerifiedArgs,
)]
pub struct CheckVerified {
    workspace_root: PathBuf,
    context: treeproof::Context,
}

impl CheckVerified {
    /// Check the git repository that contains `workspace_root`.
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_root: workspace_root.into(),
            context: treeproof::Context::default(),
        }
    }

    /// Read receipts from the store of `context` (use the same context as [`super::RunTests`]).
    pub fn with_context(mut self, context: treeproof::Context) -> Self {
        self.context = context;
        self
    }
}

#[async_trait]
impl ToolRuntime for CheckVerified {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let CheckVerifiedArgs { .. } = serde_json::from_value(args)?;
        let workspace_root = self.workspace_root.clone();
        let context = self.context.clone();
        let report = tokio::task::spawn_blocking(move || {
            let repo_root = treeproof::git::repo_root(&workspace_root)?;
            // The committed policy, not the working copy the agent can edit.
            let config = treeproof::config::Config::load_committed(&repo_root)?;
            treeproof::gate::evaluate_in(&repo_root, &config, None, &context)
        })
        .await
        .map_err(|e| runtime_error(format!("verification task failed: {e}")))?
        .map_err(|e| runtime_error(e.to_string()))?;

        let receipt = report.receipt.as_ref().map(|r| {
            json!({
                "result": r.result,
                "command": r.command,
                "passed": r.counts.passed,
                "finished_at": r.finished_at,
            })
        });
        Ok(json!({
            "success": true,
            "verified": report.ok,
            "tree": report.tree,
            "reason": report.reason,
            "receipt": receipt,
        }))
    }
}
