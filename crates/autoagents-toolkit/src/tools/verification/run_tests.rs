use std::path::PathBuf;
use std::time::Duration;

use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolRuntime, ToolT},
};
use autoagents_derive::{ToolInput, tool};
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::runtime_error;

/// Characters of test output returned to the model.
const OUTPUT_TAIL_CHARS: usize = 6000;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct RunTestsArgs {
    #[input(
        description = "Set to true to execute the tests even if this exact tree already has a cached passing run"
    )]
    no_cache: Option<bool>,
}

#[tool(
    name = "run_tests",
    description = "Run the project's test suite on the workspace as it is now. Returns PASSED, WARNED or FAILED with test counts and a signed receipt for the current git tree. Run it after your last edit and before you say the work is done.",
    input = RunTestsArgs,
)]
pub struct RunTests {
    workspace_root: PathBuf,
    command: Vec<String>,
    timeout: Option<Duration>,
    context: treeproof::Context,
}

impl RunTests {
    /// Run `command` (program and arguments, no shell) in `workspace_root`, which must be
    /// inside a git repository.
    pub fn new(workspace_root: impl Into<PathBuf>, command: Vec<String>) -> Self {
        Self {
            workspace_root: workspace_root.into(),
            command,
            timeout: Some(Duration::from_secs(30 * 60)),
            context: treeproof::Context::default(),
        }
    }

    /// Pick the usual test command for the project (`cargo test`, `go test ./...`,
    /// `python -m pytest` or `npm test`) from the files at the repository root.
    pub fn detect(workspace_root: impl Into<PathBuf>) -> Result<Self, ToolCallError> {
        let workspace_root = workspace_root.into();
        let repo_root = treeproof::git::repo_root(&workspace_root)
            .map_err(|e| runtime_error(format!("not a git repository: {e}")))?;
        // The committed policy, not the working copy the agent can edit.
        let config = treeproof::config::Config::load_committed(&repo_root)
            .map_err(|e| runtime_error(e.to_string()))?;
        let suggested = treeproof::gate::suggest_command(&repo_root, &config);
        if suggested.starts_with('<') {
            return Err(runtime_error(format!(
                "no test command detected in {}; use RunTests::new with an explicit command",
                repo_root.display()
            )));
        }
        let command = suggested.split_whitespace().map(String::from).collect();
        Ok(Self::new(workspace_root, command))
    }

    /// Kill the test command after `timeout` (default 30 minutes); `None` waits forever.
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// Keep receipts and the signing key in the locations of `context` instead of the
    /// defaults (`TREEPROOF_CACHE_DIR` / `TREEPROOF_KEY_FILE`, then the user directories).
    pub fn with_context(mut self, context: treeproof::Context) -> Self {
        self.context = context;
        self
    }

    /// The command this tool runs.
    pub fn command(&self) -> &[String] {
        &self.command
    }
}

#[async_trait]
impl ToolRuntime for RunTests {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let RunTestsArgs { no_cache } = serde_json::from_value(args)?;
        if self.command.is_empty() {
            return Err(runtime_error("RunTests has no command configured"));
        }
        debug!(
            "run_tests: {:?} in {}",
            self.command,
            self.workspace_root.display()
        );

        let options =
            treeproof::run::RunOptions::new(self.command.clone(), self.workspace_root.clone())
                .with_no_cache(no_cache.unwrap_or(false))
                .with_timeout(self.timeout);
        let context = self.context.clone();
        let outcome =
            tokio::task::spawn_blocking(move || treeproof::run::run_in(&options, &context))
                .await
                .map_err(|e| runtime_error(format!("test task failed: {e}")))?
                .map_err(|e| runtime_error(e.to_string()))?;

        // The last OUTPUT_TAIL_CHARS characters, without copying the whole log.
        let output_tail: &str = match outcome
            .output
            .char_indices()
            .rev()
            .nth(OUTPUT_TAIL_CHARS - 1)
        {
            Some((start, _)) => &outcome.output[start..],
            None => &outcome.output,
        };
        Ok(json!({
            "success": outcome.passed(),
            "result": outcome.result,
            "reason": outcome.reason,
            "status": outcome.status,
            "tree": outcome.tree,
            "command": self.command.join(" "),
            "passed": outcome.counts.passed,
            "failed": outcome.counts.failed,
            "skipped": outcome.counts.skipped,
            "notes": outcome.notes,
            "receipt_path": outcome.receipt_path.display().to_string(),
            "output_tail": output_tail,
        }))
    }
}
