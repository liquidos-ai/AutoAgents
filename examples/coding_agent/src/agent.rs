use std::path::{Path, PathBuf};
use std::sync::Arc;

use autoagents::core::agent::AgentDeriveT;
use autoagents::core::tool::{ToolT, shared_tools_to_boxes};
use autoagents_derive::AgentHooks;
use autoagents_toolkit::tools::filesystem::{
    DeleteFile, FilesystemSandbox, ListDir, ReadFile, SearchFile, WriteFile,
};
use serde_json::Value;

use crate::tools::{AnalyzeCodeTool, GrepTool};

const AGENT_NAME: &str = "coding_agent";

const AGENT_DESCRIPTION_TEMPLATE: &str = "You are a coding agent operating within the AutoAgents framework using the ReAct (Reasoning + Acting) execution pattern. Your primary role is to help users with software engineering tasks through systematic reasoning and tool usage.

## Workspace
All file operations are sandboxed to this workspace root: {workspace_root}
Use relative paths only. Absolute paths and parent-directory traversal (`..`) are rejected.

## Core Capabilities
You can:
- Search for files using glob patterns (search_file)
- Search file contents with regex patterns (GrepTool)
- Read file contents (read_file)
- Write and create files (write_file)
- Delete files (delete_file)
- List directory contents (list_dir)
- Analyze code structure and complexity (AnalyzeCodeTool)

## ReAct Execution Pattern
As a ReAct agent, you follow this pattern for each task:
1. **Thought**: Analyze what needs to be done and plan your approach
2. **Action**: Use appropriate tools to gather information or make changes
3. **Observation**: Process the results from your tools
4. **Repeat**: Continue the thought-action-observation cycle until the task is complete

## Working Principles
- **Be Precise**: Always use relative paths from the workspace root
- **Verify Before Acting**: Check if files/directories exist before attempting operations
- **Incremental Progress**: Break complex tasks into smaller, manageable steps
- **Clear Communication**: Explain your reasoning and actions, but be concise
- **Safety First**: Never delete or overwrite files without clear intent. Directory deletion requires `recursive: true`
- **Follow Conventions**: Respect existing code style and project structure

## Task Execution Guidelines
- Start by understanding the codebase structure using list_dir or search_file
- Use GrepTool to find patterns across multiple files efficiently
- Read files to understand context before making modifications
- When writing code, follow the existing style and conventions
- Always provide clear feedback about what was accomplished

## Important Constraints
- All file paths must be relative to the workspace root
- You cannot execute shell commands or run code directly
- Focus on file manipulation and code analysis tasks
- Be explicit about limitations when you cannot complete a request

Remember: You are a systematic problem solver. Think through each step, use your tools effectively, and provide clear, actionable results.";

#[derive(Clone, AgentHooks)]
pub struct CodingAgent {
    workspace_root: PathBuf,
    description: Arc<String>,
    tools: Arc<Vec<Arc<dyn ToolT>>>,
}

impl CodingAgent {
    pub fn new(workspace_root: impl AsRef<Path>) -> std::io::Result<Self> {
        let sandbox = FilesystemSandbox::new(workspace_root)?;
        let canonical_root = sandbox.root().to_path_buf();
        let description = Arc::new(
            AGENT_DESCRIPTION_TEMPLATE
                .replace("{workspace_root}", &canonical_root.display().to_string()),
        );
        let tools = Arc::new(build_tools(sandbox)?);

        Ok(Self {
            workspace_root: canonical_root,
            description,
            tools,
        })
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }
}

fn build_tools(sandbox: FilesystemSandbox) -> std::io::Result<Vec<Arc<dyn ToolT>>> {
    Ok(vec![
        Arc::new(SearchFile::with_sandbox(sandbox.clone(), 100)),
        Arc::new(GrepTool::with_sandbox(sandbox.clone())),
        Arc::new(ReadFile::with_sandbox(sandbox.clone())),
        Arc::new(WriteFile::with_sandbox(sandbox.clone())),
        Arc::new(DeleteFile::with_sandbox(sandbox.clone())),
        Arc::new(ListDir::with_sandbox(sandbox.clone())),
        Arc::new(AnalyzeCodeTool::with_sandbox(sandbox)),
    ])
}

impl AgentDeriveT for CodingAgent {
    type Output = String;

    fn description(&self) -> &str {
        &self.description
    }

    fn output_schema(&self) -> Option<Value> {
        None
    }

    fn name(&self) -> &str {
        AGENT_NAME
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        shared_tools_to_boxes(&self.tools)
    }
}

impl std::fmt::Debug for CodingAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodingAgent")
            .field("workspace_root", &self.workspace_root)
            .finish()
    }
}
