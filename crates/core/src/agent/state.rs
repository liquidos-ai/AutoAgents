use crate::agent::task::Task;
use crate::tool::ToolCallResult;

/// State tracking for agent execution
#[derive(Debug, Default, Clone)]
pub struct AgentState {
    /// Tool calls made during execution
    pub tool_calls: Vec<ToolCallResult>,
    /// Tasks that have been executed
    pub task_history: Vec<Task>,
}

impl AgentState {
    pub fn new() -> Self {
        Self {
            tool_calls: vec![],
            task_history: vec![],
        }
    }

    pub fn record_tool_call(&mut self, tool_call: ToolCallResult) {
        self.tool_calls.push(tool_call);
    }

    pub fn record_task(&mut self, task: Task) {
        self.task_history.push(task);
    }
}
