use crate::agent::task::Task;
use crate::tool::ToolCallResult;
use autoagents_llm::chat::Usage;

/// State tracking for agent execution
#[derive(Debug, Default, Clone)]
pub struct AgentState {
    /// Tool calls made during execution
    pub tool_calls: Vec<ToolCallResult>,
    /// Tasks that have been executed
    pub task_history: Vec<Task>,
    pub total_usage: TokenUsage,
}

/// Aggregated token usage across agent's lifetime
#[derive(Debug, Default, Clone)]
pub struct TokenUsage {
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub total_tokens: u64,
    pub llm_call_count: u32,
    pub usage_history: Vec<Usage>,
}

impl TokenUsage {
    /// Calculate average tokens per LLM call
    pub fn average_tokens_per_call(&self) -> f64 {
        if self.llm_call_count == 0 {
            0.0
        } else {
            self.total_tokens as f64 / self.llm_call_count as f64
        }
    }

    /// Get the most recent usage record
    pub fn last_usage(&self) -> Option<&Usage> {
        self.usage_history.last()
    }

    /// Add usage from an LLM call
    pub fn add_usage(&mut self, usage: Usage) {
        self.total_prompt_tokens += usage.prompt_tokens as u64;
        self.total_completion_tokens += usage.completion_tokens as u64;
        self.total_tokens += usage.total_tokens as u64;
        self.llm_call_count += 1;
        self.usage_history.push(usage);
    }
}

impl AgentState {
    pub fn new() -> Self {
        Self {
            tool_calls: vec![],
            task_history: vec![],
            total_usage: TokenUsage::default(),
        }
    }

    pub fn record_tool_call(&mut self, tool_call: ToolCallResult) {
        self.tool_calls.push(tool_call);
    }

    pub fn record_task(&mut self, task: Task) {
        self.task_history.push(task);
    }

    // Record token usage from LLM response
    pub fn record_usage(&mut self, usage: Usage) {
        self.total_usage.add_usage(usage);
    }

    // Get current token usage
    pub fn get_usage(&self) -> &TokenUsage {
        &self.total_usage
    }

    // Reset token usage counters
    pub fn reset_usage(&mut self) {
        self.total_usage = TokenUsage::default();
    }
}
