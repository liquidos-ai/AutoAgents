use crate::agent::task::Task;
use crate::agent::{AgentDeriveT, Context};
use crate::tool::ToolCallResult;
use async_trait::async_trait;
use autoagents_llm::ToolCall;
use serde_json::Value;

#[derive(PartialEq)]
pub enum HookOutcome {
    Continue,
    Abort,
}

#[async_trait]
pub trait AgentHooks: AgentDeriveT + Send + Sync {
    /// Hook called when builder creates a new instance of BaseAgent
    async fn on_agent_create(&self) {}
    /// Called when the Agent Execution is Triggered, Ability to Abort is Given for users
    async fn on_run_start(&self, _task: &Task, _ctx: &Context) -> HookOutcome {
        HookOutcome::Continue
    }
    /// Called when the Agent Execution is Completed
    async fn on_run_complete(&self, _task: &Task, _result: &Self::Output, _ctx: &Context) {}
    /// Called when an executor turn is started, useful for multi-turn Executors like ReAct
    async fn on_turn_start(&self, _turn_index: usize, _ctx: &Context) {}
    /// Called when an executor turn is completed, useful for multi-turn Executors like ReAct
    async fn on_turn_complete(&self, _turn_index: usize, _ctx: &Context) {}
    //. Run the hook before executing the tool_call giving ability to Abort or Continue
    async fn on_tool_call(&self, _tool_call: &ToolCall, _ctx: &Context) -> HookOutcome {
        HookOutcome::Continue
    }
    /// Called before executing the tool
    async fn on_tool_start(&self, _tool_call: &ToolCall, _ctx: &Context) {}
    /// Called post execution of tool with results
    async fn on_tool_result(
        &self,
        _tool_call: &ToolCall,
        _result: &ToolCallResult,
        _ctx: &Context,
    ) {
    }
    /// Called if the execution of the tool failed
    async fn on_tool_error(&self, _tool_call: &ToolCall, _err: Value, _ctx: &Context) {}
    /// Called when an Actor Agent post-shutdown, This has no effect on DirectAgent, It only works for ActorBased Agents
    async fn on_agent_shutdown(&self) {}
}
