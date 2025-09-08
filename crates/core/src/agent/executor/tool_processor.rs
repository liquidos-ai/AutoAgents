use crate::protocol::Event;
use crate::tool::{ToolCallResult, ToolT};
use autoagents_llm::{FunctionCall, ToolCall};
use serde_json::Value;

#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc;

#[cfg(target_arch = "wasm32")]
use futures::channel::mpsc;

use crate::agent::{AgentHooks, Context, HookOutcome};
#[cfg(target_arch = "wasm32")]
use futures::SinkExt;

/// Handles all tool-related operations in a centralized manner
pub struct ToolProcessor;

impl ToolProcessor {
    /// Process multiple tool calls and return results
    pub async fn process_tool_calls(
        tools: &[Box<dyn ToolT>],
        tool_calls: Vec<ToolCall>,
        tx_event: Option<mpsc::Sender<Event>>,
    ) -> Vec<ToolCallResult> {
        let mut results = Vec::new();

        for call in &tool_calls {
            let result = Self::process_single_tool_call(tools, call, &tx_event).await;
            results.push(result);
        }

        results
    }

    /// Process a single tool call (with hooks)
    pub(crate) async fn process_single_tool_call_with_hooks<H: AgentHooks>(
        hooks: &H,
        context: &Context,
        tools: &[Box<dyn ToolT>],
        call: &ToolCall,
        tx_event: &Option<mpsc::Sender<Event>>,
    ) -> Option<ToolCallResult> {
        // Run hook before execution
        match hooks.on_tool_call(call, context).await {
            HookOutcome::Abort => {
                return None; // skip execution
            }
            HookOutcome::Continue => {}
        }

        let tool_name = call.function.name.clone();
        let tool_args = call.function.arguments.clone();

        // Send tool call requested event
        Self::send_event(
            tx_event,
            Event::ToolCallRequested {
                id: call.id.clone(),
                tool_name: tool_name.clone(),
                arguments: tool_args.clone(),
            },
        )
        .await;

        //Run the tool start hook
        hooks.on_tool_start(call, context).await;

        let result = Self::process_single_tool_call(tools, call, tx_event).await;

        //Run on tool result hook
        if result.success {
            hooks.on_tool_result(call, &result, context).await;
        } else {
            hooks
                .on_tool_error(call, result.result.clone(), context)
                .await;
        }

        Some(result)
    }

    /// Process a single tool call
    pub(crate) async fn process_single_tool_call(
        tools: &[Box<dyn ToolT>],
        call: &ToolCall,
        tx_event: &Option<mpsc::Sender<Event>>,
    ) -> ToolCallResult {
        let tool_name = call.function.name.clone();
        let tool_args = call.function.arguments.clone();

        // Send tool call requested event
        Self::send_event(
            tx_event,
            Event::ToolCallRequested {
                id: call.id.clone(),
                tool_name: tool_name.clone(),
                arguments: tool_args.clone(),
            },
        )
        .await;

        // Find and execute the tool
        let result = match tools.iter().find(|t| t.name() == tool_name) {
            Some(tool) => Self::execute_tool(tool.as_ref(), &tool_name, &tool_args),
            None => Self::create_error_result(
                &tool_name,
                &tool_args,
                &format!("Tool '{tool_name}' not found"),
            ),
        };

        // Send completion or failure event
        Self::send_tool_result_event(tx_event, call, &result).await;

        result
    }

    /// Execute a tool and return the result
    fn execute_tool(tool: &dyn ToolT, tool_name: &str, tool_args: &str) -> ToolCallResult {
        match serde_json::from_str::<Value>(tool_args) {
            Ok(parsed_args) => match tool.execute(parsed_args) {
                Ok(output) => ToolCallResult {
                    tool_name: tool_name.to_string(),
                    success: true,
                    arguments: serde_json::from_str(tool_args).unwrap_or(Value::Null),
                    result: output,
                },
                Err(e) => Self::create_error_result(
                    tool_name,
                    tool_args,
                    &format!("Tool execution failed: {e}"),
                ),
            },
            Err(e) => Self::create_error_result(
                tool_name,
                tool_args,
                &format!("Failed to parse arguments: {e}"),
            ),
        }
    }

    /// Create an error result for tool execution
    fn create_error_result(tool_name: &str, tool_args: &str, error: &str) -> ToolCallResult {
        ToolCallResult {
            tool_name: tool_name.to_string(),
            success: false,
            arguments: serde_json::from_str(tool_args).unwrap_or(Value::Null),
            result: serde_json::json!({"error": error}),
        }
    }

    /// Send an event if tx is available
    async fn send_event(tx: &Option<mpsc::Sender<Event>>, event: Event) {
        if let Some(tx) = tx {
            #[cfg(not(target_arch = "wasm32"))]
            let _ = tx.send(event).await;
        }
    }

    /// Send tool result event (success or failure)
    async fn send_tool_result_event(
        tx: &Option<mpsc::Sender<Event>>,
        call: &ToolCall,
        result: &ToolCallResult,
    ) {
        let event = if result.success {
            Event::ToolCallCompleted {
                id: call.id.clone(),
                tool_name: result.tool_name.clone(),
                result: result.result.clone(),
            }
        } else {
            Event::ToolCallFailed {
                id: call.id.clone(),
                tool_name: result.tool_name.clone(),
                error: result.result.to_string(),
            }
        };

        Self::send_event(tx, event).await;
    }

    /// Convert tool results to tool calls for memory storage
    pub fn create_result_tool_calls(
        tool_calls: &[ToolCall],
        tool_results: &[ToolCallResult],
    ) -> Vec<ToolCall> {
        tool_calls
            .iter()
            .zip(tool_results)
            .map(|(call, result)| {
                let result_content = Self::extract_result_content(result);
                ToolCall {
                    id: call.id.clone(),
                    call_type: call.call_type.clone(),
                    function: FunctionCall {
                        name: call.function.name.clone(),
                        arguments: result_content,
                    },
                }
            })
            .collect()
    }

    /// Extract content from tool result
    fn extract_result_content(result: &ToolCallResult) -> String {
        if result.success {
            match &result.result {
                Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            }
        } else {
            serde_json::json!({"error": format!("{:?}", result.result)}).to_string()
        }
    }
}
