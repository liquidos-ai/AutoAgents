use crate::tool::{ToolCallResult, ToolT};
use autoagents_llm::{FunctionCall, ToolCall};
use autoagents_protocol::{ActorID, Event, SubmissionId};
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

#[derive(Debug, Clone, Copy)]
pub struct ToolCallContext {
    pub sub_id: SubmissionId,
    pub actor_id: ActorID,
}

impl ToolCallContext {
    pub fn new(sub_id: SubmissionId, actor_id: ActorID) -> Self {
        Self { sub_id, actor_id }
    }
}

impl ToolProcessor {
    /// Process multiple tool calls and return results
    pub async fn process_tool_calls(
        tools: &[Box<dyn ToolT>],
        tool_calls: Vec<ToolCall>,
        context: ToolCallContext,
        tx_event: Option<mpsc::Sender<Event>>,
    ) -> Vec<ToolCallResult> {
        let mut results = Vec::new();

        for call in &tool_calls {
            let result = Self::process_single_tool_call(tools, call, context, &tx_event).await;
            results.push(result);
        }

        results
    }

    /// Process a single tool call (with hooks)
    pub(crate) async fn process_single_tool_call_with_hooks<H: AgentHooks>(
        hooks: &H,
        context: &Context,
        submission_id: SubmissionId,
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

        //Run the tool start hook
        hooks.on_tool_start(call, context).await;

        let tool_context = ToolCallContext::new(submission_id, context.config().id);
        let result = Self::process_single_tool_call(tools, call, tool_context, tx_event).await;

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
        context: ToolCallContext,
        tx_event: &Option<mpsc::Sender<Event>>,
    ) -> ToolCallResult {
        let tool_name = call.function.name.clone();
        let tool_args = call.function.arguments.clone();

        // Send tool call requested event
        Self::send_event(
            tx_event,
            Event::ToolCallRequested {
                sub_id: context.sub_id,
                actor_id: context.actor_id,
                id: call.id.clone(),
                tool_name: tool_name.clone(),
                arguments: tool_args.clone(),
            },
        )
        .await;

        // Find and execute the tool
        let result = match tools.iter().find(|t| t.name() == tool_name) {
            Some(tool) => Self::execute_tool(tool.as_ref(), &tool_name, &tool_args).await,
            None => Self::create_error_result(
                &tool_name,
                &tool_args,
                &format!("Tool '{tool_name}' not found"),
            ),
        };

        // Send completion or failure event
        Self::send_tool_result_event(tx_event, call, &result, context).await;

        result
    }

    /// Execute a tool and return the result
    async fn execute_tool(tool: &dyn ToolT, tool_name: &str, tool_args: &str) -> ToolCallResult {
        match serde_json::from_str::<Value>(tool_args) {
            Ok(parsed_args) => match tool.execute(parsed_args).await {
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
        context: ToolCallContext,
    ) {
        let event = if result.success {
            Event::ToolCallCompleted {
                sub_id: context.sub_id,
                actor_id: context.actor_id,
                id: call.id.clone(),
                tool_name: result.tool_name.clone(),
                result: result.result.clone(),
            }
        } else {
            Event::ToolCallFailed {
                sub_id: context.sub_id,
                actor_id: context.actor_id,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{ToolCallError, ToolRuntime, ToolT};
    use async_trait::async_trait;
    use serde_json::json;

    #[derive(Debug)]
    struct MockTool {
        name: String,
        should_fail: bool,
    }

    impl MockTool {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                should_fail: false,
            }
        }
        fn failing(name: &str) -> Self {
            Self {
                name: name.to_string(),
                should_fail: true,
            }
        }
    }

    impl ToolT for MockTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "mock"
        }
        fn args_schema(&self) -> Value {
            json!({"type": "object"})
        }
    }

    #[async_trait]
    impl ToolRuntime for MockTool {
        async fn execute(&self, _args: Value) -> Result<Value, ToolCallError> {
            if self.should_fail {
                return Err(ToolCallError::RuntimeError("fail".to_string().into()));
            }
            Ok(json!({"ok": true}))
        }
    }

    fn make_tool_call(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    #[tokio::test]
    async fn test_process_tool_calls_valid_tool() {
        let tools: Vec<Box<dyn ToolT>> = vec![Box::new(MockTool::new("tool_a"))];
        let calls = vec![make_tool_call("1", "tool_a", r#"{"x":1}"#)];
        let ctx = ToolCallContext::new(
            autoagents_protocol::SubmissionId::new_v4(),
            autoagents_protocol::ActorID::new_v4(),
        );
        let results = ToolProcessor::process_tool_calls(&tools, calls, ctx, None).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
        assert_eq!(results[0].tool_name, "tool_a");
    }

    #[tokio::test]
    async fn test_process_tool_calls_tool_not_found() {
        let tools: Vec<Box<dyn ToolT>> = vec![Box::new(MockTool::new("tool_a"))];
        let calls = vec![make_tool_call("1", "nonexistent", r#"{}"#)];
        let ctx = ToolCallContext::new(
            autoagents_protocol::SubmissionId::new_v4(),
            autoagents_protocol::ActorID::new_v4(),
        );
        let results = ToolProcessor::process_tool_calls(&tools, calls, ctx, None).await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert!(results[0].result.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_process_tool_calls_multiple_tools() {
        let tools: Vec<Box<dyn ToolT>> = vec![
            Box::new(MockTool::new("tool_a")),
            Box::new(MockTool::new("tool_b")),
        ];
        let calls = vec![
            make_tool_call("1", "tool_a", r#"{}"#),
            make_tool_call("2", "tool_b", r#"{}"#),
        ];
        let ctx = ToolCallContext::new(
            autoagents_protocol::SubmissionId::new_v4(),
            autoagents_protocol::ActorID::new_v4(),
        );
        let results = ToolProcessor::process_tool_calls(&tools, calls, ctx, None).await;
        assert_eq!(results.len(), 2);
        assert!(results[0].success);
        assert!(results[1].success);
    }

    #[tokio::test]
    async fn test_process_single_tool_call_failure() {
        let tools: Vec<Box<dyn ToolT>> = vec![Box::new(MockTool::failing("fail_tool"))];
        let call = make_tool_call("1", "fail_tool", r#"{}"#);
        let ctx = ToolCallContext::new(
            autoagents_protocol::SubmissionId::new_v4(),
            autoagents_protocol::ActorID::new_v4(),
        );
        let result = ToolProcessor::process_single_tool_call(&tools, &call, ctx, &None).await;
        assert!(!result.success);
        assert!(result.result.to_string().contains("execution failed"));
    }

    #[tokio::test]
    async fn test_process_single_tool_call_invalid_args() {
        let tools: Vec<Box<dyn ToolT>> = vec![Box::new(MockTool::new("tool_a"))];
        let call = make_tool_call("1", "tool_a", "not json");
        let ctx = ToolCallContext::new(
            autoagents_protocol::SubmissionId::new_v4(),
            autoagents_protocol::ActorID::new_v4(),
        );
        let result = ToolProcessor::process_single_tool_call(&tools, &call, ctx, &None).await;
        assert!(!result.success);
        assert!(result.result.to_string().contains("parse arguments"));
    }

    #[test]
    fn test_create_result_tool_calls() {
        let calls = vec![make_tool_call("c1", "tool_a", r#"{"x":1}"#)];
        let results = vec![ToolCallResult {
            tool_name: "tool_a".to_string(),
            success: true,
            arguments: json!({"x": 1}),
            result: json!("done"),
        }];
        let result_calls = ToolProcessor::create_result_tool_calls(&calls, &results);
        assert_eq!(result_calls.len(), 1);
        assert_eq!(result_calls[0].id, "c1");
        assert_eq!(result_calls[0].function.arguments, "done");
    }

    #[test]
    fn test_extract_result_content_string() {
        let result = ToolCallResult {
            tool_name: "t".to_string(),
            success: true,
            arguments: json!({}),
            result: Value::String("hello".to_string()),
        };
        let content = ToolProcessor::extract_result_content(&result);
        assert_eq!(content, "hello");
    }

    #[test]
    fn test_extract_result_content_object() {
        let result = ToolCallResult {
            tool_name: "t".to_string(),
            success: true,
            arguments: json!({}),
            result: json!({"key": "value"}),
        };
        let content = ToolProcessor::extract_result_content(&result);
        assert!(content.contains("key"));
        assert!(content.contains("value"));
    }

    #[test]
    fn test_extract_result_content_failure() {
        let result = ToolCallResult {
            tool_name: "t".to_string(),
            success: false,
            arguments: json!({}),
            result: json!({"error": "bad"}),
        };
        let content = ToolProcessor::extract_result_content(&result);
        assert!(content.contains("error"));
    }

    #[tokio::test]
    async fn test_send_event_with_none_tx() {
        // Should not panic
        ToolProcessor::send_event(
            &None,
            autoagents_protocol::Event::StreamComplete {
                sub_id: autoagents_protocol::SubmissionId::new_v4(),
            },
        )
        .await;
    }
}
