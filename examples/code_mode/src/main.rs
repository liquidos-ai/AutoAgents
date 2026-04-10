mod tools;

use crate::tools::{AddNumbers, MultiplyNumbers};
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{
    CodeActAgent, CodeActAgentOutput, CodeActExecutionRecord,
};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentOutputT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use autoagents_derive::{AgentHooks, agent};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

const DEFAULT_PROMPT: &str = "What is the current time right now?";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CodeModeOutput {
    result: String,
    executions: Vec<CodeActExecutionRecord>,
}

//This enforces the output to be string only
impl AgentOutputT for CodeModeOutput {
    fn output_schema() -> &'static str {
        String::output_schema()
    }

    fn structured_output_format() -> Value {
        String::structured_output_format()
    }
}

impl From<CodeActAgentOutput> for CodeModeOutput {
    fn from(value: CodeActAgentOutput) -> Self {
        Self {
            result: value.response,
            executions: value.executions,
        }
    }
}

#[agent(
    name = "code_mode_agent",
    description = "You are a general-purpose CodeAct assistant. Solve user requests by writing one concise TypeScript script. Use standard JavaScript globals such as Date, Math, JSON, Array, and string utilities for generic tasks. Use the provided tools when they make the solution clearer, especially for arithmetic. Log important intermediate values with console.log when useful. Return a plain string answer, not JSON. The script must return its final value from the top level with `return ...;` or a trailing expression. Imports are not available in the sandbox.",
    tools = [AddNumbers, MultiplyNumbers],
    output = CodeModeOutput,
)]
#[derive(Clone, Default, AgentHooks)]
struct CodeModeAgent;

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), Error> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let prompt_args: Vec<String> = std::env::args().skip(1).collect();
    let prompt = if prompt_args.is_empty() {
        DEFAULT_PROMPT.to_string()
    } else {
        prompt_args.join(" ")
    };

    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("gpt-4o")
        .build()
        .expect("failed to build OpenAI provider");

    let memory = Box::new(SlidingWindowMemory::new(8));
    let agent = CodeActAgent::new(CodeModeAgent);
    let handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(memory)
        .build()
        .await?;

    println!("Prompt:\n{}\n", prompt);

    let result: CodeModeOutput = handle.agent.run(Task::new(prompt)).await?;

    println!("Response");
    println!("  {}", result.result);
    print_execution_trace(&result.executions);

    Ok(())
}

fn print_execution_trace(executions: &[CodeActExecutionRecord]) {
    println!("\nCodeAct Trace");
    println!("  Sandbox executions: {}", executions.len());

    for execution in executions {
        println!(
            "\n  - {} | success={} | tool_calls={} | duration={}ms",
            execution.execution_id,
            execution.success,
            execution.tool_calls.len(),
            execution.duration_ms,
        );

        if !execution.console.is_empty() {
            println!("    Console:");
            for line in &execution.console {
                println!("      {line}");
            }
        }

        if !execution.tool_calls.is_empty() {
            println!("    Tool Calls:");
            for tool_call in &execution.tool_calls {
                let arguments = format_json_value(&tool_call.arguments);
                let result = format_json_value(&tool_call.result);
                println!(
                    "      {} | success={} | args={} | result={}",
                    tool_call.tool_name, tool_call.success, arguments, result,
                );
            }
        }

        if let Some(result) = &execution.result {
            println!("    Result: {}", format_json_value(result));
        }

        if let Some(error) = &execution.error {
            println!("    Error: {error}");
        }

        if !execution.source.is_empty() {
            println!("    Source:");
            for line in execution.source.lines() {
                println!("      {line}");
            }
        }
    }
}

fn format_json_value(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<invalid json>".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents::protocol::ToolCallResult;
    use serde_json::json;

    fn sample_execution() -> CodeActExecutionRecord {
        CodeActExecutionRecord {
            execution_id: "exec-1".to_string(),
            source: "return 42;".to_string(),
            console: vec!["starting".to_string(), "finished".to_string()],
            tool_calls: vec![ToolCallResult {
                tool_name: "AddNumbers".to_string(),
                success: true,
                arguments: json!({"left": 20, "right": 22}),
                result: json!(42),
            }],
            result: Some(json!(42)),
            success: true,
            error: Some("handled warning".to_string()),
            duration_ms: 13,
        }
    }

    #[test]
    fn code_mode_output_preserves_executor_response_and_trace() {
        let executions = vec![sample_execution()];
        let output = CodeModeOutput::from(CodeActAgentOutput {
            response: "42".to_string(),
            executions: executions.clone(),
            done: true,
        });

        assert_eq!(output.result, "42");
        assert_eq!(output.executions.len(), 1);
        assert_eq!(
            output.executions[0].execution_id,
            executions[0].execution_id
        );
    }

    #[test]
    fn execution_trace_formatting_handles_all_optional_sections() {
        let execution = sample_execution();
        assert_eq!(
            format_json_value(&json!({"answer": 42})),
            r#"{"answer":42}"#
        );
        assert_eq!(format_json_value(&json!("done")), r#""done""#);

        print_execution_trace(&[execution]);
    }
}
