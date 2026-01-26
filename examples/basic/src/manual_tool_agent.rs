use autoagents::async_trait;
/// This example demonstrates creating tools and agents without macros and is helpful when dynamic tool instance creation is required
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgent, ReActAgentOutput};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentHooks, AgentOutputT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::core::tool::{ToolCallError, ToolRuntime, ToolT, shared_tools_to_boxes};
use autoagents::llm::LLMProvider;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

// Creat AdditionToolInput Struct for parsing the args
#[derive(Debug, Serialize, Deserialize)]
pub struct AdditionToolInput {
    left: i32,
    right: i32,
}

#[derive(Debug)]
pub struct AdditionTool {}

// Define the ToolRuntime for AdditionTool
#[async_trait]
impl ToolRuntime for AdditionTool {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        println!("Running addition tool");
        let typed_args: AdditionToolInput = serde_json::from_value(args)?;
        let result = typed_args.left + typed_args.right;
        Ok(result.into())
    }
}

// Define the Tool Trait for Addition Tool
impl ToolT for AdditionTool {
    fn name(&self) -> &'static str {
        "AdditionTool"
    }

    fn description(&self) -> &'static str {
        "Use this tool for Adding two numbers"
    }

    // The schema should be valid as the the LLM instance tool calling api
    fn args_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "left": {"type": "number"},
                "right": {"type": "number"}
            },
            "required": ["left", "right"]
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubtractToolInput {
    left: i32,
    right: i32,
}

#[derive(Debug)]
pub struct SubtractTool {}

#[async_trait]
impl ToolRuntime for SubtractTool {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        println!("Running subtract tool");
        let typed_args: SubtractToolInput = serde_json::from_value(args)?;
        let result = typed_args.left - typed_args.right;
        Ok(result.into())
    }
}

impl ToolT for SubtractTool {
    fn name(&self) -> &'static str {
        "SubtractTool"
    }

    fn description(&self) -> &'static str {
        "Use this tool for Subtracting two numbers"
    }

    fn args_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "left": {"type": "number"},
                "right": {"type": "number"}
            },
            "required": ["left", "right"]
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOut {
    pub out: String,
}

// Implement trait for structured json output for agent, this is not required if you want string
impl AgentOutputT for AgentOut {
    fn output_schema() -> &'static str {
        r#"{"type":"object","properties":{"out":{"type":"string"}},"required":["out"]}"#
    }

    fn structured_output_format() -> Value {
        json!({
            "name": "AgentOut",
            "description": "agent output schema",
            "schema": {
                "type": "object",
                "properties": {
                    "out": {"type": "string"}
                },
                "required": ["out"]
            },
            "strict": true
        })
    }
}

// Implement conversion from ReActAgent executor to the current Agent output structure
impl From<ReActAgentOutput> for AgentOut {
    fn from(value: ReActAgentOutput) -> Self {
        if let Ok(value) = serde_json::from_str::<AgentOut>(&value.response) {
            value
        } else {
            AgentOut {
                out: "Error".to_string(),
            }
        }
    }
}

#[derive(Default, Debug)]
pub struct Agent {
    tools: Vec<Arc<dyn ToolT>>,
}

// implement the Agent Trait
impl AgentDeriveT for Agent {
    type Output = AgentOut;
    fn description(&self) -> &'static str {
        "You are an helpful assistant"
    }

    fn name(&self) -> &'static str {
        "Agent"
    }

    //If the value returned is None then its considered as String
    fn output_schema(&self) -> Option<Value> {
        Some(AgentOut::structured_output_format())
    }

    // In this example as the agent have dynamic tools and ToolT does not have clone you can use this helper method for easy tool creation
    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        shared_tools_to_boxes(&self.tools)
    }
}

impl AgentHooks for Agent {}

pub async fn run_agent(llm: Arc<dyn LLMProvider>, mode: &str) -> Result<(), Error> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    // Dynamically insert the tools at runtime
    let agent: ReActAgent<Agent> = if mode == "addition" {
        let addition_tool = Arc::new(AdditionTool {});
        ReActAgent::new(Agent {
            tools: vec![addition_tool],
        })
    } else {
        let subtract_tool = Arc::new(SubtractTool {});
        ReActAgent::new(Agent {
            tools: vec![subtract_tool],
        })
    };
    let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    println!("Running basic agent with direct run method");
    let result = agent_handle
        .agent
        .run(Task::new("What is 20 + 10?"))
        .await?;
    println!("Result: {:?}", result.out);
    Ok(())
}
