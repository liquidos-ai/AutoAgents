use autoagents::core::agent::{AgentBuilder, AgentExecutor, ExecutorConfig, StreamingAgentExecutor};
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{Runtime, SingleThreadedRuntime};
use autoagents::core::tool::{ToolRuntime, ToolT};
use autoagents::llm::LLMProvider;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

// Simple tool for demonstration
#[derive(Debug)]
struct SimpleTool;

#[async_trait::async_trait]
impl ToolRuntime for SimpleTool {
    fn execute(&self, _args: Value) -> Result<Value, autoagents::core::tool::ToolCallError> {
        Ok(serde_json::json!({"result": "Tool executed successfully!"}))
    }
}

#[async_trait::async_trait]
impl ToolT for SimpleTool {
    fn name(&self) -> &'static str {
        "simple_tool"
    }

    fn description(&self) -> &'static str {
        "A simple tool for demonstration"
    }

    fn args_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"]
        })
    }

    fn run(&self, args: Value) -> Result<Value, autoagents::core::tool::ToolCallError> {
        self.execute(args)
    }
}

// Simple agent output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleOutput {
    message: String,
}

impl From<SimpleOutput> for Value {
    fn from(output: SimpleOutput) -> Self {
        serde_json::json!({
            "message": output.message,
        })
    }
}

impl autoagents::core::agent::AgentOutputT for SimpleOutput {
    fn output_schema() -> &'static str {
        "SimpleOutput"
    }

    fn structured_output_format() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"]
        })
    }
}

// Simple agent
#[derive(Debug)]
pub struct SimpleAgent;

#[async_trait::async_trait]
impl AgentExecutor for SimpleAgent {
    type Output = SimpleOutput;
    type Error = Error;

    fn config(&self) -> autoagents::core::agent::ExecutorConfig {
        ExecutorConfig::default()
    }

    async fn execute(
        &self,
        _llm: Arc<dyn LLMProvider>,
        _memory: Option<Arc<tokio::sync::RwLock<Box<dyn autoagents::core::memory::MemoryProvider>>>>,
        _tools: Vec<Box<dyn ToolT>>,
        _agent_config: &autoagents::core::agent::AgentConfig,
        task: autoagents::core::runtime::Task,
        _state: Arc<tokio::sync::RwLock<autoagents::core::agent::AgentState>>,
        _tx_event: tokio::sync::mpsc::Sender<Event>,
    ) -> Result<Self::Output, Self::Error> {
        Ok(SimpleOutput {
            message: format!("Processed: {}", task.prompt),
        })
    }
}

#[async_trait::async_trait]
impl StreamingAgentExecutor for SimpleAgent {
    fn supports_streaming(&self) -> bool {
        true
    }
}

#[async_trait::async_trait]
impl autoagents::core::agent::AgentDeriveT for SimpleAgent {
    type Output = SimpleOutput;

    fn description(&self) -> &'static str {
        "A simple agent for demonstration"
    }

    fn output_schema(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"]
        }))
    }

    fn name(&self) -> &'static str {
        "simple_agent"
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        vec![Box::new(SimpleTool)]
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    println!("🚀 Streaming Demo - AutoAgents Framework");
    println!("This demo shows streaming functionality with tool calling\n");

    // Create a mock LLM provider
    let llm = Arc::new(MockLLMProvider);
    
    // Create memory
    let memory = Box::new(SlidingWindowMemory::new(10));
    
    // Create runtime
    let runtime = SingleThreadedRuntime::new(None);
    
    // Create the simple agent
    let simple_agent = SimpleAgent;
    
    // Build the agent
    let agent = AgentBuilder::new(simple_agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .with_memory(memory)
        .build()
        .await?;

    println!("✅ Agent created successfully");
    println!("📊 Agent supports streaming: {}", agent.supports_streaming());
    
    // Create environment
    let mut environment = autoagents::core::environment::Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await?;

    // Get event receiver
    let receiver = environment.take_event_receiver(None).await?;
    
    // Handle events in a separate task
    let event_handle = tokio::spawn(async move {
        handle_events(receiver).await;
    });

    // Send a test message
    println!("\n📝 Sending test message...");
    runtime
        .send_message("Hello, this is a test message!".to_string(), agent.id())
        .await?;

    // Run the environment
    environment.run().await;
    
    // Wait for events to be processed
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    
    // Shutdown
    environment.shutdown().await;
    
    // Wait for event handler to finish
    let _ = event_handle.await;

    println!("\n✅ Demo completed successfully!");
    Ok(())
}

async fn handle_events(mut event_stream: ReceiverStream<Event>) {
    while let Some(event) = event_stream.next().await {
        match event {
            Event::TaskStarted {
                agent_id,
                task_description,
                ..
            } => {
                println!(
                    "📋 Task Started - Agent: {:?}, Task: {}",
                    agent_id, task_description
                );
            }
            Event::StreamTextChunk { chunk, is_final, .. } => {
                print!("{}", chunk);
                if is_final {
                    println!();
                }
            }
            Event::StreamToolCallStart { tool_call, .. } => {
                println!(
                    "\n🔧 Tool Call Started: {} with args: {}",
                    tool_call.function.name, tool_call.function.arguments
                );
            }
            Event::StreamToolCallEnd { tool_call_id, .. } => {
                println!("✅ Tool Call Completed: {}", tool_call_id);
            }
            Event::ToolCallRequested {
                tool_name,
                arguments,
                ..
            } => {
                println!(
                    "🔧 Tool Call Requested: {} with args: {}",
                    tool_name, arguments
                );
            }
            Event::ToolCallCompleted { tool_name, result, .. } => {
                println!(
                    "✅ Tool Call Completed: {} - Result: {:?}",
                    tool_name, result
                );
            }
            Event::TaskComplete { result, .. } => {
                match result {
                    TaskResult::Value(val) => {
                        println!("\n🎉 Task completed with result: {:?}", val);
                    }
                    TaskResult::Failure(error) => {
                        println!("\n❌ Task failed with error: {}", error);
                    }
                    TaskResult::Aborted => {
                        println!("\n⏹️ Task was aborted");
                    }
                }
            }
            _ => {
                // Ignore other events for this demo
            }
        }
    }
}

// Mock LLM provider for demonstration
#[derive(Debug)]
struct MockLLMProvider;

#[async_trait::async_trait]
impl autoagents::llm::chat::ChatProvider for MockLLMProvider {
    async fn chat_with_tools(
        &self,
        _messages: &[autoagents::llm::chat::ChatMessage],
        _tools: Option<&[autoagents::llm::chat::Tool]>,
        _json_schema: Option<autoagents::llm::chat::StructuredOutputFormat>,
    ) -> Result<Box<dyn autoagents::llm::chat::ChatResponse>, autoagents::llm::error::LLMError> {
        // Mock response with tool calls
        let tool_call = autoagents::llm::ToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: autoagents::llm::FunctionCall {
                name: "simple_tool".to_string(),
                arguments: r#"{"message": "Hello from tool!"}"#.to_string(),
            },
        };

        Ok(Box::new(MockChatResponse {
            text: Some("I'll help you with that.".to_string()),
            tool_calls: Some(vec![tool_call]),
        }))
    }
}

#[async_trait::async_trait]
impl autoagents::llm::completion::CompletionProvider for MockLLMProvider {
    async fn complete(
        &self,
        _req: &autoagents::llm::completion::CompletionRequest,
        _json_schema: Option<autoagents::llm::chat::StructuredOutputFormat>,
    ) -> Result<autoagents::llm::completion::CompletionResponse, autoagents::llm::error::LLMError> {
        Ok(autoagents::llm::completion::CompletionResponse {
            text: "Mock completion response".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl autoagents::llm::embedding::EmbeddingProvider for MockLLMProvider {
    async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, autoagents::llm::error::LLMError> {
        Ok(vec![vec![0.1, 0.2, 0.3]])
    }
}

impl autoagents::llm::models::ModelsProvider for MockLLMProvider {}

impl autoagents::llm::LLMProvider for MockLLMProvider {
    fn tools(&self) -> Option<&[autoagents::llm::chat::Tool]> {
        None
    }
}

#[derive(Debug)]
struct MockChatResponse {
    text: Option<String>,
    tool_calls: Option<Vec<autoagents::llm::ToolCall>>,
}

impl autoagents::llm::chat::ChatResponse for MockChatResponse {
    fn text(&self) -> Option<String> {
        self.text.clone()
    }

    fn tool_calls(&self) -> Option<Vec<autoagents::llm::ToolCall>> {
        self.tool_calls.clone()
    }
}

impl std::fmt::Display for MockChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MockChatResponse")
    }
} 