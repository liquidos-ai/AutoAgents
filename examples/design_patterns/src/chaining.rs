use autoagents::async_trait;
use autoagents::core::actor::Topic;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{ActorAgent, AgentBuilder, AgentHooks, Context};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::runtime::{SingleThreadedRuntime, TypedRuntime};
use autoagents::llm::LLMProvider;
use autoagents_derive::agent;
use std::sync::Arc;

/// First agent in the chain that extracts technical specifications from text
/// This agent parses natural language descriptions and identifies key technical details
#[agent(
    name = "agent_1",
    description = "Extract the technical specifications from the given text"
)]
pub struct Agent1 {}

#[async_trait]
impl AgentHooks for Agent1 {
    /// After completing the extraction, automatically forward the result to Agent2
    /// This creates a chain where Agent1's output becomes Agent2's input
    async fn on_run_complete(&self, _task: &Task, result: &Self::Output, ctx: &Context) {
        // Publish the extracted specifications to the next agent in the chain
        let _ = ctx
            .publish(Topic::<Task>::new("agent_2"), Task::new(result))
            .await;
    }
}

/// Second agent in the chain that transforms specifications into structured JSON
/// This agent takes the extracted specifications and formats them into a consistent structure
#[agent(
    name = "agent_2",
    description = "Transform the specifications provided into a valid JSON with 'cpu', 'memory', and 'storage' as keys"
)]
pub struct Agent2 {}

#[async_trait]
impl AgentHooks for Agent2 {
    /// Outputs the final result
    async fn on_run_complete(&self, _task: &Task, result: &Self::Output, _ctx: &Context) {
        println!("Final Results:\n {result}");
    }
}

/// Demonstrates the Chaining Design Pattern for Agents
///
/// This pattern shows how to create a sequential processing pipeline where:
/// 1. Agent1 processes the initial input (extracts specifications)
/// 2. Agent1's output is automatically forwarded to Agent2
/// 3. Agent2 transforms the data into a final structured format
///
/// Key concepts demonstrated:
/// - Sequential agent execution
/// - Automatic message passing between agents
/// - Topic-based communication
/// - Result transformation pipeline
///
/// Use cases:
/// - Data extraction and transformation pipelines
/// - Multi-step document processing
/// - Sequential validation workflows
/// - ETL (Extract, Transform, Load) operations
pub async fn run(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("Starting Chaining Pattern Example");
    println!("==================================");
    println!("This example demonstrates sequential agent processing:");
    println!("1. Agent1 extracts technical specifications from text");
    println!("2. Agent1 automatically forwards results to Agent2");
    println!("3. Agent2 transforms the specs into structured JSON\n");

    // Shared memory for maintaining conversation context across the chain
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    // Topics for pub/sub communication between agents
    let topic1 = Topic::<Task>::new("agent_1");
    let topic2 = Topic::<Task>::new("agent_2");

    // Create the agent instances
    let agent1 = BasicAgent::new(Agent1 {});
    let agent2 = BasicAgent::new(Agent2 {});

    // Create a single-threaded runtime for agent execution
    let runtime = SingleThreadedRuntime::new(None);

    // Build and register Agent1 with the runtime
    // Agent1 subscribes to topic1 and will process messages published to it
    let _ = AgentBuilder::<_, ActorAgent>::new(agent1)
        .llm(llm.clone())
        .runtime(runtime.clone())
        .subscribe(topic1.clone())
        .memory(sliding_window_memory.clone())
        .build()
        .await?;

    // Build and register Agent2 with the runtime
    // Agent2 subscribes to topic2 and will process messages from Agent1
    let _ = AgentBuilder::<_, ActorAgent>::new(agent2)
        .llm(llm)
        .runtime(runtime.clone())
        .subscribe(topic2.clone())
        .memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    // Start the chain by publishing a task to Agent1
    // This message will flow through the chain: Agent1 -> Agent2
    println!("Publishing initial task to start the chain...\n");
    runtime
        .publish(&topic1, Task::new("The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."))
        .await?;

    // Run the environment and wait for completion or interruption
    tokio::select! {
        _ = environment.run() => {
            println!("\nChaining pipeline completed successfully.");
        }
        _ = tokio::signal::ctrl_c() => {
            println!("\nCtrl+C detected. Shutting down...");
            environment.shutdown().await;
        }
    }

    Ok(())
}
