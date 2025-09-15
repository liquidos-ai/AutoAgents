use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::error::Error;
use autoagents::llm::LLMProvider;
use autoagents_derive::{agent, AgentHooks};
use serde_json::Value;
use std::sync::Arc;

/// Routing agent that analyzes requests and determines the appropriate handler
/// Acts as a intelligent dispatcher, routing tasks to specialized agents based on content
///
/// This agent demonstrates:
/// - Intent classification
/// - Decision-based routing
/// - Simple pattern matching with LLM understanding
#[agent(
    name = "routing_agent",
    description = "Analyze the user's request and determine which specialist handler should process it.
    - If the request is related to booking flights or hotels,
      output 'booker'.
    - For questions related to cities, output 'info'.
    - If the request is unclear or doesn't fit either category,
      output 'unclear'.
    ONLY output one word: 'booker', 'info', or 'unclear'"
)]
#[derive(AgentHooks)]
pub struct Agent {}

/// Handles booking-related requests
/// Simulates a specialized booking service that would handle reservations
fn booking_handler(request: String) -> String {
    // Simulates the Booking Agent handling a request.
    println!("--- DELEGATING TO BOOKING HANDLER ---");
    format!("Booking Handler processed request: '{request}'. Result: Simulated booking action.")
}

/// Handles informational queries
/// Simulates a knowledge-based agent that provides information about locations
fn info_handler(request: String) -> String {
    // Simulates the Info Agent handling a request.
    println!("--- DELEGATING TO INFO HANDLER ---");
    format!("Info Handler processed request: '{request}'. Result: Simulated information retrieval.")
}

/// Handles requests that couldn't be properly classified
/// Provides feedback when the routing agent cannot determine the appropriate handler
fn unclear_handler(request: String) -> String {
    // Handles requests that couldn't be delegated.
    println!("--- DELEGATING TO UNCLEAR HANDLER ---");
    format!("Coordinator could not delegate request: '{request}'. Please clarify.")
}

/// Routes the request to the appropriate handler based on the routing decision
///
/// This function demonstrates:
/// - Pattern matching on routing decisions
/// - Delegation to specialized handlers
/// - Fallback handling for edge cases
fn handle_routing(mode: String, request: String) -> String {
    match mode.as_ref() {
        "booker" => booking_handler(request),
        "info" => info_handler(request),
        "unclear" => unclear_handler(request),
        _ => String::from("Unknown routing mode"),
    }
}

/// Demonstrates the Routing Design Pattern for Agents
///
/// This pattern shows how to:
/// 1. Use an LLM-powered agent to classify and route requests
/// 2. Delegate work to specialized handlers based on classification
/// 3. Handle edge cases and unclear requests gracefully
///
/// Architecture:
/// ```
///       User Request
///            |
///      Routing Agent
///      (classifies)
///            |
///     ╔══════╬══════╗
///     ║      ║      ║
///   Booking Info  Unclear
///   Handler Handler Handler
/// ```
///
/// Key concepts:
/// - Intent classification using LLM
/// - Conditional routing logic
/// - Specialized handler delegation
/// - Direct agent usage (no actor system needed for simple routing)
///
/// Use cases:
/// - Customer service routing
/// - Task delegation systems
/// - Multi-domain assistants
/// - Request triage and prioritization
///
/// Note: This example uses DirectAgent instead of ActorAgent since
/// routing is a simple, synchronous operation that doesn't require
/// the overhead of an actor system.
pub async fn run(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("Starting Routing Pattern Example");
    println!("=================================");
    println!("This example demonstrates intelligent request routing:");
    println!("- The routing agent classifies each request");
    println!("- Requests are delegated to appropriate handlers");
    println!("- Each handler specializes in its domain\n");

    // Memory for maintaining conversation context
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    // Create the routing agent
    let agent = BasicAgent::new(Agent {});

    // Build the agent with DirectAgent type for synchronous execution
    // DirectAgent is more suitable for simple routing logic
    let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm.clone())
        .memory(sliding_window_memory.clone())
        .build()
        .await?;

    // Example 1: Booking request
    println!("--- Running with a booking request ---");
    let task = String::from("Book me a flight to London.");
    let result = agent_handle.agent.run(Task::new(task.clone())).await?;
    println!("Routing decision: {}", result);
    println!("Final Result: {}\n", handle_routing(result, task));

    // Example 2: Information request
    println!("--- Running with an info request ---");
    let task = String::from("What is the capital of Italy?");
    let result = agent_handle.agent.run(Task::new(task.clone())).await?;
    println!("Routing decision: {}", result);
    println!("Final Result: {}\n", handle_routing(result, task));

    // Example 3: Unclear request
    println!("--- Running with an unclear request ---");
    let task = String::from("Tell me about quantum physics.");
    let result = agent_handle.agent.run(Task::new(task.clone())).await?;
    println!("Routing decision: {}", result);
    println!("Final Result: {}\n", handle_routing(result, task));

    Ok(())
}
