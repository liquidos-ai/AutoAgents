use autoagents::async_trait;
use autoagents::core::actor::Topic;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{ActorAgent, AgentBuilder, AgentHooks, Context};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::protocol::{Event, SubmissionId};
use autoagents::core::runtime::{SingleThreadedRuntime, TypedRuntime};
use autoagents::core::tool::ToolT;
use autoagents::core::utils::BoxEventStream;
use autoagents::llm::LLMProvider;
use autoagents_derive::{agent, AgentHooks};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::Instant;
use tokio_stream::StreamExt;

/// Agent responsible for creating concise summaries
/// Processes topics independently and in parallel with other analysis agents
#[agent(
    name = "summarize",
    description = "Your role is to Summarize the following topic concisely."
)]
#[derive(AgentHooks)]
pub struct SummarizeAgent {}

/// Agent that generates thought-provoking questions
/// Works in parallel to explore different aspects of the topic
#[agent(
    name = "questions",
    description = "Your role is to Generate three interesting questions about the following topic."
)]
#[derive(AgentHooks)]
pub struct QuestionsAgent {}

/// Agent that extracts key terminology and concepts
/// Identifies important terms that define the topic domain
#[agent(
    name = "key_terms",
    description = "Your role is to Identify 5-10 key terms from the following topic, separated by commas."
)]
#[derive(AgentHooks)]
pub struct TermsAgent {}

/// Synthesis agent that combines outputs from parallel agents
/// This agent waits for all parallel agents to complete before creating a unified response
#[agent(
    name = "synthesis_agent",
    description = "Your role is to Synthesize a comprehensive answer based on given Summary, Related Question, Key Terms."
)]
pub struct SynthesisAgent {
    start_time: Instant,
}

#[async_trait]
impl AgentHooks for SynthesisAgent {
    /// Outputs the final synthesized result and measures total execution time
    async fn on_run_complete(&self, _task: &Task, result: &Self::Output, _ctx: &Context) {
        println!("Final Results:\n {result}");
        let end_time = self.start_time.elapsed().as_secs();
        println!("Time Elapsed: {end_time}")
    }
}

/// Demonstrates the Parallel Design Pattern for Agents
///
/// This pattern shows how to:
/// 1. Execute multiple agents concurrently for different analyses
/// 2. Collect results from parallel agents
/// 3. Synthesize the parallel results into a final output
///
/// Flow:
/// ```
///     Input Topic
///         |
///     ╔═══╬═══╗
///     ║   ║   ║
///     v   v   v
/// Summary Questions Terms  (executed in parallel)
///     ║   ║   ║
///     ╚═══╬═══╝
///         |
///    Synthesis
///         |
///    Final Output
/// ```
///
/// Key concepts:
/// - Parallel agent execution for improved performance
/// - Result aggregation from multiple agents
/// - Synthesis of diverse perspectives
/// - Event-driven coordination
///
/// Use cases:
/// - Multi-aspect document analysis
/// - Parallel data processing pipelines
/// - Distributed task execution
/// - Complex analysis requiring multiple perspectives
pub async fn run(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("Starting Parallel Pattern Example");
    println!("==================================");
    println!("This example demonstrates parallel agent processing:");
    println!("1. Three agents analyze a topic in parallel");
    println!("2. Results are collected as they complete");
    println!("3. A synthesis agent combines all results\n");

    // Shared memory for maintaining context
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    // Define topics for each agent
    let summarize_topic = Topic::<Task>::new("summarize");
    let questions_topic = Topic::<Task>::new("questions");
    let terms_topic = Topic::<Task>::new("terms");
    let synthesis_topic = Topic::<Task>::new("synthesis");

    // Create agent instances
    let summarize_agent = BasicAgent::new(SummarizeAgent {});
    let questions_agent = BasicAgent::new(QuestionsAgent {});
    let terms_agent = BasicAgent::new(TermsAgent {});
    let synthesis_agent = BasicAgent::new(SynthesisAgent {
        start_time: Instant::now(),
    });

    let runtime = SingleThreadedRuntime::new(None);

    // Build and register the summarization agent
    let _ = AgentBuilder::<_, ActorAgent>::new(summarize_agent)
        .llm(llm.clone())
        .runtime(runtime.clone())
        .subscribe(summarize_topic.clone())
        .memory(sliding_window_memory.clone())
        .build()
        .await?;

    // Build and register the questions agent
    let _ = AgentBuilder::<_, ActorAgent>::new(questions_agent)
        .llm(llm.clone())
        .runtime(runtime.clone())
        .subscribe(questions_topic.clone())
        .memory(sliding_window_memory.clone())
        .build()
        .await?;

    // Build and register the terms extraction agent
    let _ = AgentBuilder::<_, ActorAgent>::new(terms_agent)
        .llm(llm.clone())
        .runtime(runtime.clone())
        .subscribe(terms_topic.clone())
        .memory(sliding_window_memory.clone())
        .build()
        .await?;

    // Build and register the synthesis agent
    let _ = AgentBuilder::<_, ActorAgent>::new(synthesis_agent)
        .llm(llm.clone())
        .runtime(runtime.clone())
        .subscribe(synthesis_topic.clone())
        .memory(sliding_window_memory.clone())
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    let receiver = environment.take_event_receiver(None).await?;

    // Create the task that will be processed by multiple agents
    let task = Task::new("The history of space exploration");

    // Set up custom event handler to coordinate parallel results
    handle_events(receiver, task.submission_id, runtime.clone());

    println!("--- Running Parallel LangChain Example for Topic: '{task:?}' ---");

    // Publish the task to all three agents simultaneously
    // They will process in parallel
    runtime.publish(&summarize_topic, task.clone()).await?;

    runtime.publish(&questions_topic, task.clone()).await?;

    runtime.publish(&terms_topic, task.clone()).await?;

    // Run the environment
    tokio::select! {
        _ = environment.run() => {
            println!("Environment finished running.");
        }
        _ = tokio::signal::ctrl_c() => {
            println!("Ctrl+C detected. Shutting down...");
            environment.shutdown().await;
        }
    }

    Ok(())
}

/// Custom event handler that collects results from parallel agents
/// Once all parallel agents complete, it triggers the synthesis agent
///
/// This demonstrates:
/// - Result collection from multiple async agents
/// - Conditional triggering based on completion status
/// - Structured data preparation for synthesis
pub fn handle_events(
    mut event_stream: BoxEventStream<Event>,
    submission_id: SubmissionId,
    runtime: Arc<SingleThreadedRuntime>,
) {
    tokio::spawn(async move {
        // HashMap to collect results from parallel agents
        let mut results: HashMap<String, String> = HashMap::new();
        let expected_keys = ["summarize", "questions", "key_terms"];

        while let Some(event) = event_stream.next().await {
            match event {
                Event::TaskComplete {
                    result,
                    sub_id,
                    actor_name,
                    ..
                } => {
                    // Only process events for our specific submission
                    if sub_id == submission_id {
                        results.insert(actor_name.clone(), result.clone());
                    }

                    // Check if all parallel agents have completed
                    if expected_keys.iter().all(|k| results.contains_key(*k)) {
                        // Create a structured JSON payload for the synthesis agent
                        // This ensures the synthesis agent can reliably parse the different components
                        let payload = json!({
                            "summary": results.get("summarize").unwrap_or(&"".to_string()),
                            "questions": results.get("questions").unwrap_or(&"".to_string()),
                            "key_terms": results.get("terms").unwrap_or(&"".to_string()),
                        });

                        // Trigger the synthesis agent with combined results
                        if let Err(e) = runtime
                            .publish(
                                &Topic::<Task>::new("synthesis"),
                                Task::new(payload.to_string()),
                            )
                            .await
                        {
                            eprintln!("Failed to publish synthesis task: {e}");
                        }

                        // Clear results for potential next round
                        results.clear();
                    }
                }
                _ => { /* ignore other events */ }
            }
        }
    });
}
