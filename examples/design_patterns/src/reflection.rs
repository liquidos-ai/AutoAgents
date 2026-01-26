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

#[agent(
    name = "code_generator",
    description = "Generate or refine Python code based on task requirements and critiques.
    You are an expert Python developer. Generate clean, well-documented code that follows best practices.

    IMPORTANT: Your task is to create a Python function named `calculate_factorial` that:
    1. Accepts a single integer `n` as input
    2. Calculates its factorial (n!)
    3. Includes a clear docstring explaining what the function does
    4. Handles edge cases: The factorial of 0 is 1
    5. Handles invalid input: Raises a ValueError if the input is negative

    Output ONLY the Python code, no explanations or markdown formatting."
)]
pub struct CodeGenerator {
    iteration: Arc<std::sync::atomic::AtomicUsize>,
}

impl CodeGenerator {
    pub fn new() -> Self {
        Self {
            iteration: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl AgentHooks for CodeGenerator {
    async fn on_run_start(
        &self,
        _task: &Task,
        _ctx: &Context,
    ) -> autoagents::core::agent::HookOutcome {
        let iteration = self
            .iteration
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;

        if iteration == 1 {
            println!("\n>>> STAGE 1: GENERATING initial code...");
        } else {
            println!("\n>>> STAGE 1: REFINING code based on previous critique...");
        }

        autoagents::core::agent::HookOutcome::Continue
    }

    async fn on_run_complete(&self, _task: &Task, result: &Self::Output, ctx: &Context) {
        let iteration = self.iteration.load(std::sync::atomic::Ordering::SeqCst);

        println!(
            "\n{} REFLECTION LOOP: ITERATION {} {}",
            "=".repeat(25),
            iteration,
            "=".repeat(25)
        );
        println!("\n--- Generated Code (v{}) ---\n{}", iteration, result);

        let critique_task = format!(
            "Original Task:\n\
            Create a Python function named `calculate_factorial` that:\n\
            1. Accepts a single integer `n` as input\n\
            2. Calculates its factorial (n!)\n\
            3. Includes a clear docstring explaining what the function does\n\
            4. Handles edge cases: The factorial of 0 is 1\n\
            5. Handles invalid input: Raises a ValueError if the input is negative\n\n\
            Code to Review:\n{}\n\n\
            Critically evaluate this code. Look for:\n\
            - Correctness of the implementation\n\
            - Proper handling of edge cases (0! = 1)\n\
            - Proper error handling for negative inputs\n\
            - Clear and accurate docstring\n\
            - Code style and best practices\n\
            \n\
            If the code perfectly meets ALL requirements, respond with exactly: CODE_IS_PERFECT\n\
            Otherwise, provide specific, actionable critiques to improve the code.",
            result
        );

        let _ = ctx
            .publish(Topic::<Task>::new("code_critic"), Task::new(critique_task))
            .await;
    }
}

#[agent(
    name = "code_critic",
    description = "Critically evaluate Python code and provide constructive feedback.
    You are a senior software engineer and expert in Python.
    Your role is to perform meticulous code reviews.
    Look for bugs, style issues, missing edge cases, and areas for improvement.

    If the code is perfect and meets all requirements, respond with exactly: CODE_IS_PERFECT
    Otherwise, provide specific, actionable feedback for improvement."
)]
pub struct CodeCritic {
    max_iterations: usize,
    current_iteration: Arc<std::sync::atomic::AtomicUsize>,
}

impl CodeCritic {
    pub fn new(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            current_iteration: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl AgentHooks for CodeCritic {
    async fn on_run_start(
        &self,
        _task: &Task,
        _ctx: &Context,
    ) -> autoagents::core::agent::HookOutcome {
        println!("\n>>> STAGE 2: REFLECTING on the generated code...");
        autoagents::core::agent::HookOutcome::Continue
    }

    async fn on_run_complete(&self, task: &Task, result: &Self::Output, ctx: &Context) {
        let iteration = self
            .current_iteration
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;

        // Extract the code from the task
        let code = task
            .prompt
            .split("Code to Review:\n")
            .nth(1)
            .unwrap_or("")
            .split("\n\nCritically")
            .next()
            .unwrap_or("");

        if result.contains("CODE_IS_PERFECT") {
            println!("\n--- Critique ---\nNo further critiques found. The code is satisfactory.");
            println!("\n{} FINAL RESULT {}", "=".repeat(30), "=".repeat(30));
            println!("\nFinal refined code after the reflection process:\n");
            println!("{}", code);

            // Environment will naturally shut down when no more messages are being processed
        } else if iteration < self.max_iterations {
            println!("\n--- Critique ---\n{}", result);

            let refinement_task = format!(
                "Previous code:\n{}\n\nCritique from code review:\n{}\n\n\
                Please refine the code based on the critique above. \
                Generate an improved version that addresses all the feedback.\n\
                Remember to output ONLY the Python code, no explanations.",
                code, result
            );

            let _ = ctx
                .publish(
                    Topic::<Task>::new("code_generator"),
                    Task::new(refinement_task),
                )
                .await;
        } else {
            println!("\n--- Max iterations reached ---");
            println!("Final critique:\n{}", result);
            println!("\n{} FINAL RESULT {}", "=".repeat(30), "=".repeat(30));
            println!("\nFinal code after {} iterations:\n", self.max_iterations);
            println!("{}", code);
        }
    }
}

/// This example demonstrates the Reflection Design Pattern for Agents
///
/// The pattern involves:
/// 1. A CodeGenerator agent that creates or refines Python code
/// 2. A CodeCritic agent that evaluates the code and provides feedback
/// 3. An iterative loop where code is progressively improved based on critiques
///
/// This mimics the reflection loop from the Python example where:
/// - An initial code solution is generated
/// - The code is critically evaluated
/// - Based on the critique, the code is refined
/// - The process repeats until the code is satisfactory or max iterations are reached
pub async fn run(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("Starting Reflection Pattern Example");
    println!("====================================");
    println!(
        "This example demonstrates an AI reflection loop to progressively improve a Python function."
    );
    println!(
        "The agents will work together to generate, critique, and refine a factorial function.\n"
    );

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(20));

    let generator_topic = Topic::<Task>::new("code_generator");
    let critic_topic = Topic::<Task>::new("code_critic");

    let code_generator = BasicAgent::new(CodeGenerator::new());
    let code_critic = BasicAgent::new(CodeCritic::new(3)); // Max 3 iterations

    let runtime = SingleThreadedRuntime::new(None);

    // Build the code generator agent
    let _ = AgentBuilder::<_, ActorAgent>::new(code_generator)
        .llm(llm.clone())
        .runtime(runtime.clone())
        .subscribe(generator_topic.clone())
        .memory(sliding_window_memory.clone())
        .build()
        .await?;

    // Build the code critic agent
    let _ = AgentBuilder::<_, ActorAgent>::new(code_critic)
        .llm(llm)
        .runtime(runtime.clone())
        .subscribe(critic_topic.clone())
        .memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    // Start the reflection loop by publishing the initial task
    runtime
        .publish(
            &generator_topic,
            Task::new("Generate a Python function named calculate_factorial"),
        )
        .await?;

    // Run for a limited time to allow the reflection loop to complete
    tokio::select! {
        _ = environment.run() => {
            println!("\nReflection loop completed.");
        }
        _ = tokio::time::sleep(tokio::time::Duration::from_secs(30)) => {
            println!("\nReflection loop timeout - shutting down.");
            environment.shutdown().await;
        }
        _ = tokio::signal::ctrl_c() => {
            println!("\nCtrl+C detected. Shutting down...");
            environment.shutdown().await;
        }
    }

    Ok(())
}
