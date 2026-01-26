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

/// Strategic planner agent that creates and manages multi-step execution plans
/// Unlike simple chaining, this agent creates a complete plan with multiple steps
/// and can adaptively modify the plan based on intermediate results
#[agent(
    name = "strategic_planner",
    description = "You are a strategic planner who breaks down complex tasks into detailed, executable plans.

    Given a complex task, create a comprehensive plan with the following structure:
    1. GOAL: Clear statement of what needs to be achieved
    2. STEPS: Numbered list of specific actions to take (aim for 3-5 steps)
    3. EXPECTED_OUTPUTS: What each step should produce
    4. SUCCESS_CRITERIA: How to measure if the plan is working

    Format your response exactly like this:
    GOAL: [Clear goal statement]
    STEPS:
    1. [First specific action]
    2. [Second specific action]
    3. [Third specific action]
    EXPECTED_OUTPUTS:
    - Step 1: [Expected output]
    - Step 2: [Expected output]
    - Step 3: [Expected output]
    SUCCESS_CRITERIA: [How to measure success]"
)]
pub struct StrategicPlanner {
    plans_created: Arc<std::sync::atomic::AtomicUsize>,
}

impl StrategicPlanner {
    pub fn new() -> Self {
        Self {
            plans_created: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl AgentHooks for StrategicPlanner {
    async fn on_run_complete(&self, _task: &Task, result: &Self::Output, ctx: &Context) {
        let plan_number = self
            .plans_created
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;

        println!(
            "\n{} STRATEGIC PLAN #{} CREATED {}",
            "=".repeat(20),
            plan_number,
            "=".repeat(20)
        );
        println!("{}", result);

        // Parse the plan to extract individual steps
        let steps = extract_steps_from_plan(result);

        if !steps.is_empty() {
            println!("\n📋 PLAN CONTAINS {} EXECUTABLE STEPS", steps.len());

            // Start executing the plan by sending the first step to the executor
            let execution_task = format!(
                "EXECUTE STEP 1 OF PLAN:\n{}\n\nFULL PLAN CONTEXT:\n{}\n\n\
                Execute this step and report your progress. If you complete this step successfully, \
                I will send you the next step.",
                steps[0], result
            );

            let _ = ctx
                .publish(
                    Topic::<Task>::new("plan_executor"),
                    Task::new(execution_task),
                )
                .await;
        }
    }
}

/// Plan executor agent that carries out individual steps and provides feedback
/// This agent executes steps one by one and can request plan modifications if needed
#[agent(
    name = "plan_executor",
    description = "You are a plan executor who carries out specific steps from a strategic plan.

    Your job is to:
    1. Execute the given step thoroughly
    2. Report what you accomplished
    3. Identify any issues or roadblocks
    4. Suggest if the plan needs adjustment

    Format your response as:
    STEP_EXECUTED: [What step you just completed]
    RESULTS: [What you accomplished]
    STATUS: [SUCCESS/PARTIAL/BLOCKED]
    NEXT_ACTION: [What should happen next - continue to next step, revise plan, or complete]
    NOTES: [Any important observations or suggestions]"
)]
pub struct PlanExecutor {
    steps_executed: Arc<std::sync::atomic::AtomicUsize>,
}

impl PlanExecutor {
    pub fn new() -> Self {
        Self {
            steps_executed: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl AgentHooks for PlanExecutor {
    async fn on_run_start(
        &self,
        _task: &Task,
        _ctx: &Context,
    ) -> autoagents::core::agent::HookOutcome {
        let step_num = self
            .steps_executed
            .load(std::sync::atomic::Ordering::SeqCst)
            + 1;
        println!("\n⚡ EXECUTION PHASE: Executing step {}...", step_num);
        autoagents::core::agent::HookOutcome::Continue
    }

    async fn on_run_complete(&self, task: &Task, result: &Self::Output, ctx: &Context) {
        let step_num = self
            .steps_executed
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;

        println!(
            "\n{} STEP {} EXECUTION COMPLETE {}",
            "=".repeat(15),
            step_num,
            "=".repeat(15)
        );
        println!("{}", result);

        // Parse the execution result
        let status = extract_status_from_result(result);

        match status.as_str() {
            "SUCCESS" => {
                // Check if there are more steps to execute
                if let Some(full_plan) = extract_full_plan_from_task(&task.prompt) {
                    let all_steps = extract_steps_from_plan(&full_plan);
                    if step_num < all_steps.len() {
                        // Continue to next step
                        let next_step_task = format!(
                            "EXECUTE STEP {} OF PLAN:\n{}\n\nFULL PLAN CONTEXT:\n{}\n\n\
                            PREVIOUS STEP RESULTS:\n{}\n\n\
                            Execute this step building on the previous results.",
                            step_num + 1,
                            all_steps[step_num],
                            full_plan,
                            result
                        );

                        println!("\n➡️  CONTINUING TO STEP {}", step_num + 1);
                        let _ = ctx
                            .publish(
                                Topic::<Task>::new("plan_executor"),
                                Task::new(next_step_task),
                            )
                            .await;
                    } else {
                        // Plan completed successfully
                        println!(
                            "\n{} PLAN EXECUTION COMPLETE {}",
                            "=".repeat(20),
                            "=".repeat(20)
                        );
                        println!("✅ All {} steps executed successfully!", all_steps.len());
                        println!("\n📊 FINAL RESULTS SUMMARY:");
                        println!("{}", result);
                    }
                }
            }
            "PARTIAL" => {
                println!("\n⚠️  Step partially completed - continuing with adjustments");
                // Could implement plan revision here
                let revision_task = format!(
                    "REVISE PLAN due to execution partial completion :\n\nORIGINAL TASK: {}\n\nPARTIAL RESULT: {}\n\n
                    Create a revised plan that addresses this.",
                    extract_original_task(&task.prompt), result,
                );
                let _ = ctx
                    .publish(
                        Topic::<Task>::new("strategic_planner"),
                        Task::new(revision_task),
                    )
                    .await;
            }
            "BLOCKED" => {
                println!("\n🔄 Step blocked - requesting plan revision...");
                let revision_task = format!(
                    "REVISE PLAN due to execution roadblock:\n\nORIGINAL TASK: {}\n\nBLOCKED AT STEP: {}\n\nISSUE: {}\n\n\
                    Create a revised plan that addresses this roadblock.",
                    extract_original_task(&task.prompt),
                    step_num,
                    result
                );
                let _ = ctx
                    .publish(
                        Topic::<Task>::new("strategic_planner"),
                        Task::new(revision_task),
                    )
                    .await;
            }
            _ => {
                println!("\n❓ Unclear status - assuming continuation");
            }
        }
    }
}

/// Helper function to extract steps from a structured plan
fn extract_steps_from_plan(plan: &str) -> Vec<String> {
    let mut steps = Vec::new();
    let lines: Vec<&str> = plan.lines().collect();
    let mut in_steps_section = false;

    for line in lines {
        if line.starts_with("STEPS:") {
            in_steps_section = true;
            continue;
        }
        if in_steps_section {
            if line.starts_with("EXPECTED_OUTPUTS:") || line.starts_with("SUCCESS_CRITERIA:") {
                break;
            }
            if let Some(step) = line
                .strip_prefix(|c: char| c.is_ascii_digit())
                .and_then(|s| s.strip_prefix(". "))
            {
                steps.push(step.trim().to_string());
            }
        }
    }

    steps
}

/// Helper function to extract status from execution result
fn extract_status_from_result(result: &str) -> String {
    result
        .lines()
        .find(|line| line.starts_with("STATUS:"))
        .and_then(|line| line.strip_prefix("STATUS:"))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "SUCCESS".to_string())
}

/// Helper function to extract full plan from task context
fn extract_full_plan_from_task(task_prompt: &str) -> Option<String> {
    task_prompt
        .split("FULL PLAN CONTEXT:\n")
        .nth(1)
        .and_then(|s| s.split("\n\nExecute this step").next())
        .map(|s| s.trim().to_string())
}

/// Helper function to extract original task
fn extract_original_task(task_prompt: &str) -> String {
    // This is a simplified extraction - in practice you might store this more systematically
    task_prompt
        .split("ORIGINAL TASK:")
        .nth(1)
        .and_then(|s| s.split("BLOCKED AT STEP:").next())
        .unwrap_or("Unknown task")
        .trim()
        .to_string()
}

/// Demonstrates the Planning Design Pattern for Agents
///
/// This pattern differs from simple chaining by implementing:
/// 1. **Strategic Planning**: Creates comprehensive multi-step plans
/// 2. **Adaptive Execution**: Can modify plans based on execution feedback
/// 3. **Progress Tracking**: Monitors step-by-step progress
/// 4. **Error Handling**: Can revise plans when steps fail or are blocked
///
/// Flow:
/// ```
///     Complex Task
///          |
///    Strategic Planner
///    (creates detailed plan)
///          |
///       Plan Created
///          |
///    ╔═══════════════╗
///    ║ Plan Executor ║ <-- Executes each step
///    ║ (step by step)║ <-- Reports progress
///    ╚═══════════════╝ <-- Can request revision
///          |
///    [Success/Revision/Continue]
/// ```
///
/// Key differences from chaining:
/// - **Multi-step awareness**: Planner sees the entire task scope
/// - **Adaptive behavior**: Can revise plans based on execution results
/// - **Progress monitoring**: Tracks completion of individual steps
/// - **Error recovery**: Handles roadblocks by replanning
///
/// Use cases:
/// - Complex project management
/// - Multi-stage problem solving
/// - Research workflows with feedback loops
/// - Adaptive content creation pipelines
/// - Strategic analysis with course correction
pub async fn run(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("Starting Planning Design Pattern Example");
    println!("========================================");
    println!("This pattern demonstrates strategic planning with adaptive execution:");
    println!("1. Strategic planner creates a comprehensive multi-step plan");
    println!("2. Plan executor carries out steps one by one");
    println!("3. Executor provides feedback and can request plan revisions");
    println!("4. Pattern handles roadblocks by replanning\n");

    // Shared memory for maintaining context between planning phases
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(30));

    // Define topics for agent communication
    let planner_topic = Topic::<Task>::new("strategic_planner");
    let executor_topic = Topic::<Task>::new("plan_executor");

    // Create agent instances
    let planner_agent = BasicAgent::new(StrategicPlanner::new());
    let executor_agent = BasicAgent::new(PlanExecutor::new());

    let runtime = SingleThreadedRuntime::new(None);

    // Build and register the strategic planner
    let _ = AgentBuilder::<_, ActorAgent>::new(planner_agent)
        .llm(llm.clone())
        .runtime(runtime.clone())
        .subscribe(planner_topic.clone())
        .memory(sliding_window_memory.clone())
        .build()
        .await?;

    // Build and register the plan executor
    let _ = AgentBuilder::<_, ActorAgent>::new(executor_agent)
        .llm(llm)
        .runtime(runtime.clone())
        .subscribe(executor_topic.clone())
        .memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    // Define a complex task that requires strategic planning
    let complex_task = "Create a comprehensive technical blog post about 'Implementing Multi-Agent Systems in Rust' \
                       that includes code examples, architecture diagrams, performance benchmarks, \
                       and a working demo application. The post should be suitable for publication \
                       on a major tech blog and demonstrate both theoretical understanding and practical implementation.";

    println!("## Running Strategic Planning Example ##");
    println!("Complex Task: {}\n", complex_task);

    // Start the strategic planning process
    runtime
        .publish(&planner_topic, Task::new(complex_task))
        .await?;

    // Run the environment with extended timeout for complex planning
    tokio::select! {
        _ = environment.run() => {
            println!("\nStrategic planning process completed successfully.");
        }
        _ = tokio::time::sleep(tokio::time::Duration::from_secs(90)) => {
            println!("\nPlanning process timeout - shutting down.");
            environment.shutdown().await;
        }
        _ = tokio::signal::ctrl_c() => {
            println!("\nCtrl+C detected. Shutting down...");
            environment.shutdown().await;
        }
    }

    Ok(())
}
