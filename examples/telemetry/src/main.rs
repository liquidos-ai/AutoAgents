use autoagents::async_trait;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgent, ReActAgentOutput};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{ActorAgent, AgentBuilder, DirectAgent};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::runtime::{SingleThreadedRuntime, TypedRuntime};
use autoagents::core::tool::{ToolCallError, ToolRuntime};
use autoagents::core::utils::BoxEventStream;
use autoagents::llm::LLMProvider;
use autoagents::prelude::{AgentOutputT, LLMBuilder, ToolInputT, ToolT};
use autoagents::protocol::Event;
use autoagents_derive::{AgentHooks, AgentOutput, ToolInput, agent, tool};
use autoagents_telemetry::{LangfuseRegion, LangfuseTelemetry, TelemetryProvider, Tracer};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_stream::StreamExt;

#[derive(Debug, Clone, ValueEnum)]
enum Mode {
    Direct,
    Actor,
}

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(long, default_value = "direct")]
    mode: Mode,
    #[arg(long, default_value = "What is (10 + 5) * 3?")]
    prompt: String,
    #[arg(long, default_value_t = false)]
    stdout: bool,
    #[arg(long, default_value = "us")]
    langfuse_region: String,
    #[arg(long, default_value_t = false)]
    otlp_debug: bool,
    #[arg(long, default_value = "autoagents-testing")]
    service_name: String,
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct AdditionArgs {
    #[input(description = "Left Operand for addition")]
    left: i64,
    #[input(description = "Right Operand for addition")]
    right: i64,
}

#[tool(
    name = "Addition",
    description = "Use this tool to Add two numbers",
    input = AdditionArgs,
)]
struct Addition {}

#[async_trait]
impl ToolRuntime for Addition {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let typed_args: AdditionArgs = serde_json::from_value(args)?;
        let result = typed_args.left + typed_args.right;
        Ok(result.into())
    }
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct MultiplicationArgs {
    #[input(description = "Left operand for multiplication")]
    left: i32,
    #[input(description = "Right operand for multiplication")]
    right: i32,
}

#[tool(name = "multiplication", description = "Multiply two numbers together", input = MultiplicationArgs)]
struct Multiplication {}

#[async_trait]
impl ToolRuntime for Multiplication {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let typed_args: MultiplicationArgs = serde_json::from_value(args)?;
        let result = typed_args.left * typed_args.right;
        Ok(result.into())
    }
}

#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct MathAgentOutput {
    #[output(description = "The addition result")]
    value: i64,
    #[output(description = "Explanation of the logic")]
    explanation: String,
    #[output(description = "If user asks other than math questions, use this to answer them.")]
    generic: Option<String>,
}

#[agent(
    name = "math_agent",
    description = "You solve math problems by calling tools step-by-step.

RULES:
1. ALWAYS call tools for calculations - NEVER calculate yourself
2. For multi-step problems, call tools multiple times in sequence
3. After ALL calculations are done then only return the final answer.

Example for \"What is (20 + 30) * 10?\":
- Step 1: Call addition(20, 30) → get 50
- Step 2: Call multiplication(50, 10) → get 500
- Step 3: Return \"First added 20+30=50, then multiplied 50*10=500\"

CRITICAL: Your final response MUST be valid JSON with 'value' and 'explanation' fields.",
    tools = [Addition, Multiplication],
    output = MathAgentOutput,
)]
#[derive(Default, Clone, AgentHooks)]
pub struct MathAgent {}

impl From<ReActAgentOutput> for MathAgentOutput {
    fn from(output: ReActAgentOutput) -> Self {
        output.parse_or_map(|resp| MathAgentOutput {
            value: 0,
            explanation: resp.to_string(),
            generic: None,
        })
    }
}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), Error> {
    let args = Args::parse();
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let llm: Arc<autoagents::llm::backends::openai::OpenAI> =
        LLMBuilder::<autoagents::llm::backends::openai::OpenAI>::new()
            .api_key(api_key)
            .model("gpt-4o")
            .max_tokens(512)
            .temperature(0.2)
            .build()
            .expect("Failed to build LLM");

    match args.mode {
        Mode::Direct => run_direct(llm, &args).await?,
        Mode::Actor => run_actor(llm, &args).await?,
    }

    Ok(())
}

async fn run_direct(llm: Arc<dyn LLMProvider>, args: &Args) -> Result<(), Error> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
    let mut agent_handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(MathAgent {}))
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    let telemetry_provider = telemetry_provider(args).unwrap();
    let mut tracer = Tracer::from_direct(telemetry_provider, &mut agent_handle);
    tracer
        .start()
        .map_err(|err| Error::CustomError(err.to_string()))?;

    handle_events("direct", agent_handle.subscribe_events());

    let result = agent_handle
        .agent
        .run(Task::new(args.prompt.clone()))
        .await?;

    println!("Result: {:?}", result);

    tracer
        .shutdown()
        .await
        .map_err(|err| Error::CustomError(err.to_string()))?;

    Ok(())
}

async fn run_actor(llm: Arc<dyn LLMProvider>, args: &Args) -> Result<(), Error> {
    let runtime = SingleThreadedRuntime::new(None);

    let mut environment = Environment::new(None);
    environment.register_runtime(runtime.clone()).await?;

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
    let actor_handle = AgentBuilder::<_, ActorAgent>::new(ReActAgent::new(MathAgent {}))
        .llm(llm)
        .memory(sliding_window_memory)
        .runtime(runtime.clone())
        .build()
        .await?;

    let telemetry_provider = telemetry_provider(args).unwrap();
    let mut tracer = Tracer::from_environment(telemetry_provider, &mut environment, None)
        .await
        .map_err(|err| Error::CustomError(err.to_string()))?;
    tracer
        .start()
        .map_err(|err| Error::CustomError(err.to_string()))?;

    environment.run_background().await?;

    runtime
        .send_message(Task::new(args.prompt.clone()), actor_handle.addr())
        .await?;

    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

    tracer
        .shutdown()
        .await
        .map_err(|err| Error::CustomError(err.to_string()))?;

    environment.shutdown().await;
    Ok(())
}

fn telemetry_provider(args: &Args) -> Result<Arc<dyn TelemetryProvider>, ()> {
    let langfuse_pk = std::env::var("LANGFUSE_PUBLIC_KEY").ok();
    let langfuse_sk = std::env::var("LANGFUSE_SECRET_KEY").ok();

    if let (Some(pk), Some(sk)) = (langfuse_pk.as_ref(), langfuse_sk.as_ref()) {
        tracing::info!(
            target: "autoagents.telemetry",
            langfuse_region = %args.langfuse_region,
            service_name = %args.service_name,
            "Langfuse telemetry enabled (env keys detected)"
        );
        let region = match args.langfuse_region.as_str() {
            "eu" => LangfuseRegion::Eu,
            "us" => LangfuseRegion::Us,
            custom => LangfuseRegion::Custom(custom.to_string()),
        };

        let provider = Arc::new(
            LangfuseTelemetry::new(pk, sk)
                .with_region(region)
                .with_stdout(args.stdout)
                .with_service_name(args.service_name.clone())
                .with_http_debug(args.otlp_debug),
        );
        return Ok(provider);
    }

    tracing::warn!(
        target: "autoagents.telemetry",
        "Langfuse keys not found in environment; falling back to manual OTLP config"
    );
    Err(())
}

fn handle_events(label: &str, mut event_stream: BoxEventStream<Event>) {
    let label = label.to_string();
    tokio::spawn(async move {
        while let Some(event) = event_stream.next().await {
            match event {
                Event::TaskStarted {
                    actor_id,
                    task_description,
                    ..
                } => {
                    println!("[{label}] Task started: actor={actor_id} task={task_description}");
                }
                Event::ToolCallRequested { tool_name, .. } => {
                    println!("[{label}] Tool requested: {tool_name}");
                }
                Event::ToolCallCompleted { tool_name, .. } => {
                    println!("[{label}] Tool completed: {tool_name}");
                }
                Event::TurnStarted { turn_number, .. } => {
                    println!("[{label}] Turn {turn_number} started");
                }
                Event::TurnCompleted {
                    turn_number,
                    final_turn,
                    ..
                } => {
                    println!("[{label}] Turn {turn_number} completed (final={final_turn})");
                }
                Event::TaskComplete { .. } => {
                    println!("[{label}] Task completed");
                }
                Event::TaskError { error, .. } => {
                    println!("[{label}] Task error: {error}");
                }
                _ => {}
            }
        }
    });
}
