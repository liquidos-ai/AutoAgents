use async_trait::async_trait;
use autoagents_core::agent::memory::SlidingWindowMemory;
use autoagents_core::agent::prebuilt::executor::ReActAgent;
use autoagents_core::agent::task::Task;
use autoagents_core::agent::{
    ActorAgent, ActorAgentHandle, AgentBuilder, AgentDeriveT, AgentHooks, DirectAgent,
    DirectAgentHandle,
};
use autoagents_core::runtime::{Runtime, SingleThreadedRuntime};
use autoagents_core::tool::{ToolT, to_llm_tool};
use autoagents_llm::chat::{
    ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StructuredOutputFormat, Tool,
};
use autoagents_llm::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use autoagents_llm::embedding::EmbeddingProvider;
use autoagents_llm::error::LLMError;
use autoagents_llm::models::ModelsProvider;
use autoagents_llm::{FunctionCall, LLMProvider, ToolCall};
use autoagents_protocol::Event;
use criterion::{Criterion, criterion_group, criterion_main};
use futures_util::StreamExt;
use serde::Serialize;
use serde_json::{Value, json};
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use tokio::runtime::Runtime as TokioRuntime;
use tokio::time::{Duration, sleep};

#[derive(Debug, Clone)]
struct BenchAgent;

const BENCH_TOOL_NAME: &str = "bench_tool";
const BENCH_TOOL_CALL_ID: &str = "bench_tool_call";
const DEFAULT_TOOL_ARGS: &str = "{}";
const DEFAULT_LLM_DELAY_MS: u64 = 5;
const DEFAULT_RESPONSE_TEXT: &str = "ok";
const DEFAULT_RUNTIME_BUFFER: usize = 1024;

#[derive(Debug, Clone)]
struct BenchTool;

#[async_trait]
impl autoagents_core::tool::ToolRuntime for BenchTool {
    async fn execute(&self, args: Value) -> Result<Value, autoagents_core::tool::ToolCallError> {
        Ok(json!({"ok": true, "args": args}))
    }
}

impl ToolT for BenchTool {
    fn name(&self) -> &str {
        BENCH_TOOL_NAME
    }

    fn description(&self) -> &str {
        "Benchmark tool that echoes inputs"
    }

    fn args_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            },
            "required": []
        })
    }
}

fn default_tool_call() -> ToolCall {
    ToolCall {
        id: BENCH_TOOL_CALL_ID.to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: BENCH_TOOL_NAME.to_string(),
            arguments: DEFAULT_TOOL_ARGS.to_string(),
        },
    }
}

fn bench_tool_defs() -> Vec<Tool> {
    let tool: Box<dyn ToolT> = Box::new(BenchTool);
    vec![to_llm_tool(&tool)]
}

fn bench_tool_result_message() -> ChatMessage {
    ChatMessage {
        role: ChatRole::Tool,
        message_type: MessageType::ToolResult(vec![default_tool_call()]),
        content: String::default(),
    }
}

async fn await_task_complete(
    events: &mut autoagents_core::utils::BoxEventStream<Event>,
    submission_id: autoagents_protocol::SubmissionId,
) -> Result<(), String> {
    loop {
        match events.next().await {
            Some(Event::TaskComplete { sub_id, .. }) if sub_id == submission_id => return Ok(()),
            Some(Event::TaskError { sub_id, error, .. }) if sub_id == submission_id => {
                return Err(error);
            }
            Some(_) => {}
            None => return Err("Event stream closed before task completion".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
struct MockLlmProvider {
    delay: Duration,
    response_text: String,
    default_tool_call: ToolCall,
}

impl MockLlmProvider {
    fn new(delay_ms: u64) -> Self {
        Self {
            delay: Duration::from_millis(delay_ms),
            response_text: DEFAULT_RESPONSE_TEXT.to_string(),
            default_tool_call: default_tool_call(),
        }
    }

    async fn simulate_latency(&self) {
        if self.delay.as_nanos() > 0 {
            sleep(self.delay).await;
        }
    }

    fn should_return_tool_call(&self, messages: &[ChatMessage], tools: Option<&[Tool]>) -> bool {
        if tools.is_none() {
            return false;
        }

        !messages
            .iter()
            .any(|message| matches!(message.role, ChatRole::Tool))
    }

    fn build_tool_calls(&self, tools: Option<&[Tool]>) -> Option<Vec<ToolCall>> {
        let tools = tools?;
        let tool = tools.first()?;
        let mut call = self.default_tool_call.clone();
        call.function.name = tool.function.name.clone();
        Some(vec![call])
    }
}

#[async_trait]
impl ChatProvider for MockLlmProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.simulate_latency().await;

        let tool_calls = if self.should_return_tool_call(messages, tools) {
            self.build_tool_calls(tools)
        } else {
            None
        };

        Ok(Box::new(MockChatResponse {
            text: Some(self.response_text.clone()),
            tool_calls,
        }))
    }
}

#[async_trait]
impl CompletionProvider for MockLlmProvider {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        self.simulate_latency().await;
        Ok(CompletionResponse {
            text: self.response_text.clone(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for MockLlmProvider {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        self.simulate_latency().await;
        Ok(input.iter().map(|_| vec![0.0, 1.0]).collect())
    }
}

#[async_trait]
impl ModelsProvider for MockLlmProvider {}

impl LLMProvider for MockLlmProvider {}

#[derive(Debug)]
struct MockChatResponse {
    text: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

impl ChatResponse for MockChatResponse {
    fn text(&self) -> Option<String> {
        self.text.clone()
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.tool_calls.clone()
    }
}

impl std::fmt::Display for MockChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text.as_deref().unwrap_or(""))
    }
}

impl AgentDeriveT for BenchAgent {
    type Output = String;

    fn description(&self) -> &str {
        "Benchmark agent"
    }

    fn output_schema(&self) -> Option<Value> {
        None
    }

    fn name(&self) -> &str {
        "bench_agent"
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        vec![Box::new(BenchTool) as Box<dyn ToolT>]
    }
}

impl AgentHooks for BenchAgent {}

#[derive(Debug, Clone)]
struct BenchConfig {
    prompt: String,
    llm_delay_ms: u64,
}

impl BenchConfig {
    fn from_env() -> Self {
        Self {
            prompt: env_or_default(
                "AUTOAGENTS_BENCH_PROMPT",
                "Answer with a single number: 2+2".to_string(),
            ),
            llm_delay_ms: env_u64("AUTOAGENTS_BENCH_LLM_DELAY_MS", DEFAULT_LLM_DELAY_MS),
        }
    }
}

fn env_or_default(key: &str, default: String) -> String {
    std::env::var(key).unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .map(|value| {
            value
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("Invalid {key} value (expected u64): {value}"))
        })
        .unwrap_or(default)
}

fn build_llm(config: &BenchConfig) -> Arc<dyn LLMProvider> {
    Arc::new(MockLlmProvider::new(config.llm_delay_ms))
}

async fn build_direct_agent(
    llm: Arc<dyn LLMProvider>,
) -> DirectAgentHandle<ReActAgent<BenchAgent>> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(8));
    let agent = ReActAgent::new(BenchAgent);
    AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await
        .expect("Failed to build agent")
}

async fn build_actor_agent(
    llm: Arc<dyn LLMProvider>,
    runtime: Arc<SingleThreadedRuntime>,
) -> ActorAgentHandle<ReActAgent<BenchAgent>> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(8));
    let agent = ReActAgent::new(BenchAgent);
    AgentBuilder::<_, ActorAgent>::new(agent)
        .llm(llm)
        .memory(sliding_window_memory)
        .runtime(runtime)
        .build()
        .await
        .expect("Failed to build actor agent")
}

type BenchDirectAgent = autoagents_core::agent::BaseAgent<ReActAgent<BenchAgent>, DirectAgent>;

#[derive(Debug, Serialize)]
struct MemorySample {
    rss_before_kb: u64,
    rss_after_kb: u64,
    rss_delta_kb: u64,
}

#[derive(Debug, Serialize)]
struct MemoryReport {
    platform: String,
    available: bool,
    samples: Vec<MemorySample>,
}

static MEMORY_SAMPLES: OnceLock<Mutex<Vec<MemorySample>>> = OnceLock::new();

fn memory_samples() -> &'static Mutex<Vec<MemorySample>> {
    MEMORY_SAMPLES.get_or_init(|| Mutex::new(Vec::new()))
}

#[cfg(target_os = "linux")]
fn read_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let value = rest.split_whitespace().next()?;
            return value.parse::<u64>().ok();
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn read_rss_kb() -> Option<u64> {
    None
}

fn write_memory_report() {
    let samples = memory_samples()
        .lock()
        .map(|mut guard| std::mem::take(&mut *guard))
        .unwrap_or_default();

    let report = MemoryReport {
        platform: std::env::consts::OS.to_string(),
        available: cfg!(target_os = "linux"),
        samples,
    };

    let out_dir = PathBuf::from("target")
        .join("criterion")
        .join("agent_runtime");
    if let Err(err) = std::fs::create_dir_all(&out_dir) {
        eprintln!("Failed to create memory report directory: {err}");
        return;
    }

    let path = out_dir.join("memory.json");
    match serde_json::to_string_pretty(&report) {
        Ok(json) => {
            if let Err(err) = std::fs::write(&path, json) {
                eprintln!("Failed to write memory report: {err}");
            }
        }
        Err(err) => eprintln!("Failed to serialize memory report: {err}"),
    }
}

struct OverheadReportInputs<'a> {
    runtime: &'a TokioRuntime,
    llm: &'a Arc<dyn LLMProvider>,
    direct_agent: &'a BenchDirectAgent,
    actor_ref: &'a autoagents_core::ractor::ActorRef<Task>,
    actor_events: &'a mut autoagents_core::utils::BoxEventStream<Event>,
    prompt: &'a str,
    tools: &'a [Tool],
    tool_result_message: &'a ChatMessage,
}

fn run_overhead_report(inputs: OverheadReportInputs<'_>) -> Result<(), String> {
    let OverheadReportInputs {
        runtime,
        llm,
        direct_agent,
        actor_ref,
        actor_events,
        prompt,
        tools,
        tool_result_message,
    } = inputs;
    let iterations = 10u32;
    let message = ChatMessage::user().content(prompt.to_string()).build();
    let llm_messages = vec![message.clone()];
    let llm_followup = vec![message.clone(), tool_result_message.clone()];

    let llm_no_tools_start = Instant::now();
    for _ in 0..iterations {
        let msg = message.clone();
        let _ = runtime.block_on(llm.chat(std::slice::from_ref(&msg), None));
    }
    let llm_no_tools_elapsed = llm_no_tools_start.elapsed();

    let llm_with_tools_start = Instant::now();
    for _ in 0..iterations {
        let _ = runtime.block_on(llm.chat_with_tools(&llm_messages, Some(tools), None));
        let _ = runtime.block_on(llm.chat_with_tools(&llm_followup, Some(tools), None));
    }
    let llm_with_tools_elapsed = llm_with_tools_start.elapsed();

    let direct_start = Instant::now();
    for _ in 0..iterations {
        let _ = runtime.block_on(direct_agent.run(Task::new(prompt.to_string())));
    }
    let direct_elapsed = direct_start.elapsed();

    let actor_start = Instant::now();
    for _ in 0..iterations {
        let task = Task::new(prompt.to_string());
        let submission_id = task.submission_id;
        actor_ref
            .cast(task)
            .map_err(|e| format!("Failed to send actor task: {e}"))?;
        runtime.block_on(await_task_complete(actor_events, submission_id))?;
    }
    let actor_elapsed = actor_start.elapsed();

    let llm_no_tools_avg_ms = llm_no_tools_elapsed.as_secs_f64() * 1000.0 / f64::from(iterations);
    let llm_with_tools_avg_ms =
        llm_with_tools_elapsed.as_secs_f64() * 1000.0 / f64::from(iterations);
    let direct_avg_ms = direct_elapsed.as_secs_f64() * 1000.0 / f64::from(iterations);
    let actor_avg_ms = actor_elapsed.as_secs_f64() * 1000.0 / f64::from(iterations);
    let tool_cycle_overhead_ms = llm_with_tools_avg_ms - llm_no_tools_avg_ms;
    let direct_overhead_ms = direct_avg_ms - llm_with_tools_avg_ms;
    let actor_overhead_ms = actor_avg_ms - llm_with_tools_avg_ms;
    let direct_overhead_pct = if llm_with_tools_avg_ms > 0.0 {
        (direct_overhead_ms / llm_with_tools_avg_ms) * 100.0
    } else {
        0.0
    };
    let actor_overhead_pct = if llm_with_tools_avg_ms > 0.0 {
        (actor_overhead_ms / llm_with_tools_avg_ms) * 100.0
    } else {
        0.0
    };

    eprintln!("Overhead summary ({} iters):", iterations);
    eprintln!("  llm_no_tools={:.3} ms", llm_no_tools_avg_ms);
    eprintln!("  llm_with_tools={:.3} ms", llm_with_tools_avg_ms);
    eprintln!("  tool_cycle_overhead={:.3} ms", tool_cycle_overhead_ms);
    eprintln!(
        "  direct_agent={:.3} ms (overhead +{:.3} ms, {:.1}%)",
        direct_avg_ms, direct_overhead_ms, direct_overhead_pct
    );
    eprintln!(
        "  actor_agent={:.3} ms (overhead +{:.3} ms, {:.1}%)",
        actor_avg_ms, actor_overhead_ms, actor_overhead_pct
    );

    Ok(())
}

fn bench_agent_runtime(c: &mut Criterion) {
    let config = BenchConfig::from_env();
    {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to build tokio runtime");

        let llm = build_llm(&config);
        let direct_handle = runtime.block_on(build_direct_agent(llm.clone()));
        let DirectAgentHandle {
            agent: direct_agent,
            rx,
            ..
        } = direct_handle;

        runtime.spawn(async move {
            let mut rx = rx;
            while rx.next().await.is_some() {}
        });

        let actor_runtime = SingleThreadedRuntime::new(Some(DEFAULT_RUNTIME_BUFFER));
        let mut actor_events = runtime
            .block_on(actor_runtime.take_event_receiver())
            .expect("Failed to acquire actor event receiver");
        let actor_runtime_task = {
            let actor_runtime = actor_runtime.clone();
            runtime.spawn(async move {
                let _ = actor_runtime.run().await;
            })
        };
        let actor_handle = runtime.block_on(build_actor_agent(llm.clone(), actor_runtime.clone()));
        let actor_ref = actor_handle.actor_ref.clone();

        let prompt = config.prompt.clone();
        let llm_message = ChatMessage::user().content(prompt.clone()).build();
        let tool_defs = bench_tool_defs();
        let tool_result_message = bench_tool_result_message();
        let llm_messages = vec![llm_message.clone()];
        let llm_followup = vec![llm_message.clone(), tool_result_message.clone()];

        let _ = runtime.block_on(direct_agent.run(Task::new(prompt.clone())));
        let warm_task = Task::new(prompt.clone());
        let warm_id = warm_task.submission_id;
        actor_ref
            .cast(warm_task)
            .expect("Failed to dispatch warm-up actor task");
        runtime
            .block_on(await_task_complete(&mut actor_events, warm_id))
            .expect("Warm-up actor task failed");

        run_overhead_report(OverheadReportInputs {
            runtime: &runtime,
            llm: &llm,
            direct_agent: &direct_agent,
            actor_ref: &actor_ref,
            actor_events: &mut actor_events,
            prompt: &prompt,
            tools: &tool_defs,
            tool_result_message: &tool_result_message,
        })
        .expect("Failed to compute overhead report");

        let mut group = c.benchmark_group("agent_runtime");

        group.bench_function("llm_no_tools", |b| {
            b.iter(|| {
                let message = llm_message.clone();
                let result = runtime.block_on(llm.chat(std::slice::from_ref(&message), None));
                black_box(result).ok();
            });
        });

        group.bench_function("llm_with_tools", |b| {
            b.iter(|| {
                let result =
                    runtime.block_on(llm.chat_with_tools(&llm_messages, Some(&tool_defs), None));
                let follow_up =
                    runtime.block_on(llm.chat_with_tools(&llm_followup, Some(&tool_defs), None));
                black_box(result).ok();
                black_box(follow_up).ok();
            });
        });

        group.bench_function("direct_agent", |b| {
            b.iter(|| {
                let task = Task::new(prompt.clone());
                let result = runtime.block_on(direct_agent.run(task));
                black_box(result).ok();
            });
        });

        group.bench_function("actor_agent", |b| {
            b.iter(|| {
                let task = Task::new(prompt.clone());
                let submission_id = task.submission_id;
                actor_ref.cast(task).expect("Failed to dispatch actor task");
                runtime
                    .block_on(await_task_complete(&mut actor_events, submission_id))
                    .expect("Actor task failed");
            });
        });

        group.bench_function("direct_agent_rss_delta", |b| {
            b.iter_custom(|iters| {
                let mut local_samples = Vec::new();
                let start = Instant::now();

                for _ in 0..iters {
                    let rss_before = read_rss_kb();
                    let task = Task::new(prompt.clone());
                    let result = runtime.block_on(direct_agent.run(task));
                    black_box(result).ok();
                    let rss_after = read_rss_kb();

                    if let (Some(before), Some(after)) = (rss_before, rss_after) {
                        local_samples.push(MemorySample {
                            rss_before_kb: before,
                            rss_after_kb: after,
                            rss_delta_kb: after.saturating_sub(before),
                        });
                    }
                }

                if !local_samples.is_empty()
                    && let Ok(mut guard) = memory_samples().lock()
                {
                    guard.extend(local_samples);
                }

                start.elapsed()
            });
        });

        group.finish();

        let _ = runtime.block_on(actor_runtime.stop());
        actor_runtime_task.abort();
    }

    write_memory_report();
}

criterion_group! {
    name = benches;
    config = Criterion::default().configure_from_args();
    targets = bench_agent_runtime
}
criterion_main!(benches);
