use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::ReActAgent;
use autoagents::core::agent::skill::{
    SkillConfiguration, SkillRefreshStrategy, SkillRegistry, SkillSession, TrustedSkillPolicy,
};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, DirectAgent, DirectAgentHandle};
use autoagents::core::error::Error;
use autoagents::core::tool::ToolT;
use autoagents::core::utils::BoxEventStream;
use autoagents::llm::backends::openai::{OpenAI, OpenAIApiMode};
use autoagents::llm::builder::LLMBuilder;
use autoagents::protocol::{Event, SkillEventKind};
use autoagents_derive::AgentHooks;
use serde_json::Value;
use std::env;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::StreamExt;

#[derive(Clone, Debug, AgentHooks)]
struct SkillsAgent;

impl AgentDeriveT for SkillsAgent {
    type Output = String;

    fn name(&self) -> &str {
        "skills_agent"
    }

    fn description(&self) -> &str {
        "Answer the user's request accurately. Use an available Agent Skill whenever its description matches the request."
    }

    fn output_schema(&self) -> Option<serde_json::Value> {
        None
    }

    fn tools(&self) -> Vec<Box<dyn ToolT>> {
        Vec::new()
    }
}

struct SkillsExample;

struct SkillsConsole;

impl SkillsConsole {
    fn print_configuration(base_url: &str, model: &str, skills_directory: &Path) {
        println!("[server] {base_url}");
        println!("[model]  {model}");
        println!("[skills] {}", skills_directory.display());
    }

    fn print_discovered_skills(registry: &SkillRegistry) {
        let snapshot = registry.snapshot();
        println!("[skills] discovered {} skill(s) at boot", snapshot.len());
        for skill in snapshot.skills() {
            println!(
                "         - {}: {}",
                skill.metadata().name(),
                skill.metadata().description()
            );
        }
    }

    fn print_json(value: &Value) {
        match serde_json::to_string_pretty(value) {
            Ok(formatted) => println!("{formatted}"),
            Err(_) => println!("{value}"),
        }
    }

    async fn report_events(mut events: BoxEventStream<Event>) {
        while let Some(event) = events.next().await {
            match event {
                Event::TaskStarted {
                    task_description, ..
                } => println!("\n[agent] task started: {task_description}"),
                Event::TurnStarted {
                    turn_number,
                    max_turns,
                    ..
                } => println!("[agent] turn {turn_number}/{max_turns}"),
                Event::ToolCallRequested {
                    tool_name,
                    arguments,
                    ..
                } => {
                    println!("[tool →] {tool_name}");
                    match serde_json::from_str::<Value>(&arguments) {
                        Ok(arguments) => Self::print_json(&arguments),
                        Err(_) => println!("{arguments}"),
                    }
                }
                Event::ToolCallCompleted {
                    tool_name, result, ..
                } => {
                    println!("[tool ✓] {tool_name}");
                    Self::print_json(&result);
                }
                Event::ToolCallFailed {
                    tool_name, error, ..
                } => println!("[tool ✗] {tool_name}: {error}"),
                Event::Skill { event } => match event.event {
                    SkillEventKind::CatalogChanged {
                        generation,
                        added,
                        updated,
                        removed,
                    } => println!(
                        "[skills] generation {generation}: +{:?} ~{:?} -{:?}",
                        added, updated, removed
                    ),
                    SkillEventKind::ActivationRequested { skill_name } => {
                        println!("[skill →] activate {skill_name}")
                    }
                    SkillEventKind::Activated {
                        skill_name,
                        newly_activated,
                    } => println!("[skill ✓] active {skill_name} (new: {newly_activated})"),
                    SkillEventKind::Deactivated { skill_name, reason } => {
                        println!("[skill] deactivated {skill_name} ({reason:?})")
                    }
                    SkillEventKind::ResourceAccessRequested { skill_name, path } => {
                        println!("[skill →] read {skill_name}/{path}")
                    }
                    SkillEventKind::ResourceAccessed {
                        skill_name,
                        path,
                        bytes,
                    } => println!("[skill ✓] read {skill_name}/{path} ({bytes} bytes)"),
                    SkillEventKind::OperationFailed { message, .. } => {
                        println!("[skill ✗] {message}")
                    }
                },
                Event::TurnCompleted {
                    turn_number,
                    final_turn,
                    ..
                } => println!("[agent] turn {turn_number} complete (final: {final_turn})"),
                Event::TaskComplete { .. } => {
                    println!("[agent] task complete");
                    return;
                }
                Event::TaskError { error, .. } => {
                    println!("[agent] task failed: {error}");
                    return;
                }
                _ => {}
            }
        }
    }

    fn read_prompt() -> Result<Option<String>, Error> {
        print!("\nskills> ");
        io::stdout()
            .flush()
            .map_err(|error| Error::CustomError(format!("cannot flush prompt: {error}")))?;
        let mut prompt = String::new();
        let read = io::stdin()
            .read_line(&mut prompt)
            .map_err(|error| Error::CustomError(format!("cannot read prompt: {error}")))?;
        if read == 0 {
            return Ok(None);
        }
        Ok(Some(prompt.trim().to_string()))
    }
}

impl SkillsExample {
    async fn run() -> Result<(), Error> {
        env_logger::init();
        let api_key = env::var("LLAMA_SERVER_API_KEY")
            .or_else(|_| env::var("OPENAI_API_KEY"))
            .unwrap_or_else(|_| "local".to_string());
        let base_url =
            env::var("LLAMA_SERVER_URL").unwrap_or_else(|_| "http://127.0.0.1:8880/v1".to_string());
        let model = env::var("LLAMA_SERVER_MODEL")
            .unwrap_or_else(|_| "Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf".to_string());
        let skills_directory = env::var_os("AUTOAGENTS_SKILLS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("examples/skills_agent/skills"));
        let prompt = env::args().skip(1).collect::<Vec<_>>().join(" ");

        let llm = LLMBuilder::<OpenAI>::new()
            .api_key(api_key)
            .base_url(&base_url)
            .model(&model)
            .api_mode(OpenAIApiMode::ChatCompletions)
            .max_tokens(1024)
            .temperature(0.2)
            .build()?;
        let skills = Arc::new(SkillRegistry::from_directory(
            &skills_directory,
            SkillRefreshStrategy::Watch {
                debounce: Duration::from_millis(150),
                fallback_interval: Duration::from_secs(30),
            },
        ));
        let session = Arc::new(SkillSession::new());
        let skill_configuration =
            SkillConfiguration::new(Arc::clone(&skills), session, Arc::new(TrustedSkillPolicy));
        SkillsConsole::print_configuration(&base_url, &model, &skills_directory);
        let mut handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(SkillsAgent))
            .llm(llm)
            .memory(Box::new(SlidingWindowMemory::new(20)))
            .skills(skill_configuration)
            .build()
            .await?;
        SkillsConsole::print_discovered_skills(&skills);
        if !prompt.is_empty() {
            Self::run_prompt(&mut handle, prompt).await?;
            return Ok(());
        }

        println!(
            "[ready] enter a prompt, `/skills` to refresh the catalog, `/reset-skills`, or `/exit`"
        );
        while let Some(prompt) = SkillsConsole::read_prompt()? {
            match prompt.as_str() {
                "" => {}
                "/exit" => break,
                "/skills" => {
                    let report = skills.refresh_now().await;
                    println!(
                        "[skills] generation {}: {} discovered, +{:?} ~{:?} -{:?}",
                        report.generation(),
                        report.discovered(),
                        report.added(),
                        report.updated(),
                        report.removed()
                    );
                    SkillsConsole::print_discovered_skills(&skills);
                }
                "/reset-skills" => {
                    handle.agent.reset_skill_session().await;
                    println!("[skills] conversation activation state reset");
                }
                _ => Self::run_prompt(&mut handle, prompt).await?,
            }
        }
        Ok(())
    }

    async fn run_prompt(
        handle: &mut DirectAgentHandle<ReActAgent<SkillsAgent>>,
        prompt: String,
    ) -> Result<(), Error> {
        let reporter = tokio::spawn(SkillsConsole::report_events(handle.subscribe_events()));
        let response = handle.agent.run(Task::new(prompt)).await?;
        reporter
            .await
            .map_err(|error| Error::CustomError(format!("console reporter failed: {error}")))?;
        println!("\n[response]\n{response}");
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    SkillsExample::run().await
}
