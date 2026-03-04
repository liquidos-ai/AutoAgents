use std::io::{self, Write};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use autoagents::core::agent::error::RunnableAgentError;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent, DirectAgentHandle};
use autoagents::llm::LLMProvider;
use autoagents::llm::error::LLMError;
use autoagents::llm::optim::{CacheConfig, CacheLayer, ChatCacheKeyMode};
use autoagents::llm::pipeline::PipelineBuilder;
use autoagents_derive::{AgentHooks, agent};
use autoagents_guardrails::guards::{PromptInjectionGuard, RegexPiiRedactionGuard};
use autoagents_guardrails::{EnforcementPolicy, Guardrails};
use autoagents_llamacpp::{LlamaCppProvider, ModelSource};

#[agent(
    name = "safe_local_optimizer_agent",
    description = "You are a secure local assistant named Tess. Keep responses concise and practical.",
    tools = [],
)]
#[derive(Default, Clone, AgentHooks)]
struct SafeLocalOptimizerAgent;

fn read_prompt_from_stdin() -> Result<String> {
    print!("Enter your prompt: ");
    io::stdout()
        .flush()
        .context("failed to flush stdout for prompt")?;

    let mut prompt = String::new();
    io::stdin()
        .read_line(&mut prompt)
        .context("failed to read prompt from stdin")?;

    Ok(prompt.trim().to_string())
}

async fn build_optimized_local_agent()
-> Result<DirectAgentHandle<BasicAgent<SafeLocalOptimizerAgent>>> {
    let llm: Arc<dyn LLMProvider> = PipelineBuilder::new(Arc::new(
        LlamaCppProvider::builder()
            .model_source(ModelSource::HuggingFace {
                repo_id: "Qwen/Qwen3-VL-8B-Instruct-GGUF".to_string(),
                filename: Some("Qwen3VL-8B-Instruct-Q8_0.gguf".to_string()),
                mmproj_filename: Some("mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf".to_string()),
            })
            .n_ctx(4096)
            .max_tokens(256)
            .temperature(0.2)
            .build()
            .await
            .context("failed to build llama.cpp provider")?,
    ) as Arc<dyn LLMProvider>)
    .add_layer(CacheLayer::new(CacheConfig {
        chat_key_mode: ChatCacheKeyMode::UserPromptOnly,
        ttl: Some(Duration::from_secs(900)),
        max_size: Some(512),
        ..CacheConfig::default()
    }))
    .add_layer(
        Guardrails::builder()
            .input_guard(RegexPiiRedactionGuard::default())
            .input_guard(PromptInjectionGuard::default())
            .enforcement_policy(EnforcementPolicy::Block)
            .build()
            .layer(),
    )
    .build();

    let agent = AgentBuilder::<_, DirectAgent>::new(BasicAgent::new(SafeLocalOptimizerAgent))
        .llm(llm)
        .memory(Box::new(SlidingWindowMemory::new(20)))
        .build()
        .await
        .context("failed to build direct agent")?;

    Ok(agent)
}

#[tokio::main]
async fn main() -> Result<()> {
    autoagents::init_logging();
    let agent = build_optimized_local_agent().await?;
    println!("Conversation started. Type 'exit' or 'quit' to stop.");

    loop {
        let prompt = read_prompt_from_stdin()?;
        if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
            break;
        }
        if prompt.is_empty() {
            continue;
        }

        match agent.agent.run(Task::new(prompt)).await {
            Ok(response) => println!("\nResponse:\n{response}\n"),
            Err(error) => match &error {
                RunnableAgentError::LLMError(LLMError::GuardrailBlocked { .. }) => {
                    println!("\nBlocked by guardrails:\n{error}\n");
                    continue;
                }
                RunnableAgentError::LLMError(LLMError::GuardrailExecutionFailed { .. }) => {
                    return Err(anyhow::anyhow!("guardrail execution failed: {error}"));
                }
                _ => {
                    return Err(anyhow::anyhow!("agent execution failed: {error}"));
                }
            },
        }
    }

    Ok(())
}
