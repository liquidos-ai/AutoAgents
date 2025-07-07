# AutoAgents

**AutoAgents** is a distributed multi-agent framework written in Rust that enables you to build, deploy, and coordinate multiple intelligent agents. AutoAgents leverages modern language models and tool integrations to provide robust, scalable, and flexible solutions for various use cases—from natural language processing and automated chat to advanced tool orchestration.

---

## Features

- **Distributed Architecture:** Coordinate multiple agents across different nodes.
- **Multi-Agent Coordination:** Allow agents to interact, share knowledge, and collaborate on complex tasks.
- **Tool Integration:** Seamlessly integrate with external tools and services to extend agent capabilities.
- **Extensible Framework:** Easily add new providers and tools using our intuitive plugin system.

---

## Supported Providers

**multiple LLM backends** in a single project: [OpenAI](https://openai.com), [Anthropic (Claude)](https://www.anthropic.com), [Ollama](https://github.com/ollama/ollama), [DeepSeek](https://www.deepseek.com), [xAI](https://x.ai), [Phind](https://www.phind.com), [Groq](https://www.groq.com), [Google](https://cloud.google.com/gemini).

*Note: Provider support is actively evolving.*


## Instructions to build
```sh
git clone https://github.com/yourusername/autoagents.git
cd autoagents
cargo build --release
```

## Usage

### LLM Tool Calling
```rs
#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WeatherArgs {}

#[tool(
    name = "WeatherTool",
    description = "Use this tool to get current weather in celcius",
    input = WeatherArgs,
    output = String
)]
fn get_weather(args: WeatherArgs) -> String {
    println!("ToolCall: GetWeather {:?}", args);
    "23 degree".into()
}

#[agent(
    name = "general_agent",
    prompt = "You are general assistant and will answer user quesries in crips manner.",
    tools = [WeatherTool]
)]
pub struct WeatherAgent {}

fn handle_events(event_stream: Option<ReceiverStream<Event>>) {
    if let Some(mut event_stream) = event_stream {
        tokio::spawn(async move {
            while let Some(event) = event_stream.next().await {
                println!("Event: {:?}", event);
            }
        });
    }
}

pub async fn simple_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    // Build a Simple agent
    let agent = SimpleAgentBuilder::from_agent(WeatherAgent {})
        .with_llm(llm)
        .build()
        .unwrap();
    let mut environment = Environment::new(None).await;
    let receiver = environment.take_event_receiver(None).await;
    handle_events(receiver);

    let agent_id = environment.register_agent(agent, None).await?;
    let _ = environment
        .add_task(agent_id, "What is the current weather??")
        .await?;
    let _ = environment
        .add_task(
            agent_id,
            "What is the weather you just said? reply back in beautiful format.",
        )
        .await?;

    let result = environment.run_all(agent_id, None).await?;
    println!("Result {:?}", result);
    let _ = environment.shutdown().await;
    Ok(())
}
```

## TODO
- [ ] [FEATURE] Add Cost Tracking in LLM Crate
- [ ] [REFACTOR] Proper implemetation of streaming and HTTP requests
- [ ] [FEATURE] Add ReAct Agent type
- [ ] [REFACTOR] Refactor the Provider implementation where, Provider needs to do as little implemetation as possible,
  i.e take up the heavy lifting
- [ ] [COVERAGE] Add tests
