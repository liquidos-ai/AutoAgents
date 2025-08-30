use async_trait::async_trait;
use autoagents::core::actor::Topic;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentExecutor, Context, ExecutorConfig};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{ClusterClientRuntime, ClusterHostRuntime};
use autoagents::core::runtime::{Runtime, TypedRuntime};
use autoagents::core::tool::ToolT;
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::chat::{ChatMessage, ChatRole, MessageType};
use autoagents_derive::agent;
use colored::*;
use serde_json::Value;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

#[agent(
    name = "research_agent",
    description = "You are a research agent that gathers information and facts about topics. Your job is to research the given topic thoroughly and provide comprehensive information that can be used for analysis.",
    tools = [],
)]
pub struct ResearchAgent {}

#[agent(
    name = "analysis_agent",
    description = "You are an analysis agent that receives research data and performs deep analysis. Your job is to analyze the research provided, identify key insights, trends, and provide recommendations based on the data.",
    tools = [],
)]
pub struct AnalysisAgent {}

#[async_trait]
impl AgentExecutor for ResearchAgent {
    type Output = String;
    type Error = Error;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig { max_turns: 10 }
    }

    async fn execute(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        println!("🔍 [ResearchAgent] Starting research on: {}", task.prompt);

        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            message_type: MessageType::Text,
            content: format!(
                "{} Focus on gathering factual information, statistics, current trends, and key data points. Provide a comprehensive research summary that will be useful for further analysis.",
                context.config().description
            ),
        }];

        let research_prompt = format!(
            "Research the following topic thoroughly: {}

Provide:
1. Key facts and current statistics
2. Recent developments and trends
3. Main challenges or opportunities
4. Relevant background context
5. Important data points for analysis

Format your response as a detailed research report.",
            task.prompt
        );

        let chat_msg = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: research_prompt,
        };
        messages.push(chat_msg);

        let response = context
            .llm()
            .chat(&messages, None, context.config().output_schema.clone())
            .await?;
        let research_data = response.text().unwrap_or_default();

        println!("📊 [ResearchAgent] Research completed. Sending to AnalysisAgent...");

        // Send research data to analysis agent running on different node
        let analysis_task = Task::new(format!(
            "RESEARCH DATA FOR ANALYSIS:

Original Topic: {}

Research Findings:
{}

Please analyze this research data and provide insights, recommendations, and strategic conclusions.",
            task.prompt, research_data
        ));

        // Send to cluster for distributed processing - the runtime will handle cross-cluster forwarding
        context
            .publish(Topic::<Task>::new("analysis_agent"), analysis_task)
            .await?;

        Ok(format!("Research completed for topic: '{}'. Data sent to AnalysisAgent for further processing.", task.prompt))
    }
}

#[async_trait]
impl AgentExecutor for AnalysisAgent {
    type Output = String;
    type Error = Error;

    fn config(&self) -> ExecutorConfig {
        ExecutorConfig { max_turns: 10 }
    }

    async fn execute(
        &self,
        task: &Task,
        context: Arc<Context>,
    ) -> Result<Self::Output, Self::Error> {
        println!("🧠 [AnalysisAgent] Received research data for analysis");

        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            message_type: MessageType::Text,
            content: format!(
                "{} You will receive research data and should provide deep analytical insights, identify patterns, make recommendations, and draw strategic conclusions.",
                context.config().description
            ),
        }];

        let analysis_prompt = format!(
            "{}

Based on this research data, provide:
1. Key insights and patterns identified
2. Strategic recommendations
3. Risk assessment and opportunities
4. Actionable next steps
5. Executive summary of findings

Provide a comprehensive analysis report.",
            task.prompt
        );

        let chat_msg = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: analysis_prompt,
        };
        messages.push(chat_msg);

        let response = context
            .llm()
            .chat(&messages, None, context.config().output_schema.clone())
            .await?;
        let analysis_result = response.text().unwrap_or_default();

        println!("📈 [AnalysisAgent] Analysis completed!");
        println!("\n{}", "=".repeat(80));
        println!("🎯 FINAL ANALYSIS REPORT:");
        println!("{}", "=".repeat(80));
        println!("{}", analysis_result);
        println!("{}\n", "=".repeat(80));

        Ok(analysis_result)
    }
}

pub async fn run_research_agent(
    llm: Arc<OpenAI>,
    node_name: String,
    port: u16,
    host_addr: String,
    host: String,
) -> Result<(), Error> {
    println!(
        "🔍 Initializing ResearchAgent cluster client on port {}",
        port
    );

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
    let research_topic = Topic::<Task>::new("research_agent");

    // Create cluster client runtime for ResearchAgent - it will connect to dedicated cluster host
    let runtime = ClusterClientRuntime::new(
        "research_client".to_string(),
        host_addr.clone(),
        node_name,
        "cluster-cookie".to_string(),
        port,
        host,
    );

    let research_agent = ResearchAgent {};

    // Build and register ResearchAgent
    let _ = AgentBuilder::new(research_agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic(research_topic.clone())
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Cluster communication is now handled automatically by the ClusterRuntime

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    // Start the runtime and environment
    tokio::spawn(async move {
        if let Err(e) = environment.run().await {
            eprintln!("Environment error: {}", e);
        }
    });

    // Connection to host is handled automatically in ClusterClientRuntime
    println!(
        "🌐 ClusterClientRuntime will connect to cluster host at {}",
        host_addr
    );
    sleep(Duration::from_secs(2)).await;

    // If this is the initiating node, send research tasks after cluster is ready
    if port == 9001 {
        sleep(Duration::from_secs(3)).await;

        let research_topics = vec!["Artificial Intelligence trends in 2024"];

        for topic in research_topics {
            println!("📋 Starting research task: {}", topic);
            runtime.publish(&research_topic, Task::new(topic)).await?;
            sleep(Duration::from_secs(5)).await; // Space out the tasks
        }
    }

    // Keep running until Ctrl+C
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for Ctrl+C");
    println!("🔍 Shutting down ResearchAgent...");
    if let Err(e) = runtime.stop().await {
        eprintln!("Error stopping runtime: {}", e);
    }

    Ok(())
}

pub async fn run_analysis_agent(
    llm: Arc<OpenAI>,
    node_name: String,
    port: u16,
    host_addr: String,
    host: String,
) -> Result<(), Error> {
    println!(
        "🧠 Initializing AnalysisAgent cluster client on port {}",
        port
    );

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
    let analysis_topic = Topic::<Task>::new("analysis_agent");

    // Create cluster client runtime for AnalysisAgent - it will connect to dedicated cluster host
    let runtime = ClusterClientRuntime::new(
        "analysis_client".to_string(),
        host_addr.clone(),
        node_name,
        "cluster-cookie".to_string(),
        port,
        host,
    );

    let analysis_agent = AnalysisAgent {};

    // Build and register AnalysisAgent
    let _ = AgentBuilder::new(analysis_agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic(analysis_topic.clone())
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Cluster communication is now handled automatically by the ClusterRuntime

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    // Start the runtime and environment
    tokio::spawn(async move {
        if let Err(e) = environment.run().await {
            eprintln!("Environment error: {}", e);
        }
    });

    // Connection to host is handled automatically in ClusterClientRuntime
    println!(
        "🌐 ClusterClientRuntime will connect to cluster host at {}",
        host_addr
    );

    println!("🧠 AnalysisAgent client ready to receive research data for analysis...");

    // Keep running until Ctrl+C
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for Ctrl+C");
    println!("🧠 Shutting down AnalysisAgent...");
    if let Err(e) = runtime.stop().await {
        eprintln!("Error stopping runtime: {}", e);
    }

    Ok(())
}

pub async fn run_cluster_host(node_name: String, port: u16, host: String) -> Result<(), Error> {
    println!("🏠 Initializing ClusterHostRuntime on port {}", port);

    // Create cluster host runtime - this coordinates all client connections and routes events
    let runtime = ClusterHostRuntime::new(node_name, "cluster-cookie".to_string(), port, host);

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    // Start the runtime and environment
    tokio::spawn(async move {
        if let Err(e) = environment.run().await {
            eprintln!("Environment error: {}", e);
        }
    });

    println!("🏠 ClusterHostRuntime ready to coordinate client connections and route events...");

    // Keep running until Ctrl+C
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for Ctrl+C");
    println!("🏠 Shutting down ClusterHostRuntime...");
    if let Err(e) = runtime.stop().await {
        eprintln!("Error stopping runtime: {}", e);
    }

    Ok(())
}

fn handle_events(mut event_stream: ReceiverStream<Event>) {
    tokio::spawn(async move {
        while let Some(event) = event_stream.next().await {
            match event {
                Event::NewTask { actor_id: _, task } => {
                    println!("{}", format!("📨 New TASK: {:?}", task).green());
                }
                Event::TaskComplete {
                    result: TaskResult::Value(val),
                    ..
                } => {
                    let agent_out: String = serde_json::from_value(val).unwrap();
                    println!("{}", format!("✅ Agent Response: {}", agent_out).green());
                }
                _ => {
                    //
                }
            }
        }
    });
}
