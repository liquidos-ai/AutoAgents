// This example demonstrates various LLM Provider methods
use autoagents::llm::builder::{FunctionBuilder, ParamBuilder};
use autoagents::{
    llm::{chat::ChatMessage, LLMProvider},
    prelude::Error,
};
use std::sync::Arc;
use tokio_stream::StreamExt;

pub async fn run_llm(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    let message = ChatMessage::user().content("Hello, Who are you?").build();

    //Single Chat Request
    let chat_response = llm.chat(&[message.clone()], None).await?;
    println!("Chat Response: {:?}", chat_response);

    //Stream without structured output
    let mut stream = llm.chat_stream(&[message.clone()], None).await?;
    while let Some(result) = stream.next().await {
        match result {
            Ok(output) => {
                println!("{}", format!("Streaming Response: {}", output));
            }
            _ => {
                //
            }
        }
    }
    println!("Running Stram with structured");

    //Stream with structured output
    let mut stream = llm
        .chat_stream_struct(&[message.clone()], None, None)
        .await?;
    while let Some(result) = stream.next().await {
        match result {
            Ok(output) => {
                println!("{}", format!("Streaming Response: {:?}", output));
            }
            _ => {
                //
            }
        }
    }
    println!("Running Stram With Tool struct");

    let tool = FunctionBuilder::new("weather_function")
        .description("Use this tool to get the weather in a specific city")
        .param(
            ParamBuilder::new("city")
                .type_of("string")
                .description("The city to get the weather for"),
        )
        .required(vec!["city".to_string()])
        .build();

    //Stream with structured output
    let message = ChatMessage::user()
        .content("Hello, What is the current weather in new york?")
        .build();
    let mut stream = llm
        .chat_stream_struct(&[message.clone()], Some(&[tool.clone()]), None)
        .await?;
    while let Some(result) = stream.next().await {
        match result {
            Ok(output) => {
                println!("{}", format!("Streaming Response: {:?}", output));
            }
            _ => {
                //
            }
        }
    }

    println!("Running Stram With Tool");
    //Stream with structured output
    let message = ChatMessage::user()
        .content("Hello, What is the current weather in new york?")
        .build();
    let mut stream = llm
        .chat_stream_with_tools(&[message.clone()], Some(&[tool]), None)
        .await?;
    while let Some(result) = stream.next().await {
        match result {
            Ok(output) => {
                println!("{}", format!("Streaming Response: {:?}", output));
            }
            _ => {
                //
            }
        }
    }

    Ok(())
}
