#![allow(dead_code, unused_imports)]
use autoagents::{
    llm::chat::{ChatMessage, ChatRole, MessageType},
    llm::LLMProvider,
};
use autoagents_derive::{tool, ToolInput};
use autoagents_llm::{ToolInputT, ToolT};
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{alloc::System, time::SystemTime};

#[derive(Serialize, Deserialize, ToolInput)]
pub struct GetCurrentWeatherArgs {}

#[tool(
    name = "GetCurrentWeather",
    description = "Use this tool to get the current Weather",
    input = GetCurrentWeatherArgs,
    output = String
)]
fn get_current_weather(_args: GetCurrentWeatherArgs) -> String {
    // Dummy implementation.
    format!("Current Time is {:?}", SystemTime::now())
}

pub async fn stream(llm: impl LLMProvider) {
    // llm.register_tool(Box::new(GetCurrentWeather));
    let mut stream = llm
        .chat_stream(
            vec![
                ChatMessage {
                    role: ChatRole::Assistant,
                    message_type: MessageType::Text,
                    content: "You are an Assistant and your name is Lilly".into(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    message_type: MessageType::Text,
                    content: "Hello!".into(),
                },
            ]
            .as_slice(),
        )
        .await
        .unwrap();

    // Iterate over the stream as chunks arrive.
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(response) => {
                // Here, `response` is of type TextGenerationResponse.
                println!("Received chunk: {}", response);
            }
            Err(e) => {
                eprintln!("Error in streaming response: {}", e);
            }
        }
    }
}
