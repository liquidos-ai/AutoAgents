use autoagents::core::error::Error;
use autoagents::core::runtime::{GrpcClientConfig, GrpcRuntimeClient};
use colored::*;
use std::time::Duration;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Error> {
    env_logger::init();

    println!("{}", "gRPC Hello World Client".green().bold());
    println!("{}", "======================".green());

    // Create a temporary client ID
    let client_id = Uuid::new_v4();
    let client_name = "hello_client".to_string();

    // Configure gRPC client
    let config = GrpcClientConfig {
        server_addr: "http://127.0.0.1:50051".to_string(),
        connect_timeout: Duration::from_secs(10),
        request_timeout: Duration::from_secs(30),
        keep_alive_interval: Duration::from_secs(10),
        max_retries: 3,
    };

    println!("{}", "Connecting to gRPC server...".yellow());

    // Connect to server
    let mut client = match GrpcRuntimeClient::connect(config, client_id, client_name.clone()).await
    {
        Ok(c) => c,
        Err(e) => {
            println!("{}", format!("❌ Failed to connect: {}", e).red());
            println!("{}", "Make sure the server is running!".yellow());
            return Err(e);
        }
    };

    println!("{}", "✅ Connected to server!".green());

    // Example 1: Send a greeting
    println!("\n{}", "Example 1: Sending a greeting...".cyan());
    client
        .publish_message(
            "greetings".to_string(),
            "Hello! Please greet me in a creative way.".to_string(),
        )
        .await?;
    println!("{}", "✅ Greeting sent!".green());

    // Wait a bit for the response
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Example 2: Ask a question
    println!("\n{}", "Example 2: Asking a question...".cyan());
    client
        .publish_message(
            "questions".to_string(),
            "What is the capital of France?".to_string(),
        )
        .await?;
    println!("{}", "✅ Question sent!".green());

    // Wait a bit for the response
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Example 3: Another greeting
    println!("\n{}", "Example 3: Another greeting...".cyan());
    client
        .publish_message(
            "greetings".to_string(),
            "Say goodbye in 5 different languages!".to_string(),
        )
        .await?;
    println!("{}", "✅ Request sent!".green());

    // Wait for responses
    tokio::time::sleep(Duration::from_secs(3)).await;

    println!("\n{}", "Disconnecting from server...".yellow());
    client.shutdown().await?;

    println!("{}", "✅ Done!".green().bold());

    Ok(())
}
