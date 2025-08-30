# Image Chat Example

This example demonstrates how to use AutoAgents with image messages using OpenAI's vision-capable models.

## Prerequisites

You'll need an OpenAI API key and a test image file:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Place your test image as `test_img.jpg` in the project root directory.

## Usage

Simply run the example:

```bash
cargo run --bin image_chat
```

The example will:
1. Read `test_img.jpg` from the current directory
2. Send it to OpenAI's GPT-4o model with the prompt "What do you see in this image?"
3. Display the response

## Code Structure

The example is minimal and straightforward:

```rust
use autoagents::llm::{
    backends::openai::OpenAI,
    builder::LLMBuilder,
    chat::{ChatMessage, ImageMime},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Read the test image
    let image_bytes = tokio::fs::read("test_img.jpg").await?;

    // Create OpenAI client
    let llm = LLMBuilder::<OpenAI>::new()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o")
        .build()?;

    // Create message with image
    let message = ChatMessage::user()
        .content("What do you see in this image?")
        .image(ImageMime::JPEG, image_bytes)
        .build();

    // Send to OpenAI and get response
    let response = llm.chat(&[message], None, None).await?;
    println!("Response: {}", response.text().unwrap_or_default());

    Ok(())
}
```

## Requirements

- OpenAI API key set as environment variable
- Test image file named `test_img.jpg` in the project root
- Rust with tokio async runtime
