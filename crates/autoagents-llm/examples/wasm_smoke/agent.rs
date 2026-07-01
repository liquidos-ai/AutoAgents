//! Provider-level WASM smoke test for the OpenAI Responses path on WASI HTTP.
//!
//! Build:
//!   RUSTFLAGS="-C linker=wasm-component-ld" cargo build -p autoagents-llm \
//!     --target wasm32-wasip2 --features wasi-http,openai --example wasm_agent
//! Run:
//!   wasmtime run -W component-model=y -S http=true \
//!     --env BASE_URL=http://127.0.0.1:18765/v1 \
//!     --env OPENAI_API_KEY=sk-test \
//!     target/wasm32-wasip2/debug/examples/wasm_agent.wasm
//!
//! Optional environment variables:
//! - MODEL: model name (defaults to gpt-4o for the local mock)
//! - STRICT_MOCK_EXPECTATIONS: true by default; set false for real APIs whose
//!   exact text/reasoning/usage output differs from the local mock
//! - RUN_ERROR_TEST: true by default; set false for real APIs unless you intend
//!   to trigger a real 429
//! - REASONING_EFFORT: optional OpenAI Responses reasoning effort (for example,
//!   medium) for models/endpoints that expose reasoning deltas
//! - REASONING_SUMMARY: optional OpenAI Responses reasoning summary mode (for
//!   example, detailed) for endpoints that stream reasoning summaries
//! - NON_STREAM_PROMPT / STREAM_PROMPT: optional prompts for real API smoke runs

use autoagents_llm::backends::openai::{OpenAI, OpenAIApiMode};
use autoagents_llm::chat::{ChatMessage, ChatProvider};
use autoagents_llm::error::LLMError;
use futures::StreamExt;
use serde_json::{Map, Value, json};

fn main() {
    if let Err(err) = futures::executor::block_on(run()) {
        eprintln!("WASM provider smoke failed: {err}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), LLMError> {
    let base_url = std::env::var("BASE_URL")
        .unwrap_or_else(|_| panic!("Please set BASE_URL environment variable"));
    let api_key = std::env::var("OPENAI_API_KEY")
        .unwrap_or_else(|_| panic!("Please set OPENAI_API_KEY environment variable"));
    let model = std::env::var("MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
    let strict_mock_expectations = env_flag("STRICT_MOCK_EXPECTATIONS", true);
    let run_error_test = env_flag("RUN_ERROR_TEST", true);
    let reasoning_effort = std::env::var("REASONING_EFFORT").ok();
    let reasoning_summary = std::env::var("REASONING_SUMMARY").ok();

    let provider = build_provider(
        api_key,
        base_url,
        model,
        reasoning_effort,
        reasoning_summary,
    )?;

    let non_stream_prompt = std::env::var("NON_STREAM_PROMPT")
        .unwrap_or_else(|_| "Hello from provider-level WASM smoke".to_string());
    let non_stream_messages = [ChatMessage::user().content(non_stream_prompt).build()];
    let response = provider
        .chat_with_tools(&non_stream_messages, None, None)
        .await?;
    let text = response.text().unwrap_or_default();
    if strict_mock_expectations && text != "hello from mock" {
        return Err(LLMError::ResponseFormatError {
            message: "unexpected non-streaming provider response text".to_string(),
            raw_response: text,
        });
    }
    if !strict_mock_expectations && text.trim().is_empty() {
        return Err(LLMError::ResponseFormatError {
            message: "real API non-streaming response text was empty".to_string(),
            raw_response: text,
        });
    }
    println!("PROVIDER_NON_STREAM_OK text_len={}", text.len());
    if !strict_mock_expectations {
        println!("PROVIDER_NON_STREAM_TEXT {}", preview(&text, 240));
    }

    let stream_prompt = std::env::var("STREAM_PROMPT")
        .unwrap_or_else(|_| "Hello from provider-level WASM smoke stream".to_string());
    let stream_messages = [ChatMessage::user().content(stream_prompt).build()];
    let mut stream = provider
        .chat_stream_struct(&stream_messages, None, None)
        .await?;
    let mut saw_text_delta = false;
    let mut saw_reasoning_delta = false;
    let mut saw_tool_call = false;
    let mut saw_usage = false;
    let mut streamed_text = String::new();
    let mut streamed_reasoning = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = choice.delta.content.as_deref()
                && !content.is_empty()
            {
                saw_text_delta = true;
                streamed_text.push_str(content);
            }
            if let Some(reasoning) = choice.delta.reasoning_content.as_deref()
                && !reasoning.is_empty()
            {
                saw_reasoning_delta = true;
                streamed_reasoning.push_str(reasoning);
            }
            if choice
                .delta
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())
            {
                // Function-call streaming events also prove that the Responses SSE parser ran.
                saw_tool_call = true;
            }
        }
        saw_usage |= chunk.usage.is_some();
    }
    if strict_mock_expectations
        && (!saw_text_delta || !saw_reasoning_delta || !saw_tool_call || !saw_usage)
    {
        return Err(LLMError::ResponseFormatError {
            message: format!(
                "stream parser did not observe expected mock markers: text={saw_text_delta} reasoning={saw_reasoning_delta} tool_call={saw_tool_call} usage={saw_usage}"
            ),
            raw_response: String::new(),
        });
    }
    if !strict_mock_expectations && !saw_text_delta {
        return Err(LLMError::ResponseFormatError {
            message: "real API stream did not produce a text delta".to_string(),
            raw_response: String::new(),
        });
    }
    println!(
        "PROVIDER_STREAM_OK text_delta={saw_text_delta} reasoning_delta={saw_reasoning_delta} tool_call={saw_tool_call} usage={saw_usage}"
    );
    if !strict_mock_expectations {
        println!("PROVIDER_STREAM_TEXT {}", preview(&streamed_text, 240));
        if saw_reasoning_delta {
            println!(
                "PROVIDER_STREAM_REASONING {}",
                preview(&streamed_reasoning, 240)
            );
        }
    }

    if run_error_test {
        let error_messages = [ChatMessage::user()
            .content("please trigger rate_limit")
            .build()];
        match provider.chat_with_tools(&error_messages, None, None).await {
            Err(err) if err.http_status_code() == Some(429) => {
                println!("PROVIDER_ERROR_429_OK {err}");
            }
            Err(err) => return Err(err),
            Ok(response) => {
                return Err(LLMError::ResponseFormatError {
                    message: "expected provider-level 429 error, got success".to_string(),
                    raw_response: response.to_string(),
                });
            }
        }
    } else {
        println!("PROVIDER_ERROR_429_SKIPPED");
    }

    Ok(())
}

fn build_provider(
    api_key: String,
    base_url: String,
    model: String,
    reasoning_effort: Option<String>,
    reasoning_summary: Option<String>,
) -> Result<OpenAI, LLMError> {
    let mut extra_body = Map::new();
    let native_reasoning_effort = if let Some(summary) = reasoning_summary {
        let effort = reasoning_effort.unwrap_or_else(|| "medium".to_string());
        extra_body.insert(
            "reasoning".to_string(),
            json!({
                "effort": effort,
                "summary": summary,
            }),
        );
        None
    } else {
        reasoning_effort
    };
    let extra_body = if extra_body.is_empty() {
        None
    } else {
        Some(Value::Object(extra_body))
    };

    OpenAI::new(
        api_key,
        Some(base_url),
        Some(model),
        None,
        None,
        Some(30),
        None,
        None,
        None,
        None,
        None,
        None,
        native_reasoning_effort,
        None,
        OpenAIApiMode::Responses,
        extra_body,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => default,
    }
}

fn preview(text: &str, max_chars: usize) -> String {
    let mut preview: String = text.chars().take(max_chars).collect();
    if text.chars().count() > max_chars {
        preview.push('…');
    }
    preview.replace('\n', "\\n")
}
