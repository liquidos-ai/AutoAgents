//! Integration test for the grammar-without-tools chat path.
//!
//! Issue: `chat_with_tools(messages, None, Some(StructuredOutputFormat))` previously
//! returned `Provider Error: Failed to parse response: ffi error -3` on the first
//! call for chat-format handlers (verified for Gemma 4 GGUF, expected for any
//! handler whose emitted parser expects an envelope). Grammar-constrained
//! generations produce plain JSON with no envelope, so the parser's
//! `common_chat_parse` step raised an exception, surfacing as rc=-3 in
//! `llama_rs_chat_parse_to_oaicompat`.
//!
//! Unit-level coverage of `select_template_schema_and_grammar` was insufficient
//! to catch this — the helper returns `(None, Some(grammar))` correctly, but
//! the parser + grammar_lazy + trigger configuration happens inside the C++
//! chat-template engine, not at the schema-vs-grammar routing layer.
//!
//! This integration test loads a real GGUF and asserts the full call chain
//! returns schema-conforming JSON. Opt-in via `LLAMACPP_TEST_MODEL_PATH`
//! pointing at an instruction-tuned model.
//!
//! ## Why a single test
//!
//! `llama_cpp_2`'s backend is a process-global singleton: a second
//! `LlamaCppProvider::from_config` in the same process fails with
//! `BackendAlreadyInitialized`. Until upstream provides a shared backend
//! fixture or per-test process isolation, we cover the most-direct failure
//! shape (grammar + thinking + no tools) which exercises the full fix path
//! (parser-drop + grammar_lazy=false + cleared triggers).

use autoagents_llamacpp::{LlamaCppConfigBuilder, LlamaCppProvider, LlamaCppReasoningFormat};
use autoagents_llm::chat::{
    ChatMessage, ChatProvider, ChatRole, MessageType, StructuredOutputFormat,
};

/// Env var that opts this `#[ignore]`d test into a real-model run. Kept as a
/// const so spelling typos at the call site become compile errors rather than
/// silent skips.
const MODEL_PATH_ENV: &str = "LLAMACPP_TEST_MODEL_PATH";

const VERDICT_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "is_correct": { "type": "boolean" },
        "is_partial": { "type": "boolean" },
        "reasoning": { "type": "string" }
    },
    "required": ["is_correct", "is_partial", "reasoning"],
    "additionalProperties": false
}"#;

/// Strongly directive prompt — the previous failure manifested as the model
/// emitting markdown fences before the JSON body when prompts were vague.
/// Eager grammar enforcement should reject pre-`{` tokens regardless, but
/// pairing with a directive prompt keeps the test signal focused on the fix
/// rather than instruction-following variance.
const PROMPT: &str = r#"You are an expert evaluator. Judge whether the ANSWER is correct given the CONTEXT.

<question>
What does Dave do for work?
</question>

<context>
Dave is a senior software engineer at Acme Corp.
</context>

<answer>
Dave is a senior software engineer at Acme Corp.
</answer>

Respond ONLY with a JSON object matching this schema:
{
  "is_correct": <bool>,
  "is_partial": <bool>,
  "reasoning": "<one short sentence>"
}"#;

fn schema() -> StructuredOutputFormat {
    StructuredOutputFormat {
        name: "JudgeVerdict".to_string(),
        description: Some("Verdict for a context-answer pair".to_string()),
        schema: Some(serde_json::from_str(VERDICT_SCHEMA).expect("valid schema literal")),
        strict: Some(true),
    }
}

/// Extract a balanced top-level JSON object substring from raw text.
///
/// Walks from the first `{`, brace-counting (string-aware so `}` inside a
/// `"..."` literal does not close the object) until depth returns to zero.
/// This is robust against a leaked `<think>` block or reasoning preamble that
/// itself contains `{` or `}` — `find/rfind` alone would mismatch in that
/// case. With the fix applied raw output is already plain JSON and this is a
/// no-op; the matcher exists as a safety net for chat-template variations.
fn extract_json(text: &str) -> &str {
    let bytes = text.as_bytes();
    let start = text.find('{').expect("response must contain a JSON object");
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (idx, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &text[start..=idx];
                }
            }
            _ => {}
        }
    }
    panic!("response must contain a balanced JSON object");
}

/// Regression test: grammar + thinking + no tools. Pre-fix failure mode was
/// `Provider Error: Failed to parse response: ffi error -3`. Post-fix the
/// model must return schema-conforming JSON.
#[tokio::test]
#[ignore = "requires LLAMACPP_TEST_MODEL_PATH pointing at an instruction-tuned GGUF"]
async fn chat_with_tools_grammar_no_envelope_returns_schema_conforming_json() {
    let Ok(model_path) = std::env::var(MODEL_PATH_ENV) else {
        eprintln!("skip: set {MODEL_PATH_ENV} to enable");
        return;
    };

    let config = LlamaCppConfigBuilder::new()
        .model_path(&model_path)
        .max_tokens(1024)
        .temperature(0.0)
        .seed(42)
        .reasoning_format(LlamaCppReasoningFormat::Auto)
        .extra_body(serde_json::json!({
            "chat_template_kwargs": { "enable_thinking": true }
        }))
        .build();

    let provider = LlamaCppProvider::from_config(config)
        .await
        .expect("provider should load");

    let messages = vec![ChatMessage {
        role: ChatRole::User,
        message_type: MessageType::Text,
        content: PROMPT.to_string(),
    }];

    let response = provider
        .chat_with_tools(&messages, None, Some(schema()))
        .await
        .expect("chat_with_tools with grammar + no tools must not return ffi error -3");

    let text = response.text().unwrap_or_default();
    assert!(!text.is_empty(), "response content must not be empty");

    let json_slice = extract_json(&text);
    let parsed: serde_json::Value = serde_json::from_str(json_slice).unwrap_or_else(|err| {
        panic!("response must contain valid JSON ({err}); got slice: {json_slice}");
    });

    assert!(
        parsed.get("is_correct").is_some(),
        "schema requires `is_correct`, got: {parsed}"
    );
    assert!(
        parsed.get("is_partial").is_some(),
        "schema requires `is_partial`, got: {parsed}"
    );
    assert!(
        parsed.get("reasoning").is_some(),
        "schema requires `reasoning`, got: {parsed}"
    );
}
