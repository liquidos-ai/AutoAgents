use autoagents::core::tool::ToolCallError;
use once_cell::sync::Lazy;
use reqwest::Url;
use std::env;

pub mod llm_api;
pub mod recognizer;
pub mod short_answer;

pub use llm_api::WolframAlphaLLMApi;
pub use recognizer::{RecognizerMode, WolframAlphaQueryRecognizer};
pub use short_answer::{ShortAnswerUnits, WolframAlphaShortAnswer};

static WOLFRAM_APP_ID: Lazy<String> = Lazy::new(|| {
    env::var("WOLFRAM_ALPHA_APP_ID")
        .or_else(|_| env::var("WOLFRAM_APP_ID"))
        .expect("WOLFRAM_ALPHA_APP_ID or WOLFRAM_APP_ID must be set")
});

pub(crate) fn wolfram_app_id() -> String {
    WOLFRAM_APP_ID.clone()
}

pub(crate) fn wolfram_input_url(input: &str) -> Result<String, ToolCallError> {
    let mut url = Url::parse("https://www.wolframalpha.com/input")
        .map_err(|err| ToolCallError::RuntimeError(Box::new(err)))?;

    url.query_pairs_mut().append_pair("i", input);

    Ok(url.to_string())
}
