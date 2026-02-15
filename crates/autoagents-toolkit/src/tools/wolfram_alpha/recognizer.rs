use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT},
};
use autoagents_derive::{ToolInput, tool};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{wolfram_app_id, wolfram_input_url};

const RECOGNIZER_ENDPOINT: &str = "https://www.wolframalpha.com/queryrecognizer/query.jsp";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum RecognizerMode {
    #[serde(rename = "Default")]
    #[default]
    Default,
    #[serde(rename = "Voice")]
    Voice,
}

impl RecognizerMode {
    fn as_str(&self) -> &'static str {
        match self {
            RecognizerMode::Default => "Default",
            RecognizerMode::Voice => "Voice",
        }
    }
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WolframAlphaQueryRecognizerArgs {
    #[input(description = "Query to classify for Wolfram|Alpha suitability")]
    query: String,
    #[serde(default)]
    #[input(description = "Recognizer mode (Default for typed input, Voice for spoken input)")]
    mode: Option<RecognizerMode>,
}

#[tool(
    name = "wolfram_alpha_query_recognizer",
    description = "Classify whether a user query can be handled by WolframAlpha tools for exact mathematical, scientific, or symbolic computation."
    input = WolframAlphaQueryRecognizerArgs,
)]
pub struct WolframAlphaQueryRecognizer {
    app_id: String,
    client: Client,
}

impl Default for WolframAlphaQueryRecognizer {
    fn default() -> Self {
        Self::new_with_app_id(wolfram_app_id())
    }
}

impl WolframAlphaQueryRecognizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_app_id(app_id: String) -> Self {
        Self {
            app_id,
            client: Client::new(),
        }
    }

    async fn classify(&self, query: &str, mode: RecognizerMode) -> Result<Value, ToolCallError> {
        let params = vec![
            ("appid", self.app_id.clone()),
            ("mode", mode.as_str().to_string()),
            ("i", query.to_string()),
            ("output", "json".to_string()),
        ];

        let response = self
            .client
            .get(RECOGNIZER_ENDPOINT)
            .query(&params)
            .send()
            .await
            .map_err(|err| ToolCallError::RuntimeError(Box::new(err)))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|err| ToolCallError::RuntimeError(Box::new(err)))?;

        handle_response(status, body, query)
    }
}

fn handle_response(
    status: reqwest::StatusCode,
    body: String,
    query: &str,
) -> Result<Value, ToolCallError> {
    if !status.is_success() {
        return Err(ToolCallError::RuntimeError(
            format!(
                "Wolfram|Alpha Query Recognizer returned status {} with body: {}",
                status, body
            )
            .into(),
        ));
    }

    let parsed = serde_json::from_str::<Value>(&body).unwrap_or_else(|_| json!({ "raw": body }));
    let payload = parsed
        .get("queryrecognizer")
        .cloned()
        .unwrap_or_else(|| parsed.clone());

    let accepted = payload
        .get("accepted")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let domain = payload
        .get("domain")
        .and_then(Value::as_str)
        .map(|value| value.to_string());
    let score = payload
        .get("resultsignificancescore")
        .and_then(Value::as_f64);
    let timing_ms = payload.get("timing").and_then(Value::as_f64);
    let summary = payload
        .get("summarybox")
        .and_then(Value::as_str)
        .map(|value| value.to_string());

    let source = wolfram_input_url(query).ok();

    Ok(json!({
        "accepted": accepted,
        "domain": domain,
        "result_significance_score": score,
        "timing_ms": timing_ms,
        "summary_box": summary,
        "source": source,
        "raw": payload,
    }))
}

#[async_trait]
impl ToolRuntime for WolframAlphaQueryRecognizer {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let WolframAlphaQueryRecognizerArgs { query, mode } = serde_json::from_value(args)?;
        self.classify(&query, mode.unwrap_or_default()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recognizer_mode_as_str() {
        assert_eq!(RecognizerMode::Default.as_str(), "Default");
        assert_eq!(RecognizerMode::Voice.as_str(), "Voice");
    }

    #[test]
    fn test_handle_response_success() {
        let body = r#"{"queryrecognizer":{"accepted":true,"domain":"math","resultsignificancescore":0.8,"timing":12.5,"summarybox":"ok"}}"#.to_string();
        let value = handle_response(reqwest::StatusCode::OK, body, "2+2").unwrap();
        assert_eq!(value["accepted"], true);
        assert_eq!(value["domain"], "math");
        assert_eq!(value["summary_box"], "ok");
        assert!(
            value["source"]
                .as_str()
                .unwrap()
                .contains("wolframalpha.com/input")
        );
    }

    #[test]
    fn test_handle_response_invalid_json() {
        let value =
            handle_response(reqwest::StatusCode::OK, "not-json".to_string(), "2+2").unwrap();
        assert_eq!(value["raw"]["raw"], "not-json");
    }

    #[test]
    fn test_handle_response_error_status() {
        let err = handle_response(reqwest::StatusCode::BAD_REQUEST, "bad".to_string(), "2+2")
            .unwrap_err();
        assert!(err.to_string().contains("status 400"));
    }
}
