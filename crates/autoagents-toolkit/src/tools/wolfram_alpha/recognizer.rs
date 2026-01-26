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

        if !status.is_success() {
            return Err(ToolCallError::RuntimeError(
                format!(
                    "Wolfram|Alpha Query Recognizer returned status {} with body: {}",
                    status, body
                )
                .into(),
            ));
        }

        let parsed =
            serde_json::from_str::<Value>(&body).unwrap_or_else(|_| json!({ "raw": body }));
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
}

#[async_trait]
impl ToolRuntime for WolframAlphaQueryRecognizer {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let WolframAlphaQueryRecognizerArgs { query, mode } = serde_json::from_value(args)?;
        self.classify(&query, mode.unwrap_or_default()).await
    }
}
