use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT},
};
use autoagents_derive::{ToolInput, tool};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{wolfram_app_id, wolfram_input_url};

const SHORT_ANSWER_ENDPOINT: &str = "https://api.wolframalpha.com/v1/result";

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShortAnswerUnits {
    Metric,
    Imperial,
}

impl ShortAnswerUnits {
    fn as_str(&self) -> &'static str {
        match self {
            ShortAnswerUnits::Metric => "metric",
            ShortAnswerUnits::Imperial => "imperial",
        }
    }
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WolframAlphaShortAnswerArgs {
    #[input(description = "Question to send to Wolfram|Alpha Short Answers API")]
    query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[input(description = "Unit system to prefer in the response (metric or imperial)")]
    units: Option<ShortAnswerUnits>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[input(description = "Maximum time to wait for a response in seconds (default 5)")]
    timeout: Option<u16>,
}

#[tool(
    name = "wolfram_alpha_short_answer",
    description = "Fetch concise short answers via WolframAlpha Short Answer API."
    input = WolframAlphaShortAnswerArgs,
)]
pub struct WolframAlphaShortAnswer {
    app_id: String,
    client: Client,
}

impl Default for WolframAlphaShortAnswer {
    fn default() -> Self {
        Self::new_with_app_id(wolfram_app_id())
    }
}

impl WolframAlphaShortAnswer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_app_id(app_id: String) -> Self {
        Self {
            app_id,
            client: Client::new(),
        }
    }

    async fn fetch_answer(
        &self,
        query: &str,
        units: Option<ShortAnswerUnits>,
        timeout: Option<u16>,
    ) -> Result<Value, ToolCallError> {
        let mut params = vec![("appid", self.app_id.clone()), ("i", query.to_string())];

        if let Some(unit_pref) = units {
            params.push(("units", unit_pref.as_str().to_string()));
        }

        if let Some(timeout_seconds) = timeout {
            params.push(("timeout", timeout_seconds.to_string()));
        }

        let response = self
            .client
            .get(SHORT_ANSWER_ENDPOINT)
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
            if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(ToolCallError::RuntimeError(
                    format!(
                        "Wolfram|Alpha Short Answer API unauthorized ({}). Check WOLFRAM_ALPHA_APP_ID/WOLFRAM_APP_ID and that the key allows Short Answers. Body: {}",
                        status, body
                    )
                    .into(),
                ));
            }
            return Err(ToolCallError::RuntimeError(
                format!(
                    "Wolfram|Alpha Short Answer API returned status {} with body: {}",
                    status, body
                )
                .into(),
            ));
        }

        let source = wolfram_input_url(query).ok();

        Ok(json!({
            "answer": body.trim(),
            "source": source,
        }))
    }
}

#[async_trait]
impl ToolRuntime for WolframAlphaShortAnswer {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let WolframAlphaShortAnswerArgs {
            query,
            units,
            timeout,
        } = serde_json::from_value(args)?;

        self.fetch_answer(&query, units, timeout).await
    }
}
