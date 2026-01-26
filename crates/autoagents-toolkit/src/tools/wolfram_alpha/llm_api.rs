use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT},
};
use autoagents_derive::{ToolInput, tool};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{wolfram_app_id, wolfram_input_url};

const LLM_API_ENDPOINT: &str = "https://www.wolframalpha.com/api/v1/llm-api";

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WolframAlphaLLMApiArgs {
    #[input(description = "Query to compute with Wolfram|Alpha")]
    input: String,
    #[serde(rename = "maxchars", skip_serializing_if = "Option::is_none")]
    #[input(description = "Optional character limit for the response (defaults to service limit)")]
    max_chars: Option<u32>,
}

#[tool(
    name = "wolfram_alpha_llm_api",
    description = "Use this tool for exact mathematical, scientific, and symbolic computations (e.g., algebra, calculus, ODEs, statistics, physics, unit conversions, and plots) whenever precise calculation is required.",
    input = WolframAlphaLLMApiArgs,
)]
pub struct WolframAlphaLLMApi {
    app_id: String,
    client: Client,
}

impl Default for WolframAlphaLLMApi {
    fn default() -> Self {
        Self::new_with_app_id(wolfram_app_id())
    }
}

impl WolframAlphaLLMApi {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_app_id(app_id: String) -> Self {
        Self {
            app_id,
            client: Client::new(),
        }
    }

    async fn call_api(&self, input: &str, max_chars: Option<u32>) -> Result<Value, ToolCallError> {
        let mut params = vec![
            ("input", input.to_string()),
            ("appid", self.app_id.clone()),
            ("output", "json".to_string()),
        ];

        if let Some(limit) = max_chars {
            params.push(("maxchars", limit.to_string()));
        }

        let response = self
            .client
            .get(LLM_API_ENDPOINT)
            .query(&params)
            .header("Authorization", format!("Bearer {}", self.app_id))
            .header("Accept", "application/json")
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
                        "Wolfram|Alpha LLM API unauthorized ({}). Check WOLFRAM_ALPHA_APP_ID/WOLFRAM_APP_ID and that the app has LLM API access. Body: {}",
                        status, body
                    )
                    .into(),
                ));
            }
            return Err(ToolCallError::RuntimeError(
                format!(
                    "Wolfram|Alpha LLM API returned status {} with body: {}",
                    status, body
                )
                .into(),
            ));
        }

        let parsed =
            serde_json::from_str::<Value>(&body).unwrap_or_else(|_| json!({ "raw": body }));

        let result_text = parsed
            .get("result")
            .and_then(Value::as_str)
            .map(|s| s.to_string())
            .or_else(|| {
                if parsed.is_object() {
                    None
                } else {
                    Some(body.clone())
                }
            });

        let source = wolfram_input_url(input).ok();

        Ok(json!({
            "result": result_text,
            "source": source,
            "data": parsed,
        }))
    }
}

#[async_trait]
impl ToolRuntime for WolframAlphaLLMApi {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let WolframAlphaLLMApiArgs { input, max_chars } = serde_json::from_value(args)?;
        self.call_api(&input, max_chars).await
    }
}
