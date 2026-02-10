use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT},
};
use autoagents_derive::{ToolInput, tool};
use once_cell::sync::Lazy;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::env;

use crate::utils::constant::RestHeaders;

static ENV_BRAVE_SEARCH_API_KEY: &str = "BRAVE_SEARCH_API_KEY";
static ENV_BRAVE_API_KEY: &str = "BRAVE_API_KEY";

static BRAVE_SEARCH_API_KEY: Lazy<String> = Lazy::new(|| {
    env::var(ENV_BRAVE_SEARCH_API_KEY)
        .or_else(|_| env::var(ENV_BRAVE_API_KEY))
        .expect("BRAVE_SEARCH_API_KEY or BRAVE_API_KEY must be set")
});

const BRAVE_API_ENDPOINT: &str = "https://api.search.brave.com/res/v1/web/search";

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct BraveSearchArgs {
    #[input(description = "Query for search")]
    query: String,
}

#[tool(
    name = "brave_search",
    description = "Use this tool to search on Brave Search",
    input = BraveSearchArgs,
)]
pub struct BraveSearch {
    api_key: String,
}

impl Default for BraveSearch {
    fn default() -> Self {
        Self {
            api_key: BRAVE_SEARCH_API_KEY.clone(),
        }
    }
}

impl BraveSearch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_key(api_key: String) -> Self {
        Self { api_key }
    }

    async fn fetch_raw_results(&self, query: &str) -> Result<Value, ToolCallError> {
        let params = [("q", query), ("extra_snippets", "true")];

        let response = Client::new()
            .get(BRAVE_API_ENDPOINT)
            .query(&params)
            .header(
                RestHeaders::Accept.as_str(),
                RestHeaders::ApplicationJson.as_str(),
            )
            .header(RestHeaders::XSubscriptionToken.as_str(), &self.api_key)
            .send()
            .await
            .map_err(|err| ToolCallError::RuntimeError(Box::new(err)))?;

        let payload = response
            .error_for_status()
            .map_err(|err| ToolCallError::RuntimeError(Box::new(err)))?
            .json::<Value>()
            .await
            .map_err(|err| ToolCallError::RuntimeError(Box::new(err)))?;

        Ok(payload)
    }
}

#[async_trait]
impl ToolRuntime for BraveSearch {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let BraveSearchArgs { query } = serde_json::from_value(args)?;
        let payload = self.fetch_raw_results(&query).await?;

        let raw_results = payload
            .get("web")
            .and_then(|web| web.get("results"))
            .and_then(|results| results.as_array())
            .cloned()
            .unwrap_or_default();

        let summarized_results = raw_results
            .into_iter()
            .map(|item| {
                let title = item
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let link = item
                    .get("url")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();

                let mut snippet_parts: Vec<String> = Vec::new();
                if let Some(description) = item.get("description").and_then(Value::as_str)
                    && !description.is_empty()
                {
                    snippet_parts.push(description.to_string());
                }

                if let Some(extra_snippets) = item.get("extra_snippets").and_then(Value::as_array) {
                    for snippet in extra_snippets.iter().filter_map(Value::as_str) {
                        if !snippet.is_empty() {
                            snippet_parts.push(snippet.to_string());
                        }
                    }
                }

                json!({
                    "title": title,
                    "link": link,
                    "snippet": snippet_parts.join(" "),
                })
            })
            .collect::<Vec<Value>>();

        Ok(Value::Array(summarized_results))
    }
}
