use async_trait::async_trait;
use chrono::{DateTime, Utc};
use crate::{
    builder::LLMBackend,
    error::LLMError,
};
use std::fmt::Debug;

pub trait ModelListResponse {
    fn get_models(&self) -> Vec<String>;
    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>>;
    fn get_backend(&self) -> LLMBackend;
}

pub trait ModelListRawEntry: Debug {
    fn get_id(&self) -> String;
    fn get_created_at(&self) -> DateTime<Utc>;
    fn get_raw(&self) -> serde_json::Value;
}

#[derive(Debug, Clone, Default)]
pub struct ModelListRequest {
    pub filter: Option<String>,
}

/// Trait for providers that support listing and retrieving model information.
#[async_trait]
pub trait ModelsProvider {
    /// Asynchronously retrieves the list of available models ID's from the provider.
    ///
    /// # Arguments
    ///
    /// * `_request` - Optional filter by model ID
    ///
    /// # Returns
    ///
    /// List of model ID's or error
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        Err(LLMError::ProviderError(
            "List Models not supported".to_string(),
        ))
    }
}
