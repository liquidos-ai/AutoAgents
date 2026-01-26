use crate::{builder::LLMBackend, error::LLMError};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;

pub trait ModelListResponse: std::fmt::Debug {
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

/// Standard model entry structure used by OpenAI-compatible providers
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StandardModelEntry {
    pub id: String,
    pub created: Option<u64>,
    #[serde(flatten)]
    pub extra: Value,
}

impl ModelListRawEntry for StandardModelEntry {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_created_at(&self) -> DateTime<Utc> {
        self.created
            .map(|t| DateTime::from_timestamp(t as i64, 0).unwrap_or_default())
            .unwrap_or_default()
    }

    fn get_raw(&self) -> Value {
        self.extra.clone()
    }
}

/// Inner structure for model list response data (serializable)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StandardModelListResponseInner {
    pub data: Vec<StandardModelEntry>,
}

/// Standard model list response structure used by OpenAI-compatible providers
#[derive(Clone, Debug)]
pub struct StandardModelListResponse {
    pub inner: StandardModelListResponseInner,
    pub backend: LLMBackend,
}

impl ModelListResponse for StandardModelListResponse {
    fn get_models(&self) -> Vec<String> {
        self.inner.data.iter().map(|e| e.id.clone()).collect()
    }

    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>> {
        self.inner
            .data
            .iter()
            .map(|e| Box::new(e.clone()) as Box<dyn ModelListRawEntry>)
            .collect()
    }

    fn get_backend(&self) -> LLMBackend {
        self.backend.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::LLMBackend;
    use crate::error::LLMError;
    use async_trait::async_trait;
    use chrono::TimeZone;
    use chrono::{DateTime, Utc};
    use serde_json::Value;

    // Mock implementations for testing
    #[derive(Debug, Clone)]
    struct MockModelEntry {
        id: String,
        created_at: DateTime<Utc>,
        extra_data: Value,
    }

    impl ModelListRawEntry for MockModelEntry {
        fn get_id(&self) -> String {
            self.id.clone()
        }

        fn get_created_at(&self) -> DateTime<Utc> {
            self.created_at
        }

        fn get_raw(&self) -> Value {
            self.extra_data.clone()
        }
    }

    struct MockModelListResponse {
        models: Vec<String>,
        raw_entries: Vec<MockModelEntry>,
        backend: LLMBackend,
    }

    impl std::fmt::Debug for MockModelListResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockModelListResponse")
                .field("models", &self.models)
                .field("raw_entries", &self.raw_entries)
                .field("backend", &self.backend)
                .finish()
        }
    }

    impl ModelListResponse for MockModelListResponse {
        fn get_models(&self) -> Vec<String> {
            self.models.clone()
        }

        fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>> {
            self.raw_entries
                .iter()
                .map(|e| Box::new(e.clone()) as Box<dyn ModelListRawEntry>)
                .collect()
        }

        fn get_backend(&self) -> LLMBackend {
            self.backend.clone()
        }
    }

    struct MockModelsProvider {
        should_fail: bool,
        models: Vec<String>,
    }

    impl MockModelsProvider {
        fn new(models: Vec<String>) -> Self {
            Self {
                should_fail: false,
                models,
            }
        }

        fn with_failure() -> Self {
            Self {
                should_fail: true,
                models: vec![],
            }
        }
    }

    #[async_trait]
    impl ModelsProvider for MockModelsProvider {
        async fn list_models(
            &self,
            _request: Option<&ModelListRequest>,
        ) -> Result<Box<dyn ModelListResponse>, LLMError> {
            if self.should_fail {
                return Err(LLMError::ProviderError("Mock provider failed".to_string()));
            }

            let raw_entries = self
                .models
                .iter()
                .enumerate()
                .map(|(i, model)| MockModelEntry {
                    id: model.clone(),
                    created_at: Utc.timestamp_opt(1640995200, 0).unwrap(),
                    extra_data: serde_json::json!({
                        "index": i,
                        "description": format!("Model {}", model)
                    }),
                })
                .collect();

            Ok(Box::new(MockModelListResponse {
                models: self.models.clone(),
                raw_entries,
                backend: LLMBackend::OpenAI,
            }))
        }
    }

    // Default provider that always fails
    struct DefaultModelsProvider;

    #[async_trait]
    impl ModelsProvider for DefaultModelsProvider {}

    #[test]
    fn test_model_list_request_default() {
        let request = ModelListRequest::default();
        assert!(request.filter.is_none());
    }

    #[test]
    fn test_model_list_request_with_filter() {
        let request = ModelListRequest {
            filter: Some("gpt".to_string()),
        };
        assert_eq!(request.filter, Some("gpt".to_string()));
    }

    #[test]
    fn test_model_list_request_clone() {
        let request = ModelListRequest {
            filter: Some("test".to_string()),
        };
        let cloned = request.clone();
        assert_eq!(request.filter, cloned.filter);
    }

    #[test]
    fn test_model_list_request_debug() {
        let request = ModelListRequest {
            filter: Some("debug_test".to_string()),
        };
        let debug_str = format!("{request:?}");
        assert!(debug_str.contains("ModelListRequest"));
        assert!(debug_str.contains("debug_test"));
    }

    #[test]
    fn test_mock_model_entry_creation() {
        let now = Utc.timestamp_opt(1640995200, 0).unwrap();
        let entry = MockModelEntry {
            id: "test-model".to_string(),
            created_at: now,
            extra_data: serde_json::json!({"key": "value"}),
        };

        assert_eq!(entry.get_id(), "test-model");
        assert_eq!(entry.get_created_at(), now);
        assert_eq!(entry.get_raw(), serde_json::json!({"key": "value"}));
    }

    #[test]
    fn test_mock_model_entry_debug() {
        let entry = MockModelEntry {
            id: "debug-model".to_string(),
            created_at: Utc.timestamp_opt(1640995200, 0).unwrap(),
            extra_data: serde_json::json!({"debug": true}),
        };

        let debug_str = format!("{entry:?}");
        assert!(debug_str.contains("MockModelEntry"));
        assert!(debug_str.contains("debug-model"));
    }

    #[test]
    fn test_mock_model_entry_clone() {
        let entry = MockModelEntry {
            id: "clone-model".to_string(),
            created_at: Utc.timestamp_opt(1640995200, 0).unwrap(),
            extra_data: serde_json::json!({"clone": true}),
        };

        let cloned = entry.clone();
        assert_eq!(entry.get_id(), cloned.get_id());
        assert_eq!(entry.get_created_at(), cloned.get_created_at());
        assert_eq!(entry.get_raw(), cloned.get_raw());
    }

    #[test]
    fn test_mock_model_list_response_get_models() {
        let models = vec!["model1".to_string(), "model2".to_string()];
        let response = MockModelListResponse {
            models: models.clone(),
            raw_entries: vec![],
            backend: LLMBackend::OpenAI,
        };

        assert_eq!(response.get_models(), models);
    }

    #[test]
    fn test_mock_model_list_response_get_models_raw() {
        let raw_entries = vec![
            MockModelEntry {
                id: "model1".to_string(),
                created_at: Utc.timestamp_opt(1640995200, 0).unwrap(),
                extra_data: serde_json::json!({"index": 0}),
            },
            MockModelEntry {
                id: "model2".to_string(),
                created_at: Utc.timestamp_opt(1640995201, 0).unwrap(),
                extra_data: serde_json::json!({"index": 1}),
            },
        ];

        let response = MockModelListResponse {
            models: vec!["model1".to_string(), "model2".to_string()],
            raw_entries: raw_entries.clone(),
            backend: LLMBackend::Anthropic,
        };

        let raw = response.get_models_raw();
        assert_eq!(raw.len(), 2);
        assert_eq!(raw[0].get_id(), "model1");
        assert_eq!(raw[1].get_id(), "model2");
    }

    #[test]
    fn test_mock_model_list_response_get_backend() {
        let response = MockModelListResponse {
            models: vec![],
            raw_entries: vec![],
            backend: LLMBackend::Google,
        };

        assert!(matches!(response.get_backend(), LLMBackend::Google));
    }

    #[tokio::test]
    async fn test_mock_models_provider_success() {
        let models = vec!["gpt-3.5-turbo".to_string(), "gpt-4".to_string()];
        let provider = MockModelsProvider::new(models.clone());

        let result = provider.list_models(None).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.get_models(), models);
        assert_eq!(response.get_models_raw().len(), 2);
        assert!(matches!(response.get_backend(), LLMBackend::OpenAI));
    }

    #[tokio::test]
    async fn test_mock_models_provider_with_request() {
        let models = vec!["model1".to_string(), "model2".to_string()];
        let provider = MockModelsProvider::new(models.clone());
        let request = ModelListRequest {
            filter: Some("gpt".to_string()),
        };

        let result = provider.list_models(Some(&request)).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.get_models(), models);
    }

    #[tokio::test]
    async fn test_mock_models_provider_failure() {
        let provider = MockModelsProvider::with_failure();

        let result = provider.list_models(None).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Mock provider failed")
        );
    }

    #[tokio::test]
    async fn test_mock_models_provider_empty_models() {
        let provider = MockModelsProvider::new(vec![]);

        let result = provider.list_models(None).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.get_models(), Vec::<String>::new());
        assert_eq!(response.get_models_raw().len(), 0);
    }

    #[tokio::test]
    async fn test_default_models_provider_not_supported() {
        let provider = DefaultModelsProvider;

        let result = provider.list_models(None).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("List Models not supported")
        );
    }

    #[tokio::test]
    async fn test_default_models_provider_with_request() {
        let provider = DefaultModelsProvider;
        let request = ModelListRequest {
            filter: Some("test".to_string()),
        };

        let result = provider.list_models(Some(&request)).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("List Models not supported")
        );
    }

    #[test]
    fn test_model_list_raw_entry_trait_object() {
        let entry = MockModelEntry {
            id: "trait-test".to_string(),
            created_at: Utc.timestamp_opt(1640995200, 0).unwrap(),
            extra_data: serde_json::json!({"test": "data"}),
        };

        let boxed: Box<dyn ModelListRawEntry> = Box::new(entry);
        assert_eq!(boxed.get_id(), "trait-test");
        assert_eq!(boxed.get_raw(), serde_json::json!({"test": "data"}));
    }

    #[test]
    fn test_model_list_response_trait_object() {
        let response = MockModelListResponse {
            models: vec!["test-model".to_string()],
            raw_entries: vec![],
            backend: LLMBackend::Ollama,
        };

        let boxed: Box<dyn ModelListResponse> = Box::new(response);
        assert_eq!(boxed.get_models(), vec!["test-model".to_string()]);
        assert!(matches!(boxed.get_backend(), LLMBackend::Ollama));
    }

    #[test]
    fn test_model_entry_with_complex_data() {
        let complex_data = serde_json::json!({
            "capabilities": ["chat", "completion"],
            "max_tokens": 4096,
            "pricing": {
                "input": 0.0015,
                "output": 0.002
            }
        });

        let entry = MockModelEntry {
            id: "complex-model".to_string(),
            created_at: Utc.timestamp_opt(1640995200, 0).unwrap(),
            extra_data: complex_data.clone(),
        };

        assert_eq!(entry.get_raw(), complex_data);
        assert_eq!(entry.get_raw()["capabilities"][0], "chat");
        assert_eq!(entry.get_raw()["max_tokens"], 4096);
    }

    #[test]
    fn test_model_entry_with_null_data() {
        let entry = MockModelEntry {
            id: "null-model".to_string(),
            created_at: Utc.timestamp_opt(1640995200, 0).unwrap(),
            extra_data: serde_json::Value::Null,
        };

        assert_eq!(entry.get_raw(), serde_json::Value::Null);
    }

    #[test]
    fn test_model_entry_time_ordering() {
        let time1 = Utc.timestamp_opt(1640995200, 0).unwrap();
        let time2 = time1 + chrono::Duration::seconds(1);

        let entry1 = MockModelEntry {
            id: "older".to_string(),
            created_at: time1,
            extra_data: serde_json::Value::Null,
        };

        let entry2 = MockModelEntry {
            id: "newer".to_string(),
            created_at: time2,
            extra_data: serde_json::Value::Null,
        };

        assert!(entry1.get_created_at() < entry2.get_created_at());
    }

    #[test]
    fn test_backend_variants() {
        let backends = vec![
            LLMBackend::OpenAI,
            LLMBackend::Anthropic,
            LLMBackend::Ollama,
            LLMBackend::DeepSeek,
            LLMBackend::XAI,
            LLMBackend::Phind,
            LLMBackend::Google,
            LLMBackend::Groq,
            LLMBackend::AzureOpenAI,
        ];

        for backend in backends {
            let response = MockModelListResponse {
                models: vec![],
                raw_entries: vec![],
                backend: backend.clone(),
            };
            // Use a more flexible assertion since matches! doesn't work with clone
            let result_backend = response.get_backend();
            assert!(std::mem::discriminant(&result_backend) == std::mem::discriminant(&backend));
        }
    }

    #[tokio::test]
    async fn test_models_provider_error_handling() {
        let provider = MockModelsProvider::with_failure();

        let result = provider.list_models(None).await;
        match result {
            Err(LLMError::ProviderError(msg)) => {
                assert_eq!(msg, "Mock provider failed");
            }
            _ => panic!("Expected ProviderError"),
        }
    }

    #[tokio::test]
    async fn test_models_provider_with_many_models() {
        let models: Vec<String> = (0..100).map(|i| format!("model-{i:03}")).collect();
        let provider = MockModelsProvider::new(models.clone());

        let result = provider.list_models(None).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.get_models().len(), 100);
        assert_eq!(response.get_models_raw().len(), 100);
        assert_eq!(response.get_models()[0], "model-000");
        assert_eq!(response.get_models()[99], "model-099");
    }

    #[test]
    fn test_model_list_request_with_empty_filter() {
        let request = ModelListRequest {
            filter: Some("".to_string()),
        };
        assert_eq!(request.filter, Some("".to_string()));
    }

    #[test]
    fn test_model_list_request_with_special_chars() {
        let request = ModelListRequest {
            filter: Some("model-name_with.special-chars".to_string()),
        };
        assert_eq!(
            request.filter,
            Some("model-name_with.special-chars".to_string())
        );
    }

    #[test]
    fn test_model_entry_with_empty_id() {
        let entry = MockModelEntry {
            id: "".to_string(),
            created_at: Utc.timestamp_opt(1640995200, 0).unwrap(),
            extra_data: serde_json::Value::Null,
        };
        assert_eq!(entry.get_id(), "");
    }

    #[test]
    fn test_model_entry_with_unicode_id() {
        let entry = MockModelEntry {
            id: "模型-测试-🤖".to_string(),
            created_at: Utc.timestamp_opt(1640995200, 0).unwrap(),
            extra_data: serde_json::Value::Null,
        };
        assert_eq!(entry.get_id(), "模型-测试-🤖");
    }
}
