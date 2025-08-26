use std::fmt;

/// Error types that can occur when interacting with LLM providers.
#[derive(Debug)]
pub enum LLMError {
    /// HTTP request/response errors
    HttpError(String),
    /// Authentication and authorization errors
    AuthError(String),
    /// Invalid request parameters or format
    InvalidRequest(String),
    /// Errors returned by the LLM provider
    ProviderError(String),
    /// API response parsing or format error
    ResponseFormatError {
        message: String,
        raw_response: String,
    },
    /// Generic error
    Generic(String),
    /// JSON serialization/deserialization errors
    JsonError(String),
    /// Tool configuration error
    ToolConfigError(String),
    /// No Tool Support
    NoToolSupport(String),
}

impl fmt::Display for LLMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LLMError::HttpError(e) => write!(f, "HTTP Error: {e}"),
            LLMError::AuthError(e) => write!(f, "Auth Error: {e}"),
            LLMError::InvalidRequest(e) => write!(f, "Invalid Request: {e}"),
            LLMError::ProviderError(e) => write!(f, "Provider Error: {e}"),
            LLMError::Generic(e) => write!(f, "Generic Error : {e}"),
            LLMError::ResponseFormatError {
                message,
                raw_response,
            } => {
                write!(
                    f,
                    "Response Format Error: {message}. Raw response: {raw_response}"
                )
            }
            LLMError::JsonError(e) => write!(f, "JSON Parse Error: {e}"),
            LLMError::ToolConfigError(e) => write!(f, "Tool Configuration Error: {e}"),
            LLMError::NoToolSupport(e) => write!(f, "No Tool Support: {e}"),
        }
    }
}

impl std::error::Error for LLMError {}

/// Converts reqwest HTTP errors into LlmErrors
impl From<reqwest::Error> for LLMError {
    fn from(err: reqwest::Error) -> Self {
        LLMError::HttpError(err.to_string())
    }
}

impl From<serde_json::Error> for LLMError {
    fn from(err: serde_json::Error) -> Self {
        LLMError::JsonError(format!(
            "{} at line {} column {}",
            err,
            err.line(),
            err.column()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Error as JsonError;
    use std::error::Error;

    #[test]
    fn test_llm_error_display_http_error() {
        let error = LLMError::HttpError("Connection failed".to_string());
        assert_eq!(error.to_string(), "HTTP Error: Connection failed");
    }

    #[test]
    fn test_llm_error_display_auth_error() {
        let error = LLMError::AuthError("Invalid API key".to_string());
        assert_eq!(error.to_string(), "Auth Error: Invalid API key");
    }

    #[test]
    fn test_llm_error_display_invalid_request() {
        let error = LLMError::InvalidRequest("Missing required parameter".to_string());
        assert_eq!(
            error.to_string(),
            "Invalid Request: Missing required parameter"
        );
    }

    #[test]
    fn test_llm_error_display_provider_error() {
        let error = LLMError::ProviderError("Model not found".to_string());
        assert_eq!(error.to_string(), "Provider Error: Model not found");
    }

    #[test]
    fn test_llm_error_display_generic_error() {
        let error = LLMError::Generic("Something went wrong".to_string());
        assert_eq!(error.to_string(), "Generic Error : Something went wrong");
    }

    #[test]
    fn test_llm_error_display_response_format_error() {
        let error = LLMError::ResponseFormatError {
            message: "Invalid JSON".to_string(),
            raw_response: "{invalid json}".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Response Format Error: Invalid JSON. Raw response: {invalid json}"
        );
    }

    #[test]
    fn test_llm_error_display_json_error() {
        let error = LLMError::JsonError("Parse error at line 5 column 10".to_string());
        assert_eq!(
            error.to_string(),
            "JSON Parse Error: Parse error at line 5 column 10"
        );
    }

    #[test]
    fn test_llm_error_display_tool_config_error() {
        let error = LLMError::ToolConfigError("Invalid tool configuration".to_string());
        assert_eq!(
            error.to_string(),
            "Tool Configuration Error: Invalid tool configuration"
        );
    }

    #[test]
    fn test_llm_error_is_error_trait() {
        let error = LLMError::Generic("test error".to_string());
        assert!(error.source().is_none());
    }

    #[test]
    fn test_llm_error_debug_format() {
        let error = LLMError::HttpError("test".to_string());
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("HttpError"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_from_reqwest_error() {
        // Create a mock reqwest error by trying to make a request to an invalid URL
        let client = reqwest::Client::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let reqwest_error = rt
            .block_on(async {
                client
                    .get("http://invalid-url-that-does-not-exist-12345.com/")
                    .timeout(std::time::Duration::from_millis(100))
                    .send()
                    .await
            })
            .unwrap_err();

        let llm_error: LLMError = reqwest_error.into();

        match llm_error {
            LLMError::HttpError(msg) => {
                assert!(!msg.is_empty());
            }
            _ => panic!("Expected HttpError"),
        }
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_str = r#"{"invalid": json}"#;
        let json_error: JsonError =
            serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();

        let llm_error: LLMError = json_error.into();

        match llm_error {
            LLMError::JsonError(msg) => {
                assert!(msg.contains("line"));
                assert!(msg.contains("column"));
            }
            _ => panic!("Expected JsonError"),
        }
    }

    #[test]
    fn test_error_variants_equality() {
        let error1 = LLMError::HttpError("test".to_string());
        let error2 = LLMError::HttpError("test".to_string());
        let error3 = LLMError::HttpError("different".to_string());
        let error4 = LLMError::AuthError("test".to_string());

        // Note: LLMError doesn't implement PartialEq, so we test via string representation
        assert_eq!(error1.to_string(), error2.to_string());
        assert_ne!(error1.to_string(), error3.to_string());
        assert_ne!(error1.to_string(), error4.to_string());
    }

    #[test]
    fn test_response_format_error_fields() {
        let error = LLMError::ResponseFormatError {
            message: "Parse failed".to_string(),
            raw_response: "raw content".to_string(),
        };

        let display_str = error.to_string();
        assert!(display_str.contains("Parse failed"));
        assert!(display_str.contains("raw content"));
    }

    #[test]
    fn test_all_error_variants_have_display() {
        let errors = vec![
            LLMError::HttpError("http".to_string()),
            LLMError::AuthError("auth".to_string()),
            LLMError::InvalidRequest("invalid".to_string()),
            LLMError::ProviderError("provider".to_string()),
            LLMError::Generic("generic".to_string()),
            LLMError::ResponseFormatError {
                message: "format".to_string(),
                raw_response: "raw".to_string(),
            },
            LLMError::JsonError("json".to_string()),
            LLMError::ToolConfigError("tool".to_string()),
        ];

        for error in errors {
            let display_str = error.to_string();
            assert!(!display_str.is_empty());
        }
    }

    #[test]
    fn test_error_type_classification() {
        // Test that we can pattern match on different error types
        let http_error = LLMError::HttpError("test".to_string());
        match http_error {
            LLMError::HttpError(_) => {}
            _ => panic!("Expected HttpError"),
        }

        let auth_error = LLMError::AuthError("test".to_string());
        match auth_error {
            LLMError::AuthError(_) => {}
            _ => panic!("Expected AuthError"),
        }

        let response_error = LLMError::ResponseFormatError {
            message: "test".to_string(),
            raw_response: "test".to_string(),
        };
        match response_error {
            LLMError::ResponseFormatError { .. } => {}
            _ => panic!("Expected ResponseFormatError"),
        }
    }
}
