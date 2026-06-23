use std::fmt;
use std::time::Duration;

/// Maximum bytes of a provider response body included in [`Display`](fmt::Display) output.
pub const MAX_ERROR_BODY_DISPLAY_BYTES: usize = 512;

/// Phase where a guardrail violation occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardrailPhase {
    Input,
    Output,
}

/// Error types that can occur when interacting with LLM providers.
#[derive(Clone)]
pub enum LLMError {
    /// HTTP transport failures (connection, timeout, DNS, etc.).
    HttpError(String),
    /// Authentication and authorization errors (missing local API key or HTTP 401/403).
    AuthError {
        message: String,
        status_code: Option<u16>,
        response_body: Option<Box<str>>,
    },
    /// Rate limit or provider overload (HTTP 429, 529).
    RateLimitError {
        status_code: u16,
        message: String,
        response_body: Box<str>,
        retry_after: Option<Duration>,
        provider_code: Option<Box<str>>,
    },
    /// Non-success HTTP response that is not auth or rate-limit.
    HttpStatusError {
        status_code: u16,
        message: String,
        response_body: Box<str>,
        provider_code: Option<Box<str>>,
    },
    /// Invalid request parameters, format, or client-side HTTP 4xx rejection.
    InvalidRequest {
        message: String,
        status_code: Option<u16>,
        response_body: Option<Box<str>>,
    },
    /// Errors returned by the LLM provider in a parsed response payload.
    ProviderError(String),
    /// API response parsing or format error on a successful HTTP response.
    ResponseFormatError {
        message: String,
        raw_response: String,
    },
    /// Generic error (unsupported features, internal stubs).
    Generic(String),
    /// JSON serialization/deserialization errors.
    JsonError(String),
    /// Tool configuration error.
    ToolConfigError(String),
    /// Provider does not support tool calling.
    NoToolSupport(String),
    /// Guardrail blocked the request/response.
    GuardrailBlocked {
        phase: GuardrailPhase,
        guard: Box<str>,
        rule_id: Box<str>,
        category: Box<str>,
        severity: Box<str>,
        message: Box<str>,
    },
    /// Guardrail execution failed unexpectedly.
    GuardrailExecutionFailed { guard: String, message: String },
}

impl LLMError {
    /// Local configuration error when an API key is missing.
    pub fn missing_api_key(message: impl Into<String>) -> Self {
        Self::AuthError {
            message: message.into(),
            status_code: None,
            response_body: None,
        }
    }

    /// Local validation error for invalid request parameters or configuration.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            status_code: None,
            response_body: None,
        }
    }

    /// Returns `true` when the error represents a transient failure worth retrying.
    pub fn is_retryable(&self) -> bool {
        is_retryable(self)
    }

    /// HTTP status code when the error originated from a non-success response.
    pub fn http_status_code(&self) -> Option<u16> {
        match self {
            Self::AuthError { status_code, .. } => *status_code,
            Self::RateLimitError { status_code, .. } => Some(*status_code),
            Self::HttpStatusError { status_code, .. } => Some(*status_code),
            Self::InvalidRequest { status_code, .. } => *status_code,
            _ => None,
        }
    }

    /// Provider response body attached to HTTP-related errors.
    pub fn response_body(&self) -> Option<&str> {
        match self {
            Self::AuthError { response_body, .. } => response_body.as_deref(),
            Self::RateLimitError { response_body, .. } => Some(response_body),
            Self::HttpStatusError { response_body, .. } => Some(response_body),
            Self::InvalidRequest { response_body, .. } => response_body.as_deref(),
            Self::ResponseFormatError { raw_response, .. } => Some(raw_response),
            _ => None,
        }
    }

    /// Returns `true` when this `HttpError` represents a retryable transport failure.
    pub fn is_transport_retryable(&self) -> bool {
        match self {
            Self::HttpError(msg) => is_transport_retryable_message(msg),
            _ => false,
        }
    }
}

/// Returns `true` when a transport error message represents a retryable failure.
pub fn is_transport_retryable_message(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.starts_with("request timed out:")
        || m.starts_with("connection failed:")
        || m.contains("connection reset")
        || m.contains("broken pipe")
        || m.contains("dns error")
        || m.contains("dns lookup")
        || m.contains("name or service not known")
}

/// Truncates a response body for safe inclusion in log-oriented display output.
pub fn truncate_for_display(body: &str) -> String {
    if body.len() <= MAX_ERROR_BODY_DISPLAY_BYTES {
        return body.to_string();
    }

    let mut end = MAX_ERROR_BODY_DISPLAY_BYTES;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }

    format!(
        "{}... [truncated, {} bytes total]",
        &body[..end],
        body.len()
    )
}

fn write_truncated_body(f: &mut fmt::Formatter<'_>, label: &str, body: &str) -> fmt::Result {
    write!(f, ". {label}: {}", truncate_for_display(body))
}

fn debug_optional_body(body: &Option<Box<str>>) -> Option<String> {
    body.as_deref().map(truncate_for_display)
}

impl fmt::Debug for LLMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HttpError(message) => f.debug_tuple("HttpError").field(message).finish(),
            Self::AuthError {
                message,
                status_code,
                response_body,
            } => f
                .debug_struct("AuthError")
                .field("message", message)
                .field("status_code", status_code)
                .field("response_body", &debug_optional_body(response_body))
                .finish(),
            Self::RateLimitError {
                status_code,
                message,
                response_body,
                retry_after,
                provider_code,
            } => f
                .debug_struct("RateLimitError")
                .field("status_code", status_code)
                .field("message", message)
                .field("response_body", &truncate_for_display(response_body))
                .field("retry_after", retry_after)
                .field("provider_code", provider_code)
                .finish(),
            Self::HttpStatusError {
                status_code,
                message,
                response_body,
                provider_code,
            } => f
                .debug_struct("HttpStatusError")
                .field("status_code", status_code)
                .field("message", message)
                .field("response_body", &truncate_for_display(response_body))
                .field("provider_code", provider_code)
                .finish(),
            Self::InvalidRequest {
                message,
                status_code,
                response_body,
            } => f
                .debug_struct("InvalidRequest")
                .field("message", message)
                .field("status_code", status_code)
                .field("response_body", &debug_optional_body(response_body))
                .finish(),
            Self::ProviderError(message) => f.debug_tuple("ProviderError").field(message).finish(),
            Self::ResponseFormatError {
                message,
                raw_response,
            } => f
                .debug_struct("ResponseFormatError")
                .field("message", message)
                .field("raw_response", &truncate_for_display(raw_response))
                .finish(),
            Self::Generic(message) => f.debug_tuple("Generic").field(message).finish(),
            Self::JsonError(message) => f.debug_tuple("JsonError").field(message).finish(),
            Self::ToolConfigError(message) => {
                f.debug_tuple("ToolConfigError").field(message).finish()
            }
            Self::NoToolSupport(message) => f.debug_tuple("NoToolSupport").field(message).finish(),
            Self::GuardrailBlocked {
                phase,
                guard,
                rule_id,
                category,
                severity,
                message,
            } => f
                .debug_struct("GuardrailBlocked")
                .field("phase", phase)
                .field("guard", guard)
                .field("rule_id", rule_id)
                .field("category", category)
                .field("severity", severity)
                .field("message", message)
                .finish(),
            Self::GuardrailExecutionFailed { guard, message } => f
                .debug_struct("GuardrailExecutionFailed")
                .field("guard", guard)
                .field("message", message)
                .finish(),
        }
    }
}

/// Returns `true` when an HTTP status code represents a retryable server/transient error.
pub fn is_http_status_retryable(status_code: u16) -> bool {
    matches!(status_code, 408 | 500..=599)
}

/// Default retryability predicate shared by the retry layer.
pub fn is_retryable(err: &LLMError) -> bool {
    match err {
        LLMError::RateLimitError { .. } => true,
        LLMError::HttpStatusError { status_code, .. } => is_http_status_retryable(*status_code),
        LLMError::HttpError(msg) => is_transport_retryable_message(msg),
        LLMError::Generic(_)
        | LLMError::AuthError { .. }
        | LLMError::InvalidRequest { .. }
        | LLMError::GuardrailBlocked { .. }
        | LLMError::GuardrailExecutionFailed { .. }
        | LLMError::ResponseFormatError { .. }
        | LLMError::JsonError(_)
        | LLMError::ToolConfigError(_)
        | LLMError::NoToolSupport(_)
        | LLMError::ProviderError(_) => false,
    }
}

/// Default fallbackability predicate shared by the fallback layer.
pub fn is_fallbackable(err: &LLMError) -> bool {
    match err {
        LLMError::RateLimitError { .. } => true,
        LLMError::HttpStatusError { status_code, .. } => is_http_status_retryable(*status_code),
        LLMError::HttpError(msg) => is_transport_retryable_message(msg),
        LLMError::ProviderError(_)
        | LLMError::ResponseFormatError { .. }
        | LLMError::NoToolSupport(_) => true,
        LLMError::AuthError { .. }
        | LLMError::InvalidRequest { .. }
        | LLMError::JsonError(_)
        | LLMError::ToolConfigError(_)
        | LLMError::Generic(_)
        | LLMError::GuardrailBlocked { .. }
        | LLMError::GuardrailExecutionFailed { .. } => false,
    }
}

impl fmt::Display for LLMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LLMError::HttpError(e) => write!(f, "HTTP Error: {e}"),
            LLMError::AuthError {
                message,
                status_code,
                response_body,
            } => {
                if let Some(status) = status_code {
                    write!(f, "Auth Error ({status}): {message}")?;
                } else {
                    write!(f, "Auth Error: {message}")?;
                }
                if let Some(body) = response_body {
                    write_truncated_body(f, "Response", body)?;
                }
                Ok(())
            }
            LLMError::RateLimitError {
                status_code,
                message,
                response_body,
                retry_after,
                provider_code,
            } => {
                write!(f, "Rate Limit Error ({status_code}): {message}")?;
                write_truncated_body(f, "Response", response_body)?;
                if let Some(code) = provider_code {
                    write!(f, ". Provider code: {code}")?;
                }
                if let Some(retry_after) = retry_after {
                    write!(f, ". Retry-After: {}s", retry_after.as_secs())?;
                }
                Ok(())
            }
            LLMError::HttpStatusError {
                status_code,
                message,
                response_body,
                provider_code,
            } => {
                write!(f, "HTTP Status Error ({status_code}): {message}")?;
                write_truncated_body(f, "Response", response_body)?;
                if let Some(code) = provider_code {
                    write!(f, ". Provider code: {code}")?;
                }
                Ok(())
            }
            LLMError::InvalidRequest {
                message,
                status_code,
                response_body,
            } => {
                if let Some(status) = status_code {
                    write!(f, "Invalid Request ({status}): {message}")?;
                } else {
                    write!(f, "Invalid Request: {message}")?;
                }
                if let Some(body) = response_body {
                    write_truncated_body(f, "Response", body)?;
                }
                Ok(())
            }
            LLMError::ProviderError(e) => write!(f, "Provider Error: {e}"),
            LLMError::Generic(e) => write!(f, "Generic Error : {e}"),
            LLMError::ResponseFormatError {
                message,
                raw_response,
            } => {
                write!(f, "Response Format Error: {message}")?;
                write_truncated_body(f, "Raw response", raw_response)
            }
            LLMError::JsonError(e) => write!(f, "JSON Parse Error: {e}"),
            LLMError::ToolConfigError(e) => write!(f, "Tool Configuration Error: {e}"),
            LLMError::NoToolSupport(e) => write!(f, "No Tool Support: {e}"),
            LLMError::GuardrailBlocked {
                phase,
                guard,
                rule_id,
                category,
                severity,
                message,
            } => {
                let phase = match phase {
                    GuardrailPhase::Input => "input",
                    GuardrailPhase::Output => "output",
                };
                write!(
                    f,
                    "guardrail blocked {phase}: guard={guard}, rule={rule_id}, category={category}, severity={severity}, message={message}"
                )
            }
            LLMError::GuardrailExecutionFailed { guard, message } => {
                write!(
                    f,
                    "guardrail execution failed: guard={guard}, error={message}"
                )
            }
        }
    }
}

impl std::error::Error for LLMError {}

/// Converts reqwest HTTP errors into LLMErrors, preserving transport context.
#[cfg(not(target_arch = "wasm32"))]
impl From<reqwest::Error> for LLMError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            LLMError::HttpError(format!("request timed out: {err}"))
        } else if err.is_connect() {
            LLMError::HttpError(format!("connection failed: {err}"))
        } else {
            LLMError::HttpError(err.to_string())
        }
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

    #[test]
    fn test_truncate_for_display_short_body_unchanged() {
        assert_eq!(truncate_for_display("short"), "short");
    }

    #[test]
    fn test_truncate_for_display_long_body() {
        let body = "x".repeat(MAX_ERROR_BODY_DISPLAY_BYTES + 10);
        let truncated = truncate_for_display(&body);
        assert!(truncated.contains("truncated"));
        assert!(truncated.starts_with(&"x".repeat(MAX_ERROR_BODY_DISPLAY_BYTES)));
    }

    #[test]
    fn test_is_transport_retryable_message_prefixed_forms() {
        assert!(is_transport_retryable_message(
            "request timed out: operation timed out"
        ));
        assert!(is_transport_retryable_message(
            "connection failed: tcp connect error"
        ));
        assert!(!is_transport_retryable_message(
            "HTTP Status Error (400): bad request"
        ));
    }

    #[test]
    fn test_llm_error_display_auth_error_with_status_truncates_body() {
        let body = "secret".repeat(200);
        let error = LLMError::AuthError {
            message: "Unauthorized".to_string(),
            status_code: Some(401),
            response_body: Some(body.clone().into_boxed_str()),
        };
        let display = error.to_string();
        assert!(display.contains("401"));
        assert!(display.contains("truncated"));
        assert_eq!(error.response_body(), Some(body.as_str()));
    }

    #[test]
    fn test_llm_error_display_rate_limit_error_includes_status() {
        let error = LLMError::RateLimitError {
            status_code: 529,
            message: "Overloaded".to_string(),
            response_body: "overload".into(),
            retry_after: Some(Duration::from_secs(30)),
            provider_code: Some("overloaded".into()),
        };
        let display = error.to_string();
        assert!(display.contains("529"));
        assert!(display.contains("overloaded"));
        assert_eq!(error.http_status_code(), Some(529));
    }

    #[test]
    fn test_invalid_request_preserves_response_body() {
        let err = LLMError::InvalidRequest {
            message: "bad request".into(),
            status_code: Some(400),
            response_body: Some(r#"{"error":"details"}"#.into()),
        };
        assert_eq!(err.http_status_code(), Some(400));
        assert_eq!(err.response_body(), Some(r#"{"error":"details"}"#));
    }

    #[test]
    fn test_is_retryable_matrix() {
        assert!(
            LLMError::RateLimitError {
                status_code: 429,
                message: "limit".into(),
                response_body: "body".into(),
                retry_after: None,
                provider_code: None,
            }
            .is_retryable()
        );
        assert!(
            LLMError::HttpStatusError {
                status_code: 503,
                message: "down".into(),
                response_body: "body".into(),
                provider_code: None,
            }
            .is_retryable()
        );
        assert!(
            !LLMError::HttpStatusError {
                status_code: 400,
                message: "bad".into(),
                response_body: "body".into(),
                provider_code: None,
            }
            .is_retryable()
        );
        assert!(!LLMError::Generic("unsupported".into()).is_retryable());
        assert!(LLMError::HttpError("request timed out: elapsed".into()).is_retryable());
    }

    #[test]
    fn test_llm_error_debug_truncates_response_body() {
        let body = "secret".repeat(MAX_ERROR_BODY_DISPLAY_BYTES + 10);
        let error = LLMError::RateLimitError {
            status_code: 429,
            message: "limit".into(),
            response_body: body.clone().into_boxed_str(),
            retry_after: None,
            provider_code: None,
        };
        let debug = format!("{error:?}");
        assert!(debug.contains("truncated"));
        assert!(!debug.contains(&body));
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
}
