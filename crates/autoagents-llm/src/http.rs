//! Centralized HTTP response handling for LLM provider backends.

use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::{Response, StatusCode};

use crate::error::LLMError;

/// Maximum bytes read from a non-success HTTP response body.
const MAX_HTTP_ERROR_BODY_BYTES: usize = 65_536;

/// Parsed provider error payload extracted from a non-success response body.
#[derive(Debug, Default)]
struct ProviderErrorDetails {
    message: Option<String>,
    provider_code: Option<String>,
}

/// Ensures an HTTP response has a success status, mapping failures to typed errors.
///
/// On success, returns the original response unchanged so callers can decode the body.
/// On failure, reads the response body once and maps status codes to
/// [`AuthError`](LLMError::AuthError), [`RateLimitError`](LLMError::RateLimitError),
/// [`HttpStatusError`](LLMError::HttpStatusError), or [`InvalidRequest`](LLMError::InvalidRequest).
pub async fn ensure_success(response: Response, provider: &str) -> Result<Response, LLMError> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status_code = response.status().as_u16();
    let retry_after = parse_retry_after(response.headers());
    let body = read_bounded_error_body(response).await?;
    let details = parse_provider_error_body(&body);

    map_http_status_error(provider, status_code, body, details, retry_after)
}

fn default_error_message(provider: &str, status_code: u16) -> String {
    match StatusCode::from_u16(status_code) {
        Ok(status) => format!("{provider} API returned error status: {status}"),
        Err(_) => format!("{provider} API returned error status: {status_code}"),
    }
}

fn http_status_error(
    status_code: u16,
    message: String,
    response_body: impl Into<Box<str>>,
    provider_code: Option<Box<str>>,
) -> LLMError {
    LLMError::HttpStatusError {
        status_code,
        message,
        response_body: response_body.into(),
        provider_code,
    }
}

async fn read_bounded_error_body(response: Response) -> Result<String, LLMError> {
    let bytes = response.bytes().await?;
    if bytes.len() <= MAX_HTTP_ERROR_BODY_BYTES {
        return Ok(String::from_utf8_lossy(&bytes).into_owned());
    }

    let truncated = String::from_utf8_lossy(&bytes[..MAX_HTTP_ERROR_BODY_BYTES]).into_owned();
    Ok(format!(
        "{truncated}... [truncated, {} bytes total]",
        bytes.len()
    ))
}

fn map_http_status_error(
    provider: &str,
    status_code: u16,
    body: String,
    details: ProviderErrorDetails,
    retry_after: Option<Duration>,
) -> Result<Response, LLMError> {
    let message = details
        .message
        .unwrap_or_else(|| default_error_message(provider, status_code));
    let provider_code = details.provider_code.map(String::into_boxed_str);
    let response_body = body.into_boxed_str();

    match status_code {
        401 | 403 => Err(LLMError::AuthError {
            message,
            status_code: Some(status_code),
            response_body: Some(response_body),
        }),
        429 | 529 => Err(LLMError::RateLimitError {
            status_code,
            message,
            response_body,
            retry_after,
            provider_code,
        }),
        400 | 404 | 413 | 422 => Err(LLMError::InvalidRequest {
            message,
            status_code: Some(status_code),
            response_body: Some(response_body),
        }),
        _ => Err(http_status_error(
            status_code,
            message,
            response_body,
            provider_code,
        )),
    }
}

fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let value = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim();

    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    parse_retry_after_http_date(value)
}

fn parse_retry_after_http_date(value: &str) -> Option<Duration> {
    let retry_at = DateTime::parse_from_rfc2822(value)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
        .or_else(|| {
            DateTime::parse_from_rfc3339(value)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        })?;

    let now = Utc::now();
    if retry_at > now {
        retry_at.signed_duration_since(now).to_std().ok()
    } else {
        None
    }
}

fn parse_provider_error_body(body: &str) -> ProviderErrorDetails {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return ProviderErrorDetails::default();
    };

    // OpenAI-compatible: { "error": { "message", "type", "code" } }
    if let Some(error) = value.get("error") {
        return ProviderErrorDetails {
            message: error
                .get("message")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            provider_code: error
                .get("code")
                .or_else(|| error.get("type"))
                .and_then(|v| v.as_str())
                .map(str::to_string),
        };
    }

    ProviderErrorDetails {
        message: value
            .get("message")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        provider_code: value
            .get("type")
            .and_then(|v| v.as_str())
            .map(str::to_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::Client;

    async fn error_for_status(status: u16, body: &str, retry_after: Option<&str>) -> LLMError {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/error");
            if let Some(value) = retry_after {
                then.status(status).body(body).header("Retry-After", value);
            } else {
                then.status(status).body(body);
            }
        });

        let client = Client::new();
        let response = client
            .get(format!("{}/error", server.base_url()))
            .send()
            .await
            .unwrap();
        mock.assert();
        ensure_success(response, "TestProvider")
            .await
            .expect_err("expected error")
    }

    #[test]
    fn parse_retry_after_accepts_delay_seconds() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "45".parse().unwrap());
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(45)));
    }

    #[test]
    fn parse_retry_after_accepts_http_date() {
        let mut headers = reqwest::header::HeaderMap::new();
        let future = (Utc::now() + chrono::Duration::seconds(60))
            .format("%a, %d %b %Y %H:%M:%S GMT")
            .to_string();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            future.parse().expect("valid header value"),
        );
        let parsed = parse_retry_after(&headers).expect("retry-after should parse");
        assert!(parsed >= Duration::from_secs(55));
        assert!(parsed <= Duration::from_secs(65));
    }

    #[tokio::test]
    async fn maps_401_to_auth_error() {
        let err = error_for_status(401, r#"{"error":{"message":"invalid key"}}"#, None).await;
        match err {
            LLMError::AuthError {
                status_code,
                message,
                ..
            } => {
                assert_eq!(status_code, Some(401));
                assert_eq!(message, "invalid key");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_403_to_auth_error() {
        let err = error_for_status(403, r#"{"error":{"message":"forbidden"}}"#, None).await;
        match err {
            LLMError::AuthError {
                status_code,
                message,
                ..
            } => {
                assert_eq!(status_code, Some(403));
                assert_eq!(message, "forbidden");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_429_to_rate_limit_error_with_retry_after() {
        let err = error_for_status(
            429,
            r#"{"error":{"message":"rate limited","code":"rate_limit_exceeded"}}"#,
            Some("30"),
        )
        .await;
        match err {
            LLMError::RateLimitError {
                status_code,
                message,
                retry_after,
                provider_code,
                ..
            } => {
                assert_eq!(status_code, 429);
                assert_eq!(message, "rate limited");
                assert_eq!(provider_code.as_deref(), Some("rate_limit_exceeded"));
                assert_eq!(retry_after, Some(Duration::from_secs(30)));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_529_to_rate_limit_error() {
        let err = error_for_status(
            529,
            r#"{"error":{"message":"overloaded","type":"overloaded_error"}}"#,
            None,
        )
        .await;
        match err {
            LLMError::RateLimitError {
                status_code,
                message,
                provider_code,
                ..
            } => {
                assert_eq!(status_code, 529);
                assert_eq!(message, "overloaded");
                assert_eq!(provider_code.as_deref(), Some("overloaded_error"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_408_to_retryable_http_status_error() {
        let err = error_for_status(408, "request timeout", None).await;
        match err {
            LLMError::HttpStatusError { status_code, .. } => assert_eq!(status_code, 408),
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(err.is_retryable());
    }

    #[tokio::test]
    async fn maps_500_to_http_status_error() {
        let err = error_for_status(500, "provider exploded", None).await;
        match err {
            LLMError::HttpStatusError {
                status_code,
                response_body,
                ..
            } => {
                assert_eq!(status_code, 500);
                assert_eq!(response_body.as_ref(), "provider exploded");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_400_to_invalid_request_with_body() {
        let err = error_for_status(400, r#"{"error":{"message":"bad request"}}"#, None).await;
        match err {
            LLMError::InvalidRequest {
                message,
                status_code,
                response_body,
            } => {
                assert_eq!(message, "bad request");
                assert_eq!(status_code, Some(400));
                assert!(response_body.is_some());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_plain_text_503_to_http_status_error() {
        let err = error_for_status(503, "service unavailable", None).await;
        match err {
            LLMError::HttpStatusError {
                status_code,
                message,
                response_body,
                ..
            } => {
                assert_eq!(status_code, 503);
                assert_eq!(
                    message,
                    "TestProvider API returned error status: 503 Service Unavailable"
                );
                assert_eq!(response_body.as_ref(), "service unavailable");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn success_response_passes_through() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/ok");
            then.status(200).body(r#"{"ok":true}"#);
        });

        let client = Client::new();
        let response = client
            .get(format!("{}/ok", server.base_url()))
            .send()
            .await
            .unwrap();
        let response = ensure_success(response, "TestProvider").await.unwrap();
        mock.assert();
        assert_eq!(response.status(), 200);
        assert_eq!(response.text().await.unwrap(), r#"{"ok":true}"#);
    }
}
