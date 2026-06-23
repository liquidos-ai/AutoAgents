//! Centralized HTTP response handling for LLM provider backends.

use std::time::Duration;

use chrono::{DateTime, Utc};
use futures::StreamExt;
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
    // Reads at most `MAX_HTTP_ERROR_BODY_BYTES` from the wire on error responses.
    //
    // When `Content-Length` exceeds the cap, the body is not read at all and a placeholder
    // is returned instead. This avoids pulling multi-megabyte CDN/WAF HTML pages into memory.
    //
    // Otherwise the body is streamed and accumulation stops at the cap. The remainder is not
    // drained, so the connection may not be reused; that tradeoff is intentional on the
    // error path where bounded memory use matters more than keep-alive reuse.
    if let Some(content_length) = response.content_length()
        && content_length as usize > MAX_HTTP_ERROR_BODY_BYTES
    {
        return Ok(format!(
            "... [body omitted, Content-Length: {content_length} bytes]"
        ));
    }

    let mut stream = response.bytes_stream();
    let mut collected = Vec::with_capacity(MAX_HTTP_ERROR_BODY_BYTES.min(8192));
    let mut total_received = 0usize;
    let mut at_capacity = false;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        let chunk_len = chunk.len();
        total_received += chunk_len;

        if !at_capacity {
            let remaining = MAX_HTTP_ERROR_BODY_BYTES.saturating_sub(collected.len());
            if chunk_len <= remaining {
                collected.extend_from_slice(&chunk);
                if collected.len() == MAX_HTTP_ERROR_BODY_BYTES {
                    at_capacity = true;
                }
            } else {
                collected.extend_from_slice(&chunk[..remaining]);
                at_capacity = true;
            }
        }

        if at_capacity {
            break;
        }
    }

    if at_capacity {
        Ok(format_truncated_error_body(&collected, total_received))
    } else {
        Ok(String::from_utf8_lossy(&collected).into_owned())
    }
}

fn format_truncated_error_body(collected: &[u8], bytes_read: usize) -> String {
    let truncated = String::from_utf8_lossy(collected).into_owned();
    format!("{truncated}... [truncated after reading {bytes_read} bytes]")
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
    // Google: { "error": { "message", "code" (numeric), "status" } }
    if let Some(error) = value.get("error") {
        return ProviderErrorDetails {
            message: error
                .get("message")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            provider_code: error
                .get("code")
                .and_then(|v| v.as_str())
                .or_else(|| error.get("type").and_then(|v| v.as_str()))
                .or_else(|| error.get("status").and_then(|v| v.as_str()))
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
    fn parse_provider_error_body_extracts_google_status() {
        let details = parse_provider_error_body(
            r#"{"error":{"code":429,"message":"Resource exhausted","status":"RESOURCE_EXHAUSTED"}}"#,
        );
        assert_eq!(details.message.as_deref(), Some("Resource exhausted"));
        assert_eq!(details.provider_code.as_deref(), Some("RESOURCE_EXHAUSTED"));
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

    #[test]
    fn parse_retry_after_accepts_rfc3339_date() {
        let mut headers = reqwest::header::HeaderMap::new();
        let future = (Utc::now() + chrono::Duration::seconds(90)).to_rfc3339();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            future.parse().expect("valid header value"),
        );
        let parsed = parse_retry_after(&headers).expect("retry-after should parse");
        assert!(parsed >= Duration::from_secs(85));
        assert!(parsed <= Duration::from_secs(95));
    }

    #[test]
    fn parse_retry_after_past_http_date_returns_none() {
        let mut headers = reqwest::header::HeaderMap::new();
        let past = (Utc::now() - chrono::Duration::seconds(120))
            .format("%a, %d %b %Y %H:%M:%S GMT")
            .to_string();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            past.parse().expect("valid header value"),
        );
        assert_eq!(parse_retry_after(&headers), None);
    }

    #[test]
    fn parse_provider_error_body_reads_top_level_fields() {
        let details = parse_provider_error_body(r#"{"message":"fail","type":"provider_error"}"#);
        assert_eq!(details.message.as_deref(), Some("fail"));
        assert_eq!(details.provider_code.as_deref(), Some("provider_error"));
    }

    #[tokio::test]
    async fn maps_non_standard_status_code_uses_numeric_default_message() {
        let err = error_for_status(999, "non-standard", None).await;
        match err {
            LLMError::HttpStatusError {
                message,
                status_code,
                ..
            } => {
                assert_eq!(status_code, 999);
                assert!(message.contains("999"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
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
    async fn read_bounded_error_body_skips_oversized_content_length() {
        let server = httpmock::MockServer::start();
        let oversized_body = "x".repeat(MAX_HTTP_ERROR_BODY_BYTES + 1);
        let _mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/huge");
            then.status(500).body(&oversized_body);
        });

        let client = Client::new();
        let response = client
            .get(format!("{}/huge", server.base_url()))
            .send()
            .await
            .unwrap();
        let err = ensure_success(response, "TestProvider")
            .await
            .expect_err("expected error");

        match err {
            LLMError::HttpStatusError { response_body, .. } => {
                assert!(response_body.contains("body omitted"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn read_bounded_error_body_truncates_chunked_response() {
        let server = httpmock::MockServer::start();
        let overflow = 8_192;
        let oversized_body = "y".repeat(MAX_HTTP_ERROR_BODY_BYTES + overflow);
        let _mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/chunked");
            then.status(500)
                .header("Transfer-Encoding", "chunked")
                .body(&oversized_body);
        });

        let client = Client::new();
        let response = client
            .get(format!("{}/chunked", server.base_url()))
            .send()
            .await
            .unwrap();

        // Chunked responses must not advertise a Content-Length above the cap; otherwise
        // the fast-path skip would run instead of the streaming truncation under test.
        match response.content_length() {
            None => {}
            Some(content_length) => assert!(
                content_length as usize <= MAX_HTTP_ERROR_BODY_BYTES,
                "unexpected Content-Length for streaming test: {content_length}"
            ),
        }

        let err = ensure_success(response, "TestProvider")
            .await
            .expect_err("expected error");

        match err {
            LLMError::HttpStatusError { response_body, .. } => {
                let expected_prefix = "y".repeat(MAX_HTTP_ERROR_BODY_BYTES);
                assert!(
                    response_body.starts_with(&expected_prefix),
                    "expected first {MAX_HTTP_ERROR_BODY_BYTES} bytes preserved"
                );
                assert!(
                    response_body.contains("truncated after reading"),
                    "unexpected body: {response_body}"
                );
                assert!(
                    !response_body.contains("body omitted"),
                    "expected streaming truncation, not Content-Length skip"
                );
                assert!(response_body.len() < oversized_body.len());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn read_bounded_error_body_returns_small_body_unchanged() {
        let err = error_for_status(500, "small error", None).await;
        match err {
            LLMError::HttpStatusError { response_body, .. } => {
                assert_eq!(response_body.as_ref(), "small error");
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
