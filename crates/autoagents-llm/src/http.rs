//! Centralized HTTP response handling for LLM provider backends.
//!
//! The status-code mapping logic
//! ([`map_http_status_to_error`][crate::http::map_http_status_to_error] and
//! [`parse_retry_after_value`][crate::http::parse_retry_after_value]) is target-independent
//! and is reused by both the native `reqwest` transport and the WASI Preview2 transport.
//! The `reqwest::Response`-based entry point ([`ensure_success`][crate::http::ensure_success])
//! and its integration tests are only compiled on non-wasm32 targets.

use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::error::LLMError;

/// Maximum bytes read from a non-success HTTP response body.
///
/// Shared by the native `reqwest` transport (`read_bounded_error_body`) and the
/// WASI Preview2 transport (`crate::wasi_http::read_bounded_error_body`) so both
/// apply the same cap to error bodies, keeping multi-MB CDN/WAF error pages out
/// of memory (and, on `wasm32-wasip2`, out of the constrained linear memory).
pub(crate) const MAX_HTTP_ERROR_BODY_BYTES: usize = 65_536;

/// Parsed provider error payload extracted from a non-success response body.
#[derive(Debug, Default)]
struct ProviderErrorDetails {
    message: Option<String>,
    provider_code: Option<String>,
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use futures::StreamExt;
    use reqwest::Response;

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

        Err(map_http_status_to_error(
            provider,
            status_code,
            body,
            retry_after,
        ))
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
            && content_length > MAX_HTTP_ERROR_BODY_BYTES as u64
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
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::ensure_success;

/// Builds a human-readable default error message for a non-success status code.
///
/// Transport-independent (no `reqwest` dependency) so it can be shared by the
/// native and WASI Preview2 status mappers. Falls back to the numeric status when
/// the reason phrase is unknown or non-standard.
fn default_error_message(provider: &str, status_code: u16) -> String {
    let reason = http_reason_phrase(status_code).map_or_else(
        || status_code.to_string(),
        |phrase| format!("{status_code} {phrase}"),
    );
    format!("{provider} API returned error status: {reason}")
}

/// Returns the canonical HTTP reason phrase for well-known status codes.
///
/// This mirrors the subset of reason phrases that `reqwest::StatusCode` would
/// render, without requiring the `reqwest` dependency on WASI targets.
fn http_reason_phrase(status_code: u16) -> Option<&'static str> {
    // Sorted by status code; uses a linear scan because the table is small
    // (41 entries) and the function is called only on error paths.
    const PHRASES: &[(u16, &str)] = &[
        (400, "Bad Request"),
        (401, "Unauthorized"),
        (402, "Payment Required"),
        (403, "Forbidden"),
        (404, "Not Found"),
        (405, "Method Not Allowed"),
        (406, "Not Acceptable"),
        (407, "Proxy Authentication Required"),
        (408, "Request Timeout"),
        (409, "Conflict"),
        (410, "Gone"),
        (411, "Length Required"),
        (412, "Precondition Failed"),
        (413, "Payload Too Large"),
        (414, "URI Too Long"),
        (415, "Unsupported Media Type"),
        (416, "Range Not Satisfiable"),
        (417, "Expectation Failed"),
        (418, "I'm a teapot"),
        (421, "Misdirected Request"),
        (422, "Unprocessable Entity"),
        (423, "Locked"),
        (424, "Failed Dependency"),
        (425, "Too Early"),
        (426, "Upgrade Required"),
        (428, "Precondition Required"),
        (429, "Too Many Requests"),
        (431, "Request Header Fields Too Large"),
        (451, "Unavailable For Legal Reasons"),
        (500, "Internal Server Error"),
        (501, "Not Implemented"),
        (502, "Bad Gateway"),
        (503, "Service Unavailable"),
        (504, "Gateway Timeout"),
        (505, "HTTP Version Not Supported"),
        (506, "Variant Also Negotiates"),
        (507, "Insufficient Storage"),
        (508, "Loop Detected"),
        (510, "Not Extended"),
        (511, "Network Authentication Required"),
        (529, "Site Is Overloaded"),
    ];
    PHRASES
        .iter()
        .find(|(code, _)| *code == status_code)
        .map(|(_, phrase)| *phrase)
}

/// Maps a non-success HTTP status code and response body to a typed
/// [`LLMError`].
///
/// This is the shared, transport-independent status mapper used by both the
/// native `reqwest` path ([`ensure_success`]) and the WASI Preview2 transport.
/// It is intended to be called only after the caller has observed a non-2xx
/// status code, and always returns an [`LLMError`] describing the failure.
pub(crate) fn map_http_status_to_error(
    provider: &str,
    status_code: u16,
    body: String,
    retry_after: Option<Duration>,
) -> LLMError {
    debug_assert!(
        !(200..300).contains(&status_code),
        "map_http_status_to_error called on a success status {status_code}"
    );

    let details = parse_provider_error_body(&body);
    let message = details
        .message
        .unwrap_or_else(|| default_error_message(provider, status_code));
    let provider_code = details.provider_code.map(String::into_boxed_str);
    let response_body = body.into_boxed_str();

    match status_code {
        401 | 403 => LLMError::AuthError {
            message,
            status_code: Some(status_code),
            response_body: Some(response_body),
        },
        429 | 529 => LLMError::RateLimitError {
            status_code,
            message,
            response_body,
            retry_after,
            provider_code,
        },
        400 | 404 | 413 | 422 => LLMError::InvalidRequest {
            message,
            status_code: Some(status_code),
            response_body: Some(response_body),
        },
        _ => LLMError::HttpStatusError {
            status_code,
            message,
            response_body,
            retry_after,
            provider_code,
        },
    }
}

/// Parses a `Retry-After` header value into a [`Duration`].
///
/// Accepts both the delta-seconds form (e.g. `"30"`) and the HTTP-date /
/// RFC 3339 form (e.g. `"Wed, 21 Oct 2015 07:28:00 GMT"`). Past dates and
/// unparseable values return `None`. This is shared by the native and WASI
/// transports.
pub(crate) fn parse_retry_after_value(value: &str) -> Option<Duration> {
    let value = value.trim();

    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    parse_retry_after_http_date(value)
}

/// Looks up the first `Retry-After` header value (case-insensitive) in a
/// `(name, value)` header list and parses it via [`parse_retry_after_value`].
///
/// This is the transport-agnostic counterpart of the native
/// `parse_retry_after(&reqwest::HeaderMap)`. The WASI Preview2 transport
/// collects response headers into lower-cased `(String, String)` pairs (see
/// `crate::wasi_http`); this helper scans them. Its only callers live behind
/// `#[cfg(wasi_http)]`, so it is gated to that target plus `test` (so the
/// native `cargo test` suite exercises it) — otherwise it would be dead code on
/// native non-test builds.
#[cfg(any(test, wasi_http))]
pub(crate) fn find_retry_after(headers: &[(String, String)]) -> Option<Duration> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("retry-after"))
        .and_then(|(_, value)| parse_retry_after_value(value))
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let value = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    parse_retry_after_value(value)
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

    #[test]
    fn find_retry_after_is_case_insensitive() {
        let headers = vec![
            ("Retry-After".to_string(), "30".to_string()),
            ("x-foo".to_string(), "bar".to_string()),
        ];
        assert_eq!(find_retry_after(&headers), Some(Duration::from_secs(30)));

        let lower = vec![("retry-after".to_string(), "12".to_string())];
        assert_eq!(find_retry_after(&lower), Some(Duration::from_secs(12)));
    }

    #[test]
    fn find_retry_after_missing_returns_none() {
        let headers = vec![("content-type".to_string(), "text/plain".to_string())];
        assert!(find_retry_after(&headers).is_none());
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
    fn parse_provider_error_body_reads_top_level_fields() {
        let details = parse_provider_error_body(r#"{"message":"fail","type":"provider_error"}"#);
        assert_eq!(details.message.as_deref(), Some("fail"));
        assert_eq!(details.provider_code.as_deref(), Some("provider_error"));
    }

    #[test]
    fn parse_provider_error_body_handles_invalid_json() {
        let details = parse_provider_error_body("not-json");
        assert!(details.message.is_none());
        assert!(details.provider_code.is_none());
    }

    #[test]
    fn parse_retry_after_value_accepts_delay_seconds() {
        assert_eq!(parse_retry_after_value("45"), Some(Duration::from_secs(45)));
        // Surrounding whitespace is tolerated.
        assert_eq!(
            parse_retry_after_value("  12 \n"),
            Some(Duration::from_secs(12))
        );
    }

    #[test]
    fn parse_retry_after_value_accepts_http_date() {
        let future = (Utc::now() + chrono::Duration::seconds(60))
            .format("%a, %d %b %Y %H:%M:%S GMT")
            .to_string();
        let parsed = parse_retry_after_value(&future).expect("retry-after should parse");
        assert!(parsed >= Duration::from_secs(55));
        assert!(parsed <= Duration::from_secs(65));
    }

    #[test]
    fn parse_retry_after_value_accepts_rfc3339_date() {
        let future = (Utc::now() + chrono::Duration::seconds(90)).to_rfc3339();
        let parsed = parse_retry_after_value(&future).expect("retry-after should parse");
        assert!(parsed >= Duration::from_secs(85));
        assert!(parsed <= Duration::from_secs(95));
    }

    #[test]
    fn parse_retry_after_value_past_http_date_returns_none() {
        let past = (Utc::now() - chrono::Duration::seconds(120))
            .format("%a, %d %b %Y %H:%M:%S GMT")
            .to_string();
        assert_eq!(parse_retry_after_value(&past), None);
    }

    #[test]
    fn parse_retry_after_value_rejects_garbage() {
        assert_eq!(parse_retry_after_value("not a date"), None);
        assert_eq!(parse_retry_after_value(""), None);
    }

    #[test]
    fn map_http_status_to_error_401_maps_to_auth_error() {
        let err = map_http_status_to_error(
            "OpenAI",
            401,
            r#"{"error":{"message":"invalid key"}}"#.to_string(),
            None,
        );
        match err {
            LLMError::AuthError {
                status_code,
                message,
                response_body,
            } => {
                assert_eq!(status_code, Some(401));
                assert_eq!(message, "invalid key");
                assert!(response_body.is_some());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn map_http_status_to_error_429_maps_to_rate_limit_with_retry_after() {
        let err = map_http_status_to_error(
            "OpenAI",
            429,
            r#"{"error":{"message":"rate limited","code":"rate_limit_exceeded"}}"#.to_string(),
            Some(Duration::from_secs(30)),
        );
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

    #[test]
    fn map_http_status_to_error_400_maps_to_invalid_request() {
        let err = map_http_status_to_error(
            "OpenAI",
            400,
            r#"{"error":{"message":"bad request"}}"#.to_string(),
            None,
        );
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

    #[test]
    fn map_http_status_to_error_500_maps_to_http_status_error() {
        let err = map_http_status_to_error("OpenAI", 500, "provider exploded".to_string(), None);
        match err {
            LLMError::HttpStatusError {
                status_code,
                message,
                response_body,
                ..
            } => {
                assert_eq!(status_code, 500);
                assert!(message.contains("500"));
                assert_eq!(response_body.as_ref(), "provider exploded");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod native_tests {
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
    async fn maps_503_to_http_status_error_with_retry_after() {
        let err = error_for_status(503, "service unavailable", Some("300")).await;
        match err {
            LLMError::HttpStatusError {
                status_code,
                retry_after,
                ..
            } => {
                assert_eq!(status_code, 503);
                assert_eq!(retry_after, Some(Duration::from_secs(300)));
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
                content_length <= MAX_HTTP_ERROR_BODY_BYTES as u64,
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
