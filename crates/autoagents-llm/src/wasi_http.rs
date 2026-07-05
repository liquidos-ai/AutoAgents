//! WASI Preview2 (`wasm32-wasip2`) HTTP client built on `golem-wasi-http`.
//!
//! This module is compiled only for `wasm32-wasip2` with the `wasi-http`
//! feature. It provides a minimal, reqwest-style POST-with-Bearer-auth helper
//! backed by the [`golem_wasi_http::Client`], which itself wraps the `wasi:http`
//! / `wasi:io` host interfaces.
//!
//! Only the request shape needed by the OpenAI Responses backend is supported:
//! `POST` with a JSON body authenticated via `Authorization: Bearer <token>`.
//!
//! Timeouts are applied through `golem-wasi-http`'s connect / first-byte /
//! between-bytes configuration when a non-zero `timeout_seconds` is supplied.
//! A zero timeout leaves the host's defaults in place.

use std::time::Duration;

use golem_wasi_http::header::{CONTENT_TYPE, HeaderMap};
use golem_wasi_http::{
    Client, Error as GolemError, IncomingBody, InputStream, Response, StreamError,
};

use crate::error::LLMError;
use crate::http::MAX_HTTP_ERROR_BODY_BYTES;

/// A fully buffered HTTP response.
///
/// This is our own small abstraction over [`golem_wasi_http::Response`] so the
/// caller (`crates/autoagents-llm/src/backends/openai.rs`) does not depend on
/// the crate's response type. The body is read completely into memory by
/// `golem-wasi-http` before being returned here.
#[derive(Debug)]
pub(crate) struct HttpResponseData {
    /// HTTP response status code (e.g. `200`, `429`).
    pub status: u16,
    /// Response headers, lower-cased names paired with their string values.
    pub headers: Vec<(String, String)>,
    /// Decoded response body text.
    pub body: String,
}

/// Shared request builder for [`post_json_bearer`] (which buffers the whole
/// body) and [`post_json_bearer_stream`] (which leaves the body for incremental
/// reading). Returns the underlying [`golem_wasi_http::Response`] unbuffered.
fn send_post_request(
    url: &str,
    bearer_token: &str,
    json_body: &[u8],
    timeout_seconds: u64,
) -> Result<Response, LLMError> {
    // Parse the URL ourselves for a clearer error than golem's generic builder error.
    let parsed_url = url::Url::parse(url)
        .map_err(|e| LLMError::HttpError(format!("invalid request URL: {e}")))?;

    let client = build_client(timeout_seconds)?;

    if log::log_enabled!(log::Level::Trace) {
        log::trace!("WASI HTTP POST {url} ({} byte body)", json_body.len());
    }

    client
        .post(parsed_url)
        .bearer_auth(bearer_token)
        .header(CONTENT_TYPE, "application/json")
        .body(json_body.to_vec())
        .send()
        .map_err(golem_error_to_llm)
}

/// Issues a bearer-authenticated `GET` request and returns the underlying
/// [`golem_wasi_http::Response`] unbuffered.
fn send_get_request(
    url: &str,
    bearer_token: &str,
    timeout_seconds: u64,
) -> Result<Response, LLMError> {
    let parsed_url = url::Url::parse(url)
        .map_err(|e| LLMError::HttpError(format!("invalid request URL: {e}")))?;

    let client = build_client(timeout_seconds)?;

    if log::log_enabled!(log::Level::Trace) {
        log::trace!("WASI HTTP GET {url}");
    }

    client
        .get(parsed_url)
        .bearer_auth(bearer_token)
        .send()
        .map_err(golem_error_to_llm)
}

/// Reads an error (non-2xx) response body up to
/// [`MAX_HTTP_ERROR_BODY_BYTES`](crate::http::MAX_HTTP_ERROR_BODY_BYTES),
/// mirroring the native `reqwest` transport's `read_bounded_error_body`.
///
/// When `Content-Length` exceeds the cap the body is not read at all and a
/// placeholder is returned; otherwise the body is streamed through
/// [`WasiResponseStream`] and accumulation stops at the cap with a truncation
/// marker. This keeps multi-MB CDN/WAF error pages out of `wasm32-wasip2`'s
/// constrained linear memory — the same defense the native path applies.
fn read_bounded_error_body(mut response: Response) -> Result<String, LLMError> {
    if let Some(content_length) = response.content_length()
        && content_length > MAX_HTTP_ERROR_BODY_BYTES as u64
    {
        return Ok(format!(
            "... [body omitted, Content-Length: {content_length} bytes]"
        ));
    }

    let (input_stream, incoming_body) = response.get_raw_input_stream();
    let mut reader = WasiResponseStream::new(input_stream, incoming_body);
    let mut collected: Vec<u8> = Vec::with_capacity(MAX_HTTP_ERROR_BODY_BYTES.min(8192));
    let mut total_received = 0usize;

    while let Some(chunk) = reader.read_chunk()? {
        total_received += chunk.len();
        let remaining = MAX_HTTP_ERROR_BODY_BYTES.saturating_sub(collected.len());
        if chunk.len() <= remaining {
            collected.extend_from_slice(&chunk);
        } else {
            collected.extend_from_slice(&chunk[..remaining]);
        }

        if collected.len() >= MAX_HTTP_ERROR_BODY_BYTES {
            let truncated = String::from_utf8_lossy(&collected).into_owned();
            return Ok(format!(
                "{truncated}... [truncated after reading {total_received} bytes]"
            ));
        }
    }

    Ok(String::from_utf8_lossy(&collected).into_owned())
}

/// Buffers a response into [`HttpResponseData`], using full-body reads for
/// success statuses and bounded reads for non-success statuses.
fn buffer_response(response: Response) -> Result<HttpResponseData, LLMError> {
    let status = response.status().as_u16();
    let headers = collect_headers(response.headers());

    let is_success = (200..300).contains(&status);
    let body = if is_success {
        response.text().map_err(golem_error_to_llm)?
    } else {
        read_bounded_error_body(response)?
    };

    Ok(HttpResponseData {
        status,
        headers,
        body,
    })
}

/// Sends a JSON `POST` request authenticated with a bearer token and returns
/// the fully buffered response.
///
/// `timeout_seconds` of `0` leaves the host's default timeouts in place.
/// Transport failures map to [`LLMError::HttpError`]; non-success status codes
/// are returned in [`HttpResponseData`] for the caller to pass through
/// [`map_http_status_to_error`](crate::http::map_http_status_to_error).
///
/// On success (2xx) the body is buffered fully via
/// [`golem_wasi_http::Response::text`] (the Responses API returns the complete
/// JSON object, matching the native `reqwest` path). On a non-success status
/// the body is bounded by [`read_bounded_error_body`] so large error pages
/// cannot exhaust the linear memory.
pub(crate) fn post_json_bearer(
    url: &str,
    bearer_token: &str,
    json_body: &[u8],
    timeout_seconds: u64,
) -> Result<HttpResponseData, LLMError> {
    let response = send_post_request(url, bearer_token, json_body, timeout_seconds)?;
    buffer_response(response)
}

/// Sends a bearer-authenticated `GET` request and returns the fully buffered
/// response.
pub(crate) fn get_bearer(
    url: &str,
    bearer_token: &str,
    timeout_seconds: u64,
) -> Result<HttpResponseData, LLMError> {
    let response = send_get_request(url, bearer_token, timeout_seconds)?;
    buffer_response(response)
}

/// Sends a JSON `POST` request and returns the response without buffering the
/// body, so the caller can consume it incrementally as a stream of chunks.
///
/// This is the streaming counterpart of [`post_json_bearer`]. Use
/// [`StreamingResponse::into_byte_stream`] on the success path (e.g. for SSE
/// responses) and [`StreamingResponse::into_bounded_error_text`] on the error
/// path (to surface a meaningful but memory-bounded error body, mirroring the
/// native `ensure_success` behavior).
pub(crate) fn post_json_bearer_stream(
    url: &str,
    bearer_token: &str,
    json_body: &[u8],
    timeout_seconds: u64,
) -> Result<StreamingResponse, LLMError> {
    let response = send_post_request(url, bearer_token, json_body, timeout_seconds)?;
    let status = response.status().as_u16();
    let headers = collect_headers(response.headers());
    Ok(StreamingResponse {
        status,
        headers,
        response,
    })
}

/// HTTP response whose body can be consumed incrementally as a stream of
/// chunks (for SSE-style responses) or fully buffered as text (for error
/// bodies).
///
/// Returned by [`post_json_bearer_stream`]. Body-consuming methods take `self`
/// by value so the body can only be read once, matching the single-consume
/// semantics of the underlying `wasi:http` incoming body.
pub(crate) struct StreamingResponse {
    status: u16,
    headers: Vec<(String, String)>,
    response: Response,
}

impl StreamingResponse {
    /// HTTP response status code (e.g. `200`, `429`).
    pub(crate) fn status(&self) -> u16 {
        self.status
    }

    /// Response headers (lower-cased names), for `Retry-After` lookup.
    pub(crate) fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    /// Consumes the response and returns a chunk reader over the body.
    ///
    /// Use this on the success path for streaming responses.
    pub(crate) fn into_byte_stream(mut self) -> Result<WasiResponseStream, LLMError> {
        let (input_stream, incoming_body) = self.response.get_raw_input_stream();
        Ok(WasiResponseStream::new(input_stream, incoming_body))
    }

    /// Buffers the response body up to
    /// [`MAX_HTTP_ERROR_BODY_BYTES`](crate::http::MAX_HTTP_ERROR_BODY_BYTES) for
    /// use as an error body (non-success status), mirroring the native
    /// `read_bounded_error_body`. Use this on the error path to surface a
    /// meaningful but memory-bounded error body.
    pub(crate) fn into_bounded_error_text(self) -> Result<String, LLMError> {
        read_bounded_error_body(self.response)
    }
}

/// Bytes requested per `InputStream::blocking_read` call. Bounding reads lets
/// incremental SSE parsing surface events promptly; `blocking_read` returns at
/// least one byte (or `Closed`) regardless.
const WASI_READ_CHUNK_SIZE: u64 = 64 * 1024;

/// Blocking chunk reader over a WASI HTTP response body.
///
/// Wraps the raw [`InputStream`] returned by
/// [`golem_wasi_http::Response::get_raw_input_stream`]. Each
/// [`WasiResponseStream::read_chunk`] call performs a single blocking read (the
/// WASI host multiplexes other work while the current thread is blocked) and
/// returns `Ok(None)` at end of stream.
///
/// # Field / drop order
///
/// `input_stream` is declared before `incoming_body` so that, on drop, the
/// child input-stream is released before its parent incoming-body — the
/// ordering required by the `wasi:http/types` contract.
pub(crate) struct WasiResponseStream {
    input_stream: Option<InputStream>,
    incoming_body: Option<IncomingBody>,
    done: bool,
}

impl WasiResponseStream {
    fn new(input_stream: InputStream, incoming_body: IncomingBody) -> Self {
        Self {
            input_stream: Some(input_stream),
            incoming_body: Some(incoming_body),
            done: false,
        }
    }

    /// Reads up to 64KiB of the response body, blocking until at least one byte
    /// is available. Returns `Ok(None)` once the stream has ended.
    pub(crate) fn read_chunk(&mut self) -> Result<Option<Vec<u8>>, LLMError> {
        if self.done {
            return Ok(None);
        }
        let Some(stream) = self.input_stream.as_ref() else {
            return Ok(None);
        };

        match stream.blocking_read(WASI_READ_CHUNK_SIZE) {
            // `blocking_read` is documented to return at least one byte (or
            // `StreamError::Closed`); treat an empty result defensively as EOF.
            Ok(bytes) if bytes.is_empty() => {
                self.finish();
                Ok(None)
            }
            Ok(bytes) => Ok(Some(bytes)),
            Err(StreamError::Closed) => {
                self.finish();
                Ok(None)
            }
            Err(StreamError::LastOperationFailed(err)) => {
                self.finish();
                Err(LLMError::HttpError(format!(
                    "WASI response stream read failed: {err}"
                )))
            }
        }
    }

    /// Marks the reader as finished and releases both WASI handles in the order
    /// required by the `wasi:http/types` contract (input-stream first, then the
    /// parent incoming-body). Idempotent.
    fn finish(&mut self) {
        self.done = true;
        self.input_stream.take();
        self.incoming_body.take();
    }
}

/// Builds a [`Client`] with the requested timeout applied to the connect,
/// first-byte and between-bytes knobs. A zero `timeout_seconds` leaves the
/// host's defaults in place.
fn build_client(timeout_seconds: u64) -> Result<Client, LLMError> {
    let mut builder = Client::builder();
    if timeout_seconds > 0 {
        let duration = Duration::from_secs(timeout_seconds);
        // `timeout` covers first-byte and between-bytes; `connect_timeout`
        // covers the connect phase. Setting both applies the same deadline to
        // all phases the WASI HTTP client exposes.
        builder = builder.timeout(duration).connect_timeout(duration);
    }
    builder.build().map_err(golem_error_to_llm)
}

/// Collects a [`golem_wasi_http`] response's headers into name/value pairs.
///
/// [`http::HeaderName`] already normalizes names to lowercase. Values may be
/// non-UTF-8 bytes (rare for JSON APIs); such values fall back to an empty
/// string rather than surfacing the conversion error to the caller.
fn collect_headers(headers: &HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .map(|(name, value)| {
            let value_str = value.to_str().unwrap_or_default().to_string();
            (name.as_str().to_string(), value_str)
        })
        .collect()
}

/// Maps a [`golem_wasi_http::Error`] to an [`LLMError::HttpError`].
///
/// Timeouts are reported with a `"request timed out:"` prefix so the shared
/// retryability classifier
/// ([`is_transport_retryable_message`](crate::error::is_transport_retryable_message))
/// treats them as retryable.
fn golem_error_to_llm(err: GolemError) -> LLMError {
    if err.is_timeout() {
        LLMError::HttpError(format!("request timed out: {err}"))
    } else {
        LLMError::HttpError(format!("{err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // NOTE: this module only compiles for `wasm32-wasip2` + `wasi-http`, so
    // these tests run under `cargo test --target wasm32-wasip2` and the
    // `examples/wasm_smoke` harness, NOT the default native `cargo test` suite.
    // The transport-agnostic `Retry-After` lookup (`find_retry_after`) is now
    // shared in `crate::http` and covered natively in `crate::http::tests`.

    #[test]
    fn http_response_data_roundtrips_fields() {
        // Exercises the data contract the OpenAI Responses caller relies on:
        // status, headers and body are preserved verbatim, and `Retry-After`
        // is discoverable from the buffered headers.
        let data = HttpResponseData {
            status: 429,
            headers: vec![("retry-after".to_string(), "5".to_string())],
            body: r#"{"error":"rate limited"}"#.to_string(),
        };
        assert_eq!(data.status, 429);
        assert_eq!(
            crate::http::find_retry_after(&data.headers),
            Some(Duration::from_secs(5))
        );
        assert_eq!(data.body, r#"{"error":"rate limited"}"#);
    }
}
