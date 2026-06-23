//! Shared configuration constants for LLM providers.

/// Default HTTP request timeout in seconds when none is explicitly configured.
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 120;

/// Resolves the effective request timeout from an optional explicit value.
#[inline]
pub fn resolve_request_timeout(explicit: Option<u64>) -> u64 {
    explicit.unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECS)
}
