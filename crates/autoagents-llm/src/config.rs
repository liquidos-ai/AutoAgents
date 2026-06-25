//! Shared configuration constants for LLM providers.

/// Default HTTP request timeout in seconds when none is explicitly configured.
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 120;

/// Resolves the effective request timeout from an optional explicit value.
#[inline]
pub fn resolve_request_timeout(explicit: Option<u64>) -> u64 {
    explicit.unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_request_timeout_uses_explicit_value() {
        assert_eq!(resolve_request_timeout(Some(30)), 30);
    }

    #[test]
    fn resolve_request_timeout_defaults_when_unset() {
        assert_eq!(resolve_request_timeout(None), DEFAULT_REQUEST_TIMEOUT_SECS);
    }
}
