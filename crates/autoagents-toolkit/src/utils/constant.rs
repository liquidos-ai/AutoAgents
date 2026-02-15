#[cfg(feature = "search")]
pub(crate) enum RestHeaders {
    Accept,
    XSubscriptionToken,
    ApplicationJson,
}

#[cfg(feature = "search")]
impl RestHeaders {
    pub fn as_str(&self) -> &str {
        match self {
            RestHeaders::Accept => "Accept",
            RestHeaders::XSubscriptionToken => "X-Subscription-Token",
            RestHeaders::ApplicationJson => "application/json",
        }
    }
}

#[cfg(all(test, feature = "search"))]
mod tests {
    use super::*;

    #[test]
    fn test_rest_headers_returns_expected_values() {
        assert_eq!(RestHeaders::Accept.as_str(), "Accept");
        assert_eq!(RestHeaders::ApplicationJson.as_str(), "application/json");
    }
}
