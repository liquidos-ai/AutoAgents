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
