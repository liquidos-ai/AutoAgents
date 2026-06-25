use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use http::{HeaderName, HeaderValue};
use reqwest::Client;
use rmcp::{
    model::ClientInfo,
    service::{RoleClient, RunningService, ServiceExt},
    transport::StreamableHttpClientTransport,
    transport::streamable_http_client::StreamableHttpClientTransportConfig,
};

use crate::mcp::config::McpServerConfig;
use crate::mcp::security::validate_http_url;

use super::{McpError, with_timeout};

pub async fn connect_http_server(
    config: &McpServerConfig,
    legacy_sse: bool,
    allow_private_http_endpoints: bool,
) -> Result<Arc<RunningService<RoleClient, ClientInfo>>, McpError> {
    if legacy_sse {
        log::warn!(
            "MCP server '{}' uses deprecated 'sse' protocol; treating as Streamable HTTP",
            config.name
        );
    }

    let url = config.url.as_ref().ok_or_else(|| {
        McpError::ConfigError("url is required for http/sse transport".to_string())
    })?;

    validate_http_url(url, allow_private_http_endpoints).map_err(McpError::from)?;

    let timeout = Duration::from_secs(config.timeout);
    let http_client = Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| McpError::TransportError(format!("Failed to build HTTP client: {e}")))?;

    let mut transport_config = StreamableHttpClientTransportConfig::with_uri(url.as_str());
    let (auth_header, custom_headers) = split_headers(&config.headers)?;
    if let Some(auth) = auth_header {
        transport_config = transport_config.auth_header(auth);
    }
    if !custom_headers.is_empty() {
        transport_config = transport_config.custom_headers(custom_headers);
    }

    let transport = StreamableHttpClientTransport::with_client(http_client, transport_config);
    let client_info = ClientInfo::default();

    let service = with_timeout(timeout, "connect", async {
        client_info.serve(transport).await.map_err(|e| {
            McpError::ConnectionFailed(format!("Failed to connect to MCP server: {e:?}"))
        })
    })
    .await?;

    Ok(Arc::new(service))
}

fn split_headers(
    headers: &HashMap<String, String>,
) -> Result<(Option<String>, HashMap<HeaderName, HeaderValue>), McpError> {
    let mut auth_header = None;
    let mut custom_headers = HashMap::new();

    for (name, value) in headers {
        if name.eq_ignore_ascii_case("authorization") {
            if let Some(token) = value
                .strip_prefix("Bearer ")
                .or_else(|| value.strip_prefix("bearer "))
            {
                auth_header = Some(token.to_string());
            } else {
                let header_name = HeaderName::from_static("authorization");
                let header_value = HeaderValue::from_str(value).map_err(|e| {
                    McpError::ConfigError(format!("invalid authorization header value: {e}"))
                })?;
                custom_headers.insert(header_name, header_value);
            }
            continue;
        }

        let header_name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|e| McpError::ConfigError(format!("invalid header name '{name}': {e}")))?;
        let header_value = HeaderValue::from_str(value).map_err(|e| {
            McpError::ConfigError(format!("invalid header value for '{name}': {e}"))
        })?;
        custom_headers.insert(header_name, header_value);
    }

    Ok((auth_header, custom_headers))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_headers_extracts_bearer_token() {
        let mut headers = HashMap::new();
        headers.insert(
            "Authorization".to_string(),
            "Bearer secret-token".to_string(),
        );
        headers.insert("X-Custom".to_string(), "value".to_string());

        let (auth, custom) = split_headers(&headers).unwrap();
        assert_eq!(auth.as_deref(), Some("secret-token"));
        assert_eq!(custom.len(), 1);
    }

    #[test]
    fn split_headers_preserves_non_bearer_authorization() {
        let mut headers = HashMap::new();
        headers.insert(
            "Authorization".to_string(),
            "Basic dXNlcjpwYXNz".to_string(),
        );

        let (auth, custom) = split_headers(&headers).unwrap();
        assert!(auth.is_none());
        assert_eq!(custom.len(), 1);
        assert_eq!(
            custom
                .get(&HeaderName::from_static("authorization"))
                .and_then(|v| v.to_str().ok()),
            Some("Basic dXNlcjpwYXNz")
        );
    }
}
