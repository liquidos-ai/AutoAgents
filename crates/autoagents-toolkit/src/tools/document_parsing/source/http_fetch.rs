use futures::StreamExt;
use reqwest::{Client, Response, redirect::Policy};
use url::Url;

use super::error::DocumentSourceError;
use super::url_policy::{resolve_host, validate_url, validate_url_str};
use crate::tools::document_parsing::config::DocumentParserConfig;

fn build_pinned_client(
    host: &str,
    addresses: &[std::net::SocketAddr],
    config: &DocumentParserConfig,
) -> Result<Client, DocumentSourceError> {
    // A fresh client is built per hop so DNS pinning via `resolve_to_addrs` stays accurate.
    // Document parsing is low-QPS; connection reuse is less important than SSRF resistance.
    Client::builder()
        .timeout(config.request_timeout)
        .redirect(Policy::none())
        .resolve_to_addrs(host, addresses)
        .build()
        .map_err(DocumentSourceError::from)
}

pub async fn fetch_url(
    url: &str,
    config: &DocumentParserConfig,
) -> Result<(Vec<u8>, Option<String>), DocumentSourceError> {
    let mut current = validate_url_str(url, config)?;
    let mut redirects = 0usize;

    loop {
        let response = send_request(&current, config).await?;

        if response.status().is_redirection() {
            if redirects >= config.max_redirects {
                return Err(DocumentSourceError::TooManyRedirects {
                    limit: config.max_redirects,
                });
            }

            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .ok_or(DocumentSourceError::MissingRedirectLocation)?
                .to_str()
                .map_err(|_| {
                    DocumentSourceError::InvalidRedirectLocation("non-UTF-8 Location".to_string())
                })?;

            current = current.join(location).map_err(|error| {
                DocumentSourceError::InvalidRedirectLocation(format!("{location}: {error}"))
            })?;
            validate_url(&current, config)?;
            redirects += 1;
            continue;
        }

        let checked = response.error_for_status().map_err(map_status_error)?;

        let bytes = read_bounded_body(checked, config.max_download_bytes).await?;
        let filename = current
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .filter(|segment| !segment.is_empty())
            .map(|segment| segment.to_string());

        return Ok((bytes, filename));
    }
}

async fn send_request(
    url: &Url,
    config: &DocumentParserConfig,
) -> Result<Response, DocumentSourceError> {
    let host = url
        .host_str()
        .ok_or_else(|| DocumentSourceError::InvalidHost(url.to_string()))?;
    let port = url.port_or_known_default().unwrap_or(80);
    let addresses = resolve_host(host, port, config).await?;
    let client = build_pinned_client(host, &addresses, config)?;

    client
        .get(url.clone())
        .send()
        .await
        .map_err(DocumentSourceError::from)
}

fn map_status_error(error: reqwest::Error) -> DocumentSourceError {
    if let Some(status) = error.status() {
        DocumentSourceError::HttpStatus {
            status: status.as_u16(),
        }
    } else {
        DocumentSourceError::Http(error)
    }
}

async fn read_bounded_body(
    response: Response,
    max_download_bytes: usize,
) -> Result<Vec<u8>, DocumentSourceError> {
    if let Some(content_length) = response.content_length()
        && content_length > max_download_bytes as u64
    {
        return Err(DocumentSourceError::DownloadTooLarge {
            limit: max_download_bytes,
            observed: content_length as usize,
        });
    }

    let mut stream = response.bytes_stream();
    let mut collected = Vec::with_capacity(max_download_bytes.min(8192));

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(DocumentSourceError::from)?;
        let remaining = max_download_bytes.saturating_sub(collected.len());

        if chunk.len() > remaining {
            return Err(DocumentSourceError::DownloadTooLarge {
                limit: max_download_bytes,
                observed: collected.len() + chunk.len(),
            });
        }

        collected.extend_from_slice(&chunk);
    }

    Ok(collected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::{Method::GET, MockServer};
    use std::time::Duration;

    fn test_config(max_download_bytes: usize) -> DocumentParserConfig {
        DocumentParserConfig::default()
            .with_allow_private_networks(true)
            .with_allowed_hosts(vec!["127.0.0.1".to_string()])
            .expect("hosts")
            .with_max_download_bytes(max_download_bytes)
            .with_request_timeout(Duration::from_secs(5))
    }

    #[tokio::test]
    async fn fetch_returns_body_on_success() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/doc.txt");
            then.status(200).body("hello document");
        });

        let (bytes, filename) = fetch_url(
            &format!("{}/doc.txt", server.base_url()),
            &test_config(1024),
        )
        .await
        .expect("fetch succeeds");

        mock.assert();
        assert_eq!(bytes, b"hello document");
        assert_eq!(filename.as_deref(), Some("doc.txt"));
    }

    #[tokio::test]
    async fn fetch_rejects_http_error_status() {
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/missing");
            then.status(404).body("not found");
        });

        let error = fetch_url(
            &format!("{}/missing", server.base_url()),
            &test_config(1024),
        )
        .await
        .expect_err("404 should fail");

        assert!(matches!(
            error,
            DocumentSourceError::HttpStatus { status: 404 }
        ));
    }

    #[tokio::test]
    async fn fetch_rejects_oversized_content_length() {
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/huge");
            then.status(200)
                .header("Content-Length", "2048")
                .body(&"x".repeat(2048));
        });

        let error = fetch_url(&format!("{}/huge", server.base_url()), &test_config(1024))
            .await
            .expect_err("oversized body");

        assert!(matches!(
            error,
            DocumentSourceError::DownloadTooLarge { .. }
        ));
    }

    #[tokio::test]
    async fn fetch_rejects_oversized_chunked_body() {
        let server = MockServer::start();
        let body = "y".repeat(2048);
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/chunked");
            then.status(200)
                .header("Transfer-Encoding", "chunked")
                .body(&body);
        });

        let error = fetch_url(
            &format!("{}/chunked", server.base_url()),
            &test_config(1024),
        )
        .await
        .expect_err("chunked oversized body");

        assert!(matches!(
            error,
            DocumentSourceError::DownloadTooLarge { .. }
        ));
    }

    #[tokio::test]
    async fn fetch_blocks_redirect_to_disallowed_host() {
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/redirect");
            then.status(302)
                .header("Location", "http://example.com/private")
                .body("");
        });

        let config = test_config(1024);
        let error = fetch_url(&format!("{}/redirect", server.base_url()), &config)
            .await
            .expect_err("redirect to disallowed host");

        assert!(matches!(error, DocumentSourceError::HostNotAllowed(_)));
    }

    #[tokio::test]
    async fn fetch_follows_redirect_to_allowed_destination() {
        let server = MockServer::start();
        let redirect_mock = server.mock(|when, then| {
            when.method(GET).path("/redirect");
            then.status(302).header("Location", "/final.txt").body("");
        });
        let final_mock = server.mock(|when, then| {
            when.method(GET).path("/final.txt");
            then.status(200).body("redirected");
        });

        let config = test_config(1024);
        let (bytes, _) = fetch_url(&format!("{}/redirect", server.base_url()), &config)
            .await
            .expect("redirect succeeds");

        redirect_mock.assert();
        final_mock.assert();
        assert_eq!(bytes, b"redirected");
    }

    #[tokio::test]
    async fn fetch_rejects_too_many_redirects() {
        let server = MockServer::start();
        for index in 0..6 {
            let path = format!("/redirect-{index}");
            let next = format!("/redirect-{}", index + 1);
            let _mock = server.mock(move |when, then| {
                when.method(GET).path(&path);
                then.status(302).header("Location", &next).body("");
            });
        }

        let config = test_config(1024).with_max_redirects(5);
        let error = fetch_url(&format!("{}/redirect-0", server.base_url()), &config)
            .await
            .expect_err("redirect limit");

        assert!(matches!(
            error,
            DocumentSourceError::TooManyRedirects { limit: 5 }
        ));
    }
}
