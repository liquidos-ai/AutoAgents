use std::net::IpAddr;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DocumentSourceError {
    #[error("unsupported URL scheme: {0}")]
    UrlSchemeNotAllowed(String),

    #[error("credentials are not allowed in document URLs")]
    CredentialsInUrl,

    #[error("host is not allowed: {0}")]
    HostNotAllowed(String),

    #[error("port {port} is not allowed")]
    PortNotAllowed { port: u16 },

    #[error("blocked hostname: {0}")]
    BlockedHostname(String),

    #[error("invalid document parser configuration: {0}")]
    InvalidConfig(String),

    #[error(
        "local file exceeds maximum size of {limit} bytes (observed at least {observed} bytes)"
    )]
    LocalFileTooLarge { limit: usize, observed: usize },

    #[error("invalid URL host: {0}")]
    InvalidHost(String),

    #[error("private or restricted network address blocked for host {host}: {addr}")]
    PrivateNetworkBlocked { host: String, addr: IpAddr },

    #[error("failed to resolve host {host}: {source}")]
    DnsResolutionFailed {
        host: String,
        #[source]
        source: std::io::Error,
    },

    #[error("no addresses resolved for host: {0}")]
    NoResolvedAddresses(String),

    #[error("too many redirects while fetching document (limit: {limit})")]
    TooManyRedirects { limit: usize },

    #[error("redirect response missing Location header")]
    MissingRedirectLocation,

    #[error("invalid redirect Location header: {0}")]
    InvalidRedirectLocation(String),

    #[error(
        "document download exceeds maximum size of {limit} bytes (observed at least {observed} bytes)"
    )]
    DownloadTooLarge { limit: usize, observed: usize },

    #[error("HTTP request failed with status {status}")]
    HttpStatus { status: u16 },

    #[error("path {path} is outside of allowed roots")]
    PathOutsideRoot { path: PathBuf },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Http(#[from] reqwest::Error),
}
