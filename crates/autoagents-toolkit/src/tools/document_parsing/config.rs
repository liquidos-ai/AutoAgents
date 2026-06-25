use std::path::PathBuf;
use std::time::Duration;

use super::source::DocumentSourceError;

/// Default HTTP request timeout for document downloads.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Default maximum document download size (50 MiB).
pub const DEFAULT_MAX_DOWNLOAD_BYTES: usize = 52_428_800;

/// Default maximum local file read size (50 MiB).
pub const DEFAULT_MAX_LOCAL_FILE_BYTES: usize = 52_428_800;

/// Default maximum number of HTTP redirects to follow.
pub const DEFAULT_MAX_REDIRECTS: usize = 5;

/// Security and resource limits for [`super::parse_document::DocumentParser`].
#[derive(Debug, Clone)]
pub struct DocumentParserConfig {
    /// HTTP request timeout applied to each redirect hop.
    pub request_timeout: Duration,
    /// Maximum bytes read from a remote document.
    pub max_download_bytes: usize,
    /// Maximum bytes read from a local file.
    pub max_local_file_bytes: usize,
    /// Maximum redirect hops before the request is rejected.
    pub max_redirects: usize,
    /// Optional host allowlist. When set, only matching hosts may be fetched.
    pub allowed_hosts: Option<Vec<String>>,
    /// When set, only these TCP ports may be used for remote URLs.
    pub allowed_ports: Option<Vec<u16>>,
    /// When `false`, private, loopback, link-local, and metadata IP ranges are blocked.
    pub allow_private_networks: bool,
    /// Optional local filesystem roots. When set, local paths must resolve inside one of them.
    pub allowed_roots: Option<Vec<PathBuf>>,
}

impl Default for DocumentParserConfig {
    fn default() -> Self {
        Self {
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            max_download_bytes: DEFAULT_MAX_DOWNLOAD_BYTES,
            max_local_file_bytes: DEFAULT_MAX_LOCAL_FILE_BYTES,
            max_redirects: DEFAULT_MAX_REDIRECTS,
            allowed_hosts: None,
            allowed_ports: None,
            allow_private_networks: false,
            allowed_roots: None,
        }
    }
}

impl DocumentParserConfig {
    /// Validate security-related configuration.
    pub fn validate(&self) -> Result<(), DocumentSourceError> {
        if self.max_download_bytes == 0 {
            return Err(DocumentSourceError::InvalidConfig(
                "max_download_bytes must be greater than zero".into(),
            ));
        }

        if self.max_local_file_bytes == 0 {
            return Err(DocumentSourceError::InvalidConfig(
                "max_local_file_bytes must be greater than zero".into(),
            ));
        }

        if let Some(hosts) = &self.allowed_hosts
            && hosts.is_empty()
        {
            return Err(DocumentSourceError::InvalidConfig(
                "allowed_hosts must not be empty when set".into(),
            ));
        }

        if let Some(ports) = &self.allowed_ports
            && ports.is_empty()
        {
            return Err(DocumentSourceError::InvalidConfig(
                "allowed_ports must not be empty when set".into(),
            ));
        }

        if self.allow_private_networks
            && !matches!(&self.allowed_hosts, Some(hosts) if !hosts.is_empty())
        {
            return Err(DocumentSourceError::InvalidConfig(
                "allow_private_networks requires a non-empty allowed_hosts list".into(),
            ));
        }

        Ok(())
    }

    pub fn with_allowed_roots(mut self, roots: Vec<PathBuf>) -> Self {
        self.allowed_roots = Some(roots);
        self
    }

    pub fn with_allowed_hosts(mut self, hosts: Vec<String>) -> Result<Self, DocumentSourceError> {
        if hosts.is_empty() {
            return Err(DocumentSourceError::InvalidConfig(
                "allowed_hosts must not be empty".into(),
            ));
        }
        self.allowed_hosts = Some(hosts);
        Ok(self)
    }

    pub fn with_allowed_ports(mut self, ports: Vec<u16>) -> Result<Self, DocumentSourceError> {
        if ports.is_empty() {
            return Err(DocumentSourceError::InvalidConfig(
                "allowed_ports must not be empty".into(),
            ));
        }
        self.allowed_ports = Some(ports);
        Ok(self)
    }

    pub fn with_allow_private_networks(mut self, allow: bool) -> Self {
        self.allow_private_networks = allow;
        self
    }

    pub fn with_max_download_bytes(mut self, max_download_bytes: usize) -> Self {
        self.max_download_bytes = max_download_bytes;
        self
    }

    pub fn with_max_local_file_bytes(mut self, max_local_file_bytes: usize) -> Self {
        self.max_local_file_bytes = max_local_file_bytes;
        self
    }

    pub fn with_max_redirects(mut self, max_redirects: usize) -> Self {
        self.max_redirects = max_redirects;
        self
    }

    pub fn with_request_timeout(mut self, request_timeout: Duration) -> Self {
        self.request_timeout = request_timeout;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_blocks_private_networks() {
        let config = DocumentParserConfig::default();
        assert!(!config.allow_private_networks);
        assert_eq!(config.max_download_bytes, DEFAULT_MAX_DOWNLOAD_BYTES);
        assert_eq!(config.max_local_file_bytes, DEFAULT_MAX_LOCAL_FILE_BYTES);
        assert_eq!(config.request_timeout, DEFAULT_REQUEST_TIMEOUT);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn rejects_empty_allowed_hosts() {
        let result = DocumentParserConfig::default().with_allowed_hosts(vec![]);
        assert!(matches!(result, Err(DocumentSourceError::InvalidConfig(_))));
    }

    #[test]
    fn rejects_private_networks_without_allowlist() {
        let config = DocumentParserConfig::default().with_allow_private_networks(true);
        let error = config.validate().expect_err("must require allowlist");
        assert!(matches!(error, DocumentSourceError::InvalidConfig(_)));
    }

    #[test]
    fn allows_private_networks_with_allowlist() {
        let config = DocumentParserConfig::default()
            .with_allow_private_networks(true)
            .with_allowed_hosts(vec!["localhost".to_string()])
            .expect("hosts");
        assert!(config.validate().is_ok());
    }
}
