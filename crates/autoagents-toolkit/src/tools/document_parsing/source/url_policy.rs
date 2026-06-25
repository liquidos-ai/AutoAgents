use std::net::{IpAddr, SocketAddr};
use std::str::FromStr;
use std::sync::LazyLock;

use ipnet::{Ipv4Net, Ipv6Net};
use url::{Host, Url};

use super::error::DocumentSourceError;
use crate::tools::document_parsing::config::DocumentParserConfig;

const BLOCKED_IPV4_NETS_RAW: &[&str] = &[
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.0.2.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
    "255.255.255.255/32",
];

const BLOCKED_IPV6_NETS_RAW: &[&str] = &[
    "::/128",
    "::1/128",
    "100::/64",
    "2001:db8::/32",
    "fc00::/7",
    "fe80::/10",
    "ff00::/8",
];

const BLOCKED_HOSTNAMES: &[&str] = &["localhost", "metadata.google.internal", "metadata.google"];

const BLOCKED_HOSTNAME_SUFFIXES: &[&str] = &[".localhost", ".local", ".internal", ".corp"];

static BLOCKED_IPV4_NETS: LazyLock<Vec<Ipv4Net>> = LazyLock::new(|| {
    BLOCKED_IPV4_NETS_RAW
        .iter()
        .map(|net| Ipv4Net::from_str(net).expect("valid IPv4 net"))
        .collect()
});

static BLOCKED_IPV6_NETS: LazyLock<Vec<Ipv6Net>> = LazyLock::new(|| {
    BLOCKED_IPV6_NETS_RAW
        .iter()
        .map(|net| Ipv6Net::from_str(net).expect("valid IPv6 net"))
        .collect()
});

fn is_blocked_ip(addr: IpAddr, allow_private_networks: bool) -> bool {
    if allow_private_networks {
        return false;
    }

    match addr {
        IpAddr::V4(ipv4) => BLOCKED_IPV4_NETS.iter().any(|net| net.contains(&ipv4)),
        IpAddr::V6(ipv6) => {
            if ipv4_from_ipv6_mapped(ipv6)
                .is_some_and(|ipv4| BLOCKED_IPV4_NETS.iter().any(|net| net.contains(&ipv4)))
            {
                return true;
            }

            BLOCKED_IPV6_NETS.iter().any(|net| net.contains(&ipv6))
        }
    }
}

fn ipv4_from_ipv6_mapped(addr: std::net::Ipv6Addr) -> Option<std::net::Ipv4Addr> {
    let octets = addr.octets();
    if octets[0..10] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] && octets[10] == 0xff && octets[11] == 0xff {
        Some(std::net::Ipv4Addr::new(
            octets[12], octets[13], octets[14], octets[15],
        ))
    } else {
        None
    }
}

fn host_matches_allowlist(host: &str, allowed_hosts: &[String]) -> bool {
    if host_to_ip(host).is_some() {
        return host_matches_allowlist_exact(host, allowed_hosts);
    }

    let host = host.trim_end_matches('.').to_ascii_lowercase();

    allowed_hosts.iter().any(|allowed| {
        let allowed = allowed.trim().trim_end_matches('.').to_ascii_lowercase();
        if allowed.is_empty() {
            return false;
        }

        host == allowed
            || host.ends_with(&format!(".{allowed}"))
            || (allowed.starts_with('.') && host.ends_with(&allowed))
    })
}

fn host_matches_allowlist_exact(host: &str, allowed_hosts: &[String]) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    allowed_hosts.iter().any(|allowed| {
        let allowed = allowed.trim().trim_end_matches('.').to_ascii_lowercase();
        !allowed.is_empty() && host == allowed
    })
}

fn is_blocked_hostname(host: &str, allow_private_networks: bool) -> bool {
    if allow_private_networks {
        return false;
    }

    let host = host.trim_end_matches('.').to_ascii_lowercase();

    if BLOCKED_HOSTNAMES.iter().any(|blocked| *blocked == host) {
        return true;
    }

    BLOCKED_HOSTNAME_SUFFIXES
        .iter()
        .any(|suffix| host.ends_with(suffix))
}

fn parse_ip_octet(part: &str) -> Option<u8> {
    let part = part.trim();
    if part.is_empty() {
        return None;
    }

    if let Some(hex) = part.strip_prefix("0x").or_else(|| part.strip_prefix("0X")) {
        return u8::from_str_radix(hex, 16).ok();
    }

    if part.len() > 1 && part.starts_with('0') && part.chars().all(|ch| ch.is_ascii_digit()) {
        return u8::from_str_radix(part, 8).ok();
    }

    part.parse().ok()
}

fn parse_dotted_ipv4(host: &str) -> Option<IpAddr> {
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() != 4 {
        return None;
    }

    let mut octets = [0u8; 4];
    for (index, part) in parts.iter().enumerate() {
        octets[index] = parse_ip_octet(part)?;
    }

    Some(IpAddr::V4(std::net::Ipv4Addr::from(octets)))
}

fn host_to_ip(host: &str) -> Option<IpAddr> {
    if let Ok(addr) = host.parse::<IpAddr>() {
        return Some(addr);
    }

    let host = host.trim();

    if host.chars().all(|ch| ch.is_ascii_digit()) {
        let value: u32 = host.parse().ok()?;
        return Some(IpAddr::V4(std::net::Ipv4Addr::from(value.to_be_bytes())));
    }

    if let Some(hex) = host.strip_prefix("0x").or_else(|| host.strip_prefix("0X")) {
        let value = u32::from_str_radix(hex, 16).ok()?;
        return Some(IpAddr::V4(std::net::Ipv4Addr::from(value.to_be_bytes())));
    }

    parse_dotted_ipv4(host)
}

fn validate_port(url: &Url, config: &DocumentParserConfig) -> Result<(), DocumentSourceError> {
    let Some(allowed_ports) = &config.allowed_ports else {
        return Ok(());
    };

    let port = url.port_or_known_default().unwrap_or(80);
    if allowed_ports.contains(&port) {
        Ok(())
    } else {
        Err(DocumentSourceError::PortNotAllowed { port })
    }
}

fn validate_host_name(
    host: &str,
    config: &DocumentParserConfig,
) -> Result<(), DocumentSourceError> {
    if is_blocked_hostname(host, config.allow_private_networks) {
        return Err(DocumentSourceError::BlockedHostname(host.to_string()));
    }

    if let Some(allowed_hosts) = &config.allowed_hosts
        && !host_matches_allowlist(host, allowed_hosts)
    {
        return Err(DocumentSourceError::HostNotAllowed(host.to_string()));
    }

    if let Some(addr) = host_to_ip(host)
        && is_blocked_ip(addr, config.allow_private_networks)
    {
        return Err(DocumentSourceError::PrivateNetworkBlocked {
            host: host.to_string(),
            addr,
        });
    }

    Ok(())
}

fn validate_ip_literal(
    addr: IpAddr,
    display_host: &str,
    config: &DocumentParserConfig,
) -> Result<(), DocumentSourceError> {
    if is_blocked_ip(addr, config.allow_private_networks) {
        return Err(DocumentSourceError::PrivateNetworkBlocked {
            host: display_host.to_string(),
            addr,
        });
    }

    if let Some(allowed_hosts) = &config.allowed_hosts
        && !host_matches_allowlist_exact(display_host, allowed_hosts)
    {
        return Err(DocumentSourceError::HostNotAllowed(
            display_host.to_string(),
        ));
    }

    Ok(())
}

pub fn validate_url_str(
    url: &str,
    config: &DocumentParserConfig,
) -> Result<Url, DocumentSourceError> {
    let parsed = Url::parse(url)
        .map_err(|error| DocumentSourceError::InvalidHost(format!("{url}: {error}")))?;

    validate_url(&parsed, config)?;
    Ok(parsed)
}

pub fn validate_url(url: &Url, config: &DocumentParserConfig) -> Result<(), DocumentSourceError> {
    match url.scheme() {
        "http" | "https" => {}
        other => {
            return Err(DocumentSourceError::UrlSchemeNotAllowed(other.to_string()));
        }
    }

    if !url.username().is_empty() || url.password().is_some() {
        return Err(DocumentSourceError::CredentialsInUrl);
    }

    validate_port(url, config)?;

    let host = url
        .host()
        .ok_or_else(|| DocumentSourceError::InvalidHost(url.to_string()))?;

    match host {
        Host::Domain(domain) => validate_host_name(domain, config)?,
        Host::Ipv4(addr) => validate_ip_literal(IpAddr::V4(addr), &addr.to_string(), config)?,
        Host::Ipv6(addr) => validate_ip_literal(IpAddr::V6(addr), &addr.to_string(), config)?,
    }

    Ok(())
}

pub fn validate_resolved_addresses(
    host: &str,
    addresses: &[SocketAddr],
    config: &DocumentParserConfig,
) -> Result<(), DocumentSourceError> {
    if addresses.is_empty() {
        return Err(DocumentSourceError::NoResolvedAddresses(host.to_string()));
    }

    for address in addresses {
        if is_blocked_ip(address.ip(), config.allow_private_networks) {
            return Err(DocumentSourceError::PrivateNetworkBlocked {
                host: host.to_string(),
                addr: address.ip(),
            });
        }
    }

    Ok(())
}

pub async fn resolve_host(
    host: &str,
    port: u16,
    config: &DocumentParserConfig,
) -> Result<Vec<SocketAddr>, DocumentSourceError> {
    validate_host_name(host, config)?;

    if let Some(addr) = host_to_ip(host) {
        let socket = SocketAddr::new(addr, port);
        validate_resolved_addresses(host, std::slice::from_ref(&socket), config)?;
        return Ok(vec![socket]);
    }

    let lookup_target = format!("{host}:{port}");
    let mut addresses = tokio::net::lookup_host(&lookup_target)
        .await
        .map_err(|source| DocumentSourceError::DnsResolutionFailed {
            host: host.to_string(),
            source,
        })?
        .collect::<Vec<_>>();

    if addresses.is_empty() {
        return Err(DocumentSourceError::NoResolvedAddresses(host.to_string()));
    }

    addresses.sort();
    addresses.dedup();

    validate_resolved_addresses(host, &addresses, config)?;
    Ok(addresses)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DocumentParserConfig {
        DocumentParserConfig::default()
    }

    #[test]
    fn blocks_metadata_ip_literal() {
        let config = default_config();
        let error = validate_url_str("http://169.254.169.254/latest/meta-data/", &config)
            .expect_err("metadata IP should be blocked");
        assert!(matches!(
            error,
            DocumentSourceError::PrivateNetworkBlocked { .. }
        ));
    }

    #[test]
    fn blocks_loopback_ip() {
        let config = default_config();
        let error = validate_url_str("http://127.0.0.1/secret", &config).expect_err("loopback");
        assert!(matches!(
            error,
            DocumentSourceError::PrivateNetworkBlocked { .. }
        ));
    }

    #[test]
    fn blocks_decimal_ip_notation() {
        let config = default_config();
        let error = validate_url_str("http://2130706433/", &config).expect_err("decimal IP");
        assert!(matches!(
            error,
            DocumentSourceError::PrivateNetworkBlocked { .. }
        ));
    }

    #[test]
    fn blocks_octal_dotted_ip_notation() {
        let config = default_config();
        let error = validate_url_str("http://0177.0.0.1/secret", &config).expect_err("octal IP");
        assert!(matches!(
            error,
            DocumentSourceError::PrivateNetworkBlocked { .. }
        ));
    }

    #[test]
    fn blocks_localhost_hostname() {
        let config = default_config();
        let error = validate_url_str("http://localhost/secret", &config).expect_err("localhost");
        assert!(matches!(error, DocumentSourceError::BlockedHostname(_)));
    }

    #[test]
    fn blocks_internal_hostname_suffix() {
        let config = default_config();
        let error = validate_url_str("http://service.internal/doc", &config).expect_err("internal");
        assert!(matches!(error, DocumentSourceError::BlockedHostname(_)));
    }

    #[test]
    fn blocks_credentials_in_url() {
        let config = default_config();
        let error = validate_url_str("http://user:pass@example.com/doc.pdf", &config)
            .expect_err("credentials");
        assert!(matches!(error, DocumentSourceError::CredentialsInUrl));
    }

    #[test]
    fn blocks_non_http_scheme() {
        let config = default_config();
        let error = validate_url_str("file:///etc/passwd", &config).expect_err("file scheme");
        assert!(matches!(error, DocumentSourceError::UrlSchemeNotAllowed(_)));
    }

    #[test]
    fn blocks_disallowed_port() {
        let config = DocumentParserConfig::default()
            .with_allowed_ports(vec![80, 443])
            .expect("ports");
        let error =
            validate_url_str("http://example.com:8080/doc.pdf", &config).expect_err("port blocked");
        assert!(matches!(
            error,
            DocumentSourceError::PortNotAllowed { port: 8080 }
        ));
    }

    #[test]
    fn allows_public_url() {
        let config = default_config();
        validate_url_str("https://example.com/file.pdf", &config).expect("public URL allowed");
    }

    #[test]
    fn host_allowlist_rejects_unlisted_host() {
        let config = DocumentParserConfig::default()
            .with_allowed_hosts(vec!["example.com".to_string()])
            .expect("hosts");
        let error = validate_url_str("https://evil.com/file.pdf", &config).expect_err("allowlist");
        assert!(matches!(error, DocumentSourceError::HostNotAllowed(_)));
    }

    #[test]
    fn host_allowlist_accepts_subdomain() {
        let config = DocumentParserConfig::default()
            .with_allowed_hosts(vec!["example.com".to_string()])
            .expect("hosts");
        validate_url_str("https://cdn.example.com/file.pdf", &config)
            .expect("subdomain should match");
    }

    #[test]
    fn resolved_addresses_must_all_be_public() {
        let config = default_config();
        let addresses = vec![
            SocketAddr::new(IpAddr::V4(std::net::Ipv4Addr::new(8, 8, 8, 8)), 443),
            SocketAddr::new(IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 1)), 443),
        ];

        let error = validate_resolved_addresses("example.com", &addresses, &config)
            .expect_err("mixed DNS answers");
        assert!(matches!(
            error,
            DocumentSourceError::PrivateNetworkBlocked { .. }
        ));
    }

    #[test]
    fn host_allowlist_requires_exact_match_for_ip_literals() {
        let config = DocumentParserConfig::default()
            .with_allowed_hosts(vec!["8.8".to_string()])
            .expect("hosts");
        let error = validate_url_str("http://8.8.8.8/doc.pdf", &config).expect_err("suffix match");
        assert!(matches!(error, DocumentSourceError::HostNotAllowed(_)));
    }

    #[test]
    fn private_networks_allowed_when_configured() {
        let config = DocumentParserConfig::default()
            .with_allow_private_networks(true)
            .with_allowed_hosts(vec!["127.0.0.1".to_string(), "localhost".to_string()])
            .expect("hosts");
        validate_url_str("http://127.0.0.1/secret", &config).expect("loopback IP allowed");
        validate_url_str("http://localhost/secret", &config).expect("localhost allowed");
    }
}
