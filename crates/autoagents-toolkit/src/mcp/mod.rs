//! MCP (Model Context Protocol) support for AutoAgents
//!
//! This module provides integration with MCP servers, allowing AutoAgents
//! to use tools from external MCP-compatible servers.
//!
//! # Trust model
//!
//! MCP configuration — especially stdio transport — defines which host processes are
//! spawned. Treat all MCP config files as **trusted code execution boundaries**. Only
//! load configs from sources you trust. Use [`McpSecurityPolicy::secure_default`] in
//! production; stdio commands are restricted to a launcher allowlist unless you supply
//! an explicit approval hook or opt into [`McpSecurityPolicy::permissive`] for local dev.

pub mod adapter;
pub mod client;
pub mod config;
pub mod policy;
pub mod security;
pub mod tools;

pub use adapter::{McpToolAdapter, McpToolWrapper};
pub use client::{McpError, McpServerConnection, McpToolsManager};
pub use config::{Config, McpConfig, McpServerConfig};
pub use policy::{McpProcessApprover, McpProcessLaunchSpec, McpSecurityPolicy};
pub use security::{DEFAULT_STDIO_ALLOWED_COMMANDS, McpSecurityError};
pub use tools::McpTools;
