//! MCP (Model Context Protocol) support for AutoAgents
//!
//! This module provides integration with MCP servers, allowing AutoAgents
//! to use tools from external MCP-compatible servers.

pub mod adapter;
pub mod client;
pub mod config;
pub mod tools;

pub use adapter::{McpToolAdapter, McpToolWrapper};
pub use client::{
    McpError, McpProcessPolicy, McpServerConnection, McpServerInstructions, McpServerStatus,
    McpToolsManager,
};
pub use config::{
    Config, McpConfig, McpLocalServerConfig, McpRemoteServerConfig, McpServerConfig,
    McpServerTransport,
};
pub use tools::McpTools;
