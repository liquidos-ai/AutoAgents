#![cfg(not(target_arch = "wasm32"))]

use autoagents_toolkit::mcp::{McpConfig, McpServerConfig, McpToolsManager};
use std::path::PathBuf;
use std::process::Command;

fn python3_available() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

fn echo_server_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/mcp/servers/echo_server.py")
}

#[tokio::test]
async fn echo_stdio_server_lists_tools() {
    if !python3_available() {
        eprintln!("skipping MCP integration test: python3 not available");
        return;
    }

    let echo_server = echo_server_path();
    assert!(
        echo_server.is_file(),
        "expected echo server at {}",
        echo_server.display()
    );

    let mut config = McpConfig::new();
    config.add_server(
        McpServerConfig::new(
            "echo".to_string(),
            "stdio".to_string(),
            "python3".to_string(),
        )
        .with_args(vec![echo_server.to_string_lossy().to_string()])
        .with_timeout(30),
    );

    let manager = McpToolsManager::new();
    manager
        .connect_servers(&config)
        .await
        .expect("connect echo");

    assert!(manager.is_server_connected("echo").await);
    assert_eq!(manager.tool_count().await, 1);
    assert_eq!(manager.tool_names().await, vec!["echo".to_string()]);
}
