#!/usr/bin/env python3
"""Minimal stdio MCP server exposing an echo tool (stdlib only)."""

from __future__ import annotations

import json
import sys
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "autoagents-echo-mcp"
SERVER_VERSION = "1.0.0"

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the input message back",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo",
                }
            },
            "required": ["message"],
        },
    }
]


def log_error(message: str) -> None:
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def ok_response(request_id: Any, result: dict[str, Any]) -> None:
    send({"jsonrpc": "2.0", "id": request_id, "result": result})


def handle_initialize(request_id: Any, params: dict[str, Any]) -> None:
    _ = params
    ok_response(
        request_id,
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        },
    )


def handle_tools_list(request_id: Any) -> None:
    ok_response(request_id, {"tools": TOOLS})


def handle_tools_call(request_id: Any, params: dict[str, Any]) -> None:
    name = params.get("name")
    arguments = params.get("arguments") or {}

    if name != "echo":
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": f"Unknown tool: {name}"},
            }
        )
        return

    message = arguments.get("message", "")
    ok_response(
        request_id,
        {
            "content": [{"type": "text", "text": str(message)}],
            "isError": False,
        },
    )


def handle_message(message: dict[str, Any]) -> None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        handle_initialize(request_id, params)
        return

    if method == "notifications/initialized":
        return

    if method == "tools/list":
        handle_tools_list(request_id)
        return

    if method == "tools/call":
        handle_tools_call(request_id, params)
        return

    if request_id is not None:
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        )


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            log_error(f"invalid JSON-RPC message: {exc}")
            continue
        if isinstance(message, dict):
            handle_message(message)


if __name__ == "__main__":
    main()
