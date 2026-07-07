//! Mock OpenAI Responses API server for provider-level WASM smoke tests.
//!
//! This intentionally uses only `std::net` + `std::io` for HTTP request parsing
//! so the smoke test exercises the WASI HTTP client against raw socket I/O
//! without adding a test-server runtime dependency.

use serde_json::Value;
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

const HOST: &str = "127.0.0.1";
const PORT: u16 = 18765;
const RETRY_AFTER_SECONDS: &str = "5";

/// Parsed HTTP request components returned after reading the wire.
struct ParsedRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

fn main() -> io::Result<()> {
    let listener = TcpListener::bind((HOST, PORT))?;
    eprintln!("[MOCK] Starting mock server on {HOST}:{PORT}");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                thread::spawn(|| {
                    if let Err(err) = handle_client(stream) {
                        eprintln!("[MOCK] Error: {err}");
                    }
                });
            }
            Err(err) => eprintln!("[MOCK] accept error: {err}"),
        }
    }

    Ok(())
}

fn handle_client(mut stream: TcpStream) -> io::Result<()> {
    let mut data = Vec::new();
    let mut buf = [0_u8; 65_536];

    // Read and parse the HTTP request.
    let Some((method, path, req_json)) = read_request(&mut stream, &mut data, &mut buf)? else {
        return Ok(());
    };

    // Route and write the response.
    route_response(&mut stream, &method, &path, &req_json)
}

/// Reads raw bytes until the header terminator is found, then parses the request
/// line, headers, body, and JSON payload. Returns `None` when an error response
/// has already been written (e.g. bad request / invalid JSON).
fn read_request(
    stream: &mut TcpStream,
    data: &mut Vec<u8>,
    buf: &mut [u8; 65_536],
) -> io::Result<Option<(String, String, Value)>> {
    loop {
        let n = stream.read(buf)?;
        if n == 0 {
            return Ok(None);
        }
        data.extend_from_slice(&buf[..n]);

        let Some(header_end) = find_header_end(data) else {
            continue;
        };

        let Some(result) = parse_request(data, header_end, stream, buf)? else {
            return Ok(None);
        };

        let req_json = parse_body(stream, &result.body)?;
        let (method, path) = (result.method, result.path);

        eprintln!(
            "[MOCK] {method} {path} chunked={} len={}",
            result
                .headers
                .get("transfer-encoding")
                .map(|v| v.contains("chunked"))
                .unwrap_or(false),
            result.body.len()
        );

        if let Some(object) = req_json.as_object() {
            let keys = object.keys().cloned().collect::<Vec<_>>();
            eprintln!("[MOCK] parsed keys={keys:?}");
        }

        return Ok(Some((method, path, req_json)));
    }
}

/// Parses the request line and headers from the received data. Returns
/// `(method, path, headers, body_bytes)` or sends an error response and
/// returns `None` if the request line is missing.
fn parse_request(
    data: &[u8],
    header_end: usize,
    stream: &mut TcpStream,
    buf: &mut [u8; 65_536],
) -> io::Result<Option<ParsedRequest>> {
    let header_text = String::from_utf8_lossy(&data[..header_end]);
    let mut lines = header_text.trim().lines();
    let Some(request_line) = lines.next() else {
        send_response(
            stream,
            400,
            "application/json",
            b"{\"error\":\"bad request\"}",
            &[],
        )?;
        return Ok(None);
    };

    let mut request_parts = request_line.splitn(3, ' ');
    let method = request_parts.next().unwrap_or_default().to_string();
    let path = request_parts.next().unwrap_or_default().to_string();

    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    let body = extract_body(data, header_end, stream, &headers, buf)?;

    Ok(Some(ParsedRequest {
        method,
        path,
        headers,
        body,
    }))
}

/// Reads the HTTP body from the data buffer (and optionally the stream) based on
/// Transfer-Encoding or Content-Length.
fn extract_body(
    data: &[u8],
    header_end: usize,
    stream: &mut TcpStream,
    headers: &HashMap<String, String>,
    buf: &mut [u8; 65_536],
) -> io::Result<Vec<u8>> {
    let body_start = header_end + 4;
    let is_chunked = headers.get("transfer-encoding").is_some_and(|value| {
        value
            .split(',')
            .any(|part| part.trim().eq_ignore_ascii_case("chunked"))
    });
    let content_len = headers
        .get("content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);

    if is_chunked {
        let mut raw_body = data[body_start..].to_vec();
        while !chunked_body_complete(&raw_body) {
            let n = stream.read(buf)?;
            if n == 0 {
                break;
            }
            raw_body.extend_from_slice(&buf[..n]);
        }
        Ok(decode_chunked(&raw_body))
    } else if content_len > 0 {
        let mut body_data = data[body_start..].to_vec();
        while body_data.len() < content_len {
            let remaining = content_len - body_data.len();
            let read_len = remaining.min(buf.len());
            let n = stream.read(&mut buf[..read_len])?;
            if n == 0 {
                break;
            }
            body_data.extend_from_slice(&buf[..n]);
        }
        body_data.truncate(content_len);
        Ok(body_data)
    } else {
        Ok(data[body_start..].to_vec())
    }
}

/// Tries to parse the body bytes as JSON. Returns `Value::Object(Map::default())`
/// for empty/whitespace-only bodies.
fn parse_body(stream: &mut TcpStream, body: &[u8]) -> io::Result<Value> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Ok(Value::Object(serde_json::Map::<String, Value>::default()));
    }
    match serde_json::from_slice(body) {
        Ok(value) => Ok(value),
        Err(err) => {
            eprintln!("[MOCK] JSON error: {err}");
            send_response(
                stream,
                400,
                "application/json",
                b"{\"error\":\"invalid json\"}",
                &[],
            )?;
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid JSON body",
            ))
        }
    }
}

/// Routes the parsed request to the appropriate response handler.
fn route_response(
    stream: &mut TcpStream,
    method: &str,
    path: &str,
    req_json: &Value,
) -> io::Result<()> {
    if method == "POST" && path == "/v1/responses" {
        if req_json.get("test_error").and_then(Value::as_str) == Some("rate_limit")
            || contains_marker(req_json, "rate_limit")
        {
            send_response(
                stream,
                429,
                "application/json",
                b"{\"error\":\"rate limited\"}\n",
                &[("Retry-After", RETRY_AFTER_SECONDS)],
            )
        } else if req_json
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            send_response(
                stream,
                200,
                "text/event-stream",
                br#"data: {"type":"response.output_text.delta","delta":"hello "}

data: {"type":"response.reasoning_text.delta","delta":"think"}

data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":""}}

data: {"type":"response.function_call_arguments.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"value\"}"}}

data: {"type":"response.done","response":{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}

data: [DONE]

"#,
                &[("Cache-Control", "no-cache")],
            )
        } else {
            send_response(
                stream,
                200,
                "application/json",
                br#"{
  "output": [
    {
      "type": "reasoning",
      "summary": [
        {
          "text": "plan"
        }
      ]
    },
    {
      "type": "function_call",
      "name": "lookup",
      "arguments": "{\"q\":\"value\"}",
      "call_id": "call_1"
    },
    {
      "type": "message",
      "content": [
        {
          "type": "output_text",
          "text": "hello from mock"
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 2,
    "output_tokens": 3,
    "total_tokens": 5
  }
}
"#,
                &[],
            )
        }
    } else if method == "GET" && path == "/v1/models" {
        send_response(
            stream,
            200,
            "application/json",
            br#"{
  "data": [
    { "id": "gpt-4.1", "created": 1 },
    { "id": "gpt-4o-mini", "created": 2 }
  ]
}
"#,
            &[],
        )
    } else if method == "POST" && path == "/v1/embeddings" {
        send_response(
            stream,
            200,
            "application/json",
            br#"{
  "data": [
    { "embedding": [0.1, 0.2, 0.3] },
    { "embedding": [0.4, 0.5, 0.6] }
  ]
}
"#,
            &[],
        )
    } else {
        send_response(
            stream,
            404,
            "application/json",
            b"{\"error\":\"not found\"}",
            &[],
        )
    }
}

fn find_header_end(data: &[u8]) -> Option<usize> {
    data.windows(4).position(|window| window == b"\r\n\r\n")
}

fn chunked_body_complete(body: &[u8]) -> bool {
    let mut pos = 0;
    while pos < body.len() {
        let Some(size_end_rel) = body[pos..].windows(2).position(|w| w == b"\r\n") else {
            return false;
        };
        let size_end = pos + size_end_rel;
        let size_line = String::from_utf8_lossy(&body[pos..size_end]);
        let size_hex = size_line.split(';').next().unwrap_or_default().trim();
        let Ok(size) = usize::from_str_radix(size_hex, 16) else {
            return false;
        };
        pos = size_end + 2;

        if size == 0 {
            return body[pos..].windows(2).any(|w| w == b"\r\n") || pos == body.len();
        }

        if body.len() < pos + size + 2 {
            return false;
        }
        pos += size;
        if body.get(pos..pos + 2) != Some(b"\r\n") {
            return false;
        }
        pos += 2;
    }

    false
}

fn decode_chunked(body: &[u8]) -> Vec<u8> {
    let mut decoded = Vec::new();
    let mut pos = 0;

    while pos < body.len() {
        let Some(size_end_rel) = body[pos..].windows(2).position(|w| w == b"\r\n") else {
            break;
        };
        let size_end = pos + size_end_rel;
        let size_line = String::from_utf8_lossy(&body[pos..size_end]);
        let size_hex = size_line.split(';').next().unwrap_or_default().trim();
        let Ok(size) = usize::from_str_radix(size_hex, 16) else {
            break;
        };
        pos = size_end + 2;

        if size == 0 {
            break;
        }
        if body.len() < pos + size {
            break;
        }
        decoded.extend_from_slice(&body[pos..pos + size]);
        pos += size;
        if body.get(pos..pos + 2) == Some(b"\r\n") {
            pos += 2;
        }
    }

    decoded
}

fn contains_marker(value: &Value, marker: &str) -> bool {
    match value {
        Value::String(text) => text.contains(marker),
        Value::Array(items) => items.iter().any(|item| contains_marker(item, marker)),
        Value::Object(map) => map.values().any(|item| contains_marker(item, marker)),
        _ => false,
    }
}

fn send_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
    extra_headers: &[(&str, &str)],
) -> io::Result<()> {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        429 => "Too Many Requests",
        _ => "Unknown",
    };

    write!(
        stream,
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: {content_type}\r\nConnection: close\r\n"
    )?;
    for (name, value) in extra_headers {
        write!(stream, "{name}: {value}\r\n")?;
    }
    write!(stream, "Content-Length: {}\r\n\r\n", body.len())?;
    stream.write_all(body)?;
    stream.flush()
}
