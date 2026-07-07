use serde::Serialize;
use serde_json::Value;

const REDACTED: &str = "[redacted request payload]";

pub(crate) fn summarize_json_request<T: Serialize>(
    provider: &str,
    operation: &str,
    body: &T,
) -> String {
    match serde_json::to_value(body) {
        Ok(value) => summarize_value(provider, operation, &value),
        Err(_) => format!("{provider} {operation}: {REDACTED}; serialization=failed"),
    }
}

pub(crate) fn summarize_value(provider: &str, operation: &str, value: &Value) -> String {
    let model = get_string(value, &["model"])
        .or_else(|| get_string(value, &["requested_model"]))
        .unwrap_or("<unknown>");
    let stream = get_bool(value, &["stream"]);
    let message_count =
        count_first_array(value, &["messages", "input", "contents", "message_history"])
            .unwrap_or(0);
    let tool_count = count_tools(value);
    let schema = has_schema(value);
    let extra_keys = extra_top_level_keys(value);

    let mut parts = vec![
        format!("model={model}"),
        format!("messages={message_count}"),
        format!("tools={tool_count}"),
        format!("schema={schema}"),
    ];

    if let Some(stream) = stream {
        parts.push(format!("stream={stream}"));
    }

    if !extra_keys.is_empty() {
        parts.push(format!("extra_keys={}", extra_keys.join(",")));
    }

    format!("{provider} {operation}: {REDACTED}; {}", parts.join(" "))
}

fn get_string<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_str()
}

fn get_bool(value: &Value, path: &[&str]) -> Option<bool> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_bool()
}

fn count_first_array(value: &Value, keys: &[&str]) -> Option<usize> {
    keys.iter()
        .filter_map(|key| value.get(*key).and_then(Value::as_array))
        .map(Vec::len)
        .next()
}

fn count_tools(value: &Value) -> usize {
    let Some(tools) = value.get("tools").and_then(Value::as_array) else {
        return 0;
    };

    let declarations = tools
        .iter()
        .filter_map(|tool| tool.get("functionDeclarations").and_then(Value::as_array))
        .map(Vec::len)
        .sum::<usize>();

    if declarations > 0 {
        declarations
    } else {
        tools.len()
    }
}

fn has_schema(value: &Value) -> bool {
    has_key_recursive(
        value,
        &[
            "response_format",
            "response_schema",
            "output_config",
            "json_schema",
        ],
    )
}

fn has_key_recursive(value: &Value, keys: &[&str]) -> bool {
    match value {
        Value::Object(map) => map
            .iter()
            .any(|(key, value)| keys.contains(&key.as_str()) || has_key_recursive(value, keys)),
        Value::Array(values) => values.iter().any(|value| has_key_recursive(value, keys)),
        _ => false,
    }
}

fn extra_top_level_keys(value: &Value) -> Vec<String> {
    let Some(map) = value.as_object() else {
        return Vec::new();
    };

    let known = [
        "additional_extension_context",
        "allow_magic_buttons",
        "contents",
        "format",
        "generationConfig",
        "input",
        "is_vscode_extension",
        "keep_alive",
        "max_completion_tokens",
        "max_tokens",
        "max_output_tokens",
        "message_history",
        "messages",
        "model",
        "options",
        "output_config",
        "parallel_tool_calls",
        "requested_model",
        "reasoning_effort",
        "response_format",
        "search_parameters",
        "stream",
        "stream_options",
        "system",
        "temperature",
        "think",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
        "user_input",
    ];

    let mut keys = map
        .keys()
        .filter(|key| !known.contains(&key.as_str()) && !is_sensitive_key(key))
        .cloned()
        .collect::<Vec<_>>();
    keys.sort();
    keys
}

fn is_sensitive_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    [
        "api_key",
        "auth",
        "authorization",
        "content",
        "input",
        "key",
        "message",
        "output",
        "password",
        "prompt",
        "secret",
        "token",
    ]
    .iter()
    .any(|needle| key.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_summary_excludes_sensitive_values() {
        let body = json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "my password is hunter2"}
            ],
            "tools": [
                {"function": {"name": "lookup", "parameters": {"type": "object"}}}
            ],
            "response_format": {"type": "json_schema", "json_schema": {"name": "Secret"}},
            "seed": 7,
            "api_key": "sk-secret",
            "prompt_cache_key": "secret-cache"
        });

        let summary = summarize_value("Test", "request", &body);

        assert!(summary.contains("[redacted request payload]"));
        assert!(summary.contains("model=test-model"));
        assert!(summary.contains("messages=1"));
        assert!(summary.contains("tools=1"));
        assert!(summary.contains("schema=true"));
        assert!(summary.contains("extra_keys=seed"));
        assert!(!summary.contains("hunter2"));
        assert!(!summary.contains("sk-secret"));
        assert!(!summary.contains("prompt_cache_key"));
        assert!(!summary.contains("\"messages\""));
    }

    #[test]
    fn google_tool_count_uses_function_declarations() {
        let body = json!({
            "contents": [{"role": "user", "parts": [{"text": "secret"}]}],
            "tools": [{
                "functionDeclarations": [
                    {"name": "one"},
                    {"name": "two"}
                ]
            }],
            "generationConfig": {
                "response_schema": {"type": "object"}
            }
        });

        let summary = summarize_value("Google", "request", &body);

        assert!(summary.contains("messages=1"));
        assert!(summary.contains("tools=2"));
        assert!(summary.contains("schema=true"));
        assert!(!summary.contains("secret"));
    }
}
