use crate::{
    config::{
        LlamaCppChatContinuation, LlamaCppConfig, LlamaCppReasoningFormat, LlamaCppToolChoice,
    },
    error::LlamaCppProviderError,
    server_chat::{
        ServerChatMessage, ServerTemplateCaps, apply_server_workarounds, messages_from_autoagents,
        render_messages_to_json,
    },
};
use autoagents_llm::chat::{ChatMessage, StructuredOutputFormat, Tool};
use chrono::Local;
use minijinja::State;
use minijinja::value::{Kwargs, ValueKind};
use minijinja::{Environment, Error as JinjaError, ErrorKind, Template, Value as JinjaValue};
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::HashSet;

const CHATML_TEMPLATE: &str = r#"{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'user' -%}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' -%}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'tool' -%}
<|im_start|>tool
{{ message['content'] }}<|im_end|>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{%- endif -%}"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GrammarTrigger {
    Word(String),
    Pattern(String),
    #[allow(dead_code)]
    PatternFull(String),
    #[allow(dead_code)]
    Token(i32),
}

#[derive(Debug, Clone)]
pub(crate) struct RenderedChat {
    pub(crate) prompt: String,
    pub(crate) generation_prompt: String,
    pub(crate) force_pure_content: bool,
    pub(crate) is_continuation: bool,
    pub(crate) add_bos: bool,
    pub(crate) grammar: Option<String>,
    pub(crate) grammar_lazy: bool,
    pub(crate) grammar_triggers: Vec<GrammarTrigger>,
    pub(crate) preserved_tokens: Vec<String>,
    pub(crate) additional_stops: Vec<String>,
    pub(crate) parse_tool_calls: bool,
    pub(crate) tool_names: Vec<String>,
    pub(crate) reasoning_format: Option<LlamaCppReasoningFormat>,
    pub(crate) reasoning_start_tag: Option<String>,
    pub(crate) reasoning_end_tag: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct TemplateSource {
    pub(crate) source: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct TemplateTokens {
    pub(crate) bos_token: String,
    pub(crate) eos_token: String,
    pub(crate) add_bos: Option<bool>,
    pub(crate) add_eos: Option<bool>,
}

pub(crate) fn explicit_template_source(template: &str) -> TemplateSource {
    TemplateSource {
        source: normalize_template_source(template),
    }
}

pub(crate) fn normalize_template_source(template: &str) -> String {
    match template {
        "chatml" | "chat_ml" => CHATML_TEMPLATE.to_string(),
        _ => template.to_string(),
    }
}

fn strip_hf_generation_blocks(source: &str) -> String {
    let mut output = String::with_capacity(source.len());
    let mut rest = source;

    while let Some(start) = rest.find("{%") {
        let (before, after_start) = rest.split_at(start);
        output.push_str(before);

        let Some(relative_end) = after_start.find("%}") else {
            output.push_str(after_start);
            return output;
        };

        let tag = &after_start[..relative_end + 2];
        if is_hf_generation_tag(tag) {
            rest = &after_start[relative_end + 2..];
        } else {
            output.push_str(tag);
            rest = &after_start[relative_end + 2..];
        }
    }

    output.push_str(rest);
    output
}

fn is_hf_generation_tag(tag: &str) -> bool {
    let inner = tag
        .trim_start_matches("{%")
        .trim_end_matches("%}")
        .trim()
        .trim_matches('-')
        .trim();
    matches!(inner, "generation" | "endgeneration")
}

fn resolve_continuation(
    config: &LlamaCppConfig,
    _messages: &[ServerChatMessage],
) -> Result<Option<LlamaCppChatContinuation>, LlamaCppProviderError> {
    match config.continue_final_message {
        LlamaCppChatContinuation::None => Ok(None),
        LlamaCppChatContinuation::Auto | LlamaCppChatContinuation::Content => {
            Ok(Some(LlamaCppChatContinuation::Content))
        }
        LlamaCppChatContinuation::Reasoning => Ok(Some(LlamaCppChatContinuation::Reasoning)),
    }
}

pub(crate) fn render_chat_template(
    config: &LlamaCppConfig,
    template: &TemplateSource,
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    json_schema: Option<&StructuredOutputFormat>,
    grammar: Option<String>,
    tokens: &TemplateTokens,
) -> Result<RenderedChat, LlamaCppProviderError> {
    let caps = TemplateCapabilities::analyze(&template.source);
    let server_messages = messages_from_autoagents(config, messages)?;
    let continuation = resolve_continuation(config, &server_messages)?;
    let (render_messages, continuation_messages) =
        if continuation.is_some() && !server_messages.is_empty() {
            server_messages.split_at(server_messages.len() - 1)
        } else {
            (server_messages.as_slice(), [].as_slice())
        };
    let mut prepared_messages = render_messages_to_json(render_messages, caps.server_caps);
    apply_server_workarounds(&mut prepared_messages, caps.server_caps)?;
    let mut continuation_prepared_messages =
        render_messages_to_json(&server_messages, caps.server_caps);
    apply_server_workarounds(&mut continuation_prepared_messages, caps.server_caps)?;
    let tools_value = tools
        .filter(|tools| !tools.is_empty())
        .map(serde_json::to_value)
        .transpose()
        .map_err(|err| LlamaCppProviderError::Template(format!("Failed to encode tools: {err}")))?
        .unwrap_or_else(|| Value::Array(Vec::default()));
    let kwargs = chat_template_kwargs(config)?;
    let enable_thinking = resolve_enable_thinking(config, &kwargs)?;

    let mut env = Environment::new();
    env.add_filter("tojson", tojson_filter);
    env.add_function("raise_exception", raise_exception);
    env.add_function("strftime_now", strftime_now);
    env.set_unknown_method_callback(pycompat_method_callback);
    let render_source = strip_hf_generation_blocks(&template.source);
    env.add_template("chat", &render_source).map_err(|err| {
        LlamaCppProviderError::Template(format!("Invalid llama.cpp chat template: {err}"))
    })?;
    let tmpl = env
        .get_template("chat")
        .map_err(|err| LlamaCppProviderError::Template(format!("Chat template missing: {err}")))?;

    let prompt_enable_thinking = if config.force_pure_content {
        false
    } else {
        enable_thinking.unwrap_or(true)
    };
    let root_context = build_template_context(
        config,
        prepared_messages,
        tools_value,
        kwargs,
        tokens,
        prompt_enable_thinking,
    );

    let mut prompt_context = root_context.clone();
    set_add_generation_prompt(
        &mut prompt_context,
        config.add_generation_prompt && continuation.is_none(),
    );
    let mut prompt = render_template_context(&tmpl, prompt_context, tokens).map_err(|err| {
        LlamaCppProviderError::Template(format!("Failed to render chat template: {err}"))
    })?;
    let mut generation_prompt = render_generation_prompt(
        config,
        &caps,
        &tmpl,
        &root_context,
        continuation,
        continuation_messages,
        continuation_prepared_messages,
        &mut prompt,
        prompt_enable_thinking,
        tokens,
    )?;
    caps.apply_prompt_workarounds(
        &mut prompt,
        &mut generation_prompt,
        config.add_generation_prompt,
    );

    let has_tools = tools.is_some_and(|tools| !tools.is_empty())
        && !matches!(&config.tool_choice, LlamaCppToolChoice::None);
    let use_native_tools = has_tools && caps.supports_tools;
    let parse_tool_calls = has_tools && !config.force_pure_content;
    let grammar_lazy = !config.force_pure_content
        && use_native_tools
        && matches!(&config.tool_choice, LlamaCppToolChoice::Auto)
        && grammar.is_some()
        && json_schema.is_none();

    Ok(RenderedChat {
        force_pure_content: config.force_pure_content,
        is_continuation: continuation.is_some(),
        add_bos: tokenizer_add_special(tokens, &prompt),
        prompt,
        generation_prompt,
        grammar: if config.force_pure_content {
            None
        } else {
            grammar
        },
        grammar_lazy,
        grammar_triggers: if grammar_lazy {
            caps.grammar_triggers()
        } else {
            Vec::default()
        },
        preserved_tokens: caps.preserved_tokens(),
        additional_stops: caps.additional_stops(),
        parse_tool_calls,
        tool_names: tools
            .unwrap_or_default()
            .iter()
            .map(|tool| tool.function.name.clone())
            .collect(),
        reasoning_format: if config.force_pure_content {
            None
        } else {
            config.reasoning_format
        },
        reasoning_start_tag: if config.force_pure_content {
            None
        } else {
            caps.reasoning_start_tag.map(ToOwned::to_owned)
        },
        reasoning_end_tag: if config.force_pure_content {
            None
        } else {
            caps.reasoning_end_tag.map(ToOwned::to_owned)
        },
    })
}

fn render_template_context(
    tmpl: &minijinja::Template<'_, '_>,
    context: Map<String, Value>,
    tokens: &TemplateTokens,
) -> Result<String, JinjaError> {
    let mut rendered = tmpl.render(Value::Object(context))?;
    strip_tokenizer_added_specials(&mut rendered, tokens);
    Ok(rendered)
}

fn build_template_context(
    config: &LlamaCppConfig,
    prepared_messages: Vec<Value>,
    tools_value: Value,
    kwargs: Map<String, Value>,
    tokens: &TemplateTokens,
    prompt_enable_thinking: bool,
) -> Map<String, Value> {
    let mut root_context = Map::new();
    root_context.insert("messages".to_string(), Value::Array(prepared_messages));
    root_context.insert(
        "bos_token".to_string(),
        Value::String(tokens.bos_token.clone()),
    );
    root_context.insert(
        "eos_token".to_string(),
        Value::String(tokens.eos_token.clone()),
    );
    root_context.insert(
        "enable_thinking".to_string(),
        Value::Bool(prompt_enable_thinking),
    );
    root_context.insert(
        "parallel_tool_calls".to_string(),
        Value::Bool(config.parallel_tool_calls.unwrap_or(false)),
    );
    root_context.insert("tool_choice".to_string(), tool_choice_context(config));
    root_context.insert("date_string".to_string(), Value::String(date_string()));
    root_context.insert("datetime".to_string(), Value::String(datetime_string()));
    if !tools_value.as_array().is_some_and(Vec::is_empty) {
        root_context.insert("tools".to_string(), tools_value);
    }

    let mut kwargs_object = Map::new();
    for (key, value) in kwargs {
        root_context.insert(key.clone(), value.clone());
        kwargs_object.insert(key, value);
    }
    root_context.insert("kwargs".to_string(), Value::Object(kwargs_object.clone()));
    root_context.insert("extra_context".to_string(), Value::Object(kwargs_object));
    root_context
}

fn tool_choice_context(config: &LlamaCppConfig) -> Value {
    match &config.tool_choice {
        LlamaCppToolChoice::Auto => Value::String("auto".to_string()),
        LlamaCppToolChoice::Required => Value::String("required".to_string()),
        LlamaCppToolChoice::None => Value::String("none".to_string()),
        LlamaCppToolChoice::Function { name } => json!({
            "type": "function",
            "function": {
                "name": name
            }
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn render_generation_prompt(
    config: &LlamaCppConfig,
    caps: &TemplateCapabilities,
    tmpl: &Template<'_, '_>,
    root_context: &Map<String, Value>,
    continuation: Option<LlamaCppChatContinuation>,
    continuation_messages: &[ServerChatMessage],
    continuation_prepared_messages: Vec<Value>,
    prompt: &mut String,
    prompt_enable_thinking: bool,
    tokens: &TemplateTokens,
) -> Result<String, LlamaCppProviderError> {
    if continuation == Some(LlamaCppChatContinuation::Reasoning)
        && !continuation_messages.is_empty()
    {
        return render_reasoning_continuation_prompt(
            config,
            caps,
            tmpl,
            root_context,
            &continuation_messages[0],
            prompt,
            tokens,
        );
    }

    if continuation.is_some() && !continuation_messages.is_empty() {
        return render_content_continuation_prompt(
            tmpl,
            root_context,
            continuation_prepared_messages,
            prompt,
            tokens,
        );
    }

    if config.add_generation_prompt {
        let mut generation_root_context = root_context.clone();
        generation_root_context.insert(
            "enable_thinking".to_string(),
            Value::Bool(prompt_enable_thinking),
        );
        return generation_prompt_suffix(tmpl, generation_root_context, tokens);
    }

    Ok(String::default())
}

fn render_reasoning_continuation_prompt(
    config: &LlamaCppConfig,
    caps: &TemplateCapabilities,
    tmpl: &Template<'_, '_>,
    root_context: &Map<String, Value>,
    continuation_message: &ServerChatMessage,
    prompt: &mut String,
    tokens: &TemplateTokens,
) -> Result<String, LlamaCppProviderError> {
    let reasoning_prefill = continuation_message.render_content("\n");
    let base_generation_prompt = if config.add_generation_prompt {
        generation_prompt_suffix(tmpl, root_context.clone(), tokens)?
    } else {
        String::default()
    };
    let reasoning_prompt =
        caps.reasoning_continuation_prompt(prompt, &base_generation_prompt, &reasoning_prefill)?;
    prompt.push_str(&reasoning_prompt);
    Ok(reasoning_prompt)
}

fn render_content_continuation_prompt(
    tmpl: &Template<'_, '_>,
    root_context: &Map<String, Value>,
    continuation_prepared_messages: Vec<Value>,
    prompt: &mut String,
    tokens: &TemplateTokens,
) -> Result<String, LlamaCppProviderError> {
    let no_continuation_prompt = prompt.clone();
    let mut continuation_context = root_context.clone();
    continuation_context.insert(
        "messages".to_string(),
        Value::Array(continuation_prepared_messages),
    );
    set_add_generation_prompt(&mut continuation_context, false);
    let continuation_prompt =
        render_template_context(tmpl, continuation_context, tokens).map_err(|err| {
            LlamaCppProviderError::Template(format!(
                "Failed to render chat continuation prompt: {err}"
            ))
        })?;
    *prompt = continuation_prompt;
    Ok(string_suffix_after_common_prefix(&no_continuation_prompt, prompt).to_string())
}

fn generation_prompt_suffix(
    tmpl: &minijinja::Template<'_, '_>,
    root_context: Map<String, Value>,
    tokens: &TemplateTokens,
) -> Result<String, LlamaCppProviderError> {
    let no_gen_prompt =
        render_template_context(tmpl, root_context.clone(), tokens).map_err(|err| {
            LlamaCppProviderError::Template(format!(
                "Failed to render chat template without generation prompt: {err}"
            ))
        })?;
    let mut gen_context = root_context;
    set_add_generation_prompt(&mut gen_context, true);
    let gen_prompt = render_template_context(tmpl, gen_context, tokens).map_err(|err| {
        LlamaCppProviderError::Template(format!(
            "Failed to render chat template with generation prompt: {err}"
        ))
    })?;
    Ok(string_suffix_after_common_prefix(&no_gen_prompt, &gen_prompt).to_string())
}

fn set_add_generation_prompt(context: &mut Map<String, Value>, enabled: bool) {
    context.insert("add_generation_prompt".to_string(), Value::Bool(enabled));
}

fn tokenizer_add_special(tokens: &TemplateTokens, rendered_prompt: &str) -> bool {
    match (tokens.add_bos, tokens.add_eos) {
        (Some(add_bos), Some(add_eos)) => add_bos || add_eos,
        (Some(add_bos), None) => add_bos,
        (None, Some(add_eos)) => add_eos,
        (None, None) => {
            tokens.bos_token.is_empty() || !rendered_prompt.starts_with(&tokens.bos_token)
        }
    }
}

fn strip_tokenizer_added_specials(rendered: &mut String, tokens: &TemplateTokens) {
    if tokens.add_bos == Some(true)
        && !tokens.bos_token.is_empty()
        && rendered.starts_with(&tokens.bos_token)
    {
        rendered.replace_range(..tokens.bos_token.len(), "");
    }
    if tokens.add_eos == Some(true)
        && !tokens.eos_token.is_empty()
        && rendered.ends_with(&tokens.eos_token)
    {
        let end = rendered.len() - tokens.eos_token.len();
        rendered.truncate(end);
    }
}

fn string_suffix_after_common_prefix<'a>(left: &str, right: &'a str) -> &'a str {
    let mut prefix_len = 0;
    for ((left_idx, left_ch), (right_idx, right_ch)) in
        left.char_indices().zip(right.char_indices())
    {
        if left_ch != right_ch {
            break;
        }
        prefix_len = left_idx + left_ch.len_utf8();
        if prefix_len != right_idx + right_ch.len_utf8() {
            break;
        }
    }

    &right[prefix_len..]
}

fn chat_template_kwargs(
    config: &LlamaCppConfig,
) -> Result<Map<String, Value>, LlamaCppProviderError> {
    let Some(value) = config
        .extra_body
        .as_ref()
        .and_then(|body| body.get("chat_template_kwargs"))
    else {
        return Ok(Map::new());
    };
    let object = value.as_object().ok_or_else(|| {
        LlamaCppProviderError::Template(
            "llama.cpp chat_template_kwargs must be a JSON object".to_string(),
        )
    })?;

    let mut parsed = Map::new();
    for (key, value) in object {
        parsed.insert(key.clone(), value.clone());
    }
    Ok(parsed)
}

fn resolve_enable_thinking(
    config: &LlamaCppConfig,
    kwargs: &Map<String, Value>,
) -> Result<Option<bool>, LlamaCppProviderError> {
    match kwargs.get("enable_thinking") {
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(Value::String(_)) => Ok(config.enable_thinking),
        Some(_) => Err(LlamaCppProviderError::Template(
            "invalid type for `enable_thinking` (expected boolean)".to_string(),
        )),
        None => Ok(config.enable_thinking),
    }
}

fn tojson_filter(value: JinjaValue, kwargs: Kwargs) -> Result<JinjaValue, JinjaError> {
    let json = if let Ok(indent) = kwargs.get::<usize>("indent") {
        let mut buf = Vec::new();
        let spaces = b" ".repeat(indent);
        let formatter = serde_json::ser::PrettyFormatter::with_indent(&spaces);
        let mut serializer = serde_json::Serializer::with_formatter(&mut buf, formatter);
        value.serialize(&mut serializer).map_err(|err| {
            JinjaError::new(ErrorKind::BadSerialization, "cannot serialize to JSON")
                .with_source(err)
        })?;
        String::from_utf8(buf).map_err(|err| {
            JinjaError::new(ErrorKind::BadSerialization, "cannot serialize to JSON")
                .with_source(err)
        })?
    } else {
        serde_json::to_string(&value).map_err(|err| {
            JinjaError::new(ErrorKind::BadSerialization, "cannot serialize to JSON")
                .with_source(err)
        })?
    };

    let mut escaped = String::with_capacity(json.len());
    for ch in json.chars() {
        match ch {
            '<' => escaped.push_str("\\u003c"),
            '>' => escaped.push_str("\\u003e"),
            '&' => escaped.push_str("\\u0026"),
            '\'' => escaped.push_str("\\u0027"),
            _ => escaped.push(ch),
        }
    }
    Ok(JinjaValue::from_safe_string(escaped))
}

fn pycompat_method_callback(
    state: &State,
    value: &JinjaValue,
    method: &str,
    args: &[JinjaValue],
) -> Result<JinjaValue, JinjaError> {
    if value.kind() == ValueKind::Map {
        return match method {
            "items" => state.apply_filter("items", std::slice::from_ref(value)),
            "get" => map_get(value, args),
            _ => Err(JinjaError::from(ErrorKind::UnknownMethod)),
        };
    }

    let Some(text) = value.as_str() else {
        return Err(JinjaError::from(ErrorKind::UnknownMethod));
    };

    pycompat_string_method(text, method, args)
}

fn pycompat_string_method(
    text: &str,
    method: &str,
    args: &[JinjaValue],
) -> Result<JinjaValue, JinjaError> {
    match method {
        "startswith" => {
            let prefix = string_arg(method, args, 0)?;
            ensure_arg_count(method, args, 1)?;
            Ok(JinjaValue::from(text.starts_with(prefix)))
        }
        "endswith" => {
            let suffix = string_arg(method, args, 0)?;
            ensure_arg_count(method, args, 1)?;
            Ok(JinjaValue::from(text.ends_with(suffix)))
        }
        "strip" => {
            ensure_max_arg_count(method, args, 1)?;
            Ok(JinjaValue::from(strip_chars(
                text,
                optional_string_arg(args, 0)?,
            )))
        }
        "lstrip" => {
            ensure_max_arg_count(method, args, 1)?;
            Ok(JinjaValue::from(lstrip_chars(
                text,
                optional_string_arg(args, 0)?,
            )))
        }
        "rstrip" => {
            ensure_max_arg_count(method, args, 1)?;
            Ok(JinjaValue::from(rstrip_chars(
                text,
                optional_string_arg(args, 0)?,
            )))
        }
        "split" => {
            ensure_max_arg_count(method, args, 2)?;
            let sep = optional_string_arg(args, 0)?;
            let maxsplit = optional_usize_arg(args, 1)?;
            Ok(JinjaValue::from_iter(split_string(text, sep, maxsplit)))
        }
        "lower" => {
            ensure_arg_count(method, args, 0)?;
            Ok(JinjaValue::from(text.to_lowercase()))
        }
        "upper" => {
            ensure_arg_count(method, args, 0)?;
            Ok(JinjaValue::from(text.to_uppercase()))
        }
        "replace" => {
            let from = string_arg(method, args, 0)?;
            let to = string_arg(method, args, 1)?;
            ensure_max_arg_count(method, args, 3)?;
            let replaced = if let Some(count) = optional_usize_arg(args, 2)? {
                text.replacen(from, to, count)
            } else {
                text.replace(from, to)
            };
            Ok(JinjaValue::from(replaced))
        }
        _ => Err(JinjaError::from(ErrorKind::UnknownMethod)),
    }
}

fn map_get(value: &JinjaValue, args: &[JinjaValue]) -> Result<JinjaValue, JinjaError> {
    if !(1..=2).contains(&args.len()) {
        return Err(invalid_method_args("get"));
    }

    let resolved = value.get_item(&args[0])?;
    if resolved.is_undefined() {
        Ok(args.get(1).cloned().unwrap_or_else(|| JinjaValue::from(())))
    } else {
        Ok(resolved)
    }
}

fn string_arg<'a>(
    method: &str,
    args: &'a [JinjaValue],
    index: usize,
) -> Result<&'a str, JinjaError> {
    args.get(index)
        .and_then(JinjaValue::as_str)
        .ok_or_else(|| invalid_method_args(method))
}

fn optional_string_arg(args: &[JinjaValue], index: usize) -> Result<Option<&str>, JinjaError> {
    match args.get(index) {
        Some(value) => value
            .as_str()
            .map(Some)
            .ok_or_else(|| invalid_method_args("string method")),
        None => Ok(None),
    }
}

fn optional_usize_arg(args: &[JinjaValue], index: usize) -> Result<Option<usize>, JinjaError> {
    match args.get(index) {
        Some(value) => value
            .as_usize()
            .map(Some)
            .ok_or_else(|| invalid_method_args("string method")),
        None => Ok(None),
    }
}

fn ensure_arg_count(method: &str, args: &[JinjaValue], expected: usize) -> Result<(), JinjaError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(invalid_method_args(method))
    }
}

fn ensure_max_arg_count(method: &str, args: &[JinjaValue], max: usize) -> Result<(), JinjaError> {
    if args.len() <= max {
        Ok(())
    } else {
        Err(invalid_method_args(method))
    }
}

fn invalid_method_args(method: &str) -> JinjaError {
    JinjaError::new(
        ErrorKind::InvalidOperation,
        format!("invalid arguments for Python-compatible string method `{method}`"),
    )
}

fn strip_chars(text: &str, chars: Option<&str>) -> String {
    match chars {
        Some(chars) => text.trim_matches(|ch| chars.contains(ch)).to_string(),
        None => text.trim().to_string(),
    }
}

fn lstrip_chars(text: &str, chars: Option<&str>) -> String {
    match chars {
        Some(chars) => text.trim_start_matches(|ch| chars.contains(ch)).to_string(),
        None => text.trim_start().to_string(),
    }
}

fn rstrip_chars(text: &str, chars: Option<&str>) -> String {
    match chars {
        Some(chars) => text.trim_end_matches(|ch| chars.contains(ch)).to_string(),
        None => text.trim_end().to_string(),
    }
}

fn split_string(text: &str, sep: Option<&str>, maxsplit: Option<usize>) -> Vec<String> {
    match (sep, maxsplit) {
        (Some(sep), Some(maxsplit)) => text
            .splitn(maxsplit.saturating_add(1), sep)
            .map(ToOwned::to_owned)
            .collect(),
        (Some(sep), None) => text.split(sep).map(ToOwned::to_owned).collect(),
        (None, Some(maxsplit)) => text
            .split_whitespace()
            .take(maxsplit.saturating_add(1))
            .map(ToOwned::to_owned)
            .collect(),
        (None, None) => text.split_whitespace().map(ToOwned::to_owned).collect(),
    }
}

fn raise_exception(message: String) -> Result<String, JinjaError> {
    Err(JinjaError::new(ErrorKind::InvalidOperation, message))
}

fn strftime_now(format: String) -> String {
    Local::now().format(&format).to_string()
}

fn date_string() -> String {
    Local::now().format("%d %b %Y").to_string()
}

fn datetime_string() -> String {
    Local::now().format("%b %d %Y").to_string()
}

#[derive(Debug, Clone)]
struct TemplateCapabilities {
    supports_tools: bool,
    reasoning_start_tag: Option<&'static str>,
    reasoning_end_tag: Option<&'static str>,
    markers: HashSet<&'static str>,
    server_caps: ServerTemplateCaps,
}

fn supports_tool_calls(source: &str) -> bool {
    source.contains("tool_calls")
        || source.contains("<tool_call>")
        || source.contains("[TOOL_CALLS]")
        || source.contains("to=functions.")
}

fn supports_typed_content(source: &str) -> bool {
    source.contains("content[")
        || source.contains("content |")
        || source.contains("content|")
        || source.contains("message['content']")
        || source.contains("message.content")
        || source.contains("media_marker")
        || source.contains("'type'")
        || source.contains("\"type\"")
}

fn supports_string_content(source: &str) -> bool {
    !source.contains("content[0]")
        || source.contains("is string")
        || source.contains("is_string")
        || source.contains("message['content']")
}

fn supports_system_role(source: &str) -> bool {
    source.contains("system") || source.contains("developer") || !source.contains("raise_exception")
}

fn supports_object_arguments(source: &str) -> bool {
    source.contains("arguments|tojson")
        || source.contains("arguments | tojson")
        || source.contains("arguments: tool_call")
        || source.contains("arguments']")
}

fn detect_reasoning_tags(source: &str) -> (Option<&'static str>, Option<&'static str>) {
    if source.contains("<think>") || source.contains("enable_thinking") {
        (Some("<think>"), Some("</think>"))
    } else if source.contains("<|channel>thought") {
        (Some("<|channel>thought"), Some("<channel|>"))
    } else if source.contains("[THINK]") {
        (Some("[THINK]"), Some("[/THINK]"))
    } else {
        (None, None)
    }
}

fn template_markers(source: &str) -> HashSet<&'static str> {
    let mut markers = HashSet::new();
    for marker in TEMPLATE_MARKERS {
        if source.contains(marker) {
            markers.insert(*marker);
        }
    }
    markers
}

const TEMPLATE_MARKERS: &[&str] = &[
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "[TOOL_CALLS]",
    "[/TOOL_CALLS]",
    "[ARGS]",
    "[/ARGS]",
    "<think>",
    "</think>",
    "[THINK]",
    "[/THINK]",
    "<|channel|>",
    "<|start|>",
    "<|message|>",
    "<|end|>",
    "<|return|>",
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|turn>",
    "<turn|>",
    ">>>all",
    ">>>${recipient}",
    "[SYSTEM_PROMPT]",
    "[CALL_ID]",
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_argument_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_start|>",
    "<|tool_list_start|>",
    "<|tool_list_end|>",
    "<|message_sep|>\n\n",
    "<|message_sep|>\n\nfunction call<|role_sep|>\n",
    "<|role_sep|>\n",
    "｜DSML｜",
    "<｜DSML｜function_calls>",
    ">>>",
];

impl TemplateCapabilities {
    fn analyze(source: &str) -> Self {
        let supports_tools = source.contains("tools");
        let supports_tool_calls = supports_tool_calls(source);
        let supports_typed_content = supports_typed_content(source);
        let supports_string_content = supports_string_content(source);
        let supports_system_role = supports_system_role(source);
        let supports_object_arguments = supports_object_arguments(source);
        let gemma4_tool_response_compat = source.contains("'<|tool_call>call:'")
            && !source.contains("{#- OpenAI Chat Completions:");
        let reasoning = detect_reasoning_tags(source);
        let markers = template_markers(source);
        let gpt_oss_reasoning_compat = markers.contains("<|channel|>");
        let ministral_reasoning_blocks_compat = markers.contains("[SYSTEM_PROMPT]")
            && markers.contains("[TOOL_CALLS]")
            && markers.contains("[ARGS]")
            && !markers.contains("[CALL_ID]");
        let lfm2_reasoning_compat = (markers.contains("<|tool_list_start|>")
            && markers.contains("<|tool_list_end|>"))
            || (source.contains("List of tools: [") && !markers.contains("<|tool_list_start|>"));

        Self {
            supports_tools,
            reasoning_start_tag: reasoning.0,
            reasoning_end_tag: reasoning.1,
            markers,
            server_caps: ServerTemplateCaps {
                supports_string_content,
                supports_typed_content,
                supports_system_role,
                supports_tool_calls,
                supports_object_arguments,
                gpt_oss_reasoning_compat,
                ministral_reasoning_blocks_compat,
                lfm2_reasoning_compat,
                gemma4_tool_response_compat,
            },
        }
    }

    fn grammar_triggers(&self) -> Vec<GrammarTrigger> {
        let mut triggers = Vec::new();
        if self.markers.contains("<|channel|>") {
            triggers.push(GrammarTrigger::Pattern("^\\s+to$".to_string()));
            triggers.push(GrammarTrigger::Pattern(
                "^<\\|channel\\|>(?:commentary|analysis)\\s+to=functions$".to_string(),
            ));
            triggers.push(GrammarTrigger::Pattern(
                "<\\|start\\|>assistant(\\s+to)".to_string(),
            ));
            triggers.push(GrammarTrigger::Pattern(
                "<\\|start\\|>assistant(<\\|channel\\|>(?:commentary|analysis)\\s+to)".to_string(),
            ));
        }
        if self.markers.contains(">>>all") && self.markers.contains(">>>${recipient}") {
            triggers.push(GrammarTrigger::Pattern(">>>(?!all)".to_string()));
        }
        for marker in [
            "<|tool_call>",
            "<|tool_call_begin|>",
            "<|tool_call_start|>",
            "<|message_sep|>\n\nfunction call<|role_sep|>\n",
            "<｜DSML｜function_calls>",
        ] {
            if self.markers.contains(marker) {
                triggers.push(GrammarTrigger::Word(marker.to_string()));
            }
        }
        if self.markers.contains("<function=") {
            triggers.push(GrammarTrigger::Word("<function=".to_string()));
        }
        for marker in ["<tool_call>", "[TOOL_CALLS]", "{\"type\": \"function\","] {
            if self.markers.contains(marker) || marker.starts_with('{') {
                triggers.push(GrammarTrigger::Word(marker.to_string()));
            }
        }
        triggers
    }

    fn preserved_tokens(&self) -> Vec<String> {
        self.markers
            .iter()
            .map(|marker| (*marker).to_string())
            .collect()
    }

    fn additional_stops(&self) -> Vec<String> {
        Vec::new()
    }

    fn reasoning_continuation_prompt(
        &self,
        prompt: &str,
        base_generation_prompt: &str,
        reasoning_prefill: &str,
    ) -> Result<String, LlamaCppProviderError> {
        if self.markers.contains("<|channel|>")
            && self.markers.contains("<|start|>")
            && self.markers.contains("<|message|>")
        {
            return Ok(format!(
                "<|start|>assistant<|channel|>analysis<|message|>{reasoning_prefill}"
            ));
        }

        if self.markers.contains("<|channel>")
            && self.markers.contains("<channel|>")
            && self.markers.contains("<|turn>")
        {
            let model_turn = if prompt.ends_with("<turn|>\n") {
                "<|turn>model\n"
            } else {
                ""
            };
            return Ok(format!(
                "{model_turn}<|channel>thought\n{reasoning_prefill}"
            ));
        }

        if self.markers.contains("[THINK]") {
            return Ok(format!("[THINK]{reasoning_prefill}"));
        }

        if self.markers.contains("<|tool_calls_section_begin|>") && self.markers.contains("<think>")
        {
            return Ok(format!(
                "<|im_assistant|>assistant<|im_middle|><think>{reasoning_prefill}"
            ));
        }

        if self.markers.contains("｜DSML｜") && self.markers.contains("<think>") {
            return Ok(format!("<｜Assistant｜><think>{reasoning_prefill}"));
        }

        if self.markers.contains("<think>") {
            return Ok(format!(
                "{base_generation_prompt}<think>{reasoning_prefill}"
            ));
        }

        Err(LlamaCppProviderError::Template(
            "continue_final_message=reasoning requires a supported reasoning chat template"
                .to_string(),
        ))
    }

    fn apply_prompt_workarounds(
        &self,
        prompt: &mut String,
        generation_prompt: &mut String,
        add_generation_prompt: bool,
    ) {
        if !add_generation_prompt
            && self.markers.contains("<|return|>")
            && self.markers.contains("<|end|>")
            && let Some(pos) = prompt.rfind("<|return|>")
        {
            prompt.replace_range(pos..pos + "<|return|>".len(), "<|end|>");
        }

        if !add_generation_prompt {
            return;
        }

        if self.markers.contains("<|turn>") && prompt.ends_with("<turn|>\n") {
            const GEMMA4_MODEL_TURN: &str = "<|turn>model\n";
            prompt.push_str(GEMMA4_MODEL_TURN);
            *generation_prompt = GEMMA4_MODEL_TURN.to_string();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_llm::chat::{ChatRole, FunctionTool, MessageType, Tool};
    use autoagents_llm::{FunctionCall, ToolCall};
    use serde_json::json;

    fn empty_tokens() -> TemplateTokens {
        TemplateTokens::default()
    }

    fn user_message(content: &str) -> ChatMessage {
        ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: content.to_string(),
        }
    }

    #[test]
    fn renders_chat_template_kwargs_as_top_level_context() {
        let config = LlamaCppConfig {
            extra_body: Some(json!({
                "chat_template_kwargs": {
                    "custom_flag": "enabled"
                }
            })),
            ..Default::default()
        };
        let template = explicit_template_source(
            "{{ custom_flag }}:{% for message in messages %}{{ message['role'] }}={{ message['content'] }}{% endfor %}",
        );

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("template should render");

        assert_eq!(rendered.prompt, "enabled:user=hi");
    }

    #[test]
    fn preserves_string_chat_template_kwargs_without_json_retyping() {
        let config = LlamaCppConfig {
            extra_body: Some(json!({
                "chat_template_kwargs": {
                    "enable_thinking": "true",
                    "custom_label": "\"server-style\""
                }
            })),
            ..Default::default()
        };
        let template = explicit_template_source(
            "{{ enable_thinking is string }}:{{ enable_thinking }}:{{ custom_label }}",
        );

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("template should render");

        assert_eq!(rendered.prompt, "true:true:\"server-style\"");
    }

    #[test]
    fn renders_named_tool_choice_as_openai_object_context() {
        let config = LlamaCppConfig {
            tool_choice: LlamaCppToolChoice::Function {
                name: "lookup".to_string(),
            },
            ..Default::default()
        };
        let template = explicit_template_source(
            "{{ tool_choice['type'] }}:{{ tool_choice['function']['name'] }}",
        );

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("template should render");

        assert_eq!(rendered.prompt, "function:lookup");
    }

    #[test]
    fn enable_thinking_uses_kwargs_over_config_without_prompt_rewrite() {
        let config = LlamaCppConfig {
            enable_thinking: Some(false),
            extra_body: Some(json!({
                "chat_template_kwargs": {
                    "enable_thinking": true
                }
            })),
            ..Default::default()
        };
        let template = explicit_template_source(
            "{% if enable_thinking %}<think>{% endif %}{{ messages[0]['content'] }}",
        );

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("solve")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("template should render");

        assert_eq!(rendered.prompt, "<think>solve");
    }

    #[test]
    fn rejects_non_object_chat_template_kwargs() {
        let config = LlamaCppConfig {
            extra_body: Some(json!({
                "chat_template_kwargs": true
            })),
            ..Default::default()
        };
        let template = explicit_template_source("{{ messages[0]['content'] }}");

        let err = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect_err("non-object kwargs should fail");

        assert!(err.to_string().contains("chat_template_kwargs"));
    }

    #[test]
    fn tojson_filter_renders_valid_json_not_display_text() {
        let config = LlamaCppConfig {
            extra_body: Some(json!({
                "chat_template_kwargs": {
                    "payload": {
                        "name": "lookup",
                        "description": "a < b & c",
                        "arguments": {
                            "q": "rust"
                        }
                    }
                }
            })),
            ..Default::default()
        };
        let template = explicit_template_source("{{ payload|tojson }}");

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("template should render");

        assert!(rendered.prompt.contains("\"name\":\"lookup\""));
        assert!(rendered.prompt.contains("\\u003c"));
        assert!(rendered.prompt.contains("\\u0026"));
        let parsed: Value = serde_json::from_str(&rendered.prompt).expect("valid JSON output");
        assert_eq!(parsed["name"], "lookup");
        assert_eq!(parsed["description"], "a < b & c");
        assert_eq!(parsed["arguments"]["q"], "rust");
    }

    #[test]
    fn strftime_now_renders_current_time() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source("{{ strftime_now(\"%Y-%m-%d\") }}");

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("template should render");

        assert_eq!(rendered.prompt, Local::now().format("%Y-%m-%d").to_string());
    }

    #[test]
    fn template_receives_llama_server_date_context() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source("{{ date_string }}|{{ datetime }}");

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("template should render");

        assert_eq!(
            rendered.prompt,
            format!(
                "{}|{}",
                Local::now().format("%d %b %Y"),
                Local::now().format("%b %d %Y")
            )
        );
    }

    #[test]
    fn tool_template_sets_lazy_grammar_metadata() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source(
            "{% for message in messages %}{{ message['content'] }}{% endfor %}{% if tools %}<tool_call>{% endif %}",
        );
        let tool = Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "lookup".to_string(),
                description: "Lookup value".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "q": { "type": "string" }
                    },
                    "required": ["q"]
                }),
            },
        };

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            Some(&[tool]),
            None,
            Some("root ::= \"{}\"".to_string()),
            &empty_tokens(),
        )
        .expect("tool template should render");

        assert!(rendered.parse_tool_calls);
        assert!(rendered.grammar_lazy);
        assert!(
            rendered.grammar_triggers.iter().any(
                |trigger| matches!(trigger, GrammarTrigger::Word(word) if word == "<tool_call>")
            )
        );
    }

    #[test]
    fn tool_call_history_support_does_not_imply_tool_definition_support() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source(
            "{% for message in messages %}{% if message.tool_calls %}<tool_call>{{ message.tool_calls|tojson }}</tool_call>{% endif %}{{ message['content'] }}{% endfor %}",
        );
        let tool = Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "lookup".to_string(),
                description: "Lookup value".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "q": { "type": "string" }
                    },
                    "required": ["q"]
                }),
            },
        };

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            Some(&[tool]),
            None,
            Some("root ::= \"{}\"".to_string()),
            &empty_tokens(),
        )
        .expect("tool-call history template should render");

        assert!(rendered.parse_tool_calls);
        assert!(
            !rendered.grammar_lazy,
            "template has tool_call history support but cannot render tool definitions"
        );
        assert!(rendered.grammar_triggers.is_empty());
    }

    #[test]
    fn specialized_template_markers_set_server_preserved_tokens_and_triggers() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source(
            "{{ tools|tojson }}<function=</function><|tool_call_begin|><|tool_call_argument_begin|><|tool_call_end|><|message_sep|>\n\nfunction call<|role_sep|>\n<｜DSML｜function_calls>>>>all\n>>>${recipient}",
        );
        let tool = Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "lookup".to_string(),
                description: "Lookup value".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "q": { "type": "string" }
                    }
                }),
            },
        };

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            Some(&[tool]),
            None,
            Some("root ::= \"{}\"".to_string()),
            &empty_tokens(),
        )
        .expect("template should render");

        assert!(
            rendered
                .preserved_tokens
                .iter()
                .any(|token| token == "<|tool_call_begin|>")
        );
        assert!(rendered.grammar_triggers.iter().any(
            |trigger| matches!(trigger, GrammarTrigger::Word(word) if word == "<|tool_call_begin|>")
        ));
        assert!(
            rendered.grammar_triggers.iter().any(
                |trigger| matches!(trigger, GrammarTrigger::Word(word) if word == "<function=")
            )
        );
        assert!(rendered.grammar_triggers.iter().any(
            |trigger| matches!(trigger, GrammarTrigger::Pattern(pattern) if pattern == ">>>(?!all)")
        ));
    }

    #[test]
    fn gpt_oss_channel_marker_enables_reasoning_workaround_caps() {
        let caps = TemplateCapabilities::analyze(
            "{% for message in messages %}<|start|>{{ message.role }}<|channel|>final<|message|>{{ message.thinking }}{% endfor %}",
        );

        assert!(caps.server_caps.gpt_oss_reasoning_compat);
    }

    #[test]
    fn specialized_reasoning_history_rewrite_caps_match_server_detection() {
        let ministral = TemplateCapabilities::analyze("[SYSTEM_PROMPT][TOOL_CALLS][ARGS]");
        let mistral_small =
            TemplateCapabilities::analyze("[SYSTEM_PROMPT][TOOL_CALLS][ARGS][CALL_ID]");
        let lfm2 = TemplateCapabilities::analyze("<|tool_list_start|><|tool_list_end|>");
        let lfm25 = TemplateCapabilities::analyze("List of tools: []");

        assert!(ministral.server_caps.ministral_reasoning_blocks_compat);
        assert!(!mistral_small.server_caps.ministral_reasoning_blocks_compat);
        assert!(lfm2.server_caps.lfm2_reasoning_compat);
        assert!(lfm25.server_caps.lfm2_reasoning_compat);
    }

    #[test]
    fn functionary_trigger_requires_server_specialized_markers() {
        let functionary = TemplateCapabilities::analyze(">>>all\n>>>${recipient}");
        let generic = TemplateCapabilities::analyze(">>>generic");

        assert!(functionary.grammar_triggers().iter().any(
            |trigger| matches!(trigger, GrammarTrigger::Pattern(pattern) if pattern == ">>>(?!all)")
        ));
        assert!(generic.grammar_triggers().iter().all(
            |trigger| !matches!(trigger, GrammarTrigger::Pattern(pattern) if pattern == ">>>(?!all)")
        ));
    }

    #[test]
    fn real_template_rendering_goldens_cover_server_specialized_families() {
        struct GoldenCase {
            name: &'static str,
            template: &'static str,
            config: LlamaCppConfig,
            expected_prompt_contains: &'static [&'static str],
            expected_generation_prompt: &'static str,
            expected_lazy: bool,
            expected_trigger: Option<&'static str>,
            expected_preserved: Option<&'static str>,
            expected_reasoning: Option<(&'static str, &'static str)>,
        }

        let tool = Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "lookup".to_string(),
                description: "Lookup value".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "q": { "type": "string" }
                    },
                    "required": ["q"]
                }),
            },
        };

        let cases = vec![
            GoldenCase {
                name: "qwen chatml thinking",
                template: "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}{% endif %}{# <think></think> #}",
                config: LlamaCppConfig {
                    enable_thinking: Some(true),
                    reasoning_format: Some(LlamaCppReasoningFormat::Auto),
                    ..Default::default()
                },
                expected_prompt_contains: &[
                    "<|im_start|>user\nhi<|im_end|>",
                    "<|im_start|>assistant\n<think>\n",
                ],
                expected_generation_prompt: "<|im_start|>assistant\n<think>\n",
                expected_lazy: false,
                expected_trigger: None,
                expected_preserved: None,
                expected_reasoning: Some(("<think>", "</think>")),
            },
            GoldenCase {
                name: "gpt-oss native channel",
                template: "{% if tools %}{% for tool in tools %}{{ tool.function.name }}{% endfor %}{% endif %}{% for message in messages %}<|start|>{{ message['role'] }}<|channel|>final<|message|>{{ message['content'] }}<|end|>{% endfor %}{% if add_generation_prompt %}<|start|>assistant<|channel|>final<|message|>{% endif %}{# to=functions <|message|> #}",
                config: LlamaCppConfig::default(),
                expected_prompt_contains: &[
                    "lookup<|start|>user<|channel|>final<|message|>hi<|end|>",
                    "<|start|>assistant<|channel|>final<|message|>",
                ],
                expected_generation_prompt: "<|start|>assistant<|channel|>final<|message|>",
                expected_lazy: true,
                expected_trigger: Some("^\\s+to$"),
                expected_preserved: None,
                expected_reasoning: None,
            },
            GoldenCase {
                name: "functionary v3.2",
                template: ">>>all\n{% for message in messages %}{{ message['content'] }}{% endfor %}{% if add_generation_prompt %}>>>${recipient}\n{% endif %}{# tools rendered elsewhere #}",
                config: LlamaCppConfig::default(),
                expected_prompt_contains: &[">>>all\nhi>>>${recipient}\n"],
                expected_generation_prompt: ">>>${recipient}\n",
                expected_lazy: true,
                expected_trigger: Some(">>>(?!all)"),
                expected_preserved: None,
                expected_reasoning: None,
            },
            GoldenCase {
                name: "kimi k2",
                template: "{% if tools %}{% for tool in tools %}{{ tool.function.name }}{% endfor %}{% endif %}{% for message in messages %}<|im_user|>{{ message['content'] }}{% endfor %}{% if add_generation_prompt %}<|im_assistant|>{% endif %}{# <|tool_call_begin|>functions.lookup:0<|tool_call_argument_begin|><|tool_call_end|> #}",
                config: LlamaCppConfig::default(),
                expected_prompt_contains: &["lookup<|im_user|>hi<|im_assistant|>"],
                expected_generation_prompt: "<|im_assistant|>",
                expected_lazy: true,
                expected_trigger: Some("<|tool_call_begin|>"),
                expected_preserved: Some("<|tool_call_begin|>"),
                expected_reasoning: None,
            },
            GoldenCase {
                name: "lfm2 python tools",
                template: "List of tools: [{% for tool in tools %}{{ tool.function.name }}{% endfor %}]{% for message in messages %}{{ message['content'] }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}{# <|tool_call_start|><|tool_call_end|> #}",
                config: LlamaCppConfig::default(),
                expected_prompt_contains: &["List of tools: [lookup]hi<|assistant|>"],
                expected_generation_prompt: "<|assistant|>",
                expected_lazy: true,
                expected_trigger: Some("<|tool_call_start|>"),
                expected_preserved: Some("<|tool_call_start|>"),
                expected_reasoning: None,
            },
            GoldenCase {
                name: "gigachat v3",
                template: "{% if tools %}{% for tool in tools %}{{ tool.function.name }}{% endfor %}{% endif %}assistant<|role_sep|>\n{% for message in messages %}{{ message['content'] }}<|message_sep|>\n\n{% endfor %}{% if add_generation_prompt %}assistant<|role_sep|>\n{% endif %}{# <|message_sep|>\n\nfunction call<|role_sep|>\n #}",
                config: LlamaCppConfig::default(),
                expected_prompt_contains: &[
                    "lookupassistant<|role_sep|>\nhi<|message_sep|>\n\nassistant<|role_sep|>\n",
                ],
                expected_generation_prompt: "assistant<|role_sep|>\n",
                expected_lazy: true,
                expected_trigger: Some("<|message_sep|>\n\nfunction call<|role_sep|>\n"),
                expected_preserved: Some("<|message_sep|>\n\nfunction call<|role_sep|>\n"),
                expected_reasoning: None,
            },
        ];

        for case in cases {
            let rendered = render_chat_template(
                &case.config,
                &explicit_template_source(case.template),
                &[user_message("hi")],
                Some(std::slice::from_ref(&tool)),
                None,
                Some("root ::= \"{}\"".to_string()),
                &empty_tokens(),
            )
            .unwrap_or_else(|err| panic!("{} should render: {err}", case.name));

            for expected in case.expected_prompt_contains {
                assert!(
                    rendered.prompt.contains(expected),
                    "{} prompt should contain {expected:?}, got {:?}",
                    case.name,
                    rendered.prompt
                );
            }
            assert_eq!(
                rendered.generation_prompt, case.expected_generation_prompt,
                "{} generation prompt",
                case.name
            );
            assert!(
                rendered.parse_tool_calls,
                "{} should parse tool calls",
                case.name
            );
            assert_eq!(rendered.grammar_lazy, case.expected_lazy, "{}", case.name);
            if let Some(trigger) = case.expected_trigger {
                assert!(
                    rendered
                        .grammar_triggers
                        .iter()
                        .any(|candidate| match candidate {
                            GrammarTrigger::Word(word) => word == trigger,
                            GrammarTrigger::Pattern(pattern) => pattern == trigger,
                            GrammarTrigger::PatternFull(pattern) => pattern == trigger,
                            GrammarTrigger::Token(_) => false,
                        }),
                    "{} should expose trigger {trigger:?}; got {:?}",
                    case.name,
                    rendered.grammar_triggers
                );
            } else {
                assert!(
                    rendered.grammar_triggers.is_empty(),
                    "{} should not use lazy triggers",
                    case.name
                );
            }
            if let Some(preserved) = case.expected_preserved {
                assert!(
                    rendered
                        .preserved_tokens
                        .iter()
                        .any(|token| token == preserved),
                    "{} should preserve {preserved:?}",
                    case.name
                );
            }
            if let Some((start, end)) = case.expected_reasoning {
                assert_eq!(
                    rendered.reasoning_start_tag.as_deref(),
                    Some(start),
                    "{} reasoning start",
                    case.name
                );
                assert_eq!(
                    rendered.reasoning_end_tag.as_deref(),
                    Some(end),
                    "{} reasoning end",
                    case.name
                );
            }
        }
    }

    #[test]
    fn default_enable_thinking_matches_llama_server_default_true() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source("{% if enable_thinking %}<think>{% endif %}answer");

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("template should render");

        assert_eq!(rendered.prompt, "<think>answer");
    }

    #[test]
    fn gpt_oss_return_token_is_replaced_without_generation_prompt() {
        let config = LlamaCppConfig {
            add_generation_prompt: false,
            ..Default::default()
        };
        let template = explicit_template_source(
            "{% for message in messages %}<|start|>{{ message['role'] }}<|channel|>final<|message|>{{ message['content'] }}<|return|>{% endfor %}{# <|end|> #}",
        );

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("done")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("template should render");

        assert_eq!(
            rendered.prompt,
            "<|start|>user<|channel|>final<|message|>done<|end|>"
        );
        assert_eq!(rendered.generation_prompt, "");
    }

    #[test]
    fn continue_final_message_prefills_last_message_without_generation_prompt() {
        let config = LlamaCppConfig {
            continue_final_message: LlamaCppChatContinuation::Content,
            ..Default::default()
        };
        let template = explicit_template_source(
            "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
        );
        let assistant = ChatMessage {
            role: ChatRole::Assistant,
            message_type: MessageType::Text,
            content: "partial".to_string(),
        };

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi"), assistant],
            None,
            None,
            Some("root ::= \"{}\"".to_string()),
            &empty_tokens(),
        )
        .expect("continuation template should render");

        assert_eq!(rendered.prompt, "user:hi\nassistant:partial\n");
        assert_eq!(rendered.generation_prompt, "assistant:partial\n");
        assert!(rendered.is_continuation);
    }

    #[test]
    fn continue_final_message_reasoning_uses_known_template_markers() {
        let config = LlamaCppConfig {
            continue_final_message: LlamaCppChatContinuation::Reasoning,
            ..Default::default()
        };
        let template = explicit_template_source(
            "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}{# <think></think> #}",
        );
        let assistant = ChatMessage {
            role: ChatRole::Assistant,
            message_type: MessageType::Text,
            content: "plan".to_string(),
        };

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi"), assistant],
            None,
            None,
            Some("root ::= \"{}\"".to_string()),
            &empty_tokens(),
        )
        .expect("reasoning continuation should render");

        assert_eq!(rendered.prompt, "user:hi\n<|assistant|><think>plan");
        assert_eq!(rendered.generation_prompt, "<|assistant|><think>plan");
        assert!(rendered.is_continuation);
    }

    #[test]
    fn continue_final_message_reasoning_rejects_unknown_template_markers() {
        let config = LlamaCppConfig {
            continue_final_message: LlamaCppChatContinuation::Reasoning,
            ..Default::default()
        };
        let template = explicit_template_source("{{ messages|length }}");

        let err = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect_err("reasoning continuation requires a known marker family");

        assert!(
            err.to_string()
                .contains("supported reasoning chat template")
        );
    }

    #[test]
    fn force_pure_content_disables_parser_and_reasoning_prompt_only() {
        let config = LlamaCppConfig {
            force_pure_content: true,
            enable_thinking: Some(true),
            ..Default::default()
        };
        let template = explicit_template_source(
            "{% if enable_thinking %}<think>{% endif %}{% for message in messages %}{{ message['content'] }}{% endfor %}{% if tools %}<tool_call>{% endif %}{% if add_generation_prompt %}{% if enable_thinking %}<think>{% endif %}<|assistant|>{% endif %}",
        );
        let tool = Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "lookup".to_string(),
                description: "Lookup value".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "q": { "type": "string" }
                    },
                    "required": ["q"]
                }),
            },
        };

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            Some(&[tool]),
            None,
            Some("root ::= \"{}\"".to_string()),
            &empty_tokens(),
        )
        .expect("template should render");

        assert_eq!(rendered.prompt, "hi<tool_call><|assistant|>");
        assert_eq!(rendered.generation_prompt, "<|assistant|>");
        assert!(rendered.force_pure_content);
        assert!(!rendered.parse_tool_calls);
        assert!(rendered.grammar.is_none());
        assert!(!rendered.grammar_lazy);
        assert!(rendered.grammar_triggers.is_empty());
        assert_eq!(rendered.reasoning_format, None);
        assert_eq!(rendered.reasoning_start_tag, None);
        assert_eq!(rendered.reasoning_end_tag, None);
    }

    #[test]
    fn render_tracks_generation_prompt_suffix_for_grammar_prefill() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source(
            "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        );

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            Some("root ::= \"{}\"".to_string()),
            &empty_tokens(),
        )
        .expect("template should render");

        assert!(rendered.prompt.ends_with("<|im_start|>assistant\n"));
        assert_eq!(rendered.generation_prompt, "<|im_start|>assistant\n");
    }

    #[test]
    fn add_generation_prompt_false_is_present_in_template_context() {
        let config = LlamaCppConfig {
            add_generation_prompt: false,
            ..Default::default()
        };
        let template = explicit_template_source(
            "{{ add_generation_prompt is defined }}:{{ add_generation_prompt == false }}|{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}{% endfor %}",
        );

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("template should render");

        assert_eq!(rendered.prompt, "true:true|user:hi");
        assert_eq!(rendered.generation_prompt, "");
    }

    #[test]
    fn gemma4_prompt_workaround_adds_model_turn_after_bare_boundary() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source(
            "{% for message in messages %}<|turn>{{ message['role'] }}\n{{ message['content'] }}<turn|>\n{% endfor %}{% if add_generation_prompt %}<turn|>\n{% endif %}",
        );

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            Some("root ::= \"{}\"".to_string()),
            &empty_tokens(),
        )
        .expect("template should render");

        assert!(rendered.prompt.ends_with("<turn|>\n<|turn>model\n"));
        assert_eq!(rendered.generation_prompt, "<|turn>model\n");
    }

    #[test]
    fn template_receives_model_special_tokens() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source("{{ bos_token }}user{{ eos_token }}");
        let tokens = TemplateTokens {
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            add_bos: None,
            add_eos: None,
        };

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            None,
            &tokens,
        )
        .expect("template should render");

        assert_eq!(rendered.prompt, "<s>user</s>");
    }

    #[test]
    fn rendered_bos_disables_automatic_chat_bos() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source("{{ bos_token }}user");
        let tokens = TemplateTokens {
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            add_bos: None,
            add_eos: None,
        };

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            None,
            &tokens,
        )
        .expect("template should render");

        assert_eq!(rendered.prompt, "<s>user");
        assert!(!rendered.add_bos);
    }

    #[test]
    fn tokenizer_special_metadata_strips_rendered_bos_and_eos() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source("{{ bos_token }}user{{ eos_token }}");
        let tokens = TemplateTokens {
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            add_bos: Some(true),
            add_eos: Some(true),
        };

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hi")],
            None,
            None,
            None,
            &tokens,
        )
        .expect("template should render");

        assert_eq!(rendered.prompt, "user");
        assert!(rendered.add_bos);
    }

    #[test]
    fn tool_results_expand_to_openai_tool_messages() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source("{{ messages|tojson }}");
        let tool_result = ChatMessage {
            role: ChatRole::Tool,
            message_type: MessageType::ToolResult(vec![
                ToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "lookup".to_string(),
                        arguments: "{\"q\":\"rust\"}".to_string(),
                    },
                },
                ToolCall {
                    id: "call_2".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "search".to_string(),
                        arguments: "{\"q\":\"llama\"}".to_string(),
                    },
                },
            ]),
            content: String::default(),
        };

        let rendered = render_chat_template(
            &config,
            &template,
            &[tool_result],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("template should render");
        let messages: Value = serde_json::from_str(&rendered.prompt).expect("valid messages JSON");
        let messages = messages.as_array().expect("messages should be an array");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["tool_call_id"], "call_1");
        assert_eq!(messages[0]["content"], "{\"q\":\"rust\"}");
        assert_eq!(messages[1]["tool_call_id"], "call_2");
        assert!(messages[0].get("tool_calls").is_none());
    }

    #[test]
    fn supports_python_string_methods_used_by_hf_templates() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source(
            r#"{{ messages[0]['content'].startswith('<tool_response>') }}:
{{ messages[0]['content'].endswith('</tool_response>') }}:
{{ '  <think>x</think>  '.strip().split('</think>')[0].lstrip().replace('<think>', '') }}:
{{ 'MiXeD'.lower() }}:
{{ 'mixed'.upper() }}"#,
        );

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("<tool_response>ok</tool_response>")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("template should render");

        assert_eq!(rendered.prompt, "true:\ntrue:\nx:\nmixed:\nMIXED");
    }

    #[test]
    fn supports_python_map_get_used_by_hf_templates() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source(
            "{{ messages[0].get('content') }}:{{ messages[0].get('missing', 'fallback') }}:{{ messages[0].get('missing') is none }}",
        );

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hello")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("template should render");

        assert_eq!(rendered.prompt, "hello:fallback:true");
    }

    #[test]
    fn schema_hint_is_not_injected_as_system_message_for_templates() {
        let config = LlamaCppConfig::default();
        let schema = StructuredOutputFormat {
            name: "MathAgentOutput".to_string(),
            description: Some("math output".to_string()),
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "value": { "type": "integer" }
                }
            })),
            strict: Some(true),
        };
        let template = explicit_template_source(
            r#"{% for message in messages %}{% if message['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{{ message['role'] }}:{{ message['content'] }}
{% endfor %}"#,
        );

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("What is 42 + 8?")],
            None,
            Some(&schema),
            None,
            &empty_tokens(),
        )
        .expect("schema grammar must not inject a synthetic system message");

        assert!(!rendered.prompt.contains("Return a valid JSON response"));
        assert!(rendered.prompt.contains("user:What is 42 + 8?"));
    }

    #[test]
    fn hf_generation_blocks_are_render_transparent() {
        let config = LlamaCppConfig::default();
        let template = explicit_template_source(
            "{% for message in messages %}{% generation %}{{ message['content'] }}{% endgeneration %}{% endfor %}",
        );

        let rendered = render_chat_template(
            &config,
            &template,
            &[user_message("hello")],
            None,
            None,
            None,
            &empty_tokens(),
        )
        .expect("generation blocks should not break minijinja rendering");

        assert_eq!(rendered.prompt, "hello");
    }
}
