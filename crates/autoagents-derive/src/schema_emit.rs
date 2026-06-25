use proc_macro2::{Span, TokenStream};
use quote::quote;
use serde_json::{Map, Number, Value};
use syn::{Error, Result};

/// Largest integer exactly representable as `f64` (2^53).
const MAX_SAFE_INTEGER_F64: f64 = 9_007_199_254_740_992.0;

/// Validates that a JSON value is well-formed for emission into generated code.
pub(crate) fn validate_json(value: &Value, span: Span) -> Result<()> {
    match value {
        Value::Null | Value::Bool(_) | Value::String(_) => Ok(()),
        Value::Number(n) => validate_number(n, span),
        Value::Array(items) => {
            for item in items {
                validate_json(item, span)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            for value in map.values() {
                validate_json(value, span)?;
            }
            Ok(())
        }
    }
}

fn validate_number(number: &Number, span: Span) -> Result<()> {
    if number.is_i64() || number.is_u64() {
        return Ok(());
    }

    if let Some(f) = number.as_f64() {
        if !f.is_finite() {
            return Err(Error::new(
                span,
                format!("non-finite float `{f}` is not allowed in generated schema"),
            ));
        }
        if f.fract() != 0.0 {
            return Err(Error::new(
                span,
                format!(
                    "schema number `{number}` must be an integer; non-integer numbers are not supported in generated schema literals"
                ),
            ));
        }
        if f.abs() > MAX_SAFE_INTEGER_F64 {
            return Err(Error::new(
                span,
                format!(
                    "schema number `{number}` is out of range for generated schema literals (must be a whole number within ±2^53)"
                ),
            ));
        }
        let as_i64 = f as i64;
        if (as_i64 as f64) != f {
            return Err(Error::new(
                span,
                format!(
                    "schema number `{number}` is out of range for generated schema literals (must be a whole number within ±2^53)"
                ),
            ));
        }
        return Ok(());
    }

    Err(Error::new(
        span,
        format!("unsupported JSON number in generated schema: {number}"),
    ))
}

/// Converts a validated `serde_json::Value` into token trees that construct the value at runtime
/// without parsing JSON strings.
pub(crate) fn json_value_to_tokens(value: &Value, span: Span) -> Result<TokenStream> {
    match value {
        Value::Null => Ok(quote! { ::serde_json::Value::Null }),
        Value::Bool(b) => Ok(quote! { ::serde_json::Value::Bool(#b) }),
        Value::Number(n) => number_to_tokens(n, span),
        Value::String(s) => {
            let lit = proc_macro2::Literal::string(s);
            Ok(quote! { ::serde_json::Value::String(#lit.to_string()) })
        }
        Value::Array(items) => {
            let elements = items
                .iter()
                .map(|item| json_value_to_tokens(item, span))
                .collect::<Result<Vec<_>>>()?;
            Ok(quote! {
                ::serde_json::Value::Array(vec![#(#elements),*])
            })
        }
        Value::Object(map) => object_to_tokens(map, span),
    }
}

/// Parses a JSON string at macro expansion time, validates it, and emits constructor tokens.
pub(crate) fn schema_str_to_tokens(json_str: &str, span: Span) -> Result<TokenStream> {
    let value: Value = serde_json::from_str(json_str).map_err(|err| {
        Error::new(
            span,
            format!("failed to serialize tool/agent schema as valid JSON: {err}"),
        )
    })?;
    validate_json(&value, span)?;
    json_value_to_tokens(&value, span)
}

fn number_to_tokens(n: &Number, span: Span) -> Result<TokenStream> {
    if let Some(i) = n.as_i64() {
        return Ok(quote! {
            ::serde_json::Value::Number(::serde_json::Number::from(#i))
        });
    }
    if let Some(u) = n.as_u64() {
        return Ok(quote! {
            ::serde_json::Value::Number(::serde_json::Number::from(#u))
        });
    }
    if let Some(f) = n.as_f64()
        && f.is_finite()
        && f.fract() == 0.0
        && f.abs() <= MAX_SAFE_INTEGER_F64
    {
        let i = f as i64;
        if (i as f64) == f {
            return Ok(quote! {
                ::serde_json::Value::Number(::serde_json::Number::from(#i))
            });
        }
    }

    Err(Error::new(
        span,
        format!(
            "schema number `{n}` must be an integer; non-integer numbers are not supported in generated schema literals"
        ),
    ))
}

fn object_to_tokens(map: &Map<String, Value>, span: Span) -> Result<TokenStream> {
    let mut inserts = Vec::with_capacity(map.len());
    for (key, value) in map {
        let key_lit = proc_macro2::Literal::string(key);
        let value_tokens = json_value_to_tokens(value, span)?;
        inserts.push(quote! {
            map.insert(#key_lit.to_string(), #value_tokens);
        });
    }
    Ok(quote! {
        ::serde_json::Value::Object({
            let mut map = ::serde_json::Map::new();
            #(#inserts)*
            map
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn schema_str_to_tokens_parses_object() {
        let tokens = schema_str_to_tokens(r#"{"type":"object"}"#, Span::call_site()).unwrap();
        let expanded = tokens.to_string();
        assert!(expanded.contains("serde_json"));
        assert!(expanded.contains("type"));
    }

    #[test]
    fn json_value_to_tokens_handles_nested_structure() {
        let value = json!({
            "name": "test",
            "count": 3,
            "tags": ["a", "b"],
            "meta": { "ok": true }
        });
        let tokens = json_value_to_tokens(&value, Span::call_site()).unwrap();
        let expanded = tokens.to_string();
        assert!(expanded.contains("name"));
        assert!(expanded.contains("tags"));
        assert!(expanded.contains("meta"));
    }

    #[test]
    fn json_value_to_tokens_handles_whole_number_floats() {
        let value = json!(3.0);
        let tokens = json_value_to_tokens(&value, Span::call_site()).unwrap();
        assert!(tokens.to_string().contains("3"));
    }

    #[test]
    fn validate_json_rejects_fractional_number_literals() {
        let value = json!(1.5);
        let err = validate_json(&value, Span::call_site()).unwrap_err();
        assert!(err.to_string().contains("integer"));
    }

    #[test]
    fn validate_json_rejects_out_of_range_whole_number_floats() {
        let value = json!(1e19);
        let err = validate_json(&value, Span::call_site()).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn validate_json_rejects_i64_max_plus_one_whole_float() {
        let number = Number::from_f64(9223372036854775808.0).expect("finite float");
        let value = Value::Number(number);
        let err = validate_json(&value, Span::call_site()).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }
}
