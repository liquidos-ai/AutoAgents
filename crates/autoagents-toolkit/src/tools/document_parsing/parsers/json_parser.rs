use super::ParseError;
use crate::tools::document_parsing::ParsedDocument;
use serde_json::json;

pub fn parse_json(bytes: &[u8]) -> Result<ParsedDocument, ParseError> {
    let text = String::from_utf8(bytes.to_vec())?;

    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| ParseError::Format(format!("Invalid JSON: {}", e)))?;

    let pretty = serde_json::to_string_pretty(&value).unwrap_or_else(|_| text.clone());

    let type_info = match &value {
        serde_json::Value::Array(a) => format!("array[{}]", a.len()),
        serde_json::Value::Object(o) => format!("object{{{} keys}}", o.len()),
        _ => "scalar".to_string(),
    };

    Ok(ParsedDocument {
        text: pretty,
        metadata: json!({
            "format": "json",
            "type": type_info,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_object() {
        let data = br#"{"name": "Alice", "age": 30}"#;
        let result = parse_json(data).expect("Failed to parse JSON");
        assert!(result.text.contains("Alice"));
        assert!(
            result
                .metadata
                .get("type")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("object")
        );
    }

    #[test]
    fn test_parse_json_array() {
        let data = b"[1, 2, 3]";
        let result = parse_json(data).expect("Failed to parse JSON");
        assert!(
            result
                .metadata
                .get("type")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("array[3]")
        );
    }

    #[test]
    fn test_parse_json_invalid() {
        let result = parse_json(b"{invalid}");
        assert!(result.is_err());
    }
}
