use super::ParseError;
use crate::tools::document_parsing::ParsedDocument;
use serde_json::json;

pub fn parse_text(bytes: &[u8]) -> Result<ParsedDocument, ParseError> {
    let text = String::from_utf8(bytes.to_vec())?;
    let line_count = text.lines().count();

    Ok(ParsedDocument {
        text,
        metadata: json!({
            "format": "txt",
            "line_count": line_count,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_text_basic() {
        let data = b"Hello\nWorld\nLine 3";
        let result = parse_text(data).expect("Failed to parse text");
        assert_eq!(result.text, "Hello\nWorld\nLine 3");
        assert_eq!(
            result.metadata.get("line_count").unwrap().as_u64().unwrap(),
            3
        );
    }

    #[test]
    fn test_parse_text_empty() {
        let result = parse_text(b"").expect("Failed to parse empty text");
        assert_eq!(result.text, "");
    }
}
