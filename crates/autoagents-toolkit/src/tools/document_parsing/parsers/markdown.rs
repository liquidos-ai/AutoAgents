use super::ParseError;
use crate::tools::document_parsing::ParsedDocument;
use serde_json::json;

pub fn parse_markdown(bytes: &[u8]) -> Result<ParsedDocument, ParseError> {
    let text = String::from_utf8(bytes.to_vec())?;
    let line_count = text.lines().count();

    Ok(ParsedDocument {
        text,
        metadata: json!({
            "format": "markdown",
            "line_count": line_count,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_markdown_basic() {
        let data = b"# Title\n\nSome **bold** text\n\n- item 1\n- item 2";
        let result = parse_markdown(data).expect("Failed to parse markdown");
        assert!(result.text.contains("# Title"));
        assert!(result.text.contains("**bold**"));
        assert_eq!(
            result.metadata.get("format").unwrap().as_str().unwrap(),
            "markdown"
        );
    }
}
