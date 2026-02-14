use super::ParseError;
use crate::tools::document_parsing::ParsedDocument;
use serde_json::json;

pub fn parse_html(bytes: &[u8]) -> Result<ParsedDocument, ParseError> {
    let text = html2text::from_read(bytes, 120)
        .map_err(|e| ParseError::Format(format!("HTML parse error: {}", e)))?;

    Ok(ParsedDocument {
        text,
        metadata: json!({ "format": "html" }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_html_basic() {
        let html = b"<html><body><h1>Title</h1><p>Hello World</p></body></html>";
        let result = parse_html(html).expect("Failed to parse HTML");
        assert!(result.text.contains("Title"));
        assert!(result.text.contains("Hello World"));
    }

    #[test]
    fn test_parse_html_with_links() {
        let html = b"<a href=\"https://example.com\">Link</a>";
        let result = parse_html(html).expect("Failed to parse HTML");
        assert!(result.text.contains("Link"));
    }
}
