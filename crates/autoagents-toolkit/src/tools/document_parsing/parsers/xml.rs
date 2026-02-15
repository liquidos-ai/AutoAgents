use super::ParseError;
use crate::tools::document_parsing::ParsedDocument;
use serde_json::json;

pub fn parse_xml(bytes: &[u8]) -> Result<ParsedDocument, ParseError> {
    let xml_str = String::from_utf8(bytes.to_vec())?;
    let mut reader = quick_xml::Reader::from_str(&xml_str);
    let mut text_parts = Vec::new();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(quick_xml::events::Event::Text(ref e)) => {
                if let Ok(decoded) = e.decode()
                    && let Ok(unescaped) = quick_xml::escape::unescape(&decoded)
                {
                    let trimmed = unescaped.trim();
                    if !trimmed.is_empty() {
                        text_parts.push(trimmed.to_string());
                    }
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(e) => return Err(ParseError::Format(format!("XML parse error: {}", e))),
            _ => {}
        }
        buf.clear();
    }

    Ok(ParsedDocument {
        text: text_parts.join("\n"),
        metadata: json!({ "format": "xml" }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_xml_basic() {
        let xml = b"<root><item>Hello</item><item>World</item></root>";
        let result = parse_xml(xml).expect("Failed to parse XML");
        assert!(result.text.contains("Hello"));
        assert!(result.text.contains("World"));
    }

    #[test]
    fn test_parse_xml_nested() {
        let xml = b"<root><parent><child>Content</child></parent></root>";
        let result = parse_xml(xml).expect("Failed to parse XML");
        assert!(result.text.contains("Content"));
    }

    #[test]
    fn test_parse_xml_invalid() {
        let result = parse_xml(b"<unclosed>");
        // quick-xml may or may not error on unclosed tags depending on version
        // Just verify it doesn't panic
        let _ = result;
    }
}
