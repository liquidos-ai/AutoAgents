use super::ParseError;
use crate::tools::document_parsing::ParsedDocument;
use serde_json::json;

pub fn parse_pdf(bytes: &[u8]) -> Result<ParsedDocument, ParseError> {
    let text = pdf_extract::extract_text_from_mem(bytes)
        .map_err(|e| ParseError::Format(format!("PDF extraction failed: {}", e)))?;

    Ok(ParsedDocument {
        text,
        metadata: json!({ "format": "pdf" }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pdf_invalid() {
        let result = parse_pdf(b"not a pdf");
        assert!(result.is_err());
    }
}
