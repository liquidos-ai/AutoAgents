use super::ParseError;
use crate::tools::document_parsing::ParsedDocument;
use calamine::{Reader, Xlsx};
use serde_json::json;
use std::io::Cursor;

pub fn parse_xlsx(bytes: &[u8]) -> Result<ParsedDocument, ParseError> {
    let cursor = Cursor::new(bytes);
    let mut workbook: Xlsx<_> =
        Xlsx::new(cursor).map_err(|e| ParseError::Format(format!("XLSX open failed: {}", e)))?;

    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();
    let mut text_parts = Vec::new();

    for name in &sheet_names {
        if let Ok(range) = workbook.worksheet_range(name) {
            text_parts.push(format!("--- Sheet: {} ---", name));
            for row in range.rows() {
                let cells: Vec<String> = row.iter().map(|c| c.to_string()).collect();
                text_parts.push(cells.join("\t"));
            }
        }
    }

    Ok(ParsedDocument {
        text: text_parts.join("\n"),
        metadata: json!({
            "format": "xlsx",
            "sheet_names": sheet_names,
            "sheet_count": sheet_names.len(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_xlsx_invalid() {
        let result = parse_xlsx(b"not an xlsx");
        assert!(result.is_err());
    }
}
