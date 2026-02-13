use super::ParseError;
use crate::tools::document_parsing::ParsedDocument;
use serde_json::json;

pub fn parse_csv(bytes: &[u8]) -> Result<ParsedDocument, ParseError> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(bytes);

    let headers = reader
        .headers()
        .map_err(|e| ParseError::Format(format!("CSV header error: {}", e)))?
        .clone();

    let mut rows = Vec::new();
    rows.push(headers.iter().collect::<Vec<_>>().join("\t"));

    let mut row_count = 0;
    for result in reader.records() {
        let record = result.map_err(|e| ParseError::Format(format!("CSV row error: {}", e)))?;
        rows.push(record.iter().collect::<Vec<_>>().join("\t"));
        row_count += 1;
    }

    Ok(ParsedDocument {
        text: rows.join("\n"),
        metadata: json!({
            "format": "csv",
            "column_count": headers.len(),
            "row_count": row_count,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_csv_basic() {
        let csv_data = b"name,age\nAlice,30\nBob,25";
        let result = parse_csv(csv_data).expect("Failed to parse CSV");
        assert!(result.text.contains("Alice"));
        assert!(result.text.contains("Bob"));
        assert_eq!(
            result.metadata.get("row_count").unwrap().as_u64().unwrap(),
            2
        );
        assert_eq!(
            result
                .metadata
                .get("column_count")
                .unwrap()
                .as_u64()
                .unwrap(),
            2
        );
    }

    #[test]
    fn test_parse_csv_single_column() {
        let csv_data = b"value\n1\n2\n3";
        let result = parse_csv(csv_data).expect("Failed to parse CSV");
        assert!(result.text.contains("1"));
        assert_eq!(
            result.metadata.get("row_count").unwrap().as_u64().unwrap(),
            3
        );
    }
}
