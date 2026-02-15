use super::ParseError;
use crate::tools::document_parsing::ParsedDocument;
use serde_json::json;
use std::io::{Cursor, Read};

pub fn parse_docx(bytes: &[u8]) -> Result<ParsedDocument, ParseError> {
    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| ParseError::Format(format!("Not a valid DOCX/ZIP: {}", e)))?;

    let xml_content = {
        let mut file = archive
            .by_name("word/document.xml")
            .map_err(|e| ParseError::Format(format!("Missing word/document.xml: {}", e)))?;
        let mut buf = String::default();
        file.read_to_string(&mut buf)?;
        buf
    };

    let text = extract_docx_text(&xml_content)?;

    Ok(ParsedDocument {
        text,
        metadata: json!({ "format": "docx" }),
    })
}

fn extract_docx_text(xml: &str) -> Result<String, ParseError> {
    let mut reader = quick_xml::Reader::from_str(xml);
    let mut in_text = false;
    let mut parts: Vec<String> = Vec::new();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(quick_xml::events::Event::Start(ref e)) if e.name().as_ref() == b"w:t" => {
                in_text = true;
            }
            Ok(quick_xml::events::Event::Text(ref e)) if in_text => {
                if let Ok(decoded) = e.decode()
                    && let Ok(unescaped) = quick_xml::escape::unescape(&decoded)
                {
                    parts.push(unescaped.to_string());
                }
            }
            Ok(quick_xml::events::Event::End(ref e)) if e.name().as_ref() == b"w:t" => {
                in_text = false;
            }
            Ok(quick_xml::events::Event::End(ref e)) if e.name().as_ref() == b"w:p" => {
                parts.push("\n".to_string());
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(e) => return Err(ParseError::Format(format!("XML parse error: {}", e))),
            _ => {}
        }
        buf.clear();
    }

    Ok(parts.concat())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_minimal_docx(text: &str) -> Vec<u8> {
        let buf = Vec::new();
        let cursor = Cursor::new(buf);
        let mut zip = zip::ZipWriter::new(cursor);

        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("word/document.xml", options).unwrap();
        let xml = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>{}</w:t></w:r></w:p>
  </w:body>
</w:document>"#,
            text
        );
        zip.write_all(xml.as_bytes()).unwrap();

        zip.finish().unwrap().into_inner()
    }

    #[test]
    fn test_parse_docx_basic() {
        let docx_bytes = create_minimal_docx("Hello World");
        let result = parse_docx(&docx_bytes).expect("Failed to parse DOCX");
        assert!(result.text.contains("Hello World"));
    }

    #[test]
    fn test_parse_docx_invalid() {
        let result = parse_docx(b"not a zip");
        assert!(result.is_err());
    }
}
