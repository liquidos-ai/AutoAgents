use super::ParseError;
use crate::tools::document_parsing::ParsedDocument;
use serde_json::json;
use std::io::{Cursor, Read};

pub fn parse_pptx(bytes: &[u8]) -> Result<ParsedDocument, ParseError> {
    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| ParseError::Format(format!("Not a valid PPTX/ZIP: {}", e)))?;

    let mut slides: Vec<(usize, String)> = Vec::new();

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| ParseError::Format(format!("ZIP entry error: {}", e)))?;
        let name = file.name().to_string();

        if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
            let mut xml_content = String::default();
            file.read_to_string(&mut xml_content)?;
            let text = extract_ooxml_text(&xml_content, b"a:t");
            slides.push((slides.len() + 1, text));
        }
    }

    let slide_count = slides.len();
    let text = slides
        .into_iter()
        .map(|(num, text)| format!("--- Slide {} ---\n{}", num, text))
        .collect::<Vec<_>>()
        .join("\n\n");

    Ok(ParsedDocument {
        text,
        metadata: json!({
            "format": "pptx",
            "slide_count": slide_count,
        }),
    })
}

fn extract_ooxml_text(xml: &str, tag: &[u8]) -> String {
    let mut reader = quick_xml::Reader::from_str(xml);
    let mut in_target = false;
    let mut parts = Vec::new();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(quick_xml::events::Event::Start(ref e)) if e.name().as_ref() == tag => {
                in_target = true;
            }
            Ok(quick_xml::events::Event::Text(ref e)) if in_target => {
                if let Ok(text) = e.unescape() {
                    parts.push(text.to_string());
                }
            }
            Ok(quick_xml::events::Event::End(ref e)) if e.name().as_ref() == tag => {
                in_target = false;
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }

    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_minimal_pptx(text: &str) -> Vec<u8> {
        let buf = Vec::new();
        let cursor = Cursor::new(buf);
        let mut zip = zip::ZipWriter::new(cursor);

        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("ppt/slides/slide1.xml", options).unwrap();
        let xml = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:sp><p:txBody>
    <a:p><a:r><a:t>{}</a:t></a:r></a:p>
  </p:txBody></p:sp></p:spTree></p:cSld>
</p:sld>"#,
            text
        );
        zip.write_all(xml.as_bytes()).unwrap();

        zip.finish().unwrap().into_inner()
    }

    #[test]
    fn test_parse_pptx_basic() {
        let pptx_bytes = create_minimal_pptx("Slide Content");
        let result = parse_pptx(&pptx_bytes).expect("Failed to parse PPTX");
        assert!(result.text.contains("Slide Content"));
        assert_eq!(
            result
                .metadata
                .get("slide_count")
                .unwrap()
                .as_u64()
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_parse_pptx_invalid() {
        let result = parse_pptx(b"not a zip");
        assert!(result.is_err());
    }
}
