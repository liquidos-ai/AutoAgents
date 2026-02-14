use super::{DocumentFormat, parsers};
use std::fs;
use std::path::Path;

#[test]
fn test_parse_all_sample_files() {
    let test_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/tools/document_parsing/test_files");

    let files = [
        "sample.txt",
        "sample.csv",
        "sample.json",
        "sample.xml",
        "sample.html",
        "sample.md",
    ];

    for filename in &files {
        let path = test_dir.join(filename);
        println!("========================================");
        println!("Parsing: {}", path.display());
        println!("========================================");

        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                println!("[ERROR] Failed to read {}: {}\n", filename, e);
                continue;
            }
        };

        let format = match DocumentFormat::from_extension(filename) {
            Some(f) => f,
            None => {
                println!("[ERROR] Unknown format for {}\n", filename);
                continue;
            }
        };

        let result = match format {
            DocumentFormat::Txt => parsers::parse_text(&bytes),
            DocumentFormat::Csv => parsers::parse_csv(&bytes),
            DocumentFormat::Json => parsers::parse_json(&bytes),
            DocumentFormat::Xml => parsers::parse_xml(&bytes),
            DocumentFormat::Html => parsers::parse_html(&bytes),
            DocumentFormat::Markdown => parsers::parse_markdown(&bytes),
            DocumentFormat::Pdf => parsers::parse_pdf(&bytes),
            DocumentFormat::Docx => parsers::parse_docx(&bytes),
            DocumentFormat::Xlsx => parsers::parse_xlsx(&bytes),
            DocumentFormat::Pptx => parsers::parse_pptx(&bytes),
        };

        match result {
            Ok(doc) => {
                println!("Format: {}", format.as_str());
                println!("Metadata: {}", serde_json::to_string_pretty(&doc.metadata).unwrap());
                println!("--- Content ---");
                println!("{}", doc.text);
            }
            Err(e) => {
                println!("[ERROR] Parse failed: {}", e);
            }
        }
        println!();
    }
}
