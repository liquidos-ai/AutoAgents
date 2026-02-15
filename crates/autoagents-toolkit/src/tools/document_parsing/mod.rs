mod parse_document;
pub(crate) mod parsers;

pub use parse_document::DocumentParser;

use std::path::Path;

/// Supported document formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentFormat {
    Pdf,
    Docx,
    Xlsx,
    Pptx,
    Html,
    Csv,
    Json,
    Xml,
    Txt,
    Markdown,
}

impl DocumentFormat {
    /// Detect format from file extension. Returns None for unsupported formats.
    pub fn from_extension(path: &str) -> Option<Self> {
        // Strip query params for URLs
        let clean_path = path.split('?').next().unwrap_or(path);

        let ext = Path::new(clean_path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())?;

        match ext.as_str() {
            "pdf" => Some(Self::Pdf),
            "docx" => Some(Self::Docx),
            "xlsx" => Some(Self::Xlsx),
            "pptx" => Some(Self::Pptx),
            "html" | "htm" => Some(Self::Html),
            "csv" => Some(Self::Csv),
            "json" => Some(Self::Json),
            "xml" => Some(Self::Xml),
            "txt" | "text" | "log" => Some(Self::Txt),
            "md" | "markdown" => Some(Self::Markdown),
            _ => None,
        }
    }

    /// Resolve a format from a plain name string (e.g. "pdf", "csv").
    /// Useful when the caller wants to override auto-detection by specifying
    /// the format explicitly, such as parsing a `.dat` file as CSV.
    pub fn from_str_name(name: &str) -> Option<Self> {
        let dummy = format!("file.{}", name);
        Self::from_extension(&dummy)
    }

    /// Return the canonical lowercase name for this format.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Docx => "docx",
            Self::Xlsx => "xlsx",
            Self::Pptx => "pptx",
            Self::Html => "html",
            Self::Csv => "csv",
            Self::Json => "json",
            Self::Xml => "xml",
            Self::Txt => "txt",
            Self::Markdown => "markdown",
        }
    }
}

/// The result of parsing a document.
#[derive(Debug, Clone)]
pub struct ParsedDocument {
    /// Extracted text content.
    pub text: String,
    /// Optional metadata (page count, sheet names, etc.).
    pub metadata: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_format_detection() {
        assert_eq!(
            DocumentFormat::from_extension("file.pdf"),
            Some(DocumentFormat::Pdf)
        );
        assert_eq!(
            DocumentFormat::from_extension("file.DOCX"),
            Some(DocumentFormat::Docx)
        );
        assert_eq!(
            DocumentFormat::from_extension("file.xlsx"),
            Some(DocumentFormat::Xlsx)
        );
        assert_eq!(
            DocumentFormat::from_extension("file.pptx"),
            Some(DocumentFormat::Pptx)
        );
        assert_eq!(
            DocumentFormat::from_extension("file.html"),
            Some(DocumentFormat::Html)
        );
        assert_eq!(
            DocumentFormat::from_extension("file.htm"),
            Some(DocumentFormat::Html)
        );
        assert_eq!(
            DocumentFormat::from_extension("file.csv"),
            Some(DocumentFormat::Csv)
        );
        assert_eq!(
            DocumentFormat::from_extension("file.json"),
            Some(DocumentFormat::Json)
        );
        assert_eq!(
            DocumentFormat::from_extension("file.xml"),
            Some(DocumentFormat::Xml)
        );
        assert_eq!(
            DocumentFormat::from_extension("file.txt"),
            Some(DocumentFormat::Txt)
        );
        assert_eq!(
            DocumentFormat::from_extension("file.md"),
            Some(DocumentFormat::Markdown)
        );
        assert_eq!(DocumentFormat::from_extension("file.unknown"), None);
    }

    #[test]
    fn test_format_detection_url_with_query() {
        assert_eq!(
            DocumentFormat::from_extension("https://example.com/file.pdf?token=abc"),
            Some(DocumentFormat::Pdf)
        );
    }

    #[test]
    fn test_from_str_name() {
        assert_eq!(
            DocumentFormat::from_str_name("pdf"),
            Some(DocumentFormat::Pdf)
        );
        assert_eq!(
            DocumentFormat::from_str_name("csv"),
            Some(DocumentFormat::Csv)
        );
        assert_eq!(DocumentFormat::from_str_name("nope"), None);
    }

    #[test]
    fn test_parse_all_sample_files() {
        let test_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tools/document_parsing/test_files");

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
                    println!(
                        "Metadata: {}",
                        serde_json::to_string_pretty(&doc.metadata).unwrap()
                    );
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
}
