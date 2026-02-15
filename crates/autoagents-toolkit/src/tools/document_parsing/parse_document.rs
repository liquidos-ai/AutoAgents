use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT},
};
use autoagents_derive::{ToolInput, tool};
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{DocumentFormat, parsers};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct DocumentParserArgs {
    #[input(
        description = "Local file path or URL of the document to parse. Supported formats: PDF, DOCX, XLSX, PPTX, HTML, CSV, JSON, XML, TXT, Markdown"
    )]
    source: String,
    #[input(
        description = "Force a specific format instead of auto-detecting from extension. One of: pdf, docx, xlsx, pptx, html, csv, json, xml, txt, markdown"
    )]
    format: Option<String>,
}

#[tool(
    name = "parse_document",
    description = "Parse a document and extract its text content. Supports PDF, DOCX, XLSX, PPTX, HTML, CSV, JSON, XML, TXT, and Markdown. Accepts a local file path or a URL.",
    input = DocumentParserArgs,
)]
#[derive(Default)]
pub struct DocumentParser;

impl DocumentParser {
    pub fn new() -> Self {
        Self
    }

    fn is_url(source: &str) -> bool {
        source.starts_with("http://") || source.starts_with("https://")
    }

    fn resolve_format(
        source: &str,
        format_override: Option<&str>,
    ) -> Result<DocumentFormat, ToolCallError> {
        if let Some(fmt) = format_override {
            DocumentFormat::from_str_name(fmt).ok_or_else(|| {
                ToolCallError::RuntimeError(format!("Unsupported format override: {}", fmt).into())
            })
        } else {
            DocumentFormat::from_extension(source).ok_or_else(|| {
                ToolCallError::RuntimeError(
                    format!(
                        "Cannot detect document format from source: {}. Use the 'format' parameter to specify it explicitly.",
                        source
                    )
                    .into(),
                )
            })
        }
    }

    async fn load_file(path: &str) -> Result<Vec<u8>, ToolCallError> {
        tokio::fs::read(path)
            .await
            .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))
    }

    async fn fetch_url(url: &str) -> Result<(Vec<u8>, Option<String>), ToolCallError> {
        let response = reqwest::get(url)
            .await
            .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

        let filename = response
            .url()
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .map(|s| s.to_string());

        let bytes = response
            .bytes()
            .await
            .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?
            .to_vec();

        Ok((bytes, filename))
    }
}

#[async_trait]
impl ToolRuntime for DocumentParser {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let DocumentParserArgs { source, format } = serde_json::from_value(args)?;

        debug!("DocumentParser executing: source={}", source);

        let (bytes, effective_source) = if Self::is_url(&source) {
            let (bytes, filename) = Self::fetch_url(&source).await?;
            let effective = filename.unwrap_or_else(|| source.clone());
            (bytes, effective)
        } else {
            let bytes = Self::load_file(&source).await?;
            (bytes, source.clone())
        };

        let doc_format = Self::resolve_format(&effective_source, format.as_deref())?;

        let parsed = match doc_format {
            DocumentFormat::Pdf => {
                let b = bytes;
                tokio::task::spawn_blocking(move || parsers::parse_pdf(&b))
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?
            }
            DocumentFormat::Docx => {
                let b = bytes;
                tokio::task::spawn_blocking(move || parsers::parse_docx(&b))
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?
            }
            DocumentFormat::Xlsx => {
                let b = bytes;
                tokio::task::spawn_blocking(move || parsers::parse_xlsx(&b))
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?
            }
            DocumentFormat::Pptx => {
                let b = bytes;
                tokio::task::spawn_blocking(move || parsers::parse_pptx(&b))
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?
            }
            DocumentFormat::Xml => {
                let b = bytes;
                tokio::task::spawn_blocking(move || parsers::parse_xml(&b))
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?
            }
            DocumentFormat::Html => parsers::parse_html(&bytes),
            DocumentFormat::Csv => parsers::parse_csv(&bytes),
            DocumentFormat::Json => parsers::parse_json(&bytes),
            DocumentFormat::Txt => parsers::parse_text(&bytes),
            DocumentFormat::Markdown => parsers::parse_markdown(&bytes),
        }
        .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

        Ok(json!({
            "success": true,
            "source": source,
            "format": doc_format.as_str(),
            "content": parsed.text,
            "metadata": parsed.metadata,
            "content_length": parsed.text.len(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_parse_text_file() {
        let dir = tempdir().expect("Failed to create temp dir");
        let file_path = dir.path().join("test.txt");

        let mut file = std::fs::File::create(&file_path).expect("Failed to create file");
        file.write_all(b"Hello World").expect("Failed to write");
        drop(file);

        let parser = DocumentParser;
        let args = json!({
            "source": file_path.display().to_string()
        });

        let result = parser.execute(args).await.expect("Failed to parse");
        assert!(result.get("success").unwrap().as_bool().unwrap());
        assert_eq!(result.get("format").unwrap().as_str().unwrap(), "txt");
        assert!(
            result
                .get("content")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("Hello World")
        );
    }

    #[tokio::test]
    async fn test_parse_json_file() {
        let dir = tempdir().expect("Failed to create temp dir");
        let file_path = dir.path().join("data.json");

        let mut file = std::fs::File::create(&file_path).expect("Failed to create file");
        file.write_all(br#"{"key": "value"}"#)
            .expect("Failed to write");
        drop(file);

        let parser = DocumentParser;
        let args = json!({
            "source": file_path.display().to_string()
        });

        let result = parser.execute(args).await.expect("Failed to parse");
        assert_eq!(result.get("format").unwrap().as_str().unwrap(), "json");
    }

    #[tokio::test]
    async fn test_parse_csv_file() {
        let dir = tempdir().expect("Failed to create temp dir");
        let file_path = dir.path().join("data.csv");

        let mut file = std::fs::File::create(&file_path).expect("Failed to create file");
        file.write_all(b"name,age\nAlice,30")
            .expect("Failed to write");
        drop(file);

        let parser = DocumentParser;
        let args = json!({
            "source": file_path.display().to_string()
        });

        let result = parser.execute(args).await.expect("Failed to parse");
        assert_eq!(result.get("format").unwrap().as_str().unwrap(), "csv");
        assert!(
            result
                .get("content")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("Alice")
        );
    }

    #[tokio::test]
    async fn test_parse_with_format_override() {
        let dir = tempdir().expect("Failed to create temp dir");
        let file_path = dir.path().join("data.dat");

        let mut file = std::fs::File::create(&file_path).expect("Failed to create file");
        file.write_all(b"name,age\nAlice,30")
            .expect("Failed to write");
        drop(file);

        let parser = DocumentParser;
        let args = json!({
            "source": file_path.display().to_string(),
            "format": "csv"
        });

        let result = parser.execute(args).await.expect("Failed to parse");
        assert_eq!(result.get("format").unwrap().as_str().unwrap(), "csv");
    }

    #[tokio::test]
    async fn test_parse_unknown_format() {
        let dir = tempdir().expect("Failed to create temp dir");
        let file_path = dir.path().join("data.xyz");

        let mut file = std::fs::File::create(&file_path).expect("Failed to create file");
        file.write_all(b"content").expect("Failed to write");
        drop(file);

        let parser = DocumentParser;
        let args = json!({
            "source": file_path.display().to_string()
        });

        let result = parser.execute(args).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_is_url() {
        assert!(DocumentParser::is_url("https://example.com/file.pdf"));
        assert!(DocumentParser::is_url("http://example.com/file.pdf"));
        assert!(!DocumentParser::is_url("/path/to/file.pdf"));
        assert!(!DocumentParser::is_url("file.pdf"));
    }
}
