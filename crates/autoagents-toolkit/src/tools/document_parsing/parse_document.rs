use std::path::PathBuf;
use std::sync::Once;

use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolRuntime, ToolT},
};
use autoagents_derive::{ToolInput, tool};
use log::{debug, warn};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::config::DocumentParserConfig;
use super::source::{DocumentSourceError, fetch_url, load_local_file};
use super::{DocumentFormat, parsers};

static WARN_UNRESTRICTED_LOCAL: Once = Once::new();

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
pub struct DocumentParser {
    config: DocumentParserConfig,
}

impl DocumentParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn try_with_config(config: DocumentParserConfig) -> Result<Self, DocumentSourceError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn with_allowed_roots(roots: Vec<PathBuf>) -> Self {
        Self {
            config: DocumentParserConfig::default().with_allowed_roots(roots),
        }
    }

    pub fn with_allowed_hosts(hosts: Vec<String>) -> Result<Self, DocumentSourceError> {
        Self::try_with_config(DocumentParserConfig::default().with_allowed_hosts(hosts)?)
    }

    fn is_url(source: &str) -> bool {
        starts_with_ignore_ascii_case(source, "https://")
            || starts_with_ignore_ascii_case(source, "http://")
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

    fn map_source_error(error: DocumentSourceError) -> ToolCallError {
        ToolCallError::RuntimeError(Box::new(error))
    }

    fn warn_if_local_paths_unrestricted(&self) {
        if self.config.allowed_roots.is_none() {
            WARN_UNRESTRICTED_LOCAL.call_once(|| {
                warn!(
                    "DocumentParser local file paths are unrestricted; configure allowed_roots for agent deployments"
                );
            });
        }
    }
}

fn starts_with_ignore_ascii_case(value: &str, prefix: &str) -> bool {
    value
        .as_bytes()
        .get(..prefix.len())
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(prefix.as_bytes()))
}

#[async_trait]
impl ToolRuntime for DocumentParser {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let DocumentParserArgs { source, format } = serde_json::from_value(args)?;

        debug!("DocumentParser executing: source={}", source);

        let (bytes, effective_source) = if Self::is_url(&source) {
            let (bytes, filename) = fetch_url(&source, &self.config)
                .await
                .map_err(Self::map_source_error)?;
            let effective = filename.unwrap_or_else(|| source.clone());
            (bytes, effective)
        } else {
            self.warn_if_local_paths_unrestricted();
            let bytes = load_local_file(&source, &self.config)
                .await
                .map_err(Self::map_source_error)?;
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

        let parser = DocumentParser::default();
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

        let parser = DocumentParser::default();
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

        let parser = DocumentParser::default();
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

        let parser = DocumentParser::default();
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

        let parser = DocumentParser::default();
        let args = json!({
            "source": file_path.display().to_string()
        });

        let result = parser.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_parse_blocks_private_url() {
        let parser = DocumentParser::default();
        let args = json!({
            "source": "http://169.254.169.254/latest/meta-data/"
        });

        let result = parser.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_parse_blocks_local_file_outside_allowed_roots() {
        let allowed_dir = tempdir().expect("allowed dir");
        let outside_dir = tempdir().expect("outside dir");
        let outside_file = outside_dir.path().join("secret.txt");
        std::fs::write(&outside_file, b"secret").expect("write");

        let parser = DocumentParser::with_allowed_roots(vec![allowed_dir.path().to_path_buf()]);
        let args = json!({
            "source": outside_file.display().to_string()
        });

        let result = parser.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_parse_rejects_oversized_local_file() {
        let dir = tempdir().expect("tempdir");
        let file_path = dir.path().join("large.txt");
        std::fs::write(&file_path, vec![b'a'; 2048]).expect("write");

        let parser = DocumentParser::try_with_config(
            DocumentParserConfig::default().with_max_local_file_bytes(1024),
        )
        .expect("config");
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
        assert!(DocumentParser::is_url("HTTPS://example.com/file.pdf"));
        assert!(DocumentParser::is_url("HTTP://example.com/file.pdf"));
        assert!(!DocumentParser::is_url("/path/to/file.pdf"));
        assert!(!DocumentParser::is_url("file.pdf"));
    }
}
