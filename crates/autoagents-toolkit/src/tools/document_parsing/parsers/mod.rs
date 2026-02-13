mod csv_parser;
mod docx;
mod html;
mod json_parser;
mod markdown;
mod pdf;
mod plain_text;
mod pptx;
mod xlsx;
mod xml;

pub use csv_parser::parse_csv;
pub use docx::parse_docx;
pub use html::parse_html;
pub use json_parser::parse_json;
pub use markdown::parse_markdown;
pub use pdf::parse_pdf;
pub use plain_text::parse_text;
pub use pptx::parse_pptx;
pub use xlsx::parse_xlsx;
pub use xml::parse_xml;

/// Error type for document parsing failures.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("UTF-8 decoding error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("Parse error: {0}")]
    Format(String),
}
