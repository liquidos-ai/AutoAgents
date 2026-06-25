mod error;
mod http_fetch;
mod local_file;
mod url_policy;

pub use error::DocumentSourceError;
pub use http_fetch::fetch_url;
pub use local_file::load_local_file;
