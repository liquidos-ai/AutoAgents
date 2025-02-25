use futures::Stream;
use futures::StreamExt;
use reqwest::Error;
use serde_json::Value;

pub(crate) struct HTTPRequest {}

impl HTTPRequest {
    pub async fn request(url: String, body: Value) -> Result<String, Error> {
        let client = reqwest::Client::new();
        let response = client.post(&url).json(&body).send().await?;
        response.text().await
    }

    pub async fn stream_request(
        url: String,
        body: Value,
    ) -> Result<impl Stream<Item = Result<String, Error>>, Error> {
        let client = reqwest::Client::new();
        let response = client.post(&url).json(&body).send().await?;
        // We map each Bytes chunk into a String.
        let stream = response.bytes_stream().map(|result| {
            result.map(|bytes| {
                // Convert bytes to String. This uses a lossy conversion, replacing
                // invalid UTF-8 sequences with the Unicode replacement character.
                String::from_utf8_lossy(&bytes).to_string()
            })
        });
        Ok(stream)
    }
}
