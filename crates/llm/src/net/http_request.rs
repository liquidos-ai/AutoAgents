use futures::Stream;
use futures::StreamExt;
use reqwest::Error;
use serde_json::Value;

pub(crate) struct HTTPRequest {}

impl HTTPRequest {
    pub async fn request(url: &str, body: Value) -> Result<String, Error> {
        let client = reqwest::Client::new();
        let response = client.post(url).json(&body).send().await?;
        response.text().await
    }

    pub async fn request_with_headers(url: &str, body: Value, headers: Vec<(String, String)>) -> Result<String, Error> {
        let client = reqwest::Client::new();
        let mut request = client.post(url).json(&body);
        
        for (key, value) in headers {
            request = request.header(&key, &value);
        }
        
        let response = request.send().await?;
        response.text().await
    }

    pub fn request_sync(url: &str, body: Value) -> Result<String, Error> {
        let mut response = ureq::post(url).send_json(&body).unwrap();
        let body = response.body_mut();
        Ok(body.read_to_string().unwrap())
    }

    pub fn request_sync_with_headers(url: &str, body: Value, headers: Vec<(String, String)>) -> Result<String, Error> {
        let mut request = ureq::post(url);
        
        for (key, value) in headers {
            request = request.header(&key, &value);
        }
        
        let mut response = request.send_json(&body).unwrap();
        let body = response.body_mut();
        Ok(body.read_to_string().unwrap())
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

    pub async fn stream_request_with_headers(
        url: String,
        body: Value,
        headers: Vec<(String, String)>,
    ) -> Result<impl Stream<Item = Result<String, Error>>, Error> {
        let client = reqwest::Client::new();
        let mut request = client.post(&url).json(&body);
        
        for (key, value) in headers {
            request = request.header(&key, &value);
        }
        
        let response = request.send().await?;
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
