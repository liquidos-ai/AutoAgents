use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;
use std::{io::Read, path::Path};
use tokio::{fs::File, io::AsyncWriteExt};

pub async fn download_file_from_url(
    url: &str,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut resp = reqwest::get(url).await?;

    if resp.status().is_success() {
        let total_size = resp.content_length().unwrap_or(0);

        eprintln!("Downloading {} - total size: {}", path, total_size);

        let pb = ProgressBar::new(total_size);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"));

        let mut file = File::create(path).await?;
        let mut downloaded = 0;

        while let Some(chunk) = resp.chunk().await? {
            file.write_all(&chunk).await?;
            downloaded += chunk.len();
            pb.set_position(downloaded.try_into()?);
        }

        pb.finish_with_message("Download completed");
        Ok(())
    } else {
        Err(format!("Failed to download file: {}", resp.status()).into())
    }
}

pub fn load_json_file(path: &str) -> Result<Value, String> {
    let file = std::fs::File::open(path);
    if file.is_err() {
        return Err(format!("failed to open file: {}", file.err().unwrap()));
    }

    let mut data = String::new();
    file.unwrap()
        .read_to_string(&mut data)
        .map_err(|e| e.to_string())?;
    let json_value: Value = serde_json::from_str(&data).map_err(|e| e.to_string())?;

    Ok(json_value)
}
