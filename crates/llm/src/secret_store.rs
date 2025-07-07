use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::PathBuf;

/// Key used to store the default provider in the secret store
const DEFAULT_PROVIDER_KEY: &str = "default";

/// A secure storage for API keys and other sensitive information
///
/// Provides functionality to store, retrieve, and manage secrets
/// in a JSON file located in the user's home directory.
#[derive(Debug, Serialize, Deserialize)]
pub struct SecretStore {
    /// Map of secret keys to their values
    secrets: HashMap<String, String>,
    /// Path to the secrets file
    file_path: PathBuf,
}

impl SecretStore {
    /// Creates a new SecretStore instance
    ///
    /// Initializes the store with the default path (~/.llm/secrets.json)
    /// and loads any existing secrets from the file.
    ///
    /// # Returns
    ///
    /// * `io::Result<Self>` - A new SecretStore instance or an IO error
    pub fn new() -> io::Result<Self> {
        let home_dir = dirs::home_dir().expect("Could not find home directory");
        let file_path = home_dir.join(".llm").join("secrets.json");

        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut store = SecretStore {
            secrets: HashMap::new(),
            file_path,
        };

        store.load()?;
        Ok(store)
    }

    /// Loads secrets from the file system
    ///
    /// # Returns
    ///
    /// * `io::Result<()>` - Success or an IO error
    fn load(&mut self) -> io::Result<()> {
        match File::open(&self.file_path) {
            Ok(mut file) => {
                let mut contents = String::new();
                file.read_to_string(&mut contents)?;
                self.secrets = serde_json::from_str(&contents).unwrap_or_default();
                Ok(())
            }
            Err(ref e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Saves the current secrets to the file system
    ///
    /// # Returns
    ///
    /// * `io::Result<()>` - Success or an IO error
    fn save(&self) -> io::Result<()> {
        let contents = serde_json::to_string_pretty(&self.secrets)?;
        let mut file = File::create(&self.file_path)?;
        file.write_all(contents.as_bytes())?;
        Ok(())
    }

    /// Sets a secret value for the given key
    ///
    /// # Arguments
    ///
    /// * `key` - The key to store the secret under
    /// * `value` - The secret value to store
    ///
    /// # Returns
    ///
    /// * `io::Result<()>` - Success or an IO error
    pub fn set(&mut self, key: &str, value: &str) -> io::Result<()> {
        self.secrets.insert(key.to_string(), value.to_string());
        self.save()
    }

    /// Retrieves a secret value for the given key
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up
    ///
    /// # Returns
    ///
    /// * `Option<&String>` - The secret value if found, or None
    pub fn get(&self, key: &str) -> Option<&String> {
        self.secrets.get(key)
    }

    /// Deletes a secret with the given key
    ///
    /// # Arguments
    ///
    /// * `key` - The key to delete
    ///
    /// # Returns
    ///
    /// * `io::Result<()>` - Success or an IO error
    pub fn delete(&mut self, key: &str) -> io::Result<()> {
        self.secrets.remove(key);
        self.save()
    }

    /// Sets the default provider for LLM interactions
    ///
    /// # Arguments
    ///
    /// * `provider` - The provider string in format "provider:model"
    ///
    /// # Returns
    ///
    /// * `io::Result<()>` - Success or an IO error
    pub fn set_default_provider(&mut self, provider: &str) -> io::Result<()> {
        self.secrets
            .insert(DEFAULT_PROVIDER_KEY.to_string(), provider.to_string());
        self.save()
    }

    /// Retrieves the default provider for LLM interactions
    ///
    /// # Returns
    ///
    /// * `Option<&String>` - The default provider if set, or None
    pub fn get_default_provider(&self) -> Option<&String> {
        self.secrets.get(DEFAULT_PROVIDER_KEY)
    }

    /// Deletes the default provider setting
    ///
    /// # Returns
    ///
    /// * `io::Result<()>` - Success or an IO error
    pub fn delete_default_provider(&mut self) -> io::Result<()> {
        self.secrets.remove(DEFAULT_PROVIDER_KEY);
        self.save()
    }
}
