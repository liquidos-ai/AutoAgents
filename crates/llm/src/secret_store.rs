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
        // Ensure parent directory exists
        if let Some(parent) = self.file_path.parent() {
            fs::create_dir_all(parent)?;
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn create_temp_secret_store() -> (SecretStore, PathBuf) {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("secrets.json");

        let store = SecretStore {
            secrets: HashMap::new(),
            file_path: file_path.clone(),
        };

        (store, file_path)
    }

    #[test]
    fn test_secret_store_creation() {
        let (store, _) = create_temp_secret_store();
        assert!(store.secrets.is_empty());
        assert_eq!(store.size(), 0);
    }

    #[test]
    fn test_secret_store_set_and_get() {
        let (mut store, _) = create_temp_secret_store();

        let result = store.set("test_key", "test_value");
        assert!(result.is_ok());

        let value = store.get("test_key");
        assert_eq!(value, Some(&"test_value".to_string()));
    }

    #[test]
    fn test_secret_store_get_nonexistent() {
        let (store, _) = create_temp_secret_store();

        let value = store.get("nonexistent_key");
        assert!(value.is_none());
    }

    #[test]
    fn test_secret_store_delete() {
        let (mut store, _) = create_temp_secret_store();

        store.set("test_key", "test_value").unwrap();
        assert!(store.get("test_key").is_some());

        let result = store.delete("test_key");
        assert!(result.is_ok());
        assert!(store.get("test_key").is_none());
    }

    #[test]
    fn test_secret_store_delete_nonexistent() {
        let (mut store, _) = create_temp_secret_store();

        let result = store.delete("nonexistent_key");
        assert!(result.is_ok());
    }

    #[test]
    fn test_secret_store_set_default_provider() {
        let (mut store, _) = create_temp_secret_store();

        let result = store.set_default_provider("openai:gpt-4");
        assert!(result.is_ok());

        let provider = store.get_default_provider();
        assert_eq!(provider, Some(&"openai:gpt-4".to_string()));
    }

    #[test]
    fn test_secret_store_get_default_provider_none() {
        let (store, _) = create_temp_secret_store();

        let provider = store.get_default_provider();
        assert!(provider.is_none());
    }

    #[test]
    fn test_secret_store_delete_default_provider() {
        let (mut store, _) = create_temp_secret_store();

        store.set_default_provider("openai:gpt-4").unwrap();
        assert!(store.get_default_provider().is_some());

        let result = store.delete_default_provider();
        assert!(result.is_ok());
        assert!(store.get_default_provider().is_none());
    }

    #[test]
    fn test_secret_store_multiple_keys() {
        let (mut store, _) = create_temp_secret_store();

        store.set("key1", "value1").unwrap();
        store.set("key2", "value2").unwrap();
        store.set("key3", "value3").unwrap();

        assert_eq!(store.get("key1"), Some(&"value1".to_string()));
        assert_eq!(store.get("key2"), Some(&"value2".to_string()));
        assert_eq!(store.get("key3"), Some(&"value3".to_string()));
        assert_eq!(store.size(), 3);
    }

    #[test]
    fn test_secret_store_overwrite_key() {
        let (mut store, _) = create_temp_secret_store();

        store.set("key", "original_value").unwrap();
        assert_eq!(store.get("key"), Some(&"original_value".to_string()));

        store.set("key", "new_value").unwrap();
        assert_eq!(store.get("key"), Some(&"new_value".to_string()));
        assert_eq!(store.size(), 1);
    }

    #[test]
    fn test_secret_store_empty_key() {
        let (mut store, _) = create_temp_secret_store();

        let result = store.set("", "empty_key_value");
        assert!(result.is_ok());

        let value = store.get("");
        assert_eq!(value, Some(&"empty_key_value".to_string()));
    }

    #[test]
    fn test_secret_store_empty_value() {
        let (mut store, _) = create_temp_secret_store();

        let result = store.set("empty_value_key", "");
        assert!(result.is_ok());

        let value = store.get("empty_value_key");
        assert_eq!(value, Some(&"".to_string()));
    }

    #[test]
    fn test_secret_store_unicode_keys_and_values() {
        let (mut store, _) = create_temp_secret_store();

        let result = store.set("キー", "値");
        assert!(result.is_ok());

        let value = store.get("キー");
        assert_eq!(value, Some(&"値".to_string()));
    }

    #[test]
    fn test_secret_store_special_characters() {
        let (mut store, _) = create_temp_secret_store();

        let key = "special!@#$%^&*()_+-=[]{}|;':\",./<>?";
        let value = "value with spaces and symbols !@#$%^&*()";

        let result = store.set(key, value);
        assert!(result.is_ok());

        let retrieved = store.get(key);
        assert_eq!(retrieved, Some(&value.to_string()));
    }

    #[test]
    fn test_secret_store_large_value() {
        let (mut store, _) = create_temp_secret_store();

        let large_value = "x".repeat(10000);
        let result = store.set("large_key", &large_value);
        assert!(result.is_ok());

        let value = store.get("large_key");
        assert_eq!(value, Some(&large_value));
    }

    #[test]
    fn test_secret_store_persistence() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_secrets.json");

        // Create first store instance and add data
        {
            let mut store = SecretStore {
                secrets: HashMap::new(),
                file_path: file_path.clone(),
            };
            store.set("persistent_key", "persistent_value").unwrap();
            store.set_default_provider("anthropic:claude-3").unwrap();
        }

        // Create second store instance and verify data persists
        {
            let mut store = SecretStore {
                secrets: HashMap::new(),
                file_path: file_path.clone(),
            };
            store.load().unwrap();

            assert_eq!(
                store.get("persistent_key"),
                Some(&"persistent_value".to_string())
            );
            assert_eq!(
                store.get_default_provider(),
                Some(&"anthropic:claude-3".to_string())
            );
        }
    }

    #[test]
    fn test_secret_store_load_nonexistent_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("nonexistent.json");

        let mut store = SecretStore {
            secrets: HashMap::new(),
            file_path,
        };

        let result = store.load();
        assert!(result.is_ok());
        assert!(store.secrets.is_empty());
    }

    #[test]
    fn test_secret_store_load_invalid_json() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("invalid.json");

        // Write invalid JSON to file
        fs::write(&file_path, "invalid json content").unwrap();

        let mut store = SecretStore {
            secrets: HashMap::new(),
            file_path,
        };

        let result = store.load();
        assert!(result.is_ok());
        assert!(store.secrets.is_empty()); // Should default to empty HashMap
    }

    #[test]
    fn test_secret_store_load_empty_json() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("empty.json");

        // Write empty JSON object to file
        fs::write(&file_path, "{}").unwrap();

        let mut store = SecretStore {
            secrets: HashMap::new(),
            file_path,
        };

        let result = store.load();
        assert!(result.is_ok());
        assert!(store.secrets.is_empty());
    }

    #[test]
    fn test_secret_store_save_and_load_cycle() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("cycle_test.json");

        let mut store = SecretStore {
            secrets: HashMap::new(),
            file_path: file_path.clone(),
        };

        // Add multiple secrets
        store.set("api_key", "secret123").unwrap();
        store.set("username", "testuser").unwrap();
        store.set_default_provider("google:gemini-pro").unwrap();

        // Clear secrets and reload
        store.secrets.clear();
        store.load().unwrap();

        // Verify all data was restored
        assert_eq!(store.get("api_key"), Some(&"secret123".to_string()));
        assert_eq!(store.get("username"), Some(&"testuser".to_string()));
        assert_eq!(
            store.get_default_provider(),
            Some(&"google:gemini-pro".to_string())
        );
    }

    #[test]
    fn test_secret_store_concurrent_operations() {
        let (mut store, _) = create_temp_secret_store();

        // Simulate concurrent-like operations
        store.set("key1", "value1").unwrap();
        store.set("key2", "value2").unwrap();
        store.delete("key1").unwrap();
        store.set("key3", "value3").unwrap();
        store.set_default_provider("provider1").unwrap();
        store.delete_default_provider().unwrap();
        store.set_default_provider("provider2").unwrap();

        assert!(store.get("key1").is_none());
        assert_eq!(store.get("key2"), Some(&"value2".to_string()));
        assert_eq!(store.get("key3"), Some(&"value3".to_string()));
        assert_eq!(store.get_default_provider(), Some(&"provider2".to_string()));
    }

    #[test]
    fn test_secret_store_debug_impl() {
        let (store, _) = create_temp_secret_store();
        let debug_str = format!("{store:?}");
        assert!(debug_str.contains("SecretStore"));
    }

    #[test]
    fn test_secret_store_serialize_deserialize() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("serialize_test.json");

        let mut store = SecretStore {
            secrets: HashMap::new(),
            file_path: file_path.clone(),
        };

        // Add complex data
        store
            .set("nested_json", r#"{"key": "value", "array": [1, 2, 3]}"#)
            .unwrap();
        store.set("multiline", "line1\nline2\nline3").unwrap();

        // Verify file was created and contains valid JSON
        assert!(file_path.exists());
        let file_content = fs::read_to_string(&file_path).unwrap();
        let _parsed: serde_json::Value = serde_json::from_str(&file_content).unwrap();
    }

    #[test]
    fn test_secret_store_default_provider_key_constant() {
        let (mut store, _) = create_temp_secret_store();

        // Set using the constant
        store.set(DEFAULT_PROVIDER_KEY, "test_provider").unwrap();

        // Get using the method
        let provider = store.get_default_provider();
        assert_eq!(provider, Some(&"test_provider".to_string()));
    }

    #[test]
    fn test_secret_store_size_method() {
        let (mut store, _) = create_temp_secret_store();

        assert_eq!(store.size(), 0);

        store.set("key1", "value1").unwrap();
        assert_eq!(store.size(), 1);

        store.set("key2", "value2").unwrap();
        assert_eq!(store.size(), 2);

        store.delete("key1").unwrap();
        assert_eq!(store.size(), 1);
    }

    #[test]
    fn test_secret_store_is_empty() {
        let (mut store, _) = create_temp_secret_store();

        assert!(store.is_empty());

        store.set("key", "value").unwrap();
        assert!(!store.is_empty());

        store.delete("key").unwrap();
        assert!(store.is_empty());
    }

    #[test]
    fn test_secret_store_contains_key() {
        let (mut store, _) = create_temp_secret_store();

        assert!(!store.contains_key("test_key"));

        store.set("test_key", "test_value").unwrap();
        assert!(store.contains_key("test_key"));

        store.delete("test_key").unwrap();
        assert!(!store.contains_key("test_key"));
    }

    #[test]
    fn test_secret_store_keys() {
        let (mut store, _) = create_temp_secret_store();

        store.set("key1", "value1").unwrap();
        store.set("key2", "value2").unwrap();
        store.set("key3", "value3").unwrap();

        let keys: Vec<_> = store.keys().collect();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&&"key1".to_string()));
        assert!(keys.contains(&&"key2".to_string()));
        assert!(keys.contains(&&"key3".to_string()));
    }

    #[test]
    fn test_secret_store_values() {
        let (mut store, _) = create_temp_secret_store();

        store.set("key1", "value1").unwrap();
        store.set("key2", "value2").unwrap();
        store.set("key3", "value3").unwrap();

        let values: Vec<_> = store.values().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&&"value1".to_string()));
        assert!(values.contains(&&"value2".to_string()));
        assert!(values.contains(&&"value3".to_string()));
    }

    #[test]
    fn test_secret_store_clear() {
        let (mut store, _) = create_temp_secret_store();

        store.set("key1", "value1").unwrap();
        store.set("key2", "value2").unwrap();
        store.set_default_provider("provider").unwrap();

        assert_eq!(store.size(), 3);

        store.clear().unwrap();
        assert_eq!(store.size(), 0);
        assert!(store.is_empty());
    }

    // Helper methods for testing
    impl SecretStore {
        fn size(&self) -> usize {
            self.secrets.len()
        }

        fn is_empty(&self) -> bool {
            self.secrets.is_empty()
        }

        fn contains_key(&self, key: &str) -> bool {
            self.secrets.contains_key(key)
        }

        fn keys(&self) -> std::collections::hash_map::Keys<String, String> {
            self.secrets.keys()
        }

        fn values(&self) -> std::collections::hash_map::Values<String, String> {
            self.secrets.values()
        }

        fn clear(&mut self) -> io::Result<()> {
            self.secrets.clear();
            self.save()
        }
    }
}
