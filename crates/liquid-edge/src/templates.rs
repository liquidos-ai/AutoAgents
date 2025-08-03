//! Template rendering module for liquid-edge
//!
//! This module provides basic template rendering capabilities using the minijinja
//! crate when the `jinja-templates` feature is enabled.

use crate::error::{EdgeError, EdgeResult};
use std::collections::HashMap;

#[cfg(feature = "jinja-templates")]
use minijinja::Environment;

/// Template renderer for processing Jinja2-style templates
#[cfg(feature = "jinja-templates")]
pub struct TemplateRenderer {
    // Keep it simple - create environment per render to avoid lifetime issues
}

#[cfg(feature = "jinja-templates")]
impl TemplateRenderer {
    /// Create a new template renderer
    pub fn new() -> EdgeResult<Self> {
        Ok(Self {})
    }

    /// Render a template string with the given context
    pub fn render(
        &self,
        template_str: &str,
        context: &HashMap<String, serde_json::Value>,
    ) -> EdgeResult<String> {
        let env = Environment::new();

        let mut context_map = HashMap::new();

        for (key, value) in context {
            let jinja_value = Self::convert_json_to_jinja(value)?;
            context_map.insert(key.clone(), jinja_value);
        }

        // Convert serde_json::Value to minijinja::Value
        let jinja_context = minijinja::Value::from_object(context_map);

        let template = env
            .template_from_str(template_str)
            .map_err(|e| EdgeError::template(format!("Failed to parse template: {e}")))?;

        template
            .render(&jinja_context)
            .map_err(|e| EdgeError::template(format!("Failed to render template: {e}")))
    }

    /// Load template from file
    pub fn load_template<P: AsRef<std::path::Path>>(
        &mut self,
        _name: &str,
        path: P,
    ) -> EdgeResult<()> {
        let _content = std::fs::read_to_string(path)
            .map_err(|e| EdgeError::template(format!("Failed to read template file: {e}")))?;
        // Simplified: would store template for later use
        Ok(())
    }

    /// Convert serde_json::Value to minijinja::Value
    fn convert_json_to_jinja(value: &serde_json::Value) -> EdgeResult<minijinja::Value> {
        match value {
            serde_json::Value::Null => Ok(minijinja::Value::UNDEFINED),
            serde_json::Value::Bool(b) => Ok(minijinja::Value::from(*b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(minijinja::Value::from(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(minijinja::Value::from(f))
                } else {
                    Ok(minijinja::Value::from(n.to_string()))
                }
            }
            serde_json::Value::String(s) => Ok(minijinja::Value::from(s.as_str())),
            serde_json::Value::Array(arr) => {
                let mut vec = Vec::new();
                for item in arr {
                    vec.push(Self::convert_json_to_jinja(item)?);
                }
                Ok(minijinja::Value::from(vec))
            }
            serde_json::Value::Object(obj) => {
                let mut map = HashMap::new();
                for (key, val) in obj {
                    map.insert(key.clone(), Self::convert_json_to_jinja(val)?);
                }
                Ok(minijinja::Value::from_object(map))
            }
        }
    }

    /// Render a chat template with messages
    #[cfg(feature = "chat")]
    pub fn render_chat_template(
        &self,
        template_str: &str,
        messages: &[crate::traits::ChatMessage],
        add_generation_prompt: bool,
    ) -> EdgeResult<String> {
        let mut context = HashMap::new();

        // Convert messages to JSON format
        let messages_json: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| {
                serde_json::json!({
                    "role": msg.role,
                    "content": msg.content
                })
            })
            .collect();

        context.insert(
            "messages".to_string(),
            serde_json::Value::Array(messages_json),
        );
        context.insert(
            "add_generation_prompt".to_string(),
            serde_json::Value::Bool(add_generation_prompt),
        );

        self.render(template_str, &context)
    }
}

#[cfg(feature = "jinja-templates")]
impl Default for TemplateRenderer {
    fn default() -> Self {
        Self::new().expect("Failed to create default template renderer")
    }
}

#[cfg(feature = "jinja-templates")]
impl crate::traits::TemplateRenderer for TemplateRenderer {
    fn render(
        &self,
        template: &str,
        context: &HashMap<String, serde_json::Value>,
    ) -> EdgeResult<String> {
        self.render(template, context)
    }

    fn load_template<P: AsRef<std::path::Path>>(&mut self, name: &str, path: P) -> EdgeResult<()> {
        self.load_template(name, path)
    }
}

// Stub implementations when jinja-templates feature is not enabled
#[cfg(not(feature = "jinja-templates"))]
pub struct TemplateRenderer;

#[cfg(not(feature = "jinja-templates"))]
impl TemplateRenderer {
    pub fn new() -> EdgeResult<Self> {
        Err(EdgeError::feature_not_available("jinja-templates"))
    }

    pub fn render(
        &self,
        _template: &str,
        _context: &HashMap<String, serde_json::Value>,
    ) -> EdgeResult<String> {
        Err(EdgeError::feature_not_available("jinja-templates"))
    }

    pub fn set_global(&mut self, _key: &str, _value: serde_json::Value) {
        // Stub - no operation
    }

    pub fn load_template<P: AsRef<std::path::Path>>(
        &mut self,
        _name: &str,
        _path: P,
    ) -> EdgeResult<()> {
        Err(EdgeError::feature_not_available("jinja-templates"))
    }
}

#[cfg(not(feature = "jinja-templates"))]
impl Default for TemplateRenderer {
    fn default() -> Self {
        Self
    }
}

#[cfg(not(feature = "jinja-templates"))]
impl crate::traits::TemplateRenderer for TemplateRenderer {
    fn render(
        &self,
        _template: &str,
        _context: &HashMap<String, serde_json::Value>,
    ) -> EdgeResult<String> {
        Err(EdgeError::feature_not_available("jinja-templates"))
    }

    fn set_global(&mut self, _key: &str, _value: serde_json::Value) {
        // Stub - no operation
    }

    fn load_template<P: AsRef<std::path::Path>>(
        &mut self,
        _name: &str,
        _path: P,
    ) -> EdgeResult<()> {
        Err(EdgeError::feature_not_available("jinja-templates"))
    }
}

/// Template utilities
pub mod utils {
    use super::*;

    /// Validate template syntax
    #[cfg(feature = "jinja-templates")]
    pub fn validate_template(template_str: &str) -> EdgeResult<()> {
        let env = Environment::new();
        env.template_from_str(template_str)
            .map_err(|e| EdgeError::template(format!("Invalid template syntax: {e}")))?;
        Ok(())
    }

    #[cfg(not(feature = "jinja-templates"))]
    pub fn validate_template(_template_str: &str) -> EdgeResult<()> {
        Err(EdgeError::feature_not_available("jinja-templates"))
    }

    /// Extract variables used in a template (simplified implementation)
    pub fn extract_variables(template_str: &str) -> EdgeResult<Vec<String>> {
        let mut variables = Vec::new();
        let mut pos = 0;

        while pos < template_str.len() {
            if let Some(start) = template_str[pos..].find("{{") {
                let abs_start = pos + start + 2;
                if let Some(end) = template_str[abs_start..].find("}}") {
                    let abs_end = abs_start + end;
                    let var_content = template_str[abs_start..abs_end].trim();

                    // Extract variable name (first word before dot or space)
                    let var_name = var_content
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .split('.')
                        .next()
                        .unwrap_or("")
                        .to_string();

                    if !var_name.is_empty()
                        && var_name.chars().all(|c| c.is_alphanumeric() || c == '_')
                        && !variables.contains(&var_name)
                    {
                        variables.push(var_name);
                    }

                    pos = abs_end + 2;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(variables)
    }

    /// Minify template by removing unnecessary whitespace
    pub fn minify_template(template_str: &str) -> String {
        template_str
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
#[cfg(feature = "jinja-templates")]
mod tests {
    use super::*;

    #[test]
    fn test_simple_rendering() {
        let renderer = TemplateRenderer::new().unwrap();
        let mut context = HashMap::new();
        context.insert("name".to_string(), serde_json::json!("World"));

        let result = renderer.render("Hello, {{ name }}!", &context).unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_template_validation() {
        assert!(utils::validate_template("Hello, {{ name }}!").is_ok());
        assert!(utils::validate_template("Hello, {{ name").is_err()); // Invalid syntax
    }

    #[test]
    fn test_variable_extraction() {
        let variables =
            utils::extract_variables("Hello, {{ name }}! Your age is {{ age }}.").unwrap();
        assert!(variables.contains(&"name".to_string()));
        assert!(variables.contains(&"age".to_string()));
        assert_eq!(variables.len(), 2);
    }

    #[test]
    fn test_template_minification() {
        let template = r#"
        {% for item in items %}
            {{ item }}
        {% endfor %}
        "#;

        let minified = utils::minify_template(template);
        assert!(!minified.contains('\n'));
        assert!(minified.len() < template.len());
    }
}
