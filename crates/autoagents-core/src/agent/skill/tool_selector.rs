use crate::agent::skill::SkillError;
use std::sync::Arc;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SkillToolSelector {
    expression: Arc<str>,
    tool_name: Arc<str>,
    qualifier: Option<Arc<str>>,
}

impl SkillToolSelector {
    pub fn parse(expression: &str) -> Result<Self, SkillError> {
        let expression = expression.trim();
        if expression.is_empty() || expression.chars().any(char::is_whitespace) {
            return Err(SkillError::InvalidToolSelector(expression.to_string()));
        }

        let (tool_name, qualifier) = match expression.split_once('(') {
            Some((tool_name, qualifier))
                if !tool_name.is_empty()
                    && qualifier.ends_with(')')
                    && qualifier.len() > 1
                    && !qualifier[..qualifier.len() - 1].contains(['(', ')']) =>
            {
                (
                    tool_name,
                    Some(Arc::from(&qualifier[..qualifier.len() - 1])),
                )
            }
            Some(_) => return Err(SkillError::InvalidToolSelector(expression.to_string())),
            None => (expression, None),
        };

        if !tool_name.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.' | ':' | '/')
        }) {
            return Err(SkillError::InvalidToolSelector(expression.to_string()));
        }

        Ok(Self {
            expression: Arc::from(expression),
            tool_name: Arc::from(tool_name),
            qualifier,
        })
    }

    pub fn expression(&self) -> &str {
        &self.expression
    }

    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    pub fn qualifier(&self) -> Option<&str> {
        self.qualifier.as_deref()
    }
}
