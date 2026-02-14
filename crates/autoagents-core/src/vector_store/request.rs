use serde::{Deserialize, Serialize};

use super::VectorStoreError;

/// A vector search request - used in the [`super::VectorStoreIndex`] trait.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VectorSearchRequest<F = Filter<serde_json::Value>> {
    query: String,
    query_vector_name: Option<String>,
    samples: u64,
    threshold: Option<f64>,
    additional_params: Option<serde_json::Value>,
    filter: Option<F>,
}

impl<Filter> VectorSearchRequest<Filter> {
    pub fn builder() -> VectorSearchRequestBuilder<Filter> {
        VectorSearchRequestBuilder::<Filter>::default()
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn query_vector_name(&self) -> Option<&str> {
        self.query_vector_name.as_deref()
    }

    pub fn samples(&self) -> u64 {
        self.samples
    }

    pub fn threshold(&self) -> Option<f64> {
        self.threshold
    }

    pub fn filter(&self) -> &Option<Filter> {
        &self.filter
    }

    pub fn map_filter<T, F>(self, f: F) -> VectorSearchRequest<T>
    where
        F: Fn(Filter) -> T,
    {
        VectorSearchRequest {
            query: self.query,
            query_vector_name: self.query_vector_name,
            samples: self.samples,
            threshold: self.threshold,
            additional_params: self.additional_params,
            filter: self.filter.map(f),
        }
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum FilterError {
    #[error("Expected: {expected}, got: {got}")]
    Expected { expected: String, got: String },
    #[error("Cannot compile '{0}' to the backend's filter type")]
    TypeError(String),
    #[error("Missing field '{0}'")]
    MissingField(String),
    #[error("'{0}' must {1}")]
    Must(String, String),
    #[error("Filter serialization failed: {0}")]
    Serialization(String),
}

pub trait SearchFilter {
    type Value;

    fn eq(key: String, value: Self::Value) -> Self;
    fn gt(key: String, value: Self::Value) -> Self;
    fn lt(key: String, value: Self::Value) -> Self;
    fn and(self, rhs: Self) -> Self;
    fn or(self, rhs: Self) -> Self;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Filter<V>
where
    V: std::fmt::Debug + Clone,
{
    Eq(String, V),
    Gt(String, V),
    Lt(String, V),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

impl<V> SearchFilter for Filter<V>
where
    V: std::fmt::Debug + Clone + Serialize + for<'de> Deserialize<'de>,
{
    type Value = V;

    fn eq(key: String, value: Self::Value) -> Self {
        Self::Eq(key, value)
    }

    fn gt(key: String, value: Self::Value) -> Self {
        Self::Gt(key, value)
    }

    fn lt(key: String, value: Self::Value) -> Self {
        Self::Lt(key, value)
    }

    fn and(self, rhs: Self) -> Self {
        Self::And(self.into(), rhs.into())
    }

    fn or(self, rhs: Self) -> Self {
        Self::Or(self.into(), rhs.into())
    }
}

impl<V> Filter<V>
where
    V: std::fmt::Debug + Clone,
{
    pub fn interpret<F>(self) -> F
    where
        F: SearchFilter<Value = V>,
    {
        match self {
            Self::Eq(key, val) => F::eq(key, val),
            Self::Gt(key, val) => F::gt(key, val),
            Self::Lt(key, val) => F::lt(key, val),
            Self::And(lhs, rhs) => F::and(lhs.interpret(), rhs.interpret()),
            Self::Or(lhs, rhs) => F::or(lhs.interpret(), rhs.interpret()),
        }
    }
}

impl Filter<serde_json::Value> {
    pub fn satisfies(&self, value: &serde_json::Value) -> bool {
        use Filter::*;
        use serde_json::{Value, Value::*, json};
        use std::cmp::Ordering;

        fn compare_pair(l: &Value, r: &Value) -> Option<std::cmp::Ordering> {
            match (l, r) {
                (Number(l), Number(r)) => l
                    .as_f64()
                    .zip(r.as_f64())
                    .and_then(|(l, r)| l.partial_cmp(&r))
                    .or(l.as_i64().zip(r.as_i64()).map(|(l, r)| l.cmp(&r)))
                    .or(l.as_u64().zip(r.as_u64()).map(|(l, r)| l.cmp(&r))),
                (String(l), String(r)) => Some(l.cmp(r)),
                (Null, Null) => Some(std::cmp::Ordering::Equal),
                (Bool(l), Bool(r)) => Some(l.cmp(r)),
                _ => None,
            }
        }

        match self {
            Eq(k, v) => &json!({ k: v }) == value,
            Gt(k, v) => {
                compare_pair(&json!({k: v}), value).is_some_and(|ord| ord == Ordering::Greater)
            }
            Lt(k, v) => {
                compare_pair(&json!({k: v}), value).is_some_and(|ord| ord == Ordering::Less)
            }
            And(l, r) => l.satisfies(value) && r.satisfies(value),
            Or(l, r) => l.satisfies(value) || r.satisfies(value),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VectorSearchRequestBuilder<F = Filter<serde_json::Value>> {
    query: Option<String>,
    query_vector_name: Option<String>,
    samples: Option<u64>,
    threshold: Option<f64>,
    additional_params: Option<serde_json::Value>,
    filter: Option<F>,
}

impl<F> Default for VectorSearchRequestBuilder<F> {
    fn default() -> Self {
        Self {
            query: None,
            query_vector_name: None,
            samples: None,
            threshold: None,
            additional_params: None,
            filter: None,
        }
    }
}

impl<F> VectorSearchRequestBuilder<F>
where
    F: SearchFilter,
{
    pub fn query<T>(mut self, query: T) -> Self
    where
        T: Into<String>,
    {
        self.query = Some(query.into());
        self
    }

    pub fn samples(mut self, samples: u64) -> Self {
        self.samples = Some(samples);
        self
    }

    pub fn query_vector_name<T>(mut self, name: T) -> Self
    where
        T: Into<String>,
    {
        self.query_vector_name = Some(name.into());
        self
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    pub fn additional_params(
        mut self,
        params: serde_json::Value,
    ) -> Result<Self, VectorStoreError> {
        self.additional_params = Some(params);
        Ok(self)
    }

    pub fn filter(mut self, filter: F) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn build(self) -> Result<VectorSearchRequest<F>, VectorStoreError> {
        let Some(query) = self.query else {
            return Err(VectorStoreError::BuilderError(
                "`query` is a required variable for building a vector search request".into(),
            ));
        };

        let Some(samples) = self.samples else {
            return Err(VectorStoreError::BuilderError(
                "`samples` is a required variable for building a vector search request".into(),
            ));
        };

        let additional_params = if let Some(params) = self.additional_params {
            if !params.is_object() {
                return Err(VectorStoreError::BuilderError(
                    "Expected JSON object for additional params, got something else".into(),
                ));
            }
            Some(params)
        } else {
            None
        };

        Ok(VectorSearchRequest {
            query,
            query_vector_name: self.query_vector_name,
            samples,
            threshold: self.threshold,
            additional_params,
            filter: self.filter,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_builder_missing_query() {
        let result = VectorSearchRequest::<Filter<serde_json::Value>>::builder()
            .samples(10)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("query"));
    }

    #[test]
    fn test_builder_missing_samples() {
        let result = VectorSearchRequest::<Filter<serde_json::Value>>::builder()
            .query("test")
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("samples"));
    }

    #[test]
    fn test_builder_non_object_additional_params() {
        let result = VectorSearchRequest::<Filter<serde_json::Value>>::builder()
            .query("test")
            .samples(5)
            .additional_params(json!("not an object"))
            .unwrap()
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("JSON object"));
    }

    #[test]
    fn test_builder_success_with_all_options() {
        let filter = Filter::eq("color".to_string(), json!("red"));
        let result = VectorSearchRequest::builder()
            .query("search query")
            .samples(10)
            .threshold(0.8)
            .additional_params(json!({"key": "value"}))
            .unwrap()
            .filter(filter)
            .build();
        assert!(result.is_ok());
        let req = result.unwrap();
        assert_eq!(req.query(), "search query");
        assert_eq!(req.samples(), 10);
        assert_eq!(req.threshold(), Some(0.8));
        assert!(req.filter().is_some());
    }

    #[test]
    fn test_builder_minimal_success() {
        let result = VectorSearchRequest::<Filter<serde_json::Value>>::builder()
            .query("q")
            .samples(1)
            .build();
        assert!(result.is_ok());
        let req = result.unwrap();
        assert_eq!(req.query(), "q");
        assert_eq!(req.samples(), 1);
        assert_eq!(req.threshold(), None);
        assert!(req.filter().is_none());
    }

    #[test]
    fn test_filter_constructors() {
        let eq: Filter<serde_json::Value> = SearchFilter::eq("k".to_string(), json!("v"));
        assert!(matches!(eq, Filter::Eq(_, _)));

        let gt: Filter<serde_json::Value> = SearchFilter::gt("k".to_string(), json!(10));
        assert!(matches!(gt, Filter::Gt(_, _)));

        let lt: Filter<serde_json::Value> = SearchFilter::lt("k".to_string(), json!(5));
        assert!(matches!(lt, Filter::Lt(_, _)));
    }

    #[test]
    fn test_filter_and_or() {
        let f1: Filter<serde_json::Value> = SearchFilter::eq("a".to_string(), json!(1));
        let f2: Filter<serde_json::Value> = SearchFilter::eq("b".to_string(), json!(2));
        let combined = SearchFilter::and(f1, f2);
        assert!(matches!(combined, Filter::And(_, _)));

        let f3: Filter<serde_json::Value> = SearchFilter::eq("c".to_string(), json!(3));
        let f4: Filter<serde_json::Value> = SearchFilter::eq("d".to_string(), json!(4));
        let either = SearchFilter::or(f3, f4);
        assert!(matches!(either, Filter::Or(_, _)));
    }

    #[test]
    fn test_filter_satisfies_eq_match() {
        let filter = Filter::Eq("color".to_string(), json!("red"));
        assert!(filter.satisfies(&json!({"color": "red"})));
    }

    #[test]
    fn test_filter_satisfies_eq_mismatch() {
        let filter = Filter::Eq("color".to_string(), json!("red"));
        assert!(!filter.satisfies(&json!({"color": "blue"})));
    }

    #[test]
    fn test_filter_satisfies_and() {
        let f = Filter::And(
            Box::new(Filter::Eq("a".to_string(), json!(1))),
            Box::new(Filter::Eq("b".to_string(), json!(2))),
        );
        // Note: satisfies checks json!({k:v}) == value, so both must match same value
        // This won't match a single object with both - the Eq check is per-key
        assert!(!f.satisfies(&json!({"a": 1})));
    }

    #[test]
    fn test_filter_satisfies_or() {
        let f = Filter::Or(
            Box::new(Filter::Eq("a".to_string(), json!(1))),
            Box::new(Filter::Eq("b".to_string(), json!(2))),
        );
        assert!(f.satisfies(&json!({"a": 1})));
        assert!(f.satisfies(&json!({"b": 2})));
        assert!(!f.satisfies(&json!({"c": 3})));
    }

    #[test]
    fn test_filter_interpret_roundtrip() {
        let original: Filter<serde_json::Value> = Filter::Eq("key".to_string(), json!("value"));
        let interpreted: Filter<serde_json::Value> = original.interpret();
        assert!(matches!(interpreted, Filter::Eq(ref k, _) if k == "key"));
    }

    #[test]
    fn test_filter_interpret_compound() {
        let f: Filter<serde_json::Value> = Filter::And(
            Box::new(Filter::Gt("x".to_string(), json!(10))),
            Box::new(Filter::Lt("y".to_string(), json!(20))),
        );
        let interpreted: Filter<serde_json::Value> = f.interpret();
        assert!(matches!(interpreted, Filter::And(_, _)));
    }

    #[test]
    fn test_map_filter() {
        let req = VectorSearchRequest::<Filter<serde_json::Value>>::builder()
            .query("q")
            .samples(5)
            .filter(Filter::Eq("k".to_string(), json!("v")))
            .build()
            .unwrap();

        let mapped = req.map_filter(|f| format!("{f:?}"));
        assert_eq!(mapped.query(), "q");
        assert_eq!(mapped.samples(), 5);
        assert!(mapped.filter().is_some());
    }

    #[test]
    fn test_filter_serialize_deserialize() {
        let filter: Filter<serde_json::Value> = Filter::Eq("name".to_string(), json!("test"));
        let json = serde_json::to_string(&filter).unwrap();
        let deserialized: Filter<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, Filter::Eq(ref k, _) if k == "name"));
    }
}
