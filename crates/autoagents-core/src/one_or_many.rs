use serde::{Deserialize, Serialize};

/// Stores either a single item or multiple items without forcing a Vec allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    pub fn len(&self) -> usize {
        match self {
            OneOrMany::One(_) => 1,
            OneOrMany::Many(items) => items.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = &T> + '_> {
        match self {
            OneOrMany::One(item) => Box::new(std::iter::once(item)),
            OneOrMany::Many(items) => Box::new(items.iter()),
        }
    }

    pub fn into_vec(self) -> Vec<T> {
        match self {
            OneOrMany::One(item) => vec![item],
            OneOrMany::Many(items) => items,
        }
    }

    pub fn map<U, F>(self, mut f: F) -> OneOrMany<U>
    where
        F: FnMut(T) -> U,
    {
        match self {
            OneOrMany::One(item) => OneOrMany::One(f(item)),
            OneOrMany::Many(items) => OneOrMany::Many(items.into_iter().map(f).collect()),
        }
    }
}

impl<T> From<Vec<T>> for OneOrMany<T> {
    fn from(value: Vec<T>) -> Self {
        if value.len() == 1 {
            OneOrMany::One(value.into_iter().next().unwrap())
        } else {
            OneOrMany::Many(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_len_and_is_empty() {
        let one = OneOrMany::One(42);
        assert_eq!(one.len(), 1);
        assert!(!one.is_empty());
    }

    #[test]
    fn test_many_len_and_is_empty() {
        let many = OneOrMany::Many(vec![1, 2, 3]);
        assert_eq!(many.len(), 3);
        assert!(!many.is_empty());
    }

    #[test]
    fn test_many_empty_is_empty() {
        let many: OneOrMany<i32> = OneOrMany::Many(vec![]);
        assert_eq!(many.len(), 0);
        assert!(many.is_empty());
    }

    #[test]
    fn test_one_iter() {
        let one = OneOrMany::One(42);
        let items: Vec<&i32> = one.iter().collect();
        assert_eq!(items, vec![&42]);
    }

    #[test]
    fn test_many_iter() {
        let many = OneOrMany::Many(vec![1, 2, 3]);
        let items: Vec<&i32> = many.iter().collect();
        assert_eq!(items, vec![&1, &2, &3]);
    }

    #[test]
    fn test_one_into_vec() {
        let one = OneOrMany::One(42);
        assert_eq!(one.into_vec(), vec![42]);
    }

    #[test]
    fn test_many_into_vec() {
        let many = OneOrMany::Many(vec![1, 2, 3]);
        assert_eq!(many.into_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_one_map() {
        let one = OneOrMany::One(2);
        let mapped = one.map(|x| x * 10);
        assert_eq!(mapped.into_vec(), vec![20]);
    }

    #[test]
    fn test_many_map() {
        let many = OneOrMany::Many(vec![1, 2, 3]);
        let mapped = many.map(|x| x + 100);
        assert_eq!(mapped.into_vec(), vec![101, 102, 103]);
    }

    #[test]
    fn test_from_vec_single_becomes_one() {
        let result: OneOrMany<i32> = vec![42].into();
        assert!(matches!(result, OneOrMany::One(42)));
    }

    #[test]
    fn test_from_vec_multi_becomes_many() {
        let result: OneOrMany<i32> = vec![1, 2].into();
        assert!(matches!(result, OneOrMany::Many(_)));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_from_vec_empty_becomes_many() {
        let result: OneOrMany<i32> = vec![].into();
        assert!(matches!(result, OneOrMany::Many(_)));
        assert!(result.is_empty());
    }

    #[test]
    fn test_serialize_deserialize_one() {
        let one = OneOrMany::One("hello".to_string());
        let json = serde_json::to_string(&one).unwrap();
        let deserialized: OneOrMany<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.into_vec(), vec!["hello"]);
    }

    #[test]
    fn test_serialize_deserialize_many() {
        let many = OneOrMany::Many(vec![1, 2, 3]);
        let json = serde_json::to_string(&many).unwrap();
        let deserialized: OneOrMany<i32> = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.into_vec(), vec![1, 2, 3]);
    }
}
