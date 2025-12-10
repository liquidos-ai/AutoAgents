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
