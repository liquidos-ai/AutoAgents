use serde::Serialize;
use strum::{Display, EnumString};

#[derive(EnumString, Display, Serialize)]
pub(crate) enum JsonType {
    #[strum(serialize = "string")]
    String,
    // JSON Schema distinguishes whole numbers from general numeric values:
    // use `integer` for Rust integral types and `number` for floating-point types.
    #[strum(serialize = "integer")]
    Integer,
    #[strum(serialize = "number")]
    Number,
    #[strum(serialize = "boolean")]
    Boolean,
    #[strum(serialize = "object")]
    Object,
    #[strum(serialize = "array")]
    Array,
}
