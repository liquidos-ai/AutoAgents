use serde::Serialize;
use strum::{Display, EnumString};

#[derive(EnumString, Display, Serialize)]
pub(crate) enum JsonType {
    #[strum(serialize = "string")]
    String,
    #[strum(serialize = "number")]
    Number,
    #[strum(serialize = "bool")]
    Boolean,
    #[strum(serialize = "object")]
    Object,
    #[strum(serialize = "array")]
    Array,
}
