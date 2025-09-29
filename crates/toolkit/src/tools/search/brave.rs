pub struct BraveSearch {
    api_key: String,
}

impl BraveSearch {
    pub fn new(api_key: String) -> Self {
        BraveSearch { api_key }
    }
}
