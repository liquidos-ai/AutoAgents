use super::Tokenizer;

#[derive(Clone)]
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn new(_tokenizer_path: &str) -> Result<Self, String>
    where
        Self: Sized,
    {
        Ok(ByteTokenizer)
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        let bos_token = if bos {
            self.encode("[bos]", false, false)
        } else {
            vec![]
        };
        let eos_token = if eos {
            self.encode("[end]", false, false)
        } else {
            vec![]
        };
        let content = text.bytes().map(|b| b as u32);

        bos_token
            .into_iter()
            .chain(content)
            .chain(eos_token)
            .collect()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        format!("{tokens:?}")
    }

    fn bos_id(&self) -> u32 {
        0
    }

    fn eos_id(&self) -> u32 {
        1
    }

    fn stop_ids(&self) -> Vec<u32> {
        vec![2]
    }
}
