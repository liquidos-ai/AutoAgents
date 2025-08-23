use lazy_static::lazy_static;
use std::collections::HashMap;

pub fn get_vocab() -> std::collections::HashMap<char, usize> {
    let pad = "$";
    let punctuation = ";:,.!?¡¿—…\"«»“” ";
    let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

    let symbols: String = [pad, punctuation, letters, letters_ipa].concat();

    symbols
        .chars()
        .enumerate()
        .collect::<HashMap<_, _>>()
        .into_iter()
        .map(|(idx, c)| (c, idx))
        .collect()
}

pub fn get_reverse_vocab() -> HashMap<usize, char> {
    VOCAB.iter().map(|(&c, &idx)| (idx, c)).collect()
}

#[allow(dead_code)]
pub fn print_sorted_reverse_vocab() {
    let mut sorted_keys: Vec<_> = REVERSE_VOCAB.keys().collect();
    sorted_keys.sort();

    for key in sorted_keys {
        eprintln!("{}: {}", key, REVERSE_VOCAB[key]);
    }
}

lazy_static! {
    pub static ref VOCAB: HashMap<char, usize> = get_vocab();
    pub static ref REVERSE_VOCAB: HashMap<usize, char> = get_reverse_vocab();
}
