use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    static ref WHITESPACE_RE: Regex = Regex::new(r"[^\S \n]").unwrap();
    static ref MULTI_SPACE_RE: Regex = Regex::new(r"  +").unwrap();
    static ref NEWLINE_SPACE_RE: Regex = Regex::new(r"(?<=\n) +(?=\n)").unwrap();
    static ref DOCTOR_RE: Regex = Regex::new(r"\bD[Rr]\.(?= [A-Z])").unwrap();
    static ref MISTER_RE: Regex = Regex::new(r"\b(?:Mr\.|MR\.(?= [A-Z]))").unwrap();
    static ref MISS_RE: Regex = Regex::new(r"\b(?:Ms\.|MS\.(?= [A-Z]))").unwrap();
    static ref MRS_RE: Regex = Regex::new(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))").unwrap();
    static ref ETC_RE: Regex = Regex::new(r"\betc\.(?! [A-Z])").unwrap();
    static ref YEAH_RE: Regex = Regex::new(r"(?i)\b(y)eah?\b").unwrap();
    static ref NUMBERS_RE: Regex =
        Regex::new(r"\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)").unwrap();
    static ref COMMA_NUM_RE: Regex = Regex::new(r"(?<=\d),(?=\d)").unwrap();
    static ref MONEY_RE: Regex = Regex::new(
        r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b"
    )
    .unwrap();
    static ref POINT_NUM_RE: Regex = Regex::new(r"\d*\.\d+").unwrap();
    static ref RANGE_RE: Regex = Regex::new(r"(?<=\d)-(?=\d)").unwrap();
    static ref S_AFTER_NUM_RE: Regex = Regex::new(r"(?<=\d)S").unwrap();
    static ref POSSESSIVE_RE: Regex = Regex::new(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b").unwrap();
    static ref X_POSSESSIVE_RE: Regex = Regex::new(r"(?<=X')S\b").unwrap();
    static ref INITIALS_RE: Regex = Regex::new(r"(?:[A-Za-z]\.){2,} [a-z]").unwrap();
    static ref ACRONYM_RE: Regex = Regex::new(r"(?i)(?<=[A-Z])\.(?=[A-Z])").unwrap();
}

pub fn normalize_text(text: &str) -> String {
    let mut text = text.to_string();

    // Replace special quotes and brackets
    text = text.replace('\u{2018}', "'").replace('\u{2019}', "'");
    text = text.replace('«', "\u{201C}").replace('»', "\u{201D}");
    text = text.replace('\u{201C}', "\"").replace('\u{201D}', "\"");
    text = text.replace('(', "«").replace(')', "»");

    // Replace Chinese/Japanese punctuation
    let from_chars = ['、', '。', '！', '，', '：', '；', '？'];
    let to_chars = [',', '.', '!', ',', ':', ';', '?'];

    for (from, to) in from_chars.iter().zip(to_chars.iter()) {
        text = text.replace(*from, &format!("{} ", to));
    }

    // Apply regex replacements
    text = WHITESPACE_RE.replace_all(&text, " ").to_string();
    text = MULTI_SPACE_RE.replace_all(&text, " ").to_string();
    text = NEWLINE_SPACE_RE.replace_all(&text, "").to_string();
    text = DOCTOR_RE.replace_all(&text, "Doctor").to_string();
    text = MISTER_RE.replace_all(&text, "Mister").to_string();
    text = MISS_RE.replace_all(&text, "Miss").to_string();
    text = MRS_RE.replace_all(&text, "Mrs").to_string();
    text = ETC_RE.replace_all(&text, "etc").to_string();
    text = YEAH_RE.replace_all(&text, "${1}e'a").to_string();
    // Note: split_num, flip_money, and point_num functions need to be implemented
    text = COMMA_NUM_RE.replace_all(&text, "").to_string();
    text = RANGE_RE.replace_all(&text, " to ").to_string();
    text = S_AFTER_NUM_RE.replace_all(&text, " S").to_string();
    text = POSSESSIVE_RE.replace_all(&text, "'S").to_string();
    text = X_POSSESSIVE_RE.replace_all(&text, "s").to_string();

    // Handle initials and acronyms
    text = INITIALS_RE
        .replace_all(&text, |caps: &regex::Captures| caps[0].replace('.', "-"))
        .to_string();
    text = ACRONYM_RE.replace_all(&text, "-").to_string();

    text.trim().to_string()
}
