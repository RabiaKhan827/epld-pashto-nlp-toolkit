"""Core implementation of PashtoNormalizer, MorphologyAwarePashtoTokenizer, EPLDCorpus, and CrossLingualSentimentPipeline."""
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple

class PashtoNormalizer:
    CHAR_MAP: Dict[str, str] = {
        "ك": "ک", "گ": "ګ", "ى": "ی", "ي": "ی", "ۀ": "ه", "ة": "ه",
        "ٹ": "ټ", "ڈ": "ډ", "ڑ": "ړ", "ں": "ن"
    }
    PASHTO_SPECIFIC_LETTERS = frozenset({"ټ", "ډ", "ړ", "ږ", "ښ", "ګ", "ڼ", "څ", "ځ", "ې", "ۍ"})
    DIACRITICS_PATTERN = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        text = self.DIACRITICS_PATTERN.sub("", text)
        normalized = "".join(self.CHAR_MAP.get(ch, ch) for ch in text)
        normalized = normalized.replace("\u200c", " ")
        return self.WHITESPACE_PATTERN.sub(" ", normalized).strip()

    def pashto_character_ratio(self, text: str) -> float:
        tokens = [t for t in self.normalize(text).split(" ") if t]
        if not tokens:
            return 0.0
        pashto_tokens = sum(1 for tok in tokens if any(ch in self.PASHTO_SPECIFIC_LETTERS for ch in tok))
        return round(pashto_tokens / len(tokens), 4)

@dataclass(frozen=True)
class MorphologicalAnalysis:
    surface_form: str
    stem: str
    prefix: str
    suffix: str
    pos_hint: str
    inflection_feature: str

class MorphologyAwarePashtoTokenizer:
    SUFFIX_RULES: List[Tuple[str, str, str]] = [
        ("وونکو", "NOUN_AGENT", "oblique_plural"),
        ("وونکی", "NOUN_AGENT", "masculine_singular"),
        ("وونکې", "NOUN_AGENT", "feminine_singular"),
        ("ونه", "NOUN", "masculine_inanimate_plural"),
        ("انې", "NOUN", "feminine_animate_plural"),
        ("ګانې", "NOUN", "feminine_plural"),
        ("لې", "VERB_PAST", "feminine_plural_ergative"),
        ("ۍ", "NOUN_FEM", "feminine_singular"),
        ("ې", "INFLECTED", "feminine_or_oblique"),
    ]
    PREFIX_RULES: List[Tuple[str, str]] = [("نه", "NEG"), ("مه", "PROHIB"), ("را", "DIR_PROXIMAL"), ("در", "DIR_MEDIAL"), ("ور", "DIR_DISTAL")]
    ERGATIVE_PRONOUNS = {"ما", "تا", "هغې", "هغوی", "موږ", "تاسو", "دوی"}
    DIRECT_PRONOUNS = {"زه", "ته", "دی", "دا", "هغه"}

    def __init__(self, normalizer: PashtoNormalizer | None = None) -> None:
        self.normalizer = normalizer or PashtoNormalizer()

    def tokenize(self, text: str) -> List[str]:
        cleaned = self.normalizer.normalize(text)
        for p in "؟،؛.!?:()[]":
            cleaned = cleaned.replace(p, f" {p} ")
        return [tok for tok in cleaned.split() if tok]

    def analyze_token(self, token: str) -> MorphologicalAnalysis:
        norm_tok = self.normalizer.normalize(token)
        if norm_tok in self.ERGATIVE_PRONOUNS:
            return MorphologicalAnalysis(norm_tok, norm_tok, "", "", "PRON", "ergative_oblique")
        if norm_tok in self.DIRECT_PRONOUNS:
            return MorphologicalAnalysis(norm_tok, norm_tok, "", "", "PRON", "direct_nominative")
        prefix, stem = "", norm_tok
        for pfx, _ in self.PREFIX_RULES:
            if len(stem) > len(pfx) + 2 and stem.startswith(pfx):
                prefix, stem = pfx, stem[len(pfx):]
                break
        suffix, pos_hint, inflection = "", "LEX", "base"
        for sfx, pos, feat in self.SUFFIX_RULES:
            if len(stem) > len(sfx) + 2 and stem.endswith(sfx):
                suffix, stem, pos_hint, inflection = sfx, stem[:-len(sfx)], pos, feat
                break
        return MorphologicalAnalysis(norm_tok, stem, prefix, suffix, pos_hint, inflection)

class EPLDCorpus:
    METADATA = {"doi": "10.17632/vmgv4s6vrn.1", "title": "English-Pashto Language Dataset (EPLD)"}
    SENTENCES = [
        {"id": 1, "english": "This book is very useful and informative for students.", "pashto": "دا کتاب د زده کوونکو لپاره ډېر ګټور او معلوماتي دی.", "sentiment": "positive"},
        {"id": 2, "english": "The network connection was very slow and disappointing today.", "pashto": "نن د انټرنیټ اړیکه ډېره ورو او ناهیلې کوونکې وه.", "sentiment": "negative"},
        {"id": 3, "english": "We are learning natural language processing and data science.", "pashto": "موږ د طبیعي ژبې پروسس او ډیټا ساینس زده کوو.", "sentiment": "neutral"},
    ]

class CrossLingualSentimentPipeline:
    POS = {"ګټور": 1.5, "معلوماتي": 1.2, "غوره": 1.6, "روښانه": 1.4, "useful": 1.4, "informative": 1.2, "excellent": 1.6}
    NEG = {"ورو": -1.2, "ناهیلې": -1.6, "خراب": -1.6, "تېروتنې": -1.5, "slow": -1.2, "disappointing": -1.6, "errors": -1.5}

    def __init__(self) -> None:
        self.tokenizer = MorphologyAwarePashtoTokenizer()

    def predict_label(self, text: str) -> str:
        score = sum(self.POS.get(t.lower(), 0.0) + self.NEG.get(t.lower(), 0.0) for t in self.tokenizer.tokenize(text))
        return "positive" if score >= 0.8 else ("negative" if score <= -0.8 else "neutral")
