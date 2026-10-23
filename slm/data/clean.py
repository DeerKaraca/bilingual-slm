"""Light document-level cleaning, shared by both languages.

The sources (FineWeb2-HQ, FineWeb-Edu) are already filtered and deduplicated upstream,
so we only normalize Unicode and drop documents that are too short, too long or mostly
non-letters (tables, code, number dumps).
"""

import hashlib
import unicodedata

MIN_CHARS = 200
MAX_CHARS = 100_000
MIN_LETTER_RATIO = 0.6


def normalize(text: str) -> str:
    # NFC so that e.g. "ş" is always one code point, never "s" + combining cedilla.
    return unicodedata.normalize("NFC", text).strip()


def letter_ratio(text: str) -> float:
    non_space = len(text) - sum(map(str.isspace, text))
    return sum(map(str.isalpha, text)) / max(non_space, 1)


def is_good(text: str) -> bool:
    return MIN_CHARS <= len(text) <= MAX_CHARS and letter_ratio(text) >= MIN_LETTER_RATIO


def doc_hash(text: str) -> bytes:
    return hashlib.md5(text.encode()).digest()
