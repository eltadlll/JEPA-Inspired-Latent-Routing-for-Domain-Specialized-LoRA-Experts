"""Generic text normalization helpers shared across cleaners and chunkers."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

_CODE_FENCE_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_WHITESPACE = re.compile(r"[ \t]+\n")
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

_BOILERPLATE_PATTERNS = [
    re.compile(r"^cookie(s)?\s+(policy|notice|consent)", re.IGNORECASE),
    re.compile(r"^we use cookies", re.IGNORECASE),
    re.compile(r"^subscribe to our newsletter", re.IGNORECASE),
    re.compile(r"^all rights reserved", re.IGNORECASE),
    re.compile(r"^copyright\s+\d{4}", re.IGNORECASE),
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^skip to (main )?content", re.IGNORECASE),
    re.compile(r"^\s*(edit this page|improve this page|was this page helpful)", re.IGNORECASE),
    re.compile(r"^\s*(next|previous)\s*:?\s*$", re.IGNORECASE),
]


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    text = _TRAILING_WHITESPACE.sub("\n", text)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def strip_html_tags(text: str) -> str:
    return _HTML_TAG_PATTERN.sub("", text)


def is_boilerplate_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in _BOILERPLATE_PATTERNS)


def remove_boilerplate_lines(text: str) -> str:
    lines = text.split("\n")
    kept = [line for line in lines if not is_boilerplate_line(line)]
    return "\n".join(kept)


def extract_code_blocks(markdown_text: str) -> List[Tuple[str, str]]:
    """Return a list of (language, code) tuples for every fenced code block."""
    return [(lang or "text", code.strip()) for lang, code in _CODE_FENCE_PATTERN.findall(markdown_text)]


def strip_code_blocks(markdown_text: str) -> str:
    """Return the markdown text with fenced code blocks removed, useful for
    prose-only quality metrics like readability.
    """
    return _CODE_FENCE_PATTERN.sub("", markdown_text)


def code_character_ratio(text: str) -> float:
    if not text:
        return 0.0
    code_chars = sum(len(code) for _, code in extract_code_blocks(text))
    return min(1.0, code_chars / max(1, len(text)))


def simple_readability_score(text: str) -> float:
    """A lightweight, dependency-free approximation of the Flesch Reading
    Ease score, normalized to 0..1. Not a substitute for linguistic
    analysis, but sufficient as one signal among several in quality scoring.
    """
    prose = strip_code_blocks(text)
    sentences = re.split(r"[.!?]+", prose)
    sentences = [s for s in sentences if s.strip()]
    words = re.findall(r"[A-Za-z']+", prose)

    if not sentences or not words:
        return 0.0

    syllables = sum(_count_syllables(w) for w in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllables / len(words)

    flesch = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    normalized = max(0.0, min(1.0, flesch / 100.0))
    return round(normalized, 4)


def _count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    previous_was_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def clean_markdown_links(text: str, keep_link_text_only: bool = True) -> str:
    if keep_link_text_only:
        return _MARKDOWN_LINK.sub(r"\1", text)
    return text


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n...\n{text[-half:]}"
