"""Token counting utilities, backed by tiktoken with a graceful fallback."""

from __future__ import annotations

from functools import lru_cache

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dependency missing
    _TIKTOKEN_AVAILABLE = False


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str):
    """Return a tiktoken encoding, or None if tiktoken is unavailable or its
    BPE data file can't be loaded (e.g. no network access to download it on
    first use). Callers must handle None via the word-count fallback.
    """
    if not _TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in `text`. Falls back to a whitespace-based heuristic
    (~1.3 tokens per word, a reasonable approximation for English/code mixes)
    if tiktoken is unavailable, so the pipeline still runs in constrained
    environments.
    """
    if not text:
        return 0
    encoding = _get_encoding(encoding_name)
    if encoding is not None:
        return len(encoding.encode(text, disallowed_special=()))
    word_count = len(text.split())
    return int(word_count * 1.3)


def truncate_to_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    """Truncate `text` so it fits within `max_tokens`."""
    encoding = _get_encoding(encoding_name)
    if encoding is None:
        words = text.split()
        approx_words = int(max_tokens / 1.3)
        return " ".join(words[:approx_words])

    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])
