"""Deterministic hashing utilities used for document IDs and dedup keys."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def stable_hash(text: str, length: int = 16) -> str:
    """Return a stable, short hex hash for a piece of text."""
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return digest[:length]


def document_id(source: str, identifier: str) -> str:
    """Deterministic document ID derived from source + a natural identifier
    (e.g. file path or URL), so re-running the pipeline is idempotent.
    """
    return f"{source}_{stable_hash(identifier, length=12)}"


def normalize_for_similarity(text: str) -> str:
    """Aggressively normalize text so near-duplicate detection is robust to
    whitespace, casing, and unicode variation.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def simhash(text: str, hash_bits: int = 64) -> int:
    """A lightweight SimHash implementation for near-duplicate detection.

    This avoids pulling in a heavy dependency purely for locality-sensitive
    hashing of document text; the algorithm is standard SimHash over
    whitespace-tokenized shingles.
    """
    tokens = normalize_for_similarity(text).split()
    if not tokens:
        return 0

    shingles = [" ".join(tokens[i : i + 3]) for i in range(max(1, len(tokens) - 2))]
    if not shingles:
        shingles = tokens

    vector = [0] * hash_bits
    for shingle in shingles:
        shingle_hash = int(hashlib.md5(shingle.encode("utf-8")).hexdigest(), 16)
        for bit in range(hash_bits):
            bit_value = (shingle_hash >> bit) & 1
            vector[bit] += 1 if bit_value else -1

    fingerprint = 0
    for bit in range(hash_bits):
        if vector[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def simhash_similarity(hash_a: int, hash_b: int, hash_bits: int = 64) -> float:
    """Convert Hamming distance between two SimHash fingerprints into a
    0..1 similarity score (1.0 = identical).
    """
    distance = hamming_distance(hash_a, hash_b)
    return 1.0 - (distance / hash_bits)
