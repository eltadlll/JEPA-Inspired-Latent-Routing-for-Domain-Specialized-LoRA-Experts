"""Stage 4 (continued): Exact and near-duplicate detection.

Exact duplicates are caught via content hash. Near-duplicates are caught
via SimHash + Hamming-distance similarity, which scales to tens of
thousands of documents without the O(n^2) cost of pairwise diffing
(bucketed by simhash prefix to avoid full pairwise comparison at scale).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from src.processors.models import RawDocument
from src.utils.hashing import simhash, simhash_similarity
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Number of high-order bits used to bucket documents before pairwise
# comparison; documents in different buckets are never compared, which
# keeps near-duplicate detection roughly linear in practice.
_BUCKET_BITS = 8


class Deduplicator:
    def __init__(self, max_duplicate_similarity: float = 0.92, hash_bits: int = 64) -> None:
        self.max_duplicate_similarity = max_duplicate_similarity
        self.hash_bits = hash_bits

    def deduplicate(self, documents: List[RawDocument]) -> List[RawDocument]:
        documents = self._remove_exact_duplicates(documents)
        documents = self._remove_near_duplicates(documents)
        return documents

    def _remove_exact_duplicates(self, documents: List[RawDocument]) -> List[RawDocument]:
        seen_hashes: Set[str] = set()
        unique_documents: List[RawDocument] = []
        exact_duplicates = 0

        for document in documents:
            content_hash = document.metadata.content_hash
            if content_hash and content_hash in seen_hashes:
                document.is_duplicate = True
                exact_duplicates += 1
                continue
            seen_hashes.add(content_hash)
            unique_documents.append(document)

        if exact_duplicates:
            logger.info(f"Removed {exact_duplicates} exact duplicate document(s).")
        return unique_documents

    def _remove_near_duplicates(self, documents: List[RawDocument]) -> List[RawDocument]:
        buckets: Dict[int, List[Tuple[int, RawDocument]]] = defaultdict(list)

        for document in documents:
            fingerprint = simhash(document.raw_text, self.hash_bits)
            document.simhash_fingerprint = fingerprint
            bucket_key = fingerprint >> (self.hash_bits - _BUCKET_BITS)
            buckets[bucket_key].append((fingerprint, document))

        kept_documents: List[RawDocument] = []
        near_duplicates = 0

        for bucket in buckets.values():
            kept_in_bucket: List[Tuple[int, RawDocument]] = []
            for fingerprint, document in bucket:
                is_duplicate = False
                for kept_fingerprint, _ in kept_in_bucket:
                    similarity = simhash_similarity(fingerprint, kept_fingerprint, self.hash_bits)
                    if similarity >= self.max_duplicate_similarity:
                        is_duplicate = True
                        break
                if is_duplicate:
                    document.is_duplicate = True
                    near_duplicates += 1
                else:
                    kept_in_bucket.append((fingerprint, document))
                    kept_documents.append(document)

        if near_duplicates:
            logger.info(f"Removed {near_duplicates} near-duplicate document(s).")
        return kept_documents
