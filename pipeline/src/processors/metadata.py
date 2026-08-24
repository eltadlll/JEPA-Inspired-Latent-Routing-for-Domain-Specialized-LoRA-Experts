"""Stage 3: Metadata Generation.

Collectors focus on getting content; this module is responsible for
finalizing and enriching the `DocumentMetadata` attached to every
`RawDocument` before it enters the cleaning pipeline (token counts,
content hashes, and any fields the collector didn't already populate).
"""

from __future__ import annotations

from src.processors.models import RawDocument
from src.utils.hashing import stable_hash
from src.utils.logging_setup import get_logger
from src.utils.tokenizer import count_tokens

logger = get_logger(__name__)


class MetadataEnricher:
    """Finalizes metadata fields that require the fully-collected document
    text to compute (length, token count, content hash).
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name

    def enrich(self, document: RawDocument) -> RawDocument:
        document.metadata.document_length_chars = len(document.raw_text)
        document.metadata.token_count = count_tokens(document.raw_text, self.encoding_name)
        document.metadata.content_hash = stable_hash(document.raw_text)
        return document

    def enrich_batch(self, documents: list[RawDocument]) -> list[RawDocument]:
        enriched = [self.enrich(doc) for doc in documents]
        logger.info(f"Enriched metadata for {len(enriched)} documents.")
        return enriched
