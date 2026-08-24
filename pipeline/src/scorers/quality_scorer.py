"""Stage 7: Quality Scoring.

Produces a composite `quality_score` (0..1) per document from several
independent, individually-inspectable signals, so filtering decisions
(Stage 8) and dataset reports (Stage 10) can explain *why* a document
scored the way it did rather than treating quality as a black box.
"""

from __future__ import annotations

import re
from typing import Dict, List

from src.processors.models import RawDocument
from src.utils.logging_setup import get_logger
from src.utils.text_utils import code_character_ratio, simple_readability_score, strip_code_blocks

logger = get_logger(__name__)

_SOURCE_CREDIBILITY = {
    "github": 0.9,
    "documentation": 1.0,
    "huggingface": 0.75,
    "blog": 0.65,
    "paper": 0.95,
}

_INSTRUCTION_MARKERS = re.compile(
    r"\b(you should|you can|steps?|first,|then,|note that|important:|example:|usage:)\b",
    re.IGNORECASE,
)
_API_SIGNATURE_PATTERN = re.compile(r"\bdef\s+\w+\(|\bclass\s+\w+|\bfunction\s+\w+\(", re.MULTILINE)
_BROKEN_LINK_PATTERN = re.compile(r"\]\(\s*\)|\]\(#\)|http[s]?://\s")

# Weights sum to 1.0; tuned toward favoring documents that pair
# explanation with working code, since that is exactly what the target
# domain (agent engineering / LLM engineering) rewards.
_WEIGHTS = {
    "token_count": 0.10,
    "code_ratio": 0.15,
    "documentation_completeness": 0.15,
    "readability": 0.15,
    "duplicate_probability": 0.10,
    "source_credibility": 0.15,
    "example_density": 0.10,
    "instruction_density": 0.05,
    "broken_links": 0.05,
}


class QualityScorer:
    def __init__(self, min_token_count: int = 30, target_token_count: int = 800) -> None:
        self.min_token_count = min_token_count
        self.target_token_count = target_token_count

    def score(self, document: RawDocument) -> RawDocument:
        features = self._compute_features(document)
        composite = sum(_WEIGHTS[name] * value for name, value in features.items())
        document.quality_score = round(min(1.0, max(0.0, composite)), 4)
        document.quality_report = features
        return document

    def score_batch(self, documents: List[RawDocument]) -> List[RawDocument]:
        scored = [self.score(doc) for doc in documents]
        if scored:
            average = sum(d.quality_score for d in scored) / len(scored)
            logger.info(f"Scored {len(scored)} document(s); average quality = {average:.3f}")
        return scored

    def _compute_features(self, document: RawDocument) -> Dict[str, float]:
        text = document.raw_text
        token_count = document.metadata.token_count

        token_score = min(1.0, token_count / self.target_token_count) if token_count else 0.0
        if token_count < self.min_token_count:
            token_score *= 0.3  # heavily penalize near-empty documents

        code_ratio = code_character_ratio(text)
        # A moderate amount of code is ideal; pure code or pure prose scores lower.
        code_ratio_score = 1.0 - abs(code_ratio - 0.35) / 0.65

        prose = strip_code_blocks(text)
        has_headings = bool(re.search(r"^#{1,6}\s", text, re.MULTILINE))
        has_lists = bool(re.search(r"^[\-\*]\s|^\d+\.\s", text, re.MULTILINE))
        doc_completeness = sum([has_headings, has_lists, len(document.code_blocks) > 0, len(prose) > 200]) / 4.0

        readability = simple_readability_score(text)

        duplicate_probability_inverse = 0.0 if document.is_duplicate else 1.0

        source_credibility = _SOURCE_CREDIBILITY.get(document.metadata.source, 0.5)

        example_density = min(1.0, len(document.code_blocks) / 3.0)

        instruction_matches = len(_INSTRUCTION_MARKERS.findall(prose))
        instruction_density = min(1.0, instruction_matches / 5.0)

        broken_link_matches = len(_BROKEN_LINK_PATTERN.findall(text))
        broken_links_score = max(0.0, 1.0 - broken_link_matches * 0.2)

        return {
            "token_count": round(token_score, 4),
            "code_ratio": round(max(0.0, code_ratio_score), 4),
            "documentation_completeness": round(doc_completeness, 4),
            "readability": round(readability, 4),
            "duplicate_probability": round(duplicate_probability_inverse, 4),
            "source_credibility": round(source_credibility, 4),
            "example_density": round(example_density, 4),
            "instruction_density": round(instruction_density, 4),
            "broken_links": round(broken_links_score, 4),
        }
