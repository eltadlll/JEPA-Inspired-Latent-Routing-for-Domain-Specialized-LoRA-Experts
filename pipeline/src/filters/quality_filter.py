"""Stage 8: Filtering.

Removes documents below the configured quality threshold and produces a
human-readable removal report explaining exactly why each document was
dropped, so quality-threshold tuning is an evidence-based exercise
rather than guesswork.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.config.schema import QualityThresholdsConfig
from src.processors.models import RawDocument
from src.utils.io_utils import write_json
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class RemovalRecord:
    document_id: str
    reason: str
    quality_score: float | None
    token_count: int
    source: str


@dataclass
class FilterReport:
    total_input: int = 0
    total_kept: int = 0
    total_removed: int = 0
    removals: List[RemovalRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_input": self.total_input,
            "total_kept": self.total_kept,
            "total_removed": self.total_removed,
            "removal_rate": round(self.total_removed / self.total_input, 4) if self.total_input else 0.0,
            "removals_by_reason": self._count_by_reason(),
            "removals": [
                {
                    "document_id": r.document_id,
                    "reason": r.reason,
                    "quality_score": r.quality_score,
                    "token_count": r.token_count,
                    "source": r.source,
                }
                for r in self.removals
            ],
        }

    def _count_by_reason(self) -> dict:
        counts: dict = {}
        for removal in self.removals:
            counts[removal.reason] = counts.get(removal.reason, 0) + 1
        return counts


class QualityFilter:
    def __init__(self, thresholds: QualityThresholdsConfig) -> None:
        self.thresholds = thresholds

    def filter_documents(self, documents: List[RawDocument]) -> tuple[List[RawDocument], FilterReport]:
        report = FilterReport(total_input=len(documents))
        kept: List[RawDocument] = []

        for document in documents:
            reason = self._rejection_reason(document)
            if reason is None:
                kept.append(document)
            else:
                report.removals.append(
                    RemovalRecord(
                        document_id=document.document_id,
                        reason=reason,
                        quality_score=document.quality_score,
                        token_count=document.metadata.token_count,
                        source=document.metadata.source,
                    )
                )

        report.total_kept = len(kept)
        report.total_removed = len(report.removals)
        logger.info(
            f"Filtering complete: kept {report.total_kept}/{report.total_input} "
            f"({report.total_removed} removed)."
        )
        return kept, report

    def _rejection_reason(self, document: RawDocument) -> str | None:
        if document.is_duplicate:
            return "duplicate"
        if document.metadata.token_count < self.thresholds.min_token_count:
            return "below_min_token_count"
        if document.quality_score is None:
            return "missing_quality_score"
        if document.quality_score < self.thresholds.min_quality_score:
            return "below_quality_threshold"
        readability = document.quality_report.get("readability")
        if readability is not None and readability < self.thresholds.min_readability_score:
            return "below_readability_threshold"
        return None

    @staticmethod
    def save_report(report: FilterReport, output_path: Path) -> None:
        write_json(output_path, report.to_dict())
        logger.info(f"Filter report written to {output_path}")
