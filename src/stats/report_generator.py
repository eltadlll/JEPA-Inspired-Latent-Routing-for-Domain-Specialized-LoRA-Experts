"""Stage 10: Dataset Statistics.

Aggregates counters across the whole run (documents collected,
duplicates removed, category/framework/language distributions, quality
and difficulty distributions, code/text ratio) into a single JSON report,
plus a small set of PNG charts for quick visual inspection.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List

from src.filters.quality_filter import FilterReport
from src.processors.models import InstructionExample, RawDocument
from src.utils.io_utils import write_json
from src.utils.logging_setup import get_logger
from src.utils.text_utils import code_character_ratio

logger = get_logger(__name__)


class ReportGenerator:
    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        raw_documents: List[RawDocument],
        filtered_documents: List[RawDocument],
        filter_report: FilterReport,
        instruction_examples: List[InstructionExample],
    ) -> dict:
        stats = {
            "documents_collected": len(raw_documents),
            "documents_after_filtering": len(filtered_documents),
            "duplicates_removed": sum(1 for d in raw_documents if d.is_duplicate),
            "average_tokens_per_document": self._average(
                [d.metadata.token_count for d in filtered_documents]
            ),
            "average_quality_score": self._average(
                [d.quality_score for d in filtered_documents if d.quality_score is not None]
            ),
            "category_distribution": self._distribution(
                [d.metadata.category for d in filtered_documents]
            ),
            "framework_distribution": self._distribution(
                [d.metadata.framework for d in filtered_documents if d.metadata.framework]
            ),
            "source_distribution": self._distribution([d.metadata.source for d in filtered_documents]),
            "language_distribution": self._distribution(
                [d.metadata.programming_language for d in filtered_documents if d.metadata.programming_language]
            ),
            "code_to_text_ratio": self._average(
                [code_character_ratio(d.raw_text) for d in filtered_documents]
            ),
            "filtering": filter_report.to_dict(),
            "instruction_dataset": {
                "total_examples": len(instruction_examples),
                "difficulty_distribution": self._distribution(
                    [e.difficulty for e in instruction_examples]
                ),
                "category_distribution": self._distribution(
                    [e.category for e in instruction_examples]
                ),
                "template_distribution": self._distribution(
                    [e.metadata.get("template") for e in instruction_examples]
                ),
                "average_output_length_chars": self._average(
                    [len(e.output) for e in instruction_examples]
                ),
            },
        }

        report_path = self.reports_dir / "dataset_statistics.json"
        write_json(report_path, stats)
        logger.info(f"Dataset statistics written to {report_path}")

        self._generate_charts(stats)
        return stats

    @staticmethod
    def _average(values: List[float]) -> float:
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    @staticmethod
    def _distribution(values: List[str]) -> dict:
        return dict(Counter(v for v in values if v))

    def _generate_charts(self, stats: dict) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping chart generation.")
            return

        self._bar_chart(
            plt, stats["category_distribution"], "Documents by Category", "category_distribution.png"
        )
        self._bar_chart(
            plt, stats["instruction_dataset"]["difficulty_distribution"],
            "Instruction Examples by Difficulty", "difficulty_distribution.png",
        )
        self._bar_chart(
            plt, stats["instruction_dataset"]["template_distribution"],
            "Instruction Examples by Template", "template_distribution.png",
        )
        self._bar_chart(
            plt, stats["source_distribution"], "Documents by Source", "source_distribution.png"
        )

    def _bar_chart(self, plt, distribution: dict, title: str, filename: str) -> None:
        if not distribution:
            return
        labels = list(distribution.keys())
        values = list(distribution.values())

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values, color="#4C72B0")
        ax.set_title(title)
        ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        output_path = self.reports_dir / filename
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        logger.debug(f"Wrote chart: {output_path}")
