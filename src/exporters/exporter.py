"""Stage 12: Export.

Writes the final, validated instruction dataset to every configured
format (JSONL, Parquet, Arrow, CSV, Hugging Face `datasets` format),
with a deterministic train/validation split.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from src.config.schema import ExportConfig, ExportFormat
from src.processors.models import InstructionExample
from src.utils.io_utils import write_jsonl
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class DatasetExporter:
    def __init__(self, config: ExportConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, examples: List[InstructionExample]) -> List[Path]:
        train_examples, val_examples = self._split(examples)
        logger.info(
            f"Splitting dataset: {len(train_examples)} train / {len(val_examples)} validation "
            f"(train_split={self.config.train_split})"
        )

        written_paths: List[Path] = []
        for fmt in self.config.formats:
            handler = self._handler_for_format(fmt)
            written_paths.extend(handler(train_examples, val_examples))

        return written_paths

    def _split(
        self, examples: List[InstructionExample]
    ) -> Tuple[List[InstructionExample], List[InstructionExample]]:
        shuffled = list(examples)
        random.Random(self.config.shuffle_seed).shuffle(shuffled)
        split_index = int(len(shuffled) * self.config.train_split)
        return shuffled[:split_index], shuffled[split_index:]

    def _handler_for_format(self, fmt: ExportFormat):
        return {
            ExportFormat.JSONL: self._export_jsonl,
            ExportFormat.PARQUET: self._export_parquet,
            ExportFormat.ARROW: self._export_arrow,
            ExportFormat.CSV: self._export_csv,
            ExportFormat.HF_DATASET: self._export_hf_dataset,
        }[fmt]

    def _export_jsonl(self, train, val) -> List[Path]:
        train_path = self.output_dir / f"{self.config.dataset_name}.train.jsonl"
        val_path = self.output_dir / f"{self.config.dataset_name}.validation.jsonl"
        write_jsonl(train_path, (e.to_dict() for e in train))
        write_jsonl(val_path, (e.to_dict() for e in val))
        logger.info(f"Wrote JSONL: {train_path}, {val_path}")
        return [train_path, val_path]

    def _to_dataframe(self, examples: List[InstructionExample]):
        import pandas as pd

        rows = [e.to_dict() for e in examples]
        for row in rows:
            row["metadata"] = str(row["metadata"])  # flatten for tabular formats
        return pd.DataFrame(rows)

    def _export_parquet(self, train, val) -> List[Path]:
        train_path = self.output_dir / f"{self.config.dataset_name}.train.parquet"
        val_path = self.output_dir / f"{self.config.dataset_name}.validation.parquet"
        self._to_dataframe(train).to_parquet(train_path, index=False)
        self._to_dataframe(val).to_parquet(val_path, index=False)
        logger.info(f"Wrote Parquet: {train_path}, {val_path}")
        return [train_path, val_path]

    def _export_arrow(self, train, val) -> List[Path]:
        import pyarrow as pa
        import pyarrow.feather as feather

        train_path = self.output_dir / f"{self.config.dataset_name}.train.arrow"
        val_path = self.output_dir / f"{self.config.dataset_name}.validation.arrow"
        feather.write_feather(pa.Table.from_pandas(self._to_dataframe(train)), str(train_path))
        feather.write_feather(pa.Table.from_pandas(self._to_dataframe(val)), str(val_path))
        logger.info(f"Wrote Arrow: {train_path}, {val_path}")
        return [train_path, val_path]

    def _export_csv(self, train, val) -> List[Path]:
        train_path = self.output_dir / f"{self.config.dataset_name}.train.csv"
        val_path = self.output_dir / f"{self.config.dataset_name}.validation.csv"
        self._to_dataframe(train).to_csv(train_path, index=False)
        self._to_dataframe(val).to_csv(val_path, index=False)
        logger.info(f"Wrote CSV: {train_path}, {val_path}")
        return [train_path, val_path]

    def _export_hf_dataset(self, train, val) -> List[Path]:
        from datasets import Dataset, DatasetDict

        def to_hf_records(examples: List[InstructionExample]) -> List[dict]:
            return [e.to_dict() for e in examples]

        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_list(to_hf_records(train)),
                "validation": Dataset.from_list(to_hf_records(val)),
            }
        )
        out_path = self.output_dir / f"{self.config.dataset_name}_hf"
        dataset_dict.save_to_disk(str(out_path))
        logger.info(f"Wrote Hugging Face dataset: {out_path}")
        return [out_path]
