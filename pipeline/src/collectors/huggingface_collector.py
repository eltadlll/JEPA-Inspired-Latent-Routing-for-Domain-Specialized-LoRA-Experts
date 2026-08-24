"""Stage 2: Hugging Face dataset collector.

Supports streaming (to avoid downloading entire large datasets) and
partial download (capped by `max_examples`), storing dataset-level
metadata alongside each extracted example.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.collectors.base import BaseCollector, CollectorError
from src.config.schema import HuggingFaceDatasetConfig, RetryConfig
from src.processors.models import DocumentMetadata, RawDocument
from src.utils.hashing import document_id
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class HuggingFaceCollector(BaseCollector):
    source_name = "huggingface"

    def __init__(
        self,
        dataset_configs: List[HuggingFaceDatasetConfig],
        retry_config: RetryConfig,
        raw_output_dir: Path,
    ) -> None:
        super().__init__(retry_config, raw_output_dir)
        self.dataset_configs = dataset_configs

    def collect(self) -> List[RawDocument]:
        documents: List[RawDocument] = []
        for ds_config in self.dataset_configs:
            try:
                documents.extend(self._collect_dataset(ds_config))
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to collect HF dataset {ds_config.repo_id}: {exc}")
        self._log_summary(documents)
        return documents

    def _collect_dataset(self, ds_config: HuggingFaceDatasetConfig) -> List[RawDocument]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise CollectorError(
                "The `datasets` library is required for HuggingFaceCollector. "
                "Install with `pip install datasets`."
            ) from exc

        logger.info(
            f"Loading HF dataset '{ds_config.repo_id}' split='{ds_config.split}' "
            f"streaming={ds_config.streaming}"
        )
        dataset = load_dataset(
            ds_config.repo_id, split=ds_config.split, streaming=ds_config.streaming
        )

        documents: List[RawDocument] = []
        for index, example in enumerate(dataset):
            if ds_config.max_examples is not None and index >= ds_config.max_examples:
                break

            text = self._extract_text_field(example, ds_config.text_field)
            if not text or not text.strip():
                continue

            metadata = DocumentMetadata(
                document_id=document_id("huggingface", f"{ds_config.repo_id}_{index}"),
                source=self.source_name,
                repository=ds_config.repo_id,
                category=ds_config.category.value,
                url=f"https://huggingface.co/datasets/{ds_config.repo_id}",
                file_path=None,
                version=ds_config.split,
                extra={"row_index": index},
            )
            documents.append(RawDocument(raw_text=text, metadata=metadata))

        return documents

    @staticmethod
    def _extract_text_field(example: dict, text_field: str) -> Optional[str]:
        if text_field in example:
            value = example[text_field]
            return str(value) if value is not None else None

        # Fall back to concatenating all string-valued fields, which is a
        # reasonable default for instruction-style datasets with multiple
        # text columns (e.g. "instruction" + "output").
        string_parts = [str(v) for v in example.values() if isinstance(v, str) and v.strip()]
        return "\n\n".join(string_parts) if string_parts else None
