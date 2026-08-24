"""Stage 9: Instruction Dataset Generation - orchestration.

Runs every configured template against every chunk, caps the number of
examples produced per source document (to avoid over-representing very
long documents), and assembles fully-formed `InstructionExample` records
matching the required schema.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.config.schema import InstructionGenerationConfig
from src.generators.templates import get_template
from src.processors.models import Chunk, InstructionExample, RawDocument
from src.utils.hashing import stable_hash
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class InstructionGenerator:
    def __init__(self, config: InstructionGenerationConfig) -> None:
        self.config = config
        self.templates = [t for t in (get_template(name) for name in config.templates) if t is not None]

    def generate(
        self, chunks: List[Chunk], documents_by_id: Dict[str, RawDocument]
    ) -> List[InstructionExample]:
        examples: List[InstructionExample] = []
        per_document_count: Dict[str, int] = defaultdict(int)

        for chunk in chunks:
            document = documents_by_id.get(chunk.parent_document_id)
            if document is None:
                continue
            if per_document_count[chunk.parent_document_id] >= self.config.max_examples_per_document:
                continue

            for template_fn in self.templates:
                if per_document_count[chunk.parent_document_id] >= self.config.max_examples_per_document:
                    break
                try:
                    raw_examples = template_fn(chunk, document.metadata)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"Template {template_fn.__name__} failed on {chunk.chunk_id}: {exc}")
                    continue

                for raw_example in raw_examples:
                    if per_document_count[chunk.parent_document_id] >= self.config.max_examples_per_document:
                        break
                    example = self._build_example(raw_example, chunk, document)
                    if example is not None:
                        examples.append(example)
                        per_document_count[chunk.parent_document_id] += 1

        logger.info(
            f"Generated {len(examples)} instruction example(s) from {len(chunks)} chunk(s) "
            f"across {len(documents_by_id)} document(s)."
        )
        return examples

    def _build_example(
        self, raw_example: dict, chunk: Chunk, document: RawDocument
    ) -> InstructionExample | None:
        instruction = raw_example.get("instruction", "").strip()
        output = raw_example.get("output", "").strip()
        if not instruction or not output:
            return None

        example_id = stable_hash(f"{chunk.chunk_id}:{raw_example.get('template', '')}:{instruction}", 16)
        return InstructionExample(
            id=example_id,
            instruction=instruction,
            input=raw_example.get("input", ""),
            output=output,
            category=document.metadata.category,
            difficulty=raw_example.get("difficulty", "intermediate"),
            source=document.metadata.source,
            metadata={
                "template": raw_example.get("template"),
                "document_id": document.document_id,
                "chunk_id": chunk.chunk_id,
                "framework": document.metadata.framework,
                "url": document.metadata.url,
                "quality_score": document.quality_score,
            },
        )
