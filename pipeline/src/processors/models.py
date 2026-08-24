"""
Core data models that flow through every stage of the pipeline.

`RawDocument` is produced by collectors (Stage 2), enriched with
`DocumentMetadata` (Stage 3), transformed in place by cleaners (Stage 4),
split into `Chunk` objects (Stage 6), scored (Stage 7), filtered (Stage 8),
and finally converted into `InstructionExample` records (Stage 9).

Using a single, well-typed model across stages -- rather than passing
raw dicts around -- is what makes this pipeline maintainable by a team.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CodeBlock:
    """A single fenced code block extracted from a document."""

    language: str
    code: str
    surrounding_context: str = ""
    line_start: Optional[int] = None


@dataclass
class DocumentMetadata:
    """Stage 3: metadata attached to every collected document."""

    document_id: str
    source: str  # e.g. "github", "documentation", "huggingface", "blog", "paper"
    repository: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    collection_time: float = field(default_factory=time.time)
    category: str = "uncategorized"
    subcategory: Optional[str] = None
    programming_language: Optional[str] = None
    framework: Optional[str] = None
    url: Optional[str] = None
    file_path: Optional[str] = None
    document_length_chars: int = 0
    token_count: int = 0
    content_hash: str = ""
    version: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "repository": self.repository,
            "author": self.author,
            "license": self.license,
            "collection_time": self.collection_time,
            "category": self.category,
            "subcategory": self.subcategory,
            "programming_language": self.programming_language,
            "framework": self.framework,
            "url": self.url,
            "file_path": self.file_path,
            "document_length_chars": self.document_length_chars,
            "token_count": self.token_count,
            "content_hash": self.content_hash,
            "version": self.version,
            "extra": self.extra,
        }


@dataclass
class RawDocument:
    """A single collected unit of content, before or after cleaning."""

    raw_text: str
    metadata: DocumentMetadata
    code_blocks: List[CodeBlock] = field(default_factory=list)
    quality_score: Optional[float] = None
    quality_report: Dict[str, float] = field(default_factory=dict)
    simhash_fingerprint: Optional[int] = None
    is_duplicate: bool = False

    @property
    def document_id(self) -> str:
        return self.metadata.document_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.raw_text,
            "metadata": self.metadata.to_dict(),
            "code_blocks": [
                {
                    "language": cb.language,
                    "code": cb.code,
                    "surrounding_context": cb.surrounding_context,
                    "line_start": cb.line_start,
                }
                for cb in self.code_blocks
            ],
            "quality_score": self.quality_score,
            "quality_report": self.quality_report,
            "is_duplicate": self.is_duplicate,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RawDocument":
        meta_data = data["metadata"]
        metadata = DocumentMetadata(**meta_data)
        code_blocks = [
            CodeBlock(
                language=cb["language"],
                code=cb["code"],
                surrounding_context=cb.get("surrounding_context", ""),
                line_start=cb.get("line_start"),
            )
            for cb in data.get("code_blocks", [])
        ]
        return RawDocument(
            raw_text=data["text"],
            metadata=metadata,
            code_blocks=code_blocks,
            quality_score=data.get("quality_score"),
            quality_report=data.get("quality_report", {}),
            is_duplicate=data.get("is_duplicate", False),
        )


@dataclass
class Chunk:
    """Stage 6: a semantically coherent slice of a document."""

    chunk_id: str
    parent_document_id: str
    text: str
    token_count: int
    heading_path: List[str] = field(default_factory=list)
    chunk_index: int = 0
    contains_code: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "parent_document_id": self.parent_document_id,
            "text": self.text,
            "token_count": self.token_count,
            "heading_path": self.heading_path,
            "chunk_index": self.chunk_index,
            "contains_code": self.contains_code,
            "metadata": self.metadata,
        }


@dataclass
class InstructionExample:
    """Stage 9: a single fine-tuning example in Alpaca-style schema."""

    id: str
    instruction: str
    input: str
    output: str
    category: str
    difficulty: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "category": self.category,
            "difficulty": self.difficulty,
            "source": self.source,
            "metadata": self.metadata,
        }
