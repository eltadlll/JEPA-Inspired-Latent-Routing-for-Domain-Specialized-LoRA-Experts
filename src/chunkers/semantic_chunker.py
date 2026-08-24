"""Stage 6: Document Chunking.

Chunks respect markdown heading structure and code block boundaries
rather than splitting at a fixed character offset. Each chunk records
its `heading_path` (the trail of headings above it) which acts as the
parent/child relationship linking chunks back to their position in the
source document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from src.config.schema import ChunkingConfig
from src.processors.models import Chunk, RawDocument
from src.utils.hashing import stable_hash
from src.utils.logging_setup import get_logger
from src.utils.text_utils import extract_code_blocks
from src.utils.tokenizer import count_tokens

logger = get_logger(__name__)

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class _Section:
    heading_path: List[str]
    text: str
    level: int


@dataclass
class _ChunkBuilder:
    heading_path: List[str]
    parts: List[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.parts.append(text)

    def render(self) -> str:
        return "\n\n".join(p for p in self.parts if p.strip())


class SemanticChunker:
    """Splits documents along heading boundaries first, then packs the
    resulting sections into token-budgeted chunks, only splitting mid-section
    (never mid-code-block) when a section alone exceeds `max_chunk_tokens`.
    """

    def __init__(self, config: ChunkingConfig, encoding_name: str = "cl100k_base") -> None:
        self.config = config
        self.encoding_name = encoding_name

    def chunk_document(self, document: RawDocument) -> List[Chunk]:
        sections = self._split_into_sections(document.raw_text)
        chunks = self._pack_sections(sections, document.document_id)
        return chunks

    def chunk_batch(self, documents: List[RawDocument]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for document in documents:
            try:
                all_chunks.extend(self.chunk_document(document))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Chunking failed for {document.document_id}: {exc}")
        logger.info(f"Produced {len(all_chunks)} chunk(s) from {len(documents)} document(s).")
        return all_chunks

    def _split_into_sections(self, text: str) -> List[_Section]:
        headings = list(_HEADING_PATTERN.finditer(text))
        if not headings:
            return [_Section(heading_path=[], text=text, level=0)]

        sections: List[_Section] = []
        heading_stack: List[tuple] = []  # (level, title)

        preamble = text[: headings[0].start()].strip()
        if preamble:
            sections.append(_Section(heading_path=[], text=preamble, level=0))

        for i, match in enumerate(headings):
            level = len(match.group(1))
            title = match.group(2).strip()

            heading_stack = [h for h in heading_stack if h[0] < level]
            heading_stack.append((level, title))
            heading_path = [h[1] for h in heading_stack]

            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            body = text[start:end].strip()

            sections.append(_Section(heading_path=heading_path, text=body, level=level))

        return sections

    def _pack_sections(self, sections: List[_Section], parent_document_id: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        builder = _ChunkBuilder(heading_path=[])
        current_tokens = 0
        chunk_index = 0

        def flush() -> None:
            nonlocal builder, current_tokens, chunk_index
            rendered = builder.render()
            if rendered.strip():
                token_count = count_tokens(rendered, self.encoding_name)
                if token_count >= self.config.min_chunk_tokens:
                    chunks.append(self._build_chunk(rendered, builder.heading_path, parent_document_id, chunk_index))
                    chunk_index += 1
            builder = _ChunkBuilder(heading_path=builder.heading_path)
            current_tokens = 0

        for section in sections:
            section_tokens = count_tokens(section.text, self.encoding_name)

            if section_tokens > self.config.max_chunk_tokens:
                flush()
                sub_chunks = self._split_oversized_section(section, parent_document_id, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
                continue

            if current_tokens + section_tokens > self.config.chunk_size_tokens and current_tokens > 0:
                flush()

            builder.heading_path = section.heading_path
            builder.add(self._render_section_with_heading(section))
            current_tokens += section_tokens

        flush()
        return self._apply_overlap(chunks)

    def _render_section_with_heading(self, section: _Section) -> str:
        if not section.heading_path:
            return section.text
        heading_line = "#" * section.level + " " + section.heading_path[-1]
        return f"{heading_line}\n\n{section.text}" if section.text else heading_line

    def _split_oversized_section(
        self, section: _Section, parent_document_id: str, start_index: int
    ) -> List[Chunk]:
        """Split a single section that alone exceeds max_chunk_tokens.

        Splits on paragraph boundaries and never inside a fenced code block.
        """
        code_blocks = extract_code_blocks(section.text)
        if self.config.respect_code_blocks and code_blocks:
            paragraphs = self._split_preserving_code_blocks(section.text)
        else:
            paragraphs = [p for p in section.text.split("\n\n") if p.strip()]

        sub_chunks: List[Chunk] = []
        buffer: List[str] = []
        buffer_tokens = 0
        index = start_index

        for paragraph in paragraphs:
            paragraph_tokens = count_tokens(paragraph, self.encoding_name)
            if buffer_tokens + paragraph_tokens > self.config.chunk_size_tokens and buffer:
                text = "\n\n".join(buffer)
                sub_chunks.append(self._build_chunk(text, section.heading_path, parent_document_id, index))
                index += 1
                buffer, buffer_tokens = [], 0
            buffer.append(paragraph)
            buffer_tokens += paragraph_tokens

        if buffer:
            text = "\n\n".join(buffer)
            sub_chunks.append(self._build_chunk(text, section.heading_path, parent_document_id, index))

        return sub_chunks

    @staticmethod
    def _split_preserving_code_blocks(text: str) -> List[str]:
        # Treat each fenced code block as an atomic paragraph so it is
        # never split across chunks, while surrounding prose is still
        # split on blank lines as usual.
        parts: List[str] = []
        remaining = text
        fence_pattern = re.compile(r"```.*?```", re.DOTALL)
        last_end = 0
        for match in fence_pattern.finditer(remaining):
            before = remaining[last_end : match.start()]
            parts.extend(p for p in before.split("\n\n") if p.strip())
            parts.append(match.group(0))
            last_end = match.end()
        tail = remaining[last_end:]
        parts.extend(p for p in tail.split("\n\n") if p.strip())
        return parts

    def _apply_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        if self.config.chunk_overlap_tokens <= 0 or len(chunks) < 2:
            return chunks

        overlapped: List[Chunk] = [chunks[0]]
        for previous, current in zip(chunks, chunks[1:]):
            overlap_text = self._tail_tokens(previous.text, self.config.chunk_overlap_tokens)
            if overlap_text and previous.parent_document_id == current.parent_document_id:
                merged_text = f"{overlap_text}\n\n{current.text}"
                current.text = merged_text
                current.token_count = count_tokens(merged_text, self.encoding_name)
            overlapped.append(current)
        return overlapped

    def _tail_tokens(self, text: str, max_tokens: int) -> str:
        words = text.split()
        approx_word_count = max(1, int(max_tokens / 1.3))
        return " ".join(words[-approx_word_count:])

    def _build_chunk(
        self, text: str, heading_path: List[str], parent_document_id: str, chunk_index: int
    ) -> Chunk:
        chunk_id = f"{parent_document_id}_chunk{chunk_index}_{stable_hash(text, 8)}"
        return Chunk(
            chunk_id=chunk_id,
            parent_document_id=parent_document_id,
            text=text,
            token_count=count_tokens(text, self.encoding_name),
            heading_path=heading_path,
            chunk_index=chunk_index,
            contains_code="```" in text,
        )
