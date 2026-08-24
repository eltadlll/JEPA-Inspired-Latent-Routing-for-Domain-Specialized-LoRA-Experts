"""Stage 5: Code Extraction.

Separates fenced code blocks from prose, auto-labels language (falling
back to heuristic detection when a fence has no language tag), and keeps
a link between each code block and the surrounding explanatory text so
downstream chunking/instruction-generation can use both together.
"""

from __future__ import annotations

import re
from typing import List

from src.processors.models import CodeBlock, RawDocument
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

_CODE_FENCE_WITH_CONTEXT = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_CONTEXT_WINDOW_CHARS = 300

_LANGUAGE_HEURISTICS = [
    (re.compile(r"^\s*(def |class |import |from .+ import)", re.MULTILINE), "python"),
    (re.compile(r"^\s*(function |const |let |var |=>)", re.MULTILINE), "javascript"),
    (re.compile(r"^\s*(interface |type \w+ =|export )", re.MULTILINE), "typescript"),
    (re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE)\s", re.IGNORECASE | re.MULTILINE), "sql"),
    (re.compile(r"^\s*\{"), "json"),
    (re.compile(r"^\s*(FROM|RUN|CMD|COPY)\s", re.MULTILINE), "dockerfile"),
    (re.compile(r"^\s*apiVersion:|^\s*kind:", re.MULTILINE), "yaml"),
]


class CodeExtractor:
    def extract(self, document: RawDocument) -> RawDocument:
        text = document.raw_text
        code_blocks: List[CodeBlock] = []

        for match in _CODE_FENCE_WITH_CONTEXT.finditer(text):
            language = match.group(1).strip().lower()
            code = match.group(2).strip()
            if not code:
                continue
            if not language:
                language = self._guess_language(code)

            context_start = max(0, match.start() - _CONTEXT_WINDOW_CHARS)
            surrounding_context = text[context_start : match.start()].strip()
            line_start = text[: match.start()].count("\n") + 1

            code_blocks.append(
                CodeBlock(
                    language=language,
                    code=code,
                    surrounding_context=surrounding_context,
                    line_start=line_start,
                )
            )

        document.code_blocks = code_blocks
        if code_blocks and document.metadata.programming_language is None:
            document.metadata.programming_language = self._dominant_language(code_blocks)
        return document

    def extract_batch(self, documents: List[RawDocument]) -> List[RawDocument]:
        processed = [self.extract(doc) for doc in documents]
        total_blocks = sum(len(d.code_blocks) for d in processed)
        logger.info(f"Extracted {total_blocks} code block(s) across {len(processed)} document(s).")
        return processed

    @staticmethod
    def _guess_language(code: str) -> str:
        for pattern, language in _LANGUAGE_HEURISTICS:
            if pattern.search(code):
                return language
        return "text"

    @staticmethod
    def _dominant_language(code_blocks: List[CodeBlock]) -> str:
        counts: dict = {}
        for block in code_blocks:
            counts[block.language] = counts.get(block.language, 0) + 1
        return max(counts, key=counts.get)
