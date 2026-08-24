"""Stage 4: Cleaning Pipeline.

Runs a sequence of cleaning steps over every document's raw text:
boilerplate/navigation/cookie-banner removal, broken-HTML stripping,
unicode/whitespace/markdown normalization -- while explicitly preserving
code blocks, API signatures, and heading/list/table structure.
"""

from __future__ import annotations

import re
from typing import List

from src.processors.models import RawDocument
from src.utils.logging_setup import get_logger
from src.utils.text_utils import (
    clean_markdown_links,
    extract_code_blocks,
    normalize_unicode,
    normalize_whitespace,
    remove_boilerplate_lines,
    strip_html_tags,
)

logger = get_logger(__name__)

_LICENSE_BLOCK_PATTERN = re.compile(
    r"(licensed under the .*?license.*?(?:\n\n|\Z))", re.IGNORECASE | re.DOTALL
)
_CODE_FENCE_PATTERN = re.compile(r"```.*?```", re.DOTALL)
_BROKEN_HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_EXCESSIVE_HEADING_MARKS = re.compile(r"#{7,}")


class DocumentCleaner:
    """Applies a deterministic, ordered sequence of cleaning transforms.

    Code blocks are extracted and swapped for placeholders before prose
    cleaning runs, then reinserted verbatim, so cleaning steps aimed at
    prose (boilerplate removal, link stripping, entity decoding) never
    touch code content or API signatures.
    """

    def clean(self, document: RawDocument) -> RawDocument:
        text = document.raw_text

        code_blocks = extract_code_blocks(text)
        placeholders = {}
        protected_text = text
        for i, (lang, code) in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            fenced = f"```{lang}\n{code}\n```"
            protected_text = protected_text.replace(fenced, placeholder, 1)
            placeholders[placeholder] = fenced

        protected_text = self._remove_html_artifacts(protected_text)
        protected_text = self._remove_license_blocks(protected_text)
        protected_text = remove_boilerplate_lines(protected_text)
        protected_text = clean_markdown_links(protected_text, keep_link_text_only=True)
        protected_text = normalize_unicode(protected_text)
        protected_text = self._normalize_markdown_structure(protected_text)

        for placeholder, fenced in placeholders.items():
            protected_text = protected_text.replace(placeholder, fenced, 1)

        cleaned = normalize_whitespace(protected_text)
        document.raw_text = cleaned
        return document

    def clean_batch(self, documents: List[RawDocument]) -> List[RawDocument]:
        cleaned_documents = []
        for document in documents:
            try:
                cleaned_documents.append(self.clean(document))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Cleaning failed for {document.document_id}: {exc}; keeping raw text.")
                cleaned_documents.append(document)

        non_empty = [d for d in cleaned_documents if d.raw_text.strip()]
        removed_empty = len(cleaned_documents) - len(non_empty)
        if removed_empty:
            logger.info(f"Removed {removed_empty} document(s) that became empty after cleaning.")
        logger.info(f"Cleaned {len(non_empty)} document(s).")
        return non_empty

    @staticmethod
    def _remove_html_artifacts(text: str) -> str:
        # Strip stray/broken HTML tags that survived collection (e.g. from
        # markdown files that embed raw HTML snippets), and decode common
        # entities so text doesn't retain artifacts like &amp; or &nbsp;.
        text = strip_html_tags(text)
        entity_map = {
            "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
            "&#39;": "'", "&nbsp;": " ", "&apos;": "'",
        }
        for entity, replacement in entity_map.items():
            text = text.replace(entity, replacement)
        text = _BROKEN_HTML_ENTITY_PATTERN.sub("", text)
        return text

    @staticmethod
    def _remove_license_blocks(text: str) -> str:
        return _LICENSE_BLOCK_PATTERN.sub("", text)

    @staticmethod
    def _normalize_markdown_structure(text: str) -> str:
        text = _EXCESSIVE_HEADING_MARKS.sub("######", text)
        # Collapse 3+ consecutive '-' or '*' bullet markers used as visual
        # dividers (not real lists) into a standard horizontal rule.
        text = re.sub(r"^[\-\*_]{3,}\s*$", "---", text, flags=re.MULTILINE)
        # Normalize inconsistent bullet characters to '-'.
        text = re.sub(r"^[ \t]*[•‣▪]\s+", "- ", text, flags=re.MULTILINE)
        return text
