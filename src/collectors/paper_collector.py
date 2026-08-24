"""Stage 2: Research paper collector.

Resolves an arXiv ID (or a direct URL) to paper metadata via the arXiv
Atom API, then extracts full text from the PDF when available, doing a
best-effort split into sections using heuristic heading detection.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import requests

from src.collectors.base import BaseCollector
from src.collectors.docs_collector import USER_AGENT
from src.config.schema import PaperSourceConfig, RetryConfig
from src.processors.models import DocumentMetadata, RawDocument
from src.utils.hashing import document_id
from src.utils.logging_setup import get_logger
from src.utils.text_utils import normalize_whitespace

logger = get_logger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
REQUEST_TIMEOUT_SECONDS = 30

# Common paper section headings, used to segment extracted PDF text.
SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s+)?(abstract|introduction|related work|background|"
    r"methodology|methods|approach|experiments?|results|discussion|"
    r"conclusion|references|acknowledge?ments?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class PaperCollector(BaseCollector):
    source_name = "paper"

    def __init__(
        self,
        paper_sources: List[PaperSourceConfig],
        retry_config: RetryConfig,
        raw_output_dir: Path,
        session: Optional[requests.Session] = None,
    ) -> None:
        super().__init__(retry_config, raw_output_dir)
        self.paper_sources = paper_sources
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def collect(self) -> List[RawDocument]:
        documents: List[RawDocument] = []
        for paper_source in self.paper_sources:
            try:
                document = self._collect_paper(paper_source)
                if document is not None:
                    documents.append(document)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to collect paper {paper_source.identifier}: {exc}")
        self._log_summary(documents)
        return documents

    def _collect_paper(self, paper_source: PaperSourceConfig) -> Optional[RawDocument]:
        arxiv_id = self._extract_arxiv_id(paper_source.identifier)
        if arxiv_id is None:
            logger.warning(
                f"Could not resolve an arXiv ID from '{paper_source.identifier}'; skipping."
            )
            return None

        title, abstract, authors, pdf_url = self._fetch_arxiv_metadata(arxiv_id)
        if title is None:
            return None

        body_text = self._extract_pdf_text(pdf_url) if pdf_url else None
        sections = self._segment_sections(body_text) if body_text else {}

        parts = [f"# {title}", "", "## Abstract", abstract or ""]
        for heading, content in sections.items():
            parts.extend(["", f"## {heading.title()}", content])
        full_text = normalize_whitespace("\n".join(parts))

        metadata = DocumentMetadata(
            document_id=document_id("paper", arxiv_id),
            source=self.source_name,
            author=", ".join(authors) if authors else None,
            category=paper_source.category.value,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            version=arxiv_id,
            extra={"sections_found": list(sections.keys())},
        )
        return RawDocument(raw_text=full_text, metadata=metadata)

    @staticmethod
    def _extract_arxiv_id(identifier: str) -> Optional[str]:
        match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", identifier)
        return match.group(0) if match else None

    def _fetch_arxiv_metadata(self, arxiv_id: str):
        response = self.session.get(
            ARXIV_API_URL, params={"id_list": arxiv_id}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        entry = root.find("atom:entry", ARXIV_ATOM_NS)
        if entry is None:
            return None, None, [], None

        title_el = entry.find("atom:title", ARXIV_ATOM_NS)
        summary_el = entry.find("atom:summary", ARXIV_ATOM_NS)
        title = title_el.text.strip() if title_el is not None and title_el.text else None
        abstract = summary_el.text.strip() if summary_el is not None and summary_el.text else None
        authors = [
            (author.find("atom:name", ARXIV_ATOM_NS).text or "").strip()
            for author in entry.findall("atom:author", ARXIV_ATOM_NS)
        ]

        pdf_url = None
        for link in entry.findall("atom:link", ARXIV_ATOM_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break

        return title, abstract, authors, pdf_url

    def _extract_pdf_text(self, pdf_url: str) -> Optional[str]:
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning("pypdf not installed; paper body text will be limited to the abstract.")
            return None

        try:
            response = self.session.get(pdf_url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            reader = PdfReader(BytesIO(response.content))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to extract PDF text from {pdf_url}: {exc}")
            return None

    @staticmethod
    def _segment_sections(body_text: str) -> dict:
        matches = list(SECTION_HEADING_PATTERN.finditer(body_text))
        sections: dict = {}
        for i, match in enumerate(matches):
            heading = match.group(1).strip().lower()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body_text)
            content = body_text[start:end].strip()
            if content:
                sections[heading] = content[:8000]  # cap per-section length defensively
        return sections
