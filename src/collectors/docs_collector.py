"""Stage 2: Documentation site collector.

Pulls raw Markdown when a documentation source publishes it directly
(preferred, since it avoids scraping rendered HTML), and falls back to
HTML extraction via BeautifulSoup otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import requests

from src.collectors.base import BaseCollector
from src.config.schema import DocSourceConfig, RetryConfig
from src.processors.models import DocumentMetadata, RawDocument
from src.utils.hashing import document_id
from src.utils.logging_setup import get_logger
from src.utils.text_utils import normalize_whitespace, strip_html_tags

logger = get_logger(__name__)

REQUEST_TIMEOUT_SECONDS = 20
USER_AGENT = "dataset-pipeline/1.0 (+research use; contact via config)"


class DocumentationCollector(BaseCollector):
    source_name = "documentation"

    def __init__(
        self,
        doc_sources: List[DocSourceConfig],
        retry_config: RetryConfig,
        raw_output_dir: Path,
        session: Optional[requests.Session] = None,
    ) -> None:
        super().__init__(retry_config, raw_output_dir)
        self.doc_sources = doc_sources
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def collect(self) -> List[RawDocument]:
        documents: List[RawDocument] = []
        for doc_source in self.doc_sources:
            for url in doc_source.urls:
                try:
                    document = self._collect_page(doc_source, url)
                    if document is not None:
                        documents.append(document)
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Failed to collect doc page {url}: {exc}")
        self._log_summary(documents)
        return documents

    def _fetch(self, url: str) -> str:
        response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.text

    def _collect_page(self, doc_source: DocSourceConfig, url: str) -> Optional[RawDocument]:
        raw_content = self._fetch(url)

        if doc_source.format == "markdown" or url.endswith((".md", ".mdx")):
            text = normalize_whitespace(raw_content)
        elif doc_source.format == "html":
            text = self._extract_html_main_content(raw_content)
        else:
            logger.warning(f"Unsupported doc format '{doc_source.format}' for {url}; skipping.")
            return None

        if not text or len(text.strip()) < 20:
            logger.debug(f"Skipping near-empty page: {url}")
            return None

        metadata = DocumentMetadata(
            document_id=document_id("documentation", url),
            source=self.source_name,
            repository=None,
            category=doc_source.category.value,
            framework=doc_source.framework,
            url=url,
            file_path=None,
        )
        return RawDocument(raw_text=text, metadata=metadata)

    @staticmethod
    def _extract_html_main_content(html: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("beautifulsoup4 not installed; falling back to regex tag stripping.")
            return normalize_whitespace(strip_html_tags(html))

        soup = BeautifulSoup(html, "html.parser")

        for tag_name in ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(attrs={"role": "main"})
            or soup.find(id="content")
            or soup.body
            or soup
        )
        text = main.get_text(separator="\n")
        return normalize_whitespace(text)
