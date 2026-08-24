"""Stage 2: Blog collector.

Blogs are configured individually (no crawling of arbitrary link graphs,
to keep the pipeline's data provenance auditable). Markdown sources are
preferred; HTML pages are reduced to main-content text the same way the
documentation collector does.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import requests

from src.collectors.base import BaseCollector
from src.collectors.docs_collector import USER_AGENT, DocumentationCollector
from src.config.schema import BlogSourceConfig, RetryConfig
from src.processors.models import DocumentMetadata, RawDocument
from src.utils.hashing import document_id
from src.utils.logging_setup import get_logger
from src.utils.text_utils import normalize_whitespace

logger = get_logger(__name__)

REQUEST_TIMEOUT_SECONDS = 20


class BlogCollector(BaseCollector):
    source_name = "blog"

    def __init__(
        self,
        blog_sources: List[BlogSourceConfig],
        retry_config: RetryConfig,
        raw_output_dir: Path,
        session: Optional[requests.Session] = None,
    ) -> None:
        super().__init__(retry_config, raw_output_dir)
        self.blog_sources = blog_sources
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def collect(self) -> List[RawDocument]:
        documents: List[RawDocument] = []
        for blog_source in self.blog_sources:
            try:
                document = self._collect_post(blog_source)
                if document is not None:
                    documents.append(document)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to collect blog post {blog_source.url}: {exc}")
        self._log_summary(documents)
        return documents

    def _collect_post(self, blog_source: BlogSourceConfig) -> Optional[RawDocument]:
        response = self.session.get(blog_source.url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        raw_content = response.text

        if blog_source.url.endswith((".md", ".mdx")):
            text = normalize_whitespace(raw_content)
        else:
            text = DocumentationCollector._extract_html_main_content(raw_content)

        if not text or len(text.strip()) < 20:
            logger.debug(f"Skipping near-empty blog post: {blog_source.url}")
            return None

        metadata = DocumentMetadata(
            document_id=document_id("blog", blog_source.url),
            source=self.source_name,
            author=blog_source.author,
            category=blog_source.category.value,
            framework=blog_source.framework,
            url=blog_source.url,
        )
        return RawDocument(raw_text=text, metadata=metadata)
