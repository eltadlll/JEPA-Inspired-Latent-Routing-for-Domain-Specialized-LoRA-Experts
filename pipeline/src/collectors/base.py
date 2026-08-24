"""Stage 2: Raw Data Collection - base abstractions.

Every concrete collector (GitHub, documentation sites, Hugging Face,
blogs, papers) implements `BaseCollector.collect()` and returns a list of
`RawDocument` objects. Shared retry/backoff behavior lives here so
individual collectors stay focused on source-specific parsing logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.schema import RetryConfig
from src.processors.models import RawDocument
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Ignored path fragments shared by all collectors that walk a filesystem
# (GitHub clones, local doc mirrors, etc.)
IGNORED_PATH_FRAGMENTS = {
    ".git",
    "node_modules",
    "build",
    "dist",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    "site-packages",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".whl", ".so", ".dylib", ".dll",
    ".pyc", ".pyo", ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3",
    ".bin", ".exe", ".class", ".jar",
}


def is_ignored_path(path: Path) -> bool:
    parts = set(path.parts)
    if parts & IGNORED_PATH_FRAGMENTS:
        return True
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    return False


def build_retry_decorator(retry_config: RetryConfig) -> Callable:
    """Build a tenacity retry decorator from validated `RetryConfig`."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(retry_config.max_attempts),
        wait=wait_exponential(
            multiplier=retry_config.initial_backoff_seconds,
            max=retry_config.max_backoff_seconds,
            exp_base=retry_config.backoff_multiplier,
        ),
        retry=retry_if_exception_type((IOError, ConnectionError, TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying after error (attempt {retry_state.attempt_number}): "
            f"{retry_state.outcome.exception() if retry_state.outcome else 'unknown'}"
        ),
    )


class CollectorError(Exception):
    """Raised when a collector cannot produce any documents from a source."""


class BaseCollector(ABC):
    """Abstract base class for all Stage 2 collectors."""

    source_name: str = "base"

    def __init__(self, retry_config: RetryConfig, raw_output_dir: Path) -> None:
        self.retry_config = retry_config
        self.raw_output_dir = raw_output_dir
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def collect(self) -> List[RawDocument]:
        """Collect and return all documents this collector is responsible for.

        Implementations should never raise on a single failed item; they
        should log a warning, skip it, and keep going. `CollectorError`
        should only be raised if the entire source is unreachable.
        """
        raise NotImplementedError

    def _log_summary(self, documents: List[RawDocument]) -> None:
        logger.info(f"[{self.source_name}] collected {len(documents)} document(s).")
