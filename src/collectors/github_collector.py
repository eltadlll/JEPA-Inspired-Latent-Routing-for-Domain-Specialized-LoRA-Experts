"""Stage 2: GitHub collector.

Clones (or updates) configured repositories into `raw/github/<repo>` and
extracts README, docs/, examples/, tutorials/, notebooks/, and optionally
Python source, while skipping .git, node_modules, build/dist, caches,
virtualenvs, and binary files.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from src.collectors.base import BaseCollector, CollectorError, is_ignored_path
from src.config.schema import GitHubRepoConfig, RetryConfig
from src.processors.models import DocumentMetadata, RawDocument
from src.utils.hashing import document_id
from src.utils.io_utils import safe_filename
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".ipynb", ".txt"}
NOTEBOOK_EXTENSION = ".ipynb"


class GitHubCollector(BaseCollector):
    source_name = "github"

    def __init__(
        self,
        repos: List[GitHubRepoConfig],
        retry_config: RetryConfig,
        raw_output_dir: Path,
        clone_root: Optional[Path] = None,
    ) -> None:
        super().__init__(retry_config, raw_output_dir)
        self.repos = repos
        self.clone_root = clone_root or (raw_output_dir / "_repos")
        self.clone_root.mkdir(parents=True, exist_ok=True)

    def collect(self) -> List[RawDocument]:
        documents: List[RawDocument] = []
        for repo_config in self.repos:
            try:
                documents.extend(self._collect_repo(repo_config))
            except Exception as exc:  # noqa: BLE001 - a single bad repo must not kill the run
                logger.error(f"Failed to collect repo {repo_config.url}: {exc}")
        self._log_summary(documents)
        return documents

    def _collect_repo(self, repo_config: GitHubRepoConfig) -> List[RawDocument]:
        try:
            import git  # GitPython
        except ImportError as exc:
            raise CollectorError(
                "GitPython is required for GitHubCollector. Install with `pip install GitPython`."
            ) from exc

        repo_name = repo_config.url.rstrip("/").split("/")[-1].replace(".git", "")
        local_path = self.clone_root / safe_filename(repo_name)

        if local_path.exists():
            logger.info(f"Repo '{repo_name}' already cloned locally; pulling latest changes.")
            try:
                repo = git.Repo(str(local_path))
                repo.remotes.origin.pull()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Pull failed for {repo_name}, using existing local copy: {exc}")
        else:
            logger.info(f"Cloning {repo_config.url} -> {local_path}")
            clone_kwargs = {"depth": 1}
            if repo_config.branch:
                clone_kwargs["branch"] = repo_config.branch
            try:
                git.Repo.clone_from(repo_config.url, str(local_path), **clone_kwargs)
            except Exception as exc:
                raise CollectorError(f"git clone failed for {repo_config.url}: {exc}") from exc

        documents: List[RawDocument] = []
        documents.extend(self._extract_documentation(local_path, repo_config, repo_name))
        if repo_config.include_source:
            documents.extend(self._extract_source_code(local_path, repo_config, repo_name))
        return documents

    def _extract_documentation(
        self, local_path: Path, repo_config: GitHubRepoConfig, repo_name: str
    ) -> List[RawDocument]:
        documents: List[RawDocument] = []
        candidate_files: List[Path] = []

        for include_path in repo_config.include_paths:
            target = local_path / include_path
            if target.is_file():
                candidate_files.append(target)
            elif target.is_dir():
                candidate_files.extend(p for p in target.rglob("*") if p.is_file())

        for file_path in candidate_files:
            if is_ignored_path(file_path.relative_to(local_path)):
                continue
            if file_path.suffix.lower() not in DOC_EXTENSIONS:
                continue
            if file_path.stat().st_size > repo_config.max_file_size_kb * 1024:
                logger.debug(f"Skipping oversized file: {file_path}")
                continue

            text = self._read_text_file(file_path)
            if text is None or not text.strip():
                continue

            relative_path = str(file_path.relative_to(local_path))
            metadata = DocumentMetadata(
                document_id=document_id("github", f"{repo_name}/{relative_path}"),
                source=self.source_name,
                repository=repo_config.url,
                category=repo_config.category.value,
                framework=repo_config.framework,
                url=f"{repo_config.url.rstrip('.git')}/blob/main/{relative_path}",
                file_path=relative_path,
                programming_language=None,
            )
            documents.append(RawDocument(raw_text=text, metadata=metadata))

        return documents

    def _extract_source_code(
        self, local_path: Path, repo_config: GitHubRepoConfig, repo_name: str
    ) -> List[RawDocument]:
        documents: List[RawDocument] = []
        source_files = [
            p
            for p in local_path.rglob("*")
            if p.is_file()
            and p.suffix.lower() in set(repo_config.source_extensions)
            and not is_ignored_path(p.relative_to(local_path))
        ]

        for file_path in source_files:
            if file_path.stat().st_size > repo_config.max_file_size_kb * 1024:
                continue
            text = self._read_text_file(file_path)
            if text is None or not text.strip():
                continue

            relative_path = str(file_path.relative_to(local_path))
            wrapped = f"```{file_path.suffix.lstrip('.')}\n{text}\n```"
            metadata = DocumentMetadata(
                document_id=document_id("github_source", f"{repo_name}/{relative_path}"),
                source=self.source_name,
                repository=repo_config.url,
                category=repo_config.category.value,
                framework=repo_config.framework,
                url=f"{repo_config.url.rstrip('.git')}/blob/main/{relative_path}",
                file_path=relative_path,
                programming_language=file_path.suffix.lstrip("."),
            )
            documents.append(RawDocument(raw_text=wrapped, metadata=metadata))

        return documents

    @staticmethod
    def _read_text_file(file_path: Path) -> Optional[str]:
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError) as exc:
            logger.debug(f"Could not read {file_path}: {exc}")
            return None

    def cleanup_clones(self) -> None:
        """Optionally remove cloned repositories after extraction to save disk space."""
        if self.clone_root.exists():
            shutil.rmtree(self.clone_root, ignore_errors=True)
            logger.info(f"Removed cloned repos at {self.clone_root}")
