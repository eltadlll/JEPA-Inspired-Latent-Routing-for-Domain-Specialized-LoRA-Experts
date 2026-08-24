"""
Configuration schema definitions for the dataset generation pipeline.

All runtime configuration is validated through these Pydantic models before
any pipeline stage executes. This guarantees that malformed YAML fails fast,
at startup, with a clear error message rather than deep inside a collector
or exporter at 2am during an unattended run.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ExportFormat(str, Enum):
    JSONL = "jsonl"
    PARQUET = "parquet"
    ARROW = "arrow"
    CSV = "csv"
    HF_DATASET = "hf_dataset"


class DocumentCategory(str, Enum):
    AGENT_ENGINEERING = "agent_engineering"
    LLM_ENGINEERING = "llm_engineering"
    RETRIEVAL_SYSTEMS = "retrieval_systems"
    DATA_SCIENCE = "data_science"
    AI_SYSTEM_DESIGN = "ai_system_design"
    TECHNICAL_PROJECT_MANAGEMENT = "technical_project_management"
    UNCATEGORIZED = "uncategorized"


class DirectoriesConfig(BaseModel):
    """Filesystem layout used by every stage of the pipeline."""

    root: Path = Field(default=Path("data"))
    raw: Path = Field(default=Path("data/raw"))
    intermediate: Path = Field(default=Path("data/intermediate"))
    clean: Path = Field(default=Path("data/clean"))
    processed: Path = Field(default=Path("data/processed"))
    instruction: Path = Field(default=Path("data/instruction"))
    reports: Path = Field(default=Path("data/reports"))
    logs: Path = Field(default=Path("data/logs"))

    def all_paths(self) -> List[Path]:
        return [
            self.root,
            self.raw,
            self.intermediate,
            self.clean,
            self.processed,
            self.instruction,
            self.reports,
            self.logs,
        ]

    def ensure_exist(self) -> None:
        for path in self.all_paths():
            path.mkdir(parents=True, exist_ok=True)


class GitHubRepoConfig(BaseModel):
    """A single GitHub repository to mine for documentation and code."""

    url: str
    category: DocumentCategory = DocumentCategory.UNCATEGORIZED
    framework: Optional[str] = None
    branch: Optional[str] = None
    include_paths: List[str] = Field(
        default_factory=lambda: ["README.md", "docs", "examples", "tutorials", "notebooks"]
    )
    include_source: bool = False
    source_extensions: List[str] = Field(default_factory=lambda: [".py"])
    max_file_size_kb: int = 512

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://", "git@")):
            raise ValueError(f"Invalid git URL: {v}")
        return v


class DocSourceConfig(BaseModel):
    """A documentation site or set of raw markdown/HTML/PDF pages."""

    name: str
    base_url: str
    category: DocumentCategory = DocumentCategory.UNCATEGORIZED
    framework: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    format: str = Field(default="markdown", description="markdown | html | pdf")


class HuggingFaceDatasetConfig(BaseModel):
    """A Hugging Face dataset to pull (streamed or fully downloaded)."""

    repo_id: str
    split: str = "train"
    category: DocumentCategory = DocumentCategory.UNCATEGORIZED
    streaming: bool = True
    max_examples: Optional[int] = 2000
    text_field: str = "text"


class BlogSourceConfig(BaseModel):
    """A single blog post or blog index to pull markdown/HTML content from."""

    url: str
    category: DocumentCategory = DocumentCategory.UNCATEGORIZED
    framework: Optional[str] = None
    author: Optional[str] = None


class PaperSourceConfig(BaseModel):
    """A research paper, identified by arXiv id or direct PDF/abstract URL."""

    identifier: str = Field(description="arXiv ID (e.g. 2305.18290) or direct URL")
    category: DocumentCategory = DocumentCategory.UNCATEGORIZED


class SourcesConfig(BaseModel):
    github_repos: List[GitHubRepoConfig] = Field(default_factory=list)
    documentation: List[DocSourceConfig] = Field(default_factory=list)
    huggingface_datasets: List[HuggingFaceDatasetConfig] = Field(default_factory=list)
    blogs: List[BlogSourceConfig] = Field(default_factory=list)
    papers: List[PaperSourceConfig] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    strategy: str = Field(default="semantic", description="semantic | fixed")
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120
    min_chunk_tokens: int = 40
    max_chunk_tokens: int = 2000
    respect_code_blocks: bool = True

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def overlap_must_be_smaller(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size_tokens", 800)
        if v >= chunk_size:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return v


class QualityThresholdsConfig(BaseModel):
    min_quality_score: float = Field(default=0.55, ge=0.0, le=1.0)
    min_token_count: int = 30
    max_duplicate_similarity: float = Field(default=0.92, ge=0.0, le=1.0)
    min_readability_score: float = Field(default=0.3, ge=0.0, le=1.0)


class RetryConfig(BaseModel):
    max_attempts: int = 4
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    backoff_multiplier: float = 2.0


class ConcurrencyConfig(BaseModel):
    max_workers: int = 6
    github_clone_workers: int = 2
    http_request_workers: int = 8


class LoggingConfig(BaseModel):
    level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_file_name: str = "pipeline.log"
    rotation: str = "10 MB"
    retention: str = "10 days"
    serialize_json: bool = False


class TokenizerConfig(BaseModel):
    encoding_name: str = "cl100k_base"
    max_document_tokens: int = 50_000


class InstructionGenerationConfig(BaseModel):
    templates: List[str] = Field(
        default_factory=lambda: [
            "qa",
            "instruction_following",
            "code_generation",
            "architecture_explanation",
            "debugging",
            "reasoning",
            "comparison",
        ]
    )
    max_examples_per_document: int = 3
    difficulty_levels: List[str] = Field(default_factory=lambda: ["beginner", "intermediate", "advanced"])


class ExportConfig(BaseModel):
    formats: List[ExportFormat] = Field(default_factory=lambda: [ExportFormat.JSONL])
    dataset_name: str = "ai-agent-datascience-instruct"
    train_split: float = 0.9
    shuffle_seed: int = 42

    @field_validator("train_split")
    @classmethod
    def validate_split(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("train_split must be between 0 and 1 (exclusive)")
        return v


class PipelineConfig(BaseModel):
    """Root configuration object. This is what `ConfigLoader` produces."""

    project_name: str = "ai-agent-datascience-llm-dataset"
    directories: DirectoriesConfig = Field(default_factory=DirectoriesConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    quality: QualityThresholdsConfig = Field(default_factory=QualityThresholdsConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    instruction_generation: InstructionGenerationConfig = Field(
        default_factory=InstructionGenerationConfig
    )
    export: ExportConfig = Field(default_factory=ExportConfig)

    @model_validator(mode="after")
    def validate_has_sources(self) -> "PipelineConfig":
        total_sources = (
            len(self.sources.github_repos)
            + len(self.sources.documentation)
            + len(self.sources.huggingface_datasets)
            + len(self.sources.blogs)
            + len(self.sources.papers)
        )
        if total_sources == 0:
            raise ValueError(
                "No data sources configured. Add at least one source under `sources:` in the YAML config."
            )
        return self
