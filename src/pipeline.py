"""
Pipeline orchestrator.

Wires together Stages 1-12 in order. Each stage's output is also persisted
to disk under `directories.intermediate` / `directories.clean` /
`directories.processed` / `directories.instruction`, so the pipeline can be
resumed or inspected stage-by-stage without re-running earlier, expensive
stages (e.g. re-cloning GitHub repos).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from src.chunkers.semantic_chunker import SemanticChunker
from src.cleaners.cleaner import DocumentCleaner
from src.cleaners.deduplicator import Deduplicator
from src.collectors.blog_collector import BlogCollector
from src.collectors.docs_collector import DocumentationCollector
from src.collectors.github_collector import GitHubCollector
from src.collectors.huggingface_collector import HuggingFaceCollector
from src.collectors.paper_collector import PaperCollector
from src.config.schema import PipelineConfig
from src.exporters.exporter import DatasetExporter
from src.filters.quality_filter import QualityFilter
from src.generators.instruction_generator import InstructionGenerator
from src.processors.code_extractor import CodeExtractor
from src.processors.metadata import MetadataEnricher
from src.processors.models import Chunk, InstructionExample, RawDocument
from src.scorers.quality_scorer import QualityScorer
from src.stats.report_generator import ReportGenerator
from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.logging_setup import get_logger
from src.validators.dataset_validator import DatasetValidator

logger = get_logger(__name__)


class DatasetPipeline:
    """End-to-end orchestrator for Stages 2 through 12.

    (Stage 1, configuration management, has already happened by the time
    this class is constructed -- `config` here is the validated result.)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.dirs = config.directories

    # ------------------------------------------------------------------
    # Stage 2: Collection
    # ------------------------------------------------------------------
    def run_collection(self) -> List[RawDocument]:
        logger.info("=== Stage 2: Raw Data Collection ===")
        documents: List[RawDocument] = []

        if self.config.sources.github_repos:
            collector = GitHubCollector(
                self.config.sources.github_repos, self.config.retry, self.dirs.raw / "github"
            )
            documents.extend(collector.collect())

        if self.config.sources.documentation:
            collector = DocumentationCollector(
                self.config.sources.documentation, self.config.retry, self.dirs.raw / "documentation"
            )
            documents.extend(collector.collect())

        if self.config.sources.huggingface_datasets:
            collector = HuggingFaceCollector(
                self.config.sources.huggingface_datasets, self.config.retry, self.dirs.raw / "huggingface"
            )
            documents.extend(collector.collect())

        if self.config.sources.blogs:
            collector = BlogCollector(
                self.config.sources.blogs, self.config.retry, self.dirs.raw / "blogs"
            )
            documents.extend(collector.collect())

        if self.config.sources.papers:
            collector = PaperCollector(
                self.config.sources.papers, self.config.retry, self.dirs.raw / "papers"
            )
            documents.extend(collector.collect())

        logger.info(f"Stage 2 complete: {len(documents)} raw document(s) collected.")
        self._persist_documents(documents, self.dirs.raw / "collected.jsonl")
        return documents

    # ------------------------------------------------------------------
    # Stage 3: Metadata enrichment
    # ------------------------------------------------------------------
    def run_metadata_enrichment(self, documents: List[RawDocument]) -> List[RawDocument]:
        logger.info("=== Stage 3: Metadata Generation ===")
        enricher = MetadataEnricher(self.config.tokenizer.encoding_name)
        return enricher.enrich_batch(documents)

    # ------------------------------------------------------------------
    # Stage 4: Cleaning + dedup
    # ------------------------------------------------------------------
    def run_cleaning(self, documents: List[RawDocument]) -> List[RawDocument]:
        logger.info("=== Stage 4: Cleaning Pipeline ===")
        cleaner = DocumentCleaner()
        cleaned = cleaner.clean_batch(documents)

        # Re-enrich metadata since cleaning changes text length/hash/token count.
        enricher = MetadataEnricher(self.config.tokenizer.encoding_name)
        cleaned = enricher.enrich_batch(cleaned)

        deduplicator = Deduplicator(self.config.quality.max_duplicate_similarity)
        deduplicated = deduplicator.deduplicate(cleaned)

        self._persist_documents(deduplicated, self.dirs.clean / "cleaned.jsonl")
        return deduplicated

    # ------------------------------------------------------------------
    # Stage 5: Code extraction
    # ------------------------------------------------------------------
    def run_code_extraction(self, documents: List[RawDocument]) -> List[RawDocument]:
        logger.info("=== Stage 5: Code Extraction ===")
        extractor = CodeExtractor()
        return extractor.extract_batch(documents)

    # ------------------------------------------------------------------
    # Stage 6: Chunking
    # ------------------------------------------------------------------
    def run_chunking(self, documents: List[RawDocument]) -> List[Chunk]:
        logger.info("=== Stage 6: Document Chunking ===")
        chunker = SemanticChunker(self.config.chunking, self.config.tokenizer.encoding_name)
        chunks = chunker.chunk_batch(documents)
        write_jsonl(self.dirs.processed / "chunks.jsonl", (c.to_dict() for c in chunks))
        return chunks

    # ------------------------------------------------------------------
    # Stage 7: Quality scoring
    # ------------------------------------------------------------------
    def run_quality_scoring(self, documents: List[RawDocument]) -> List[RawDocument]:
        logger.info("=== Stage 7: Quality Scoring ===")
        scorer = QualityScorer(min_token_count=self.config.quality.min_token_count)
        return scorer.score_batch(documents)

    # ------------------------------------------------------------------
    # Stage 8: Filtering
    # ------------------------------------------------------------------
    def run_filtering(self, documents: List[RawDocument]):
        logger.info("=== Stage 8: Filtering ===")
        quality_filter = QualityFilter(self.config.quality)
        filtered, report = quality_filter.filter_documents(documents)
        quality_filter.save_report(report, self.dirs.reports / "filter_report.json")
        self._persist_documents(filtered, self.dirs.processed / "filtered.jsonl")
        return filtered, report

    # ------------------------------------------------------------------
    # Stage 9: Instruction generation
    # ------------------------------------------------------------------
    def run_instruction_generation(
        self, chunks: List[Chunk], documents: List[RawDocument]
    ) -> List[InstructionExample]:
        logger.info("=== Stage 9: Instruction Dataset Generation ===")
        documents_by_id = {d.document_id: d for d in documents}
        # Keep only chunks whose parent document survived filtering.
        surviving_chunks = [c for c in chunks if c.parent_document_id in documents_by_id]

        generator = InstructionGenerator(self.config.instruction_generation)
        examples = generator.generate(surviving_chunks, documents_by_id)
        write_jsonl(
            self.dirs.instruction / "instruction_examples.raw.jsonl", (e.to_dict() for e in examples)
        )
        return examples

    # ------------------------------------------------------------------
    # Stage 10: Statistics
    # ------------------------------------------------------------------
    def run_statistics(
        self, raw_documents, filtered_documents, filter_report, instruction_examples
    ) -> dict:
        logger.info("=== Stage 10: Dataset Statistics ===")
        report_generator = ReportGenerator(self.dirs.reports)
        return report_generator.generate(raw_documents, filtered_documents, filter_report, instruction_examples)

    # ------------------------------------------------------------------
    # Stage 11: Validation
    # ------------------------------------------------------------------
    def run_validation(self, examples: List[InstructionExample]):
        logger.info("=== Stage 11: Validation ===")
        validator = DatasetValidator(strict=False)
        valid_examples, report = validator.validate(examples)
        from src.utils.io_utils import write_json

        write_json(self.dirs.reports / "validation_report.json", report.to_dict())
        return valid_examples, report

    # ------------------------------------------------------------------
    # Stage 12: Export
    # ------------------------------------------------------------------
    def run_export(self, examples: List[InstructionExample]) -> List[Path]:
        logger.info("=== Stage 12: Export ===")
        exporter = DatasetExporter(self.config.export, self.dirs.instruction)
        return exporter.export(examples)

    # ------------------------------------------------------------------
    def run_full_pipeline(self) -> dict:
        """Runs every stage in order and returns the final statistics dict."""
        raw_documents = self.run_collection()
        raw_documents = self.run_metadata_enrichment(raw_documents)
        cleaned_documents = self.run_cleaning(raw_documents)
        code_extracted_documents = self.run_code_extraction(cleaned_documents)
        chunks = self.run_chunking(code_extracted_documents)
        scored_documents = self.run_quality_scoring(code_extracted_documents)
        filtered_documents, filter_report = self.run_filtering(scored_documents)
        instruction_examples = self.run_instruction_generation(chunks, filtered_documents)
        valid_examples, _validation_report = self.run_validation(instruction_examples)
        exported_paths = self.run_export(valid_examples)
        stats = self.run_statistics(raw_documents, filtered_documents, filter_report, valid_examples)

        logger.info("=== Pipeline complete ===")
        logger.info(f"Exported files: {[str(p) for p in exported_paths]}")
        return stats

    @staticmethod
    def _persist_documents(documents: List[RawDocument], path: Path) -> None:
        write_jsonl(path, (d.to_dict() for d in documents))

    @staticmethod
    def load_documents(path: Path) -> List[RawDocument]:
        return [RawDocument.from_dict(record) for record in read_jsonl(path)]
