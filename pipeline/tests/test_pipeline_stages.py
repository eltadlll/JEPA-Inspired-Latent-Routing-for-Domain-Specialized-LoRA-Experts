from src.chunkers.semantic_chunker import SemanticChunker
from src.cleaners.cleaner import DocumentCleaner
from src.cleaners.deduplicator import Deduplicator
from src.config.schema import ChunkingConfig, InstructionGenerationConfig, QualityThresholdsConfig
from src.filters.quality_filter import QualityFilter
from src.generators.instruction_generator import InstructionGenerator
from src.processors.code_extractor import CodeExtractor
from src.processors.metadata import MetadataEnricher
from src.processors.models import DocumentMetadata, RawDocument
from src.scorers.quality_scorer import QualityScorer
from src.validators.dataset_validator import DatasetValidator


def make_document(text: str, doc_id: str = "doc1", source: str = "documentation", category: str = "agent_engineering") -> RawDocument:
    metadata = DocumentMetadata(document_id=doc_id, source=source, category=category, framework="langgraph")
    return RawDocument(raw_text=text, metadata=metadata)


SAMPLE_DOC = """# Building an Agent with LangGraph

We use cookies to improve your experience.

## What is LangGraph?

LangGraph is a library for building stateful, multi-agent workflows on top of LLMs.
It should be used when you need cyclic graphs rather than simple chains.

Here is a basic example:

```python
from langgraph.graph import StateGraph

def build_graph():
    graph = StateGraph()
    graph.add_node("agent", agent_node)
    return graph.compile()
```

## Debugging common issues

If you encounter a `KeyError` when running the graph, it is usually because
a required key was not initialized in the state. First, check your reducer
functions. Then, verify the initial state passed to `.invoke()`.

All rights reserved.
"""


def test_cleaner_removes_boilerplate_and_keeps_code():
    document = make_document(SAMPLE_DOC)
    cleaned = DocumentCleaner().clean(document)
    assert "cookies" not in cleaned.raw_text.lower()
    assert "all rights reserved" not in cleaned.raw_text.lower()
    assert "StateGraph" in cleaned.raw_text
    assert "```python" in cleaned.raw_text


def test_metadata_enricher_sets_token_count_and_hash():
    document = make_document(SAMPLE_DOC)
    enriched = MetadataEnricher().enrich(document)
    assert enriched.metadata.token_count > 0
    assert enriched.metadata.document_length_chars == len(document.raw_text)
    assert len(enriched.metadata.content_hash) > 0


def test_deduplicator_removes_exact_and_near_duplicates():
    doc_a = MetadataEnricher().enrich(make_document(SAMPLE_DOC, doc_id="a"))
    doc_b = MetadataEnricher().enrich(make_document(SAMPLE_DOC, doc_id="b"))  # exact duplicate content
    doc_c = MetadataEnricher().enrich(
        make_document("Completely unrelated content about quantum computing and entanglement.", doc_id="c")
    )

    deduped = Deduplicator(max_duplicate_similarity=0.92).deduplicate([doc_a, doc_b, doc_c])
    ids = {d.document_id for d in deduped}
    assert "c" in ids
    assert len(ids) == 2  # one of a/b removed as exact duplicate


def test_code_extractor_labels_python_and_links_context():
    document = make_document(SAMPLE_DOC)
    extracted = CodeExtractor().extract(document)
    assert len(extracted.code_blocks) == 1
    block = extracted.code_blocks[0]
    assert block.language == "python"
    assert "StateGraph" in block.code
    assert "example" in block.surrounding_context.lower()


def test_chunker_respects_headings_and_code_blocks():
    document = make_document(SAMPLE_DOC)
    config = ChunkingConfig(chunk_size_tokens=200, chunk_overlap_tokens=10, min_chunk_tokens=5, max_chunk_tokens=2000)
    chunks = SemanticChunker(config).chunk_document(document)
    assert len(chunks) >= 1
    # No chunk should contain a broken/unbalanced code fence.
    for chunk in chunks:
        assert chunk.text.count("```") % 2 == 0


def test_quality_scorer_produces_bounded_score():
    document = make_document(SAMPLE_DOC)
    document = MetadataEnricher().enrich(document)
    document = CodeExtractor().extract(document)
    scored = QualityScorer(min_token_count=5).score(document)
    assert scored.quality_score is not None
    assert 0.0 <= scored.quality_score <= 1.0
    assert "code_ratio" in scored.quality_report


def test_quality_filter_removes_low_scoring_documents():
    good_doc = make_document(SAMPLE_DOC, doc_id="good")
    good_doc = MetadataEnricher().enrich(good_doc)
    good_doc = CodeExtractor().extract(good_doc)
    good_doc = QualityScorer(min_token_count=5).score(good_doc)

    bad_doc = make_document("short", doc_id="bad")
    bad_doc = MetadataEnricher().enrich(bad_doc)
    bad_doc.quality_score = 0.1
    bad_doc.quality_report = {"readability": 0.9}

    thresholds = QualityThresholdsConfig(min_quality_score=0.3, min_token_count=5)
    filtered, report = QualityFilter(thresholds).filter_documents([good_doc, bad_doc])

    kept_ids = {d.document_id for d in filtered}
    assert "bad" not in kept_ids
    assert report.total_removed >= 1


def test_instruction_generator_creates_examples_from_chunks():
    document = make_document(SAMPLE_DOC)
    document = MetadataEnricher().enrich(document)
    document = CodeExtractor().extract(document)

    config = ChunkingConfig(chunk_size_tokens=200, chunk_overlap_tokens=0, min_chunk_tokens=5, max_chunk_tokens=2000)
    chunks = SemanticChunker(config).chunk_document(document)

    gen_config = InstructionGenerationConfig(
        templates=["qa", "code_generation", "debugging"], max_examples_per_document=10
    )
    examples = InstructionGenerator(gen_config).generate(chunks, {document.document_id: document})

    assert len(examples) > 0
    for example in examples:
        assert example.instruction.strip()
        assert example.output.strip()
        assert example.category == "agent_engineering"


def test_validator_flags_missing_fields_and_deduplicates_ids():
    from src.processors.models import InstructionExample

    valid_example = InstructionExample(
        id="ex1", instruction="What is X?", input="", output="X is Y.",
        category="agent_engineering", difficulty="beginner", source="documentation",
    )
    empty_output_example = InstructionExample(
        id="ex2", instruction="What is Z?", input="", output="",
        category="agent_engineering", difficulty="beginner", source="documentation",
    )
    duplicate_id_example = InstructionExample(
        id="ex1", instruction="Another question?", input="", output="Another answer.",
        category="agent_engineering", difficulty="beginner", source="documentation",
    )

    valid, report = DatasetValidator(strict=False).validate(
        [valid_example, empty_output_example, duplicate_id_example]
    )
    assert len(valid) == 1
    assert valid[0].id == "ex1"
    assert report.total_examples == 3
    assert report.valid_examples == 1
