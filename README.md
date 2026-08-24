# AI Agent Engineering & Data Science — Instruction Dataset Pipeline

A production-grade, modular data engineering pipeline that builds a
high-quality instruction-tuning dataset for a domain-specialized small
LLM (AI Agent Engineering, LLM Engineering, Retrieval Systems, Data
Science, AI System Design, Technical Project Management).

**This pipeline does not train a model.** Its only job is to collect,
clean, chunk, score, and convert source material into a validated
instruction-tuning dataset (Alpaca-style schema), ready for LoRA/QLoRA
fine-tuning with any framework of your choice.

---

## Architecture

```
project/
├── configs/
│   ├── config.yaml                  # main pipeline configuration
│   └── config.local.example.yaml    # example override file
├── src/
│   ├── config/          # Stage 1  — schema + YAML loader
│   ├── collectors/       # Stage 2  — github / docs / huggingface / blogs / papers
│   ├── processors/       # Stage 3, 5 — metadata enrichment, code extraction, core models
│   ├── cleaners/          # Stage 4  — cleaning + dedup (exact + SimHash near-dup)
│   ├── chunkers/          # Stage 6  — heading/code-aware semantic chunking
│   ├── scorers/           # Stage 7  — multi-signal quality scoring
│   ├── filters/           # Stage 8  — threshold filtering + removal reports
│   ├── generators/        # Stage 9  — rule-based instruction-example templates
│   ├── stats/             # Stage 10 — dataset statistics + charts
│   ├── validators/        # Stage 11 — schema / dedup / encoding validation
│   ├── exporters/         # Stage 12 — JSONL / Parquet / Arrow / CSV / HF dataset
│   ├── utils/              # logging, hashing, tokenizer, text, io helpers
│   └── pipeline.py         # orchestrates all 12 stages
├── data/
│   ├── raw/ intermediate/ clean/ processed/ instruction/ reports/ logs/
├── tests/                   # pytest unit tests (27 passing)
├── main.py                  # Typer CLI entrypoint
├── requirements.txt
├── pytest.ini
└── .env.example
```

Every stage consumes and produces the same core types
(`src/processors/models.py`: `RawDocument`, `Chunk`, `InstructionExample`),
so stages are independently testable and independently re-runnable.
Intermediate results are persisted as JSONL at each stage boundary
(`data/raw/collected.jsonl`, `data/clean/cleaned.jsonl`,
`data/processed/chunks.jsonl`, `data/processed/filtered.jsonl`,
`data/instruction/instruction_examples.raw.jsonl`), so a long run can be
resumed or debugged stage-by-stage without re-collecting from scratch.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`faiss-cpu`, `sentence-transformers`, `duckdb`, and `polars` are included
for downstream use (retrieval, alternate similarity backends, alternate
tabular engines) but are not required by the pipeline itself as shipped —
if you're on a constrained machine you can trim them from
`requirements.txt` and everything above still runs.

Copy `.env.example` to `.env` if you need a `GITHUB_TOKEN` (higher GitHub
API rate limits) or `HUGGINGFACE_TOKEN` (gated datasets), and reference
them in `configs/config.yaml` as `${GITHUB_TOKEN}`.

---

## Configuration

Edit `configs/config.yaml`. Every stage's behavior is controlled here —
sources, chunking, quality thresholds, retries, concurrency, logging,
tokenizer, instruction templates, and export formats. See
`src/config/schema.py` for the full, validated schema (Pydantic will
reject a malformed config immediately with a clear error, before any
network call or file write happens).

Environment variables can be referenced in the YAML as `${VAR_NAME}` or
`${VAR_NAME:-default_value}`. Interpolation happens **after** YAML
parsing (on string values only), so it never misfires on text inside
YAML comments.

For local experiments, copy `configs/config.local.example.yaml` to
`configs/config.local.yaml`, adjust it (e.g. lower quality thresholds,
fewer export formats), and pass it as an override:

```bash
python main.py run --config configs/config.yaml --override configs/config.local.yaml
```

---

## Usage

Run the full pipeline (all 12 stages):

```bash
python main.py run --config configs/config.yaml
```

Validate a config without running anything:

```bash
python main.py validate-config --config configs/config.yaml
```

Run an individual stage against already-persisted intermediate data
(useful when iterating on cleaning/chunking/scoring logic without
re-collecting from GitHub/HF/web every time):

```bash
python main.py run-stage collect --config configs/config.yaml
python main.py run-stage clean   --config configs/config.yaml
python main.py run-stage chunk   --config configs/config.yaml
python main.py run-stage score   --config configs/config.yaml
python main.py run-stage filter  --config configs/config.yaml
```

Output dataset lands in `data/instruction/` as
`<dataset_name>.train.jsonl` / `<dataset_name>.validation.jsonl` (plus
Parquet/Arrow/CSV/HF-dataset variants if configured). Each record
matches the schema:

```json
{
  "id": "…",
  "instruction": "…",
  "input": "…",
  "output": "…",
  "category": "agent_engineering",
  "difficulty": "intermediate",
  "source": "github",
  "metadata": {
    "template": "code_generation",
    "document_id": "…",
    "chunk_id": "…",
    "framework": "langgraph",
    "url": "…",
    "quality_score": 0.71
  }
}
```

Reports land in `data/reports/`:
- `dataset_statistics.json` (+ PNG charts) — Stage 10
- `filter_report.json` — Stage 8, explains every document that was dropped and why
- `validation_report.json` — Stage 11

---

## Design notes

**Instruction generation is fully rule-based (Stage 9).** No external
LLM API call is made to synthesize instruction/output pairs — every
example is deterministically derived from the collected corpus via
regex/heuristic templates (`src/generators/templates.py`: QA,
instruction-following, code-generation, architecture-explanation,
debugging, reasoning, comparison). This was a deliberate choice for a
research pipeline: it makes the dataset **100% reproducible** from the
source list alone, with no dependency on a paid API, model version
drift, or nondeterministic sampling. If you want higher-quality,
more varied instructions later, you can swap in an LLM-based generator
behind the same `InstructionGenerator` interface — the chunk/document
inputs and `InstructionExample` output contract stay the same.

**Near-duplicate detection uses SimHash, not embeddings**, bucketed by
high-order bits before pairwise Hamming-distance comparison. This keeps
dedup roughly linear rather than O(n²), and avoids a hard dependency on
loading an embedding model just to clean text. `sentence-transformers`
+ `faiss-cpu` are included in `requirements.txt` if you want to swap in
embedding-based semantic dedup for a research-quality upgrade.

**Quality scoring is a transparent, weighted composite** of nine
independently-computed features (token count, code ratio, documentation
completeness, readability, duplicate probability, source credibility,
example density, instruction density, broken links) — see
`src/scorers/quality_scorer.py`. Every document's `quality_report` is
kept alongside its score, so filtering decisions in Stage 8 are
auditable rather than a black box.

**Tokenization degrades gracefully.** `src/utils/tokenizer.py` tries
`tiktoken`'s `cl100k_base` encoding first; if that fails (e.g. no
network access to fetch the BPE file on first use, which happens in
sandboxed/offline environments), it falls back to a word-count heuristic
(~1.3 tokens/word) so the pipeline keeps running rather than crashing.

---

## Testing

```bash
pytest
```

27 unit tests cover: hashing/simhash, tokenizer fallback, text
normalization, config loading + validation + env interpolation,
cleaning, deduplication, code extraction, semantic chunking, quality
scoring, filtering, instruction generation, and dataset validation.

---

## Extending the pipeline

- **New source type**: subclass `BaseCollector` in
  `src/collectors/base.py`, implement `collect() -> List[RawDocument]`,
  register it in `src/pipeline.py::run_collection`, and add its config
  model to `src/config/schema.py::SourcesConfig`.
- **New instruction template**: add a function with signature
  `(Chunk, DocumentMetadata) -> List[dict]` to
  `src/generators/templates.py` and register it in `TEMPLATE_REGISTRY`.
- **New export format**: add a handler method to `DatasetExporter` in
  `src/exporters/exporter.py` and a new `ExportFormat` enum value.
- **LLM-assisted instruction generation**: implement an alternate
  `InstructionGenerator`-shaped class that calls your model of choice
  per chunk; swap it into `src/pipeline.py::run_instruction_generation`.

---

## License

This pipeline is infrastructure code for building your own dataset. It
does not bundle any collected content. Respect the license of every
source you configure it to collect from (GitHub repo licenses, arXiv's
terms, Hugging Face dataset cards, individual blog terms of use) before
using or redistributing the resulting dataset.
