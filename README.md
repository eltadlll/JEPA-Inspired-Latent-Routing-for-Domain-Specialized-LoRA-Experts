# JEPA-Inspired Latent Routing for Domain-Specialized LoRA Experts

A research project that goes end-to-end: collect and clean a technical
instruction-tuning corpus, split it into five domain-specialized
datasets, fine-tune a small LLM into domain experts via LoRA, and test
whether a self-supervised JEPA-style objective can learn a better
expert-routing space than a simple baseline.

**Headline finding, reported honestly:** the JEPA-learned router did not
beat a simple Sentence Transformer baseline (80% vs. 90% routing
accuracy), and — more importantly — fine-tuning the base model into
domain experts at this data scale *reduced* answer quality rather than
improving it (raw base 5.33/10 > combined fine-tune 4.93/10 > JEPA-routed
MoE-LoRA 2.18/10, Claude-judged, n=20). See
[`RESULTS.md`](#part-3-fine-tuning--moe-lora--jepa-research) below for
the full breakdown and why.

---

## Repository structure

```
project/
├── pipeline/                # Part 1 — dataset collection pipeline
│   ├── configs/
│   ├── src/
│   │   ├── config/ collectors/ processors/ cleaners/ chunkers/
│   │   ├── scorers/ filters/ generators/ stats/ validators/ exporters/
│   │   └── pipeline.py
│   ├── data/
│   ├── tests/
│   └── main.py
├── datasets/                 # Part 2 — 5 domain-specific fine-tuning datasets
│   ├── ai_system_design.jsonl
│   ├── retrieval_systems.jsonl
│   ├── llm_engineering.jsonl
│   ├── agent_engineering.jsonl
│   ├── data_science.jsonl
│   └── eval.jsonl
├── research/                 # Part 3 — fine-tuning + MoE-LoRA + JEPA
│   ├── moe_lora_jepa_final.ipynb
│   ├── jepa_router_experiment_colab.py
│   ├── push_to_hub.py
│   └── demo/app.py
└── paper/
    ├── JEPA_LoRA_Expert_Routing_Research_Paper.docx
    └── JEPA_MoE_LoRA_Presentation.pptx
```

---

## Part 1: Dataset collection pipeline

A production-grade, modular data engineering pipeline that collects,
cleans, chunks, scores, and exports a raw instruction-tuning corpus
across six technical categories (AI Agent Engineering, LLM Engineering,
Retrieval Systems, Data Science, AI System Design, Technical Project
Management). **This pipeline does not train a model** — its only job is
to turn source material (GitHub repos, docs, Hugging Face datasets,
blogs, papers) into a validated Alpaca-style instruction dataset, ready
for LoRA/QLoRA fine-tuning with any framework.

### Architecture

Twelve independently-runnable stages, each consuming and producing the
same core types (`src/processors/models.py`: `RawDocument`, `Chunk`,
`InstructionExample`), with intermediate JSONL persisted at every stage
boundary so a long run can be resumed or debugged without re-collecting
from scratch:

| Stage | Module | Does |
|---|---|---|
| 1 | `config/` | Schema + YAML loader, env-var interpolation, Pydantic validation |
| 2 | `collectors/` | GitHub / docs / Hugging Face / blogs / papers |
| 3 | `processors/` | Metadata enrichment, code extraction |
| 4 | `cleaners/` | Cleaning + dedup (exact + SimHash near-duplicate) |
| 5 | `processors/` | Core model construction |
| 6 | `chunkers/` | Heading/code-aware semantic chunking |
| 7 | `scorers/` | Multi-signal quality scoring (9 weighted features) |
| 8 | `filters/` | Threshold filtering + removal reports |
| 9 | `generators/` | Rule-based instruction-example templates |
| 10 | `stats/` | Dataset statistics + charts |
| 11 | `validators/` | Schema / dedup / encoding validation |
| 12 | `exporters/` | JSONL / Parquet / Arrow / CSV / HF dataset |

### Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`faiss-cpu`, `sentence-transformers`, `duckdb`, and `polars` are included
for downstream use (retrieval, alternate similarity backends, alternate
tabular engines) but are not required by the pipeline itself — trim them
from `requirements.txt` on a constrained machine and everything above
still runs.

Copy `.env.example` to `.env` for `GITHUB_TOKEN` (higher API rate
limits) or `HUGGINGFACE_TOKEN` (gated datasets), referenced in
`configs/config.yaml` as `${GITHUB_TOKEN}`.

### Usage

```bash
# Full pipeline, all 12 stages
python main.py run --config configs/config.yaml

# Validate a config without running anything
python main.py validate-config --config configs/config.yaml

# Run a single stage against already-persisted intermediate data
python main.py run-stage collect --config configs/config.yaml
python main.py run-stage clean   --config configs/config.yaml
python main.py run-stage chunk   --config configs/config.yaml
python main.py run-stage score   --config configs/config.yaml
python main.py run-stage filter  --config configs/config.yaml
```

Output lands in `data/instruction/` as `<dataset_name>.train.jsonl` /
`<dataset_name>.validation.jsonl`, each record matching:

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
    "framework": "langgraph",
    "url": "…",
    "quality_score": 0.71
  }
}
```

Reports land in `data/reports/`: `dataset_statistics.json` (+ PNG
charts), `filter_report.json` (explains every dropped document and why),
`validation_report.json`.

### Design notes

- **Instruction generation is fully rule-based** (Stage 9) — no LLM API
  call synthesizes instruction/output pairs; every example is
  deterministically derived from the corpus via regex/heuristic
  templates. This makes the dataset 100% reproducible from the source
  list alone, with no API cost or model-version drift. Swap in an
  LLM-based generator later behind the same `InstructionGenerator`
  interface if you want higher-variance instructions.
- **Near-duplicate detection uses SimHash**, not embeddings — bucketed
  by high-order bits before pairwise Hamming-distance comparison, which
  keeps dedup roughly linear and avoids a hard dependency on an
  embedding model for cleaning.
- **Quality scoring is a transparent, weighted composite** of nine
  independently-computed features (token count, code ratio,
  documentation completeness, readability, duplicate probability,
  source credibility, example density, instruction density, broken
  links) — every document's full `quality_report` is kept alongside its
  score, so Stage 8's filtering decisions are auditable.
- **Tokenization degrades gracefully** — tries `tiktoken`'s
  `cl100k_base` first, falls back to a ~1.3-tokens/word heuristic if the
  BPE file isn't reachable (e.g. offline/sandboxed environments).

### Testing

```bash
pytest   # 27 unit tests: hashing/simhash, tokenizer fallback, text
         # normalization, config validation, cleaning, dedup, code
         # extraction, chunking, scoring, filtering, instruction
         # generation, dataset validation
```

### Extending the pipeline

- **New source type** — subclass `BaseCollector`
  (`src/collectors/base.py`), implement `collect() -> List[RawDocument]`,
  register in `pipeline.py::run_collection`, add config to
  `SourcesConfig`.
- **New instruction template** — add `(Chunk, DocumentMetadata) ->
  List[dict]` to `src/generators/templates.py`, register in
  `TEMPLATE_REGISTRY`.
- **New export format** — add a handler to `DatasetExporter`
  (`src/exporters/exporter.py`) and a new `ExportFormat` enum value.
- **LLM-assisted instruction generation** — implement an alternate
  `InstructionGenerator`-shaped class, swap into
  `pipeline.py::run_instruction_generation`.

---

## Part 2: Five domain-specific fine-tuning datasets

The pipeline above produced a raw corpus of 4,905 instruction examples
tagged by category. Before it was usable for domain-expert fine-tuning,
it needed one more, downstream cleaning pass — the raw corpus was ~72%
one category (`ai_system_design`, mostly FastAPI documentation) and
~45% non-English (translated framework docs), with output lengths
ranging from 31 to 240,665 characters.

**Additional cleaning applied on top of the pipeline's own Stage 7/8
scoring:**

1. **Language filter** — English-only (regex-based script detection),
   removing ~45% of rows that were translated documentation.
2. **Length filter** — outputs kept between 50 and 6,000 characters,
   removing degenerate short answers and multi-thousand-token document
   dumps that exceed a small model's training context.
3. **Per-domain quality threshold, not a global one** — the corpus's
   own `quality_score` field was not comparable across sources: rows
   sourced from Hugging Face (all of `data_science`) scored 0.55–0.67,
   while GitHub-sourced rows scored higher on the same scale. A flat
   0.65 cutoff would have deleted the entire `data_science` domain. A
   domain-aware threshold (0.55) was used instead.
4. **Domain cap** — `ai_system_design`, at 72% of the raw corpus, was
   capped at 400 examples so no single expert dominates training.

**Resulting datasets** (`datasets/`, Alpaca-style
`instruction`/`input`/`output` schema, unchanged from the pipeline's
export format):

| Domain | File | Train examples | Eval examples |
|---|---|---|---|
| AI system design | `ai_system_design.jsonl` | 400 | 3 |
| Retrieval systems | `retrieval_systems.jsonl` | 300 | 4 |
| LLM engineering | `llm_engineering.jsonl` | 289 | 6 |
| Agent engineering | `agent_engineering.jsonl` | 260 | 3 |
| Data science | `data_science.jsonl` | 209 | 4 |
| **Total** | | **1,458** | **20** |

`eval.jsonl` is drawn from a separate validation split the fine-tuned
experts never train on, with fields `{"domain", "prompt", "reference"}`.

---

## Part 3: Fine-tuning + MoE-LoRA + JEPA research

### Base model

**Qwen2.5-Coder-1.5B-Instruct**, 4-bit QLoRA, kept consistent across
every experiment in this project.

### Architecture

```
Query → Embed → Route → Select top-2 → Merge adapters → Generate
```

1. **Five LoRA experts**, one per domain (`r=16`, `α=32`, dropout 0.05,
   targeting q/k/v/o/gate/up/down projections) — 18,464,768 trainable
   parameters per expert, 1.18% of the 1.562B base model.
2. **Router** — two variants compared: (a) baseline, embeddings from
   `all-MiniLM-L6-v2`; (b) proposed, embeddings from a JEPA-inspired
   context encoder trained on this project's own corpus (context/target/
   predictor with EMA target update and a variance-regularization term
   to prevent representational collapse).
3. **Weighted merge** — top-2 experts' LoRA weight deltas combined
   linearly by the router's softmax-over-cosine-similarity weights,
   applied to the frozen base model in a single forward pass.

Full math, diagrams, and the "why this differs from a conventional
token-level sparse Transformer MoE" discussion are in the paper
(Sections 3–5) and the presentation (Slides 5–9).

### Results

**Routing accuracy** (does the router pick the right domain?), n=20:

| Router | Accuracy |
|---|---|
| Baseline (Sentence Transformer) | 90% (18/20) |
| JEPA-inspired (proposed) | 80% (16/20) |

**Generation quality** (does the finished system answer well?), Claude-judged
1–10 rubric after the Gemini judge's API quota was exhausted mid-run
(disclosed substitution — see paper Section 6.4):

| System | Correctness | Relevance | Completeness | Overall |
|---|---|---|---|---|
| Raw base model | 4.70 | 6.20 | 5.10 | 5.33 |
| Combined fine-tune | 4.55 | 5.70 | 4.55 | 4.93 |
| JEPA-routed MoE-LoRA | 1.85 | 2.75 | 1.95 | **2.18** |
| Gemini 1.5 Flash (reference) | 8.15 | 8.90 | 8.75 | 8.60 |

**Why the fine-tuned systems underperformed the untouched base model** —
three converging, non-exclusive mechanisms (paper Section 7.2c):

1. **Weight-space adapter interference** — experts are blended by
   summing weight deltas *before* one forward pass, not by blending two
   experts' output distributions; these are only equivalent if the
   model is linear, which a Transformer is not.
2. **Router confidence saturation** — top-2 route weights sit within
   0.001 of an exact 50/50 tie in half the evaluation set (mean entropy
   0.994 of a possible 1.0 bit), so the router rarely commits to one
   expert even when it should.
3. **Capacity vs. data volume** — 18.46M trainable parameters against
   209–400 examples per domain, trained in as few as 81 optimizer
   steps — a regime consistent with memorization or shortcut learning
   rather than generalization, and directly visible in the raw outputs
   (e.g. asked to explain "Output" in data science, the JEPA-routed
   system reproduces an unrelated, verbatim-memorized protein-function
   description).

### Reproducing this

```bash
# 1. Train + evaluate everything (Colab, T4 GPU)
research/moe_lora_jepa_final.ipynb

# 2. Push trained adapters + router to your Hugging Face account
python research/push_to_hub.py

# 3. Run the live demo (Gradio, deploy to an HF Space)
research/demo/app.py
```

### What this project actually demonstrates

Read plainly: the most **data-efficient** system in this comparison was
the one that used *zero* additional training data — the raw base model.
LoRA achieved its intended *parameter*-efficiency goal (1.18% trainable
parameters), but at this data scale, using that efficiency to fine-tune
domain experts reduced answer quality rather than improving it. The
paper and presentation report this as a genuine, mechanism-grounded
negative result rather than reframing it as a success, and propose a
prioritized set of fixes (paper Section 9b / presentation Slides 20–22)
— the highest-priority one being to fall back to the raw base model
whenever the router's confidence is low, since the base model already
outperforms both fine-tuned variants on this evaluation set.

---

## License

The dataset pipeline (Part 1) is infrastructure code for building your
own dataset; it does not bundle any collected content. Respect the
license of every source you configure it to collect from (GitHub repo
licenses, arXiv's terms, Hugging Face dataset cards, individual blog
terms of use) before using or redistributing the resulting dataset. The
research code and paper (Parts 2–3) are provided for reproducibility of
the reported experiments.