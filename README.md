# Document Intelligence Refinery

A production-grade, multi-stage agentic pipeline that ingests heterogeneous enterprise documents and emits structured, queryable, spatially-indexed knowledge.

## Setup (under 10 minutes)

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd refinery

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# 5. Place your documents
mkdir -p data/corpus
cp your_documents/*.pdf data/corpus/
```

## Run

```bash
# Triage a single document
python -m src.agents.triage data/corpus/your_document.pdf

# Extract a document (auto-routes to correct strategy)
python -m src.agents.extractor data/corpus/your_document.pdf

# Run on the full corpus
python scripts/run_corpus.py
```

## Run Tests

```bash
pytest tests/ -v
```

## Pipeline Stages

| Stage | File | Description |
|-------|------|-------------|
| 1. Triage | `src/agents/triage.py` | Classifies document → DocumentProfile |
| 2. Extraction | `src/agents/extractor.py` | Routes to Strategy A/B/C |
| 3. Chunking | `src/agents/chunker.py` | Produces Logical Document Units |
| 4. PageIndex | `src/agents/indexer.py` | Builds navigation tree |
| 5. Query | `src/agents/query_agent.py` | Answers with provenance |

## Extraction Strategies

| Strategy | Tool | Triggers | Cost |
|----------|------|----------|------|
| A — Fast Text | pdfplumber | native_digital + single_column | ~$0.00 |
| B — Layout-Aware | Docling | multi_column, table_heavy, mixed | ~$0.00 (local) |
| C — Vision | VLM via OpenRouter | scanned_image, low confidence | ~$0.002/page |

## Configuration

All thresholds live in `rubric/extraction_rules.yaml` — no code changes needed to onboard new document types.

## Artifacts

- `.refinery/profiles/` — DocumentProfile JSON per document
- `.refinery/extraction_ledger.jsonl` — full audit log of every extraction
- `.refinery/pageindex/` — PageIndex trees per document
