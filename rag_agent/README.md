# RAG Agent

The RAG Agent is a production-grade Retrieval-Augmented Generation service for Korean manufacturing documentation. It accepts natural language queries and returns cited, safety-checked answers by searching ingested PDF and Office documents. The agent exposes a FastAPI service with Server-Sent Events (SSE) streaming and supports two execution modes — a low-latency **Fast Lane** for simple factual queries and a reasoning-capable **Slow Lane** for complex multi-hop questions.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Ingestion Pipeline (Offline)](#ingestion-pipeline-offline)
3. [Fast Lane](#fast-lane)
4. [Slow Lane](#slow-lane)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Database Schema](#database-schema)
8. [Setup and Running](#setup-and-running)
9. [Testing](#testing)
10. [Project Structure](#project-structure)

---

## Architecture Overview

The RAG Agent is split into two phases: an **offline ingestion pipeline** that processes documents into a searchable index, and an **online query pipeline** that retrieves and generates answers at request time. An internal LLM router decides at query time whether to use the Fast Lane (single-pass retrieval + generation) or the Slow Lane (multi-step LangGraph agent).

**Technology stack:**

| Component | Technology |
|---|---|
| Document parsing | Upstage Document Parse API |
| Embeddings | Upstage Solar Embedding (4096d) |
| Vector store | Milvus |
| Sparse index | BM25 over PostgreSQL |
| Reranking | Cohere via Azure AI Foundry |
| Generation | Azure OpenAI (GPT-4.5-mini / GPT-4.1) |
| Agent orchestration | LangGraph |
| API framework | FastAPI + SSE |

---

## Ingestion Pipeline (Offline)

Documents are processed once offline and stored in a dual-index (dense + sparse) before any queries are served.

![Ingestion Pipeline](./diagrams/The_Ingestion_Pipeline__Offline_.png)

The pipeline runs in five sequential stages:

**1. Parse** — Raw PDF or Office files are submitted to the Upstage Document Parse API (`document-parse` model). The API returns structured Markdown where text and table elements are cleanly separated. Tables are preserved in Markdown table format rather than flattened to plain text, which is critical for retrieval quality on tabular manufacturing data. Large files use the async endpoint with polling.

**2. Chunk** — The structured Markdown is split using a parent-child chunking strategy. Section headers (`#`) define parent boundaries; each parent is then split into smaller child chunks for embedding. Text chunks use paragraph/sentence-aware splitting with configurable overlap. Table rows are grouped into token-aware child chunks separately by `TableChunker`, which handles token counting using the same encoding as the embedding API to avoid the tokenizer mismatch that would otherwise cause silent truncation during embedding.

**3. Embed** — Child chunks are embedded in passage mode using `solar-embedding-1-large-passage`. Batches are submitted concurrently up to the configured `embedding_batch_size`. Query-time embeddings use the separate `solar-embedding-1-large-query` model — using the correct passage/query model pair is essential for retrieval quality with asymmetric embedding models.

**4. Index** — Child embeddings are upserted into Milvus (dense vector index, IVF_FLAT). In parallel, child text is tokenized with the BM25 tokenizer (which handles both Korean morpheme splitting and English stemming) and term frequencies are written to the `bm25_index` PostgreSQL table. Document frequency statistics in `bm25_df` are maintained automatically via PostgreSQL triggers.

**5. Store** — Parent text and metadata are stored in the PostgreSQL `parents` table. The parent-child relationship is tracked via foreign keys, allowing retrieval to return child chunks but fetch the broader parent context for generation.

### Running ingestion

```bash
python scripts/ingest_document.py --file data/inputs/your_document.pdf
```

To clean and re-ingest all documents:

```bash
python scripts/clean_all_databases.py
python scripts/ingest_document.py --file data/inputs/your_document.pdf
```

---

## Fast Lane

The Fast Lane handles simple and factual queries with a target latency under 4 seconds. It runs a single-pass pipeline from query expansion through to streaming generation.

![Fast Lane Architecture](./diagrams/The_Fast_Lane_Architecture.png)

### Step 1 — Query Expansion

The original query is passed to an LLM that generates `n` semantic variants. This addresses vocabulary mismatch: a query about "스테인레스강 가공" may miss documents indexed under "STS 절삭" unless variants are generated. All variants are retrieved in parallel in the next step.

Query expansion is optional and controlled by `fast_lane.query_expansion.enabled` in config.

### Step 2 — Hybrid Retrieval

Each query variant runs through hybrid search independently, then results are merged:

- **Dense search**: The query is embedded with `solar-embedding-1-large-query` and a nearest-neighbor search is run against the Milvus collection.
- **Sparse search (BM25)**: The query is tokenized and scored against the `bm25_index` table in PostgreSQL using the BM25 formula (k1=1.5, b=0.75).
- **RRF merge**: Results from all dense and sparse searches (across all query variants) are merged using Reciprocal Rank Fusion. RRF is position-based rather than score-based, making it robust to score scale differences between the two retrieval modes.

### Step 3 — Precision (Reranking)

The top-n RRF candidates are passed to the Cohere reranker (`BGE-Reranker-v2-m3` via Azure AI Foundry). The reranker scores each candidate against the original query using cross-attention, which is more accurate than bi-encoder similarity but too slow to run over the entire index. After reranking, the parent context for the top-k chunks is fetched from PostgreSQL to give the generator richer context than the child chunk alone.

Reranking is optional and controlled by `fast_lane.reranker.enabled`.

### Step 4 — Generation and Safety

The top-k chunks with their parent context are passed to the Azure OpenAI LLM in citation mode. The system prompt instructs the model to inline citation markers (`[1]`, `[2]`, etc.) tied to source documents. The generator streams the response as SSE chunks.

After generation, a two-stage safety check runs:

1. **Heuristic check**: Regex-based patterns flag known unsafe content types.
2. **NLI check** (optional): An NLI model evaluates entailment between retrieved context and the generated answer to detect hallucinations.

If the safety check passes, the stream is forwarded to the client. If it fails, a fallback message is returned instead.

---

## Slow Lane

The Slow Lane handles complex multi-hop queries that require decomposing a question into sub-questions, gathering evidence iteratively, and synthesizing a final answer. It is implemented as a LangGraph state machine.

![Slow Lane Architecture](./diagrams/The_Slow_Lane_Architecture.png)

### Agent State

The agent maintains a typed `SlowLaneState` across all nodes containing a **scratchpad** of accumulated facts and a **context bag** of source chunks collected during execution. State persists across the entire reasoning loop.

### Reasoning Loop (LangGraph nodes)

| Node | Responsibility |
|---|---|
| `planner` | Decomposes the user query into a JSON plan of sub-queries, each annotated with which tool to call |
| `executor` | Calls the assigned tool (Fast Lane RAG or SQL) for the current sub-query and appends the result to state |
| `critic` | Validates whether the returned evidence sufficiently answers the sub-query; decides whether to continue, synthesize, or rewrite |
| `rewriter` | Rewrites a failed sub-query with different phrasing and returns to the executor |
| `synthesizer` | Combines all accumulated evidence into a final coherent answer with citations |

The graph follows the edges: `planner → executor → critic`, then conditionally to `executor` (more sub-queries), `synthesizer` (all done), or `rewriter` (evidence invalid). The rewriter loops back to `executor`, not the planner, to avoid re-planning from scratch on a single failed step.

### Tool invocation

The Slow Lane calls the Fast Lane in **tool mode** (`invoke_tool()`), which runs retrieval and generation without streaming or safety checks. The Slow Lane handles its own validation via the critic node.

---

## API Reference

The FastAPI service is defined in `api/` and started via `api/main.py`.

### `POST /query`

The primary query endpoint. Supports both streaming (SSE) and non-streaming JSON responses.

**Request body:**

```json
{
  "query": "스테인레스강 고속 가공에 적합한 코팅은?",
  "top_k": 5,
  "language": "ko",
  "streaming": true
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | User query (1–1000 chars) |
| `top_k` | integer | 5 | Number of context chunks to use (1–20) |
| `language` | string | `"ko"` | Response language: `"ko"` or `"en"` |
| `streaming` | boolean | `true` | Enable SSE streaming |

**Streaming response (SSE events):**

When `streaming: true`, the endpoint returns `text/event-stream`. Each line is a JSON event:

```
data: {"type": "status", "content": "문서 검색 중..."}
data: {"type": "citation", "data": {"id": "[1]", "file": "manual.pdf", "page": 5, "source_id": "chunk_abc"}}
data: {"type": "chunk", "content": "스테인레스강 고속 가공에는 PC8110 PVD 코팅이 적합합니다 [1]."}
data: {"type": "done", "metadata": {"latency_ms": 2840, "citation_count": 2, "safety_passed": true}}
```

| Event type | Description |
|---|---|
| `status` | Progress update (searching, reranking, generating) |
| `citation` | Source metadata emitted before the answer starts |
| `chunk` | Incremental answer text chunk |
| `done` | Final metadata: latency, citation count, safety result |
| `error` | Error with `message` and `type` fields |

**Non-streaming response:**

When `streaming: false`, returns a single JSON object:

```json
{
  "answer": "스테인레스강 고속 가공에는 PC8110 PVD 코팅이 적합합니다 [1].",
  "citations": [
    {"id": "[1]", "file": "manual.pdf", "page": 5, "source_id": "chunk_abc"}
  ],
  "metadata": {"latency_ms": 2840, "citation_count": 1, "safety_passed": true}
}
```

### `GET /query/test`

Health-check endpoint that runs a fixed test query (`"PVD 코팅이란?"`) and returns a non-streaming response. Used to verify Fast Lane initialization on startup.

---

## Configuration

All configuration is managed through Pydantic models in `src/config/models.py` with environment variable overrides via `.env`.

### Key environment variables

```env
# Upstage (parsing + embeddings)
UPSTAGE_API_KEY=up-...

# Azure OpenAI (generation)
LLM__AZURE_ENDPOINT=https://your-resource.openai.azure.com/
LLM__AZURE_API_KEY=...
LLM__AZURE_API_VERSION=2024-02-15-preview
LLM__DEFAULT_MODEL=gpt-4-5-mini
LLM__FALLBACK_MODEL=gpt-4-1

# PostgreSQL
POSTGRES_DSN=postgresql://user:password@localhost:5432/rag_db

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=rag_chunks

# Cohere reranker (Azure AI Foundry)
COHERE_RERANK_ENDPOINT=https://...
COHERE_RERANK_API_KEY=...
```

### Key config parameters

| Config path | Default | Description |
|---|---|---|
| `fast_lane.query_expansion.enabled` | `true` | Enable LLM query expansion |
| `fast_lane.retrieval.dense_top_k` | `20` | Candidates from dense search per query |
| `fast_lane.retrieval.bm25_top_k` | `20` | Candidates from BM25 per query |
| `fast_lane.retrieval.rrf_k` | `60` | RRF smoothing constant |
| `fast_lane.reranker.enabled` | `true` | Enable Cohere reranking |
| `fast_lane.safety_check.enabled` | `true` | Enable safety check |
| `fast_lane.safety_check.use_nli` | `false` | Enable NLI hallucination check |
| `llm.temperature` | `0.7` | Generation temperature |
| `llm.max_tokens` | `1000` | Max generated tokens |
| `ingestion.chunking.text.max_tokens` | `300` | Max tokens per text child chunk |
| `ingestion.chunking.table.max_rows` | `10` | Max rows per table child chunk |

---

## Database Schema

### PostgreSQL tables

The PostgreSQL schema (`scripts/db/schema.sql`) stores parent context and the BM25 index.

```
parents          — Full parent chunks with source metadata and ACL
children         — Child chunks linked to parents via foreign key
bm25_index       — Term frequencies per child chunk (JSONB)
bm25_df          — Document frequency per term (updated by trigger)
bm25_stats       — Global corpus statistics (N, avgdl)
ingestion_log    — Ingestion job tracking
```

BM25 statistics (`bm25_df`, `bm25_stats`) are maintained automatically via PostgreSQL triggers on `bm25_index` — no manual refresh is needed after ingestion.

### Milvus collection

The Milvus collection (`scripts/db/milvus_schema.py`) stores dense vector embeddings with the following fields:

```
child_id        — Primary key (VARCHAR)
embedding       — Dense vector (FLOAT_VECTOR, 4096d)
parent_id       — Foreign key reference to PostgreSQL
source_file     — Source document name
page_number     — Page in source document
```

The index type is `IVF_FLAT` with `IP` (inner product) metric, which is compatible with the normalized Solar embeddings.

---

## Setup and Running

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Node.js (only if generating `.docx` documentation)

### 1. Start infrastructure

```bash
docker-compose up -d
```

This starts Milvus and PostgreSQL. Wait ~30 seconds for Milvus to finish initializing before running setup scripts.

### 2. Initialize databases

```bash
python scripts/db/milvus_schema.py    # Create Milvus collection + index
psql -U postgres -d rag_db -f scripts/db/schema.sql  # Create PostgreSQL tables
```

### 3. Configure environment

Copy and fill in `.env`:

```bash
cp .env.example .env
# Edit .env with your API keys and connection strings
```

### 4. Ingest documents

```bash
python scripts/ingest_document.py --file data/inputs/your_document.pdf
```

### 5. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Verify

```bash
curl http://localhost:8000/query/test
```

---

## Testing

### Unit tests

```bash
pytest tests/unit/ -v
```

Unit tests cover all core components with mocked HTTP clients and do not require live API keys or database connections.

| Test file | Coverage |
|---|---|
| `test_text_chunker.py` | Paragraph/sentence splitting, overlap, token counting |
| `test_table_chunker.py` | Table row grouping, token-aware splits |
| `test_bm25_tokenizer.py` | Korean morpheme splitting, English stemming |
| `test_bm25_indexer.py` | BM25 scoring, index operations (integration-marked) |
| `test_rrf.py` | RRF merge correctness |
| `test_citation_formatter.py` | Citation numbering and metadata |
| `test_safety_check.py` | Heuristic safety patterns |
| `test_upstage_parser.py` | Parsing response handling (mocked HTTP) |
| `test_upstage_embedder.py` | Embedding batching (mocked HTTP) |
| `test_upstage_reranker.py` | Reranker async calls (mocked HTTP) |
| `test_config.py` | Config loading and validation |

### Integration and smoke tests (requires live services)

```bash
python scripts/test_fast_lane_e2e.py     # Full Fast Lane end-to-end
python scripts/test_slow_lane.py         # Full Slow Lane end-to-end
python scripts/test_router.py            # LLM router decisions
python scripts/test_streaming_generator.py  # SSE streaming
python scripts/test_rerank_endpoint.py   # Cohere reranker connectivity
python scripts/test_query_expansion.py   # Query expansion output
```

---

## Project Structure

```
rag_agent/
├── api/
│   ├── main.py             # FastAPI app + lifespan init (loads config, FastLane)
│   ├── routes.py           # /query SSE endpoint + non-streaming fallback
│   └── schemas.py          # Pydantic request/response/SSE schemas
├── config/
│   ├── default.yaml        # Default config values
│   └── routing_rules.yaml  # Fast/Slow lane routing rules
├── src/
│   ├── agents/
│   │   ├── fast_lane.py        # FastLane orchestrator
│   │   ├── slow_lane.py        # SlowLane LangGraph agent
│   │   └── slow_lane_state.py  # Typed LangGraph state
│   ├── config/
│   │   └── models.py           # All Pydantic config models
│   ├── generation/
│   │   ├── llm_client.py           # Azure OpenAI async client
│   │   ├── streaming_generator.py  # SSE streaming + citations
│   │   ├── citation_formatter.py   # Citation numbering + metadata
│   │   └── safety_check.py         # Heuristic + NLI safety checker
│   ├── ingestion/
│   │   ├── pipeline.py             # End-to-end ingestion orchestrator
│   │   ├── parsers/upstage.py      # Upstage document parser
│   │   ├── chunkers/text_chunker.py    # Text chunking
│   │   ├── chunkers/table_chunker.py   # Table chunking
│   │   ├── embedders/upstage.py    # Upstage embedding client
│   │   └── indexers/
│   │       ├── bm25_tokenizer.py   # Korean/English BM25 tokenizer
│   │       └── bm25_indexer.py     # BM25 index writer (PostgreSQL)
│   ├── retrieval/
│   │   ├── hybrid_retriever.py     # Dense + BM25 + RRF retrieval
│   │   ├── query_expansion.py      # LLM query expansion
│   │   ├── merge/rrf.py            # Reciprocal Rank Fusion
│   │   └── rerankers/cohere.py     # Cohere reranker (Azure AI Foundry)
│   ├── router/
│   │   └── llm_router.py           # Fast vs Slow lane router
│   └── utils/
│       └── logging_config.py       # Centralized logging setup
├── scripts/
│   ├── db/
│   │   ├── schema.sql          # PostgreSQL schema (DDL)
│   │   └── milvus_schema.py    # Milvus collection + index setup
│   └── ingest_document.py      # CLI ingestion runner
├── tests/unit/                 # Unit tests (no live services required)
├── docker-compose.yml          # Local Milvus + PostgreSQL stack
├── pyproject.toml
└── requirements.txt
```
