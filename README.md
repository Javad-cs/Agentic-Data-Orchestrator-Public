# Agentic Data Orchestrator

The Agentic Data Orchestrator is a multi-agent AI system for Korean manufacturing data analysis. It accepts natural language queries and intelligently routes them to the most appropriate agent: a RAG pipeline for searching unstructured documentation, a Text-to-SQL pipeline for querying structured Oracle databases, a multi-hop Slow Lane agent for complex questions that span both sources, or a direct LLM path for general knowledge queries. The system is designed to be self-improving — routing decisions are logged and analyzed to refine routing rules over time.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Modules](#modules)
3. [Data Flow](#data-flow)
4. [Repository Structure](#repository-structure)
5. [Technology Stack](#technology-stack)
6. [Prerequisites](#prerequisites)
7. [Setup and Installation](#setup-and-installation)
8. [Running the System](#running-the-system)
9. [Environment Variables Reference](#environment-variables-reference)
10. [Known Issues and Design Notes](#known-issues-and-design-notes)
11. [Future Work](#future-work)

---

## System Architecture

```
                        ┌─────────────────────────────────┐
                        │          User Query              │
                        └─────────────────────────────────┘
                                        │
                                        ▼
                        ┌─────────────────────────────────┐
                        │         Router Agent             │
                        │  Rule-Driven LLM Scoring (0-10) │
                        │  Path-Level Meta-Cache           │
                        │  Async JSONL Observability       │
                        └─────────────────────────────────┘
                                        │
              ┌─────────────────────────┼──────────────────────────┐
              │                         │                           │
              ▼                         ▼                           ▼
   ┌──────────────────┐   ┌──────────────────────────┐  ┌─────────────────┐
   │  Text-to-SQL     │   │     RAG Agent             │  │   LLM Only      │
   │  Agent           │   │                           │  │  (Chit-Chat /   │
   │  (FACT_ONLY)     │   │  Fast Lane (DOC_ONLY)     │  │   General KG)   │
   │                  │   │  Slow Lane (COMPLEX_DUAL) │  └─────────────────┘
   │  Oracle PRIMARY_DATABASE   │   │  with SQL tool as needed  │
   │  Oracle LINKED_DATABASE   │   │                           │
   └──────────────────┘   └──────────────────────────┘
              │                         │
              └─────────────────────────┘
                                        │
                                        ▼
                        ┌─────────────────────────────────┐
                        │  Execution Logs → Rule-Making   │
                        │  Expert Agent → routing_rules   │
                        │  (Self-Improvement Loop)        │
                        └─────────────────────────────────┘
```

The system has three independently-deployable modules that are orchestrated at runtime by the Router Agent:

| Module | Purpose | Key path |
|---|---|---|
| `router_agent/` | Query routing, orchestration, observability | Entry point for all queries |
| `rag_agent/` | Document retrieval and generation (RAG) | Called for DOC_ONLY and COMPLEX_DUAL paths |
| `text_to_sql_agent/` | Natural language to SQL against Oracle databases | Called for FACT_ONLY and COMPLEX_DUAL paths |

---

## Modules

### Router Agent

The Router Agent is the single entry point. It scores every incoming query across four execution paths using a rule-driven LLM evaluator, dispatches to the winning path, logs results, and supports continuous rule refinement through a Rule-Making Expert Agent feedback loop.

**Four execution paths:**

| Path | Trigger | Agent invoked |
|---|---|---|
| `FACT_ONLY` | Numbers, statistics, counts, data queries | Text-to-SQL Agent |
| `DOC_ONLY` | Definitions, explanations, process queries | RAG Agent Fast Lane |
| `COMPLEX_DUAL` | Multi-source questions requiring both data and documents | RAG Agent Slow Lane (calls both tools) |
| `LLM_ONLY` | General knowledge, chit-chat, definitions | Direct Azure OpenAI call |

→ See [`router_agent/README.md`](./router_agent/README.md) for full documentation.

---

### RAG Agent

The RAG Agent processes PDF and Office documents through an offline ingestion pipeline and serves them at query time through two execution modes:

- **Fast Lane** — single-pass: query expansion → hybrid retrieval (dense + BM25 + RRF) → Cohere reranking → streaming generation with inline citations → safety check. Target latency under 4 seconds.
- **Slow Lane** — multi-hop LangGraph agent: decomposes complex queries into sub-questions, calls Fast Lane and SQL tools iteratively, validates evidence through a critic node, synthesizes a final cited answer.

**Storage:**
- Milvus for dense vector search (Upstage Solar embeddings, 4096d)
- PostgreSQL for BM25 sparse index and parent chunk context
- FastAPI with SSE streaming for the query endpoint

→ See [`rag_agent/README.md`](./rag_agent/README.md) for full documentation.

---

### Text-to-SQL Agent

The Text-to-SQL Agent translates Korean natural language questions into Oracle SQL using a four-phase pipeline based on the BIRD paper methodology:

1. **Profiling** — column statistics, LLM-generated field descriptions, query log enrichment
2. **Schema Linking** — FAISS semantic search + MinHash LSH literal matching → 5 schema variants
3. **Algorithm 1** — iterative SQL generation and literal-coverage refinement across all 5 schema variants
4. **Candidate Voting** — 3 diverse SQL candidates (field shuffling + LLM seeds) → lint → execute → majority vote

Supports cross-database queries between the primary Oracle database (PRIMARY_DATABASE) and a linked Oracle database (LINKED_DATABASE) via database links (`@LINKED_DATABASE` suffix).

→ See [`text_to_sql_agent/README.md`](./text_to_sql_agent/README.md) for full documentation.

---

## Data Flow

### Query-time flow (online)

```
User query
  └─► Router Agent
        ├─ [cache hit] retrieve cached plan → skip LLM call
        └─ [cache miss] LLM scores query across 4 paths
              │
              ├─ FACT_ONLY ──────────────────────────────────────────────►
              │                                                           │
              │  Text-to-SQL Agent:                                       │
              │  Schema linking → Algorithm 1 → Candidate voting          │
              │  → Execute SQL on Oracle → Return result                  │
              │                                                           │
              ├─ DOC_ONLY ───────────────────────────────────────────────►
              │                                                           │
              │  RAG Agent Fast Lane:                                     │
              │  Query expansion → Hybrid search → Reranking              │
              │  → Generate with citations → Safety check                 │
              │  → Stream SSE response                                    │
              │                                                           │
              ├─ COMPLEX_DUAL ───────────────────────────────────────────►
              │                                                           │
              │  RAG Agent Slow Lane (LangGraph):                        │
              │  Planner decomposes → Executor calls RAG or SQL tool     │
              │  → Critic validates → Rewriter retries on failure        │
              │  → Synthesizer produces final cited answer               │
              │                                                           │
              └─ LLM_ONLY ──────────────────────────────────────────────►
                                                                         │
                 Direct Azure OpenAI call → Return answer               │
                                                                         │
  ◄────────────────────────────────────────────────────────────────────┘
         Final answer + Execution log written to JSONL
```

### Ingestion flow (offline, run once per document batch)

```
Raw PDFs / Office files
  └─► Upstage Document Parse API
        └─► Structured Markdown (tables preserved)
              └─► Parent-Child Chunker
                    ├─► Child embeddings (Upstage Solar)
                    │     └─► Milvus (dense vector index)
                    ├─► Child text (BM25 tokenizer: Korean + English)
                    │     └─► PostgreSQL (bm25_index table)
                    └─► Parent text + metadata
                          └─► PostgreSQL (parents table)
```

---

## Repository Structure

```
Agentic-Data-Orchestrator/
│
├── docker-compose.yml          # PostgreSQL + Milvus + agent-env services
├── Dockerfile                  # Oracle Linux 8 + Python 3.9 + Oracle Instant Client
├── requirements.txt            # Shared Python dependencies (all modules)
│
├── router_agent/               # Orchestration layer (entry point)
│   ├── config/
│   │   └── routing_rules.yaml  # Rule-based routing definitions
│   ├── src/
│   │   ├── router/             # LLM router + scoring prompts
│   │   ├── tools/              # RAGTool, SQLTool, LangGraphToolWrapper
│   │   ├── config/             # RouterSettings (Pydantic)
│   │   └── utils/              # LLM client, async JSONL logger
│   ├── scripts/                # Integration test scripts
│   └── README.md
│
├── rag_agent/                  # RAG pipeline (documents)
│   ├── api/                    # FastAPI app + SSE /query endpoint
│   ├── config/                 # YAML configs
│   ├── src/
│   │   ├── agents/             # FastLane, SlowLane (LangGraph)
│   │   ├── generation/         # LLM client, streaming generator, citations, safety
│   │   ├── ingestion/          # Parsers, chunkers, embedders, BM25 indexers
│   │   ├── retrieval/          # HybridRetriever, RRF, query expansion, rerankers
│   │   └── router/             # Internal Fast/Slow lane router
│   ├── scripts/                # DB setup, ingestion CLI, test scripts
│   ├── tests/unit/             # Unit tests (mocked HTTP)
│   └── README.md
│
└── text_to_sql_agent/          # Text-to-SQL pipeline (Oracle databases)
    ├── core/                   # LLM client interface, BaseDatabase
    ├── config/                 # Pydantic settings (dual Oracle DB)
    ├── profiling/              # ColumnProfiler, LLM summaries, SME enrichment
    ├── indexing/               # FAISS field index, MinHash LSH matcher
    ├── schema_linking/         # FocusedSchemaBuilder, 5 schema variants
    ├── sql_generation/         # Algorithm 1, SQL parser, schema augmenter
    ├── final_sql_w_cand_voting/ # VotingOrchestrator, CandidateGenerator, FewShotStore
    ├── oracle_adapter.py       # Oracle database implementation
    ├── test_scripts/           # End-to-end tests and benchmarks
    └── README.md
```

---

## Technology Stack

| Category | Technology |
|---|---|
| **Orchestration** | LangGraph (Slow Lane state machine) |
| **LLM** | Azure OpenAI (GPT-4.1-mini for routing, GPT-4.5-mini for generation, GPT-4.1 as fallback) |
| **Document parsing** | Upstage Document Parse API |
| **Embeddings** | Upstage Solar Embedding (`solar-embedding-1-large`, 4096d) |
| **Vector store** | Milvus (dense vector search, IVF_FLAT + IP metric) |
| **Sparse index** | BM25 over PostgreSQL (Korean + English tokenization) |
| **Reranking** | Cohere `BGE-Reranker-v2-m3` via Azure AI Foundry |
| **Few-shot store** | FAISS + `all-MiniLM-L6-v2` (SentenceTransformers) |
| **LSH matching** | MinHash LSH (datasketch) |
| **SQL parsing** | sqlglot (dialect-aware: Oracle, PostgreSQL, MySQL, SQLite) |
| **Databases** | Oracle 21c (PRIMARY_DATABASE + LINKED_DATABASE via DB link), PostgreSQL 15, Milvus 2.3 |
| **API framework** | FastAPI + SSE streaming |
| **Container base** | Oracle Linux 8 + Oracle Instant Client 21 |
| **Config** | Pydantic Settings (per-module, `.env` based) |

---

## Prerequisites

- Docker and Docker Compose
- Python 3.9+ (3.11 recommended for development outside Docker)
- Oracle Instant Client 21 (bundled in the Docker image)
- Access to:
  - Azure OpenAI deployment (GPT-4.1-mini, GPT-4.5-mini, GPT-4.1)
  - Upstage API (document parsing + embeddings)
  - Cohere reranking endpoint via Azure AI Foundry
  - Oracle databases (PRIMARY_DATABASE primary, LINKED_DATABASE linked)

---

## Setup and Installation

### 1. Clone and configure environment

```bash
git clone <repo-url>
cd Agentic-Data-Orchestrator
```

All three modules share a single `.env` file at the repository root. Copy and fill it in:

```bash
cp .env.example .env
```

See [Environment Variables Reference](#environment-variables-reference) below for all required keys.

### 2. Build and start the infrastructure

```bash
docker compose up -d
```

This starts three services:
- `agent-env` — the Python application container (Oracle Linux 8 + Instant Client)
- `db` — PostgreSQL 15 for RAG BM25 index and parent text store
- `milvus` — Milvus 2.3.3 for dense vector search

Wait approximately 30 seconds for Milvus to finish initializing before proceeding.

### 3. Initialize RAG databases

```bash
docker compose exec agent-env python rag_agent/scripts/db/milvus_schema.py
docker compose exec agent-env psql -U $DATABASE__POSTGRES_USER -d $DATABASE__POSTGRES_DATABASE \
    -f rag_agent/scripts/db/schema.sql
```

### 4. Ingest documents into RAG

```bash
docker compose exec agent-env python rag_agent/scripts/ingest_document.py \
    --file data/inputs/your_document.pdf
```

### 5. Build the Text-to-SQL few-shot store

This step requires Oracle database connectivity.

```bash
docker compose exec agent-env python -c "
import sys; sys.path.insert(0, 'text_to_sql_agent')
from populate_store import build_store
build_store()
"
```

### 6. Verify each module

```bash
# RAG Agent: health check endpoint
curl http://localhost:8000/query/test

# Router Agent: routing smoke test
docker compose exec agent-env python -m router_agent.scripts.test_router_basic

# Text-to-SQL Agent: Oracle pipeline test
docker compose exec agent-env python -m text_to_sql_agent.test_scripts.test_oracle_korean_pipeline
```

---

## Running the System

### Start the RAG Agent API

```bash
docker compose exec agent-env uvicorn rag_agent.api.main:app \
    --host 0.0.0.0 --port 8000 --reload
```

### Send a query through the Router

```bash
# Streaming (SSE)
curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "스테인레스강 고속 가공에 적합한 코팅은?", "language": "ko", "streaming": true}'

# Non-streaming (JSON)
curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "이번 달 PRIMARY_DATABASE 생산 실적은?", "language": "ko", "streaming": false}'
```

### Running tests

All tests must be run as modules from the repository root to ensure correct path resolution across the sibling-agent import mechanism:

```bash
# Router
python -m router_agent.scripts.test_router_basic
python -m router_agent.scripts.test_rag_tool
python -m router_agent.scripts.test_sql_tool

# RAG Agent unit tests (no live services required)
pytest rag_agent/tests/unit/ -v

# RAG Agent end-to-end (requires live services)
python -m rag_agent.scripts.test_fast_lane_e2e
python -m rag_agent.scripts.test_slow_lane

# Text-to-SQL
python -m text_to_sql_agent.test_scripts.test_phase4_pipeline
python -m text_to_sql_agent.test_scripts.test_oracle_korean_pipeline
```

---

## Environment Variables Reference

All three modules read from a single `.env` file at the repository root. The file is structured into sections by concern. All variables are optional unless marked required.

```env
################################################################################
# SHARED LLM (AZURE OPENAI) — used by all three modules
################################################################################
LLM__AZURE_ENDPOINT=            # Required. Azure OpenAI endpoint URL
LLM__AZURE_API_KEY=             # Required. Azure OpenAI API key
LLM__AZURE_API_VERSION=         # e.g. 2024-02-15-preview

LLM__DEFAULT_MODEL=             # e.g. gpt-4-5-mini  (speed-optimised)
LLM__FALLBACK_MODEL=            # e.g. gpt-4-1       (quality fallback)
LLM__TEMPERATURE=
LLM__MAX_TOKENS=
LLM__STREAMING=

################################################################################
# COMPAT ALIASES — text_to_sql_agent reads these instead of LLM__ prefix
################################################################################
AZURE_OPENAI_ENDPOINT=          # Mirror of LLM__AZURE_ENDPOINT
AZURE_OPENAI_KEY=               # Mirror of LLM__AZURE_API_KEY
AZURE_OPENAI_API_VERSION=
DEFAULT_MODEL=
FALLBACK_MODEL=

################################################################################
# TEXT-TO-SQL AGENT
################################################################################
DB_TYPE=oracle                  # oracle | postgres | mysql | sqlite
DB_DIALECT=oracle

PRIMARY_DB_NAME=PRIMARY_DATABASE
PRIMARY_DSN=                    # Required. Oracle DSN for primary DB
PRIMARY_USER=                   # Required.
PRIMARY_PASSWORD=               # Required.

LINKED_DB_NAME=LINKED_DATABASE
LINKED_DSN=                     # Required. Oracle DSN for linked DB
LINKED_USER=                    # Required.
LINKED_PASSWORD=                # Required.
LINKED_SUFFIX=@LINKED_DATABASE

USE_DOCKER=false                # Set true inside Docker to rewrite 127.0.0.1 → host.docker.internal

BIRD_ROOT_PATH=                 # Path to BIRD benchmark root (dev only)
BIRD_DATA_PATH=
PROFILE_SAMPLE_SIZE=10000
PROFILE_TOP_K=10

################################################################################
# UPSTAGE — rag_agent (parsing + embeddings)
################################################################################
UPSTAGE_API_KEY=                # Required.

################################################################################
# DATABASES — rag_agent
################################################################################
DATABASE__POSTGRES_HOST=localhost
DATABASE__POSTGRES_PORT=5432
DATABASE__POSTGRES_DATABASE=rag_db
DATABASE__POSTGRES_USER=        # Required.
DATABASE__POSTGRES_PASSWORD=    # Required.
DATABASE__POSTGRES_POOL_MIN_SIZE=
DATABASE__POSTGRES_POOL_MAX_SIZE=

DATABASE__MILVUS_URI=           # e.g. http://localhost:19530
DATABASE__MILVUS_COLLECTION_NAME=rag_chunks

################################################################################
# INGESTION / FAST LANE / SLOW LANE — rag_agent
################################################################################
INGESTION__BATCH_SIZE=

FAST_LANE__RERANKER__ENABLED=true
FAST_LANE__RERANKER__PROVIDER=cohere
FAST_LANE__RERANKER__COHERE_API_KEY=    # Required if reranker enabled
FAST_LANE__RERANKER__COHERE_BASE_URL=
FAST_LANE__RERANKER__COHERE_MODEL=
FAST_LANE__RERANKER__TOP_N=

FAST_LANE__QUERY_EXPANSION__ENABLED=true
FAST_LANE__QUERY_EXPANSION__NUM_VARIANTS=
FAST_LANE__QUERY_EXPANSION__TEMPERATURE=
FAST_LANE__QUERY_EXPANSION__PARALLEL=

FAST_LANE__SAFETY_CHECK__USE_NLI=false
FAST_LANE__SAFETY_CHECK__NLI_THRESHOLD=
FAST_LANE__SAFETY_CHECK__NLI_MODEL=

SLOW_LANE__MAX_ITERATIONS=

################################################################################
# SYSTEM / LOGGING — rag_agent
################################################################################
LOG_LEVEL=INFO
ENVIRONMENT=development

################################################################################
# ROUTER
################################################################################
ROUTER__MODEL_NAME=             # rag_agent internal Fast/Slow router model
ROUTER_MODEL=gpt-4.1-mini       # router_agent LLM routing model
ROUTER_TEMPERATURE=0.0
ROUTER_MAX_TOKENS=500

ROUTING_RULES_PATH=config/routing_rules.yaml
LOG_DIR=logs
DEFAULT_DB_NAME=

################################################################################
# OPTIONAL: sibling agent paths — router_agent
################################################################################
TEXT_TO_SQL_AGENT_PATH=../text_to_sql_agent
RAG_AGENT_PATH=../rag_agent
```

> **Note on compat aliases:** The `text_to_sql_agent` was originally developed with its own environment variable naming convention (`AZURE_OPENAI_ENDPOINT`, `DEFAULT_MODEL`, etc.). Rather than refactoring all config references, the unified `.env` includes both the canonical `LLM__` prefixed keys (used by the router and RAG agent) and the alias keys that the Text-to-SQL agent reads directly. Keep both sets in sync when updating credentials.