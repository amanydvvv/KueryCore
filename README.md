<div align="center">

# ⚡ KueryCore AI

### *Enterprise Document Intelligence & Production-Hardened Hybrid RAG Platform*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?style=for-the-badge&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16_pgvector-336791.svg?style=for-the-badge&logo=PostgreSQL&logoColor=white)](https://github.com/pgvector/pgvector)
[![React](https://img.shields.io/badge/React-19_Vite-61DAFB.svg?style=for-the-badge&logo=React&logoColor=black)](https://react.dev)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Inference-Groq_Cascade-F55036.svg?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <a href="#-system-architecture--request-lifecycle">Architecture</a> •
  <a href="#-core-capabilities">Capabilities</a> •
  <a href="#-technical-specifications">Tech Stack</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-api-contract--endpoints">API Docs</a> •
  <a href="#-environment-setup--local-development">Quickstart</a> •
  <a href="#-verification--automated-tests">Testing</a>
</p>

</div>

---

KueryCore AI is an enterprise-grade Retrieval-Augmented Generation (RAG) platform designed to ingest, index, and query unstructured documents with sub-second hybrid retrieval, real-time Server-Sent Events (SSE) streaming, deterministic PDF citation fidelity, and multi-tenant security isolation.

---

## ◈ System Architecture & Request Lifecycle

```mermaid
flowchart TD
    subgraph Ingestion["1. Multi-Stage Ingestion Pipeline"]
        Upload["Document Upload (PDF / MD)"] --> Parser["PyMuPDF Native Text Parser"]
        Parser --> BlankCheck{"Text Extracted?"}
        BlankCheck -->|No / Scanned PDF| VisionOCR["Vision OCR Fallback<br/>(Groq Qwen 3.6-27B)"]
        BlankCheck -->|Yes| ChunkEngine["Recursive Character Splitter<br/>(800 Chars / 150 Overlap)"]
        VisionOCR --> ChunkEngine
        ChunkEngine --> RawBytesStore[("PostgreSQL Store<br/>(Encrypted Raw PDF Bytes)")]
        ChunkEngine --> EmbedModel["Google Gemini Embedding Engine<br/>(text-embedding-004 · 768-dim)"]
        EmbedModel --> DenseIndex[("pgvector Store<br/>(HNSW Cosine Distance)")]
        ChunkEngine --> SparseIndex[("PostgreSQL FTS Store<br/>(tsvector english GIN)")]
    end

    subgraph RetrievalEngine["2. Two-Stage Hybrid Retrieval Pipeline"]
        UserPrompt["User Query"] --> GuardIn{"Input Guardrails<br/>(PII Redact + Injection Filter)"}
        GuardIn -->|Flagged Injection| CannedRefusal["Immediate Grounded Refusal"]
        GuardIn -->|Passed| RewriteEngine["Deictic Query Rewriter<br/>(Multi-Turn Context Resolution)"]
        RewriteEngine --> QueryCacheCheck{"In-Memory TTL LRU Cache<br/>(Tenant-Isolated Keying)"}
        QueryCacheCheck -->|Cache Hit| ContextRerank
        QueryCacheCheck -->|Cache Miss| ConcurrentSearch["Concurrent Search (asyncio.gather)"]
        DenseIndex -.-> ConcurrentSearch
        SparseIndex -.-> ConcurrentSearch
        ConcurrentSearch --> RRF["Reciprocal Rank Fusion (RRF, k=60)<br/>Top-20 Dense + Top-20 Sparse → Top-10"]
        RRF --> Reranker["Option 2b Phrase Coverage Re-Scorer<br/>(0.50 Dense + 0.30 Lexical + 0.20 Phrase)"]
        Reranker --> ContextRerank["Final Top-K Context Selection"]
    end

    subgraph GenerationEngine["3. Resilient Streaming Generation"]
        ContextRerank --> SlidingWindow["Sliding Token-Budgeted History<br/>+ Context Summarizer"]
        SlidingWindow --> PromptAssemble["Prompt Assembler<br/>(Grounding Rules + Corpus Metadata)"]
        PromptAssemble --> LLMCascade{"Resilient LLM Cascade<br/>Groq Primary → GPT-OSS → Qwen → Gemini"}
        LLMCascade --> CoTBuffer["StreamCoTBuffer State Machine<br/>(Isolates & Discards &lt;thought_process&gt;)"]
        CoTBuffer --> OutputGuard{"Output Validation<br/>(Prompt Leak & Unsafe Filter)"}
        OutputGuard --> SSEStream["SSE Stream: Real-Time Tokens<br/>+ PDF Citation Metadata Payloads"]
    end
```

---

## ◆ Core Capabilities

* **Two-Stage Hybrid Search & RRF Re-ranking:**
  * **Stage 1 (Concurrent Retrieval):** Queries `pgvector` HNSW index (top 20) and PostgreSQL Full-Text Search with `ts_rank_cd` (top 20) concurrently via `asyncio.gather`.
  * **Reciprocal Rank Fusion (RRF):** Fuses candidates using rank smoothing constant $k=60$ into a top-10 candidate pool.
  * **Stage 2 (Option 2b Re-scorer):** Normalizes three independent signals ($0.50 \cdot S_{\text{vector}} + 0.30 \cdot S_{\text{lexical}} + 0.20 \cdot S_{\text{phrase}}$) for high-precision snippet selection.
* **Resilient Multi-Provider LLM Fallback Cascade:**
  * Primary: Groq LLaMA 3.3 70B Versatile
  * Fallback 1: Groq OpenAI GPT-OSS 120B
  * Fallback 2: Groq Qwen 3.6 27B
  * Fallback 3: Google Gemini 1.5 Flash (Cross-provider cloud safety net)
* **Intelligent Document Parsing & OCR Fallback:**
  * PyMuPDF for fast digital PDF extraction.
  * Automatic image-density OCR fallback via Groq Vision (`qwen/qwen3.6-27b`) for scanned, handwritten, or image-only documents.
* **Tenant-Isolated In-Process Query Cache:**
  * Deterministic exact-normalized matching with TTL LRU eviction.
  * Tenant-isolated keys (`{user_id}.{doc_id}.{top_k}.{query}`) with automatic $O(1)$ full cache invalidation on any document upload, reindex, or deletion.
* **Strict Multi-Layered Guardrails:**
  * **Input Sanitization:** Redacts emails, phone numbers, SSNs, credit cards, and UUIDs to `[REDACTED:<type>]`.
  * **Injection Defense:** Filters known prompt injection patterns and obfuscation runs (base64/unicode).
  * **Output Validation:** Intercepts system prompt fragments and unsafe keyword phrases before streaming to client.
* **Stateless Multi-Tenant JWT Auth & Security:**
  * PBKDF2-HMAC-SHA256 password hashing.
  * RFC 7519 HMAC-SHA256 JWT access tokens paired with atomic compare-and-swap `refresh_token_jti` rotation.
  * Peer-anchored client IP rate limiting via SlowAPI (`10/minute` auth, `60/minute` chat, `5/hour` account deletion).
* **Interactive PDF Citation Viewer:**
  * Integrated React PDF viewer supporting one-click citation pill jumping directly to source pages and bounding snippets.

---

## ◇ Technical Specifications

| Subsystem | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend API** | FastAPI, Uvicorn, SlowAPI | Async REST endpoints, Server-Sent Events (SSE), and rate limiting |
| **Database** | PostgreSQL 16 + `pgvector` | Relational entity storage, HNSW vector indexing, and English FTS GIN indexes |
| **ORM & Migrations** | SQLAlchemy 2.0 (Async), Alembic | Async ORM mappings, connection pooling, and schema migration ledger |
| **Embeddings** | `langchain-google-genai` | Google `text-embedding-004` (768-dimensional dense vectors) |
| **Inference Orchestration** | `langchain-groq`, `langchain-core` | Low-latency LLM generation with LangChain `with_fallbacks()` cascade |
| **Document Parsing** | PyMuPDF (fitz), Groq Vision | Digital text extraction and scanned PDF OCR |
| **Frontend UI** | React 19, Vite, Tailwind CSS, Lucide | Modern glassmorphic SPA with real-time SSE token rendering and PDF viewer |
| **Testing** | Pytest, Pytest-Asyncio, Vitest | Comprehensive backend integration testing and frontend component unit tests |

---

## ▧ Project Structure

```
KueryCore/
├── backend/
│   ├── alembic/                 # Database schema migration versions
│   ├── app/
│   │   ├── core/                # Security, JWT, PBKDF2, and rate limiter configuration
│   │   ├── models/              # SQLAlchemy models (User, Document, Chunk, Conversation, Message, QueryLog)
│   │   ├── routers/             # FastAPI routers (auth, documents, chat, conversations, analytics)
│   │   ├── schemas/             # Pydantic request/response validation schemas
│   │   ├── services/            # Core business logic:
│   │   │   ├── email.py         # Multi-provider email delivery (Brevo API, SMTP, Resend)
│   │   │   ├── evaluation.py    # RAG benchmark eval harness & judge scoring
│   │   │   ├── generation.py    # LLM fallback cascade, prompt template, and StreamCoTBuffer
│   │   │   ├── guardrails.py    # PII redaction, prompt injection filter, output validator
│   │   │   ├── ingestion.py     # Document parser, OCR fallback, and text splitter
│   │   │   ├── memory.py        # Background conversation summarizer
│   │   │   ├── query_cache.py   # In-memory TTL LRU query cache
│   │   │   ├── retrieval.py     # Two-stage hybrid search engine (pgvector + FTS + RRF)
│   │   │   └── rewrite.py       # Conversational follow-up query rewriter
│   │   ├── config.py            # Centralized Pydantic application settings
│   │   ├── database.py          # Async SQLAlchemy engine and session factory
│   │   └── main.py              # Application lifespan, CORS, and router registration
│   ├── requirements.txt         # Backend Python dependencies
│   └── tests/                   # Pytest test suite (auth, guardrails, retrieval, cache, cot)
├── frontend/
│   ├── src/
│   │   ├── components/          # React UI components (Sidebar, ChatArea, PdfViewer, UploadPanel, AuthModal)
│   │   ├── hooks/               # Custom hooks (useConversations, useChatStream)
│   │   ├── services/            # Frontend API client and event-stream consumers
│   │   └── tests/               # Vitest component and hook unit tests
│   ├── package.json             # Frontend dependencies
│   └── vite.config.js           # Vite development and build configuration
├── docker-compose.yml           # Full-stack container orchestration
├── Dockerfile                   # Backend production container specification
└── README.md                    # Project documentation
```

---

## ◬ API Contract & Endpoints

### Authentication (`/api/auth`)

| Method | Endpoint | Description | Rate Limit |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/auth/signup` | Register a new user account | `10/minute` |
| `POST` | `/api/auth/login` | Authenticate with credentials and receive access/refresh tokens | `10/minute` |
| `POST` | `/api/auth/refresh` | Rotate refresh token with atomic CAS JTI validation | `30/minute` |
| `GET` | `/api/auth/me` | Fetch authenticated user profile and account details | - |
| `DELETE`| `/api/auth/me` | GDPR self-service account deletion and cascade data purge | `5/hour` |
| `POST` | `/api/auth/forgot-password`| Request password reset email token | `5/minute` |
| `POST` | `/api/auth/reset-password` | Set new password using verified token | `5/minute` |

### Documents (`/api/documents`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/documents/upload` | Upload PDF or Markdown document for async ingestion |
| `GET` | `/api/documents` | List all documents belonging to current authenticated tenant |
| `GET` | `/api/documents/{id}` | Retrieve document status, page count, and chunk count |
| `GET` | `/api/documents/{id}/file` | Serve raw PDF bytes for authenticated inline citation viewing |
| `DELETE`| `/api/documents/{id}` | Delete document and cascade chunks with cache invalidation |
| `POST` | `/api/documents/{id}/reindex` | Trigger re-chunking and re-embedding pipeline |

### Chat & Streaming (`/api/chat`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/chat` | Standard single-turn chat completion with retrieved citations |
| `POST` | `/api/chat/stream` | Real-time Server-Sent Events (SSE) token stream with citation metadata |

#### SSE Event Protocol:

```text
event: citation
data: {"citation_id": 1, "document_id": "uuid", "display_title": "Q3_Report.pdf", "page_number": 4, "similarity": 0.89, "snippet": "..."}

event: token
data: {"token": "KueryCore"}

event: token
data: {"token": " utilizes"}

event: end
data: {"status": "completed", "latency_ms": 380}
```

### Conversations & Analytics

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/conversations` | Paginated list of user conversation sessions |
| `GET` | `/api/conversations/{id}` | Fetch full message history for a conversation |
| `DELETE`| `/api/conversations/{id}` | Delete conversation and associated message logs |
| `GET` | `/api/analytics` | Summary of workspace documents, queries, and chunk distribution |
| `GET` | `/api/health` | Diagnostic health check (DB, embeddings, and generative LLM status) |

---

## ▧ Environment Setup & Local Development

### Prerequisites

* Python `3.11+`
* Node.js `20+` and `npm`
* PostgreSQL `16` with `pgvector` enabled

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and supply your GROQ_API_KEY, GEMINI_API_KEY, DATABASE_URL, and JWT_SECRET_KEY

# Run database migrations
alembic upgrade head

# Start FastAPI development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```

### 3. Docker Compose (Full Stack)

```bash
docker-compose up --build
```

---

## ▣ Verification & Automated Tests

```bash
# Run backend test suite
cd backend
pytest tests/ -v

# Run frontend test suite
cd frontend
npm test -- --run

# Build frontend production bundle
npm run build
```


---

## ⚠️ Database Migration Rules (Read Before Deploying)

> Ignoring these rules is what caused the Can't locate revision crash-loop on Render.

### 1. Never run migrations against production from your local machine
Always use a **local or dev database** for local development.  
Never point DATABASE_URL at the Supabase production database while running lembic upgrade head locally.

### 2. Commit migration files before applying them to production
Every file in ackend/alembic/versions/ **must be committed and pushed** to the repo before the deploy that needs it.  
If a migration runs on the production DB but the file isn't in git, Render's Docker container won't have it — causing a Can't locate revision crash-loop.

**Rule of thumb:** git add backend/alembic/versions/ && git commit **before** you run lembic upgrade head on production.

### 3. Use the Supabase Session-mode pooler connection string
| Use-case | Port | Driver prefix |
|---|---|---|
| FastAPI / asyncpg (DATABASE_URL) | **6543** (Transaction mode) or **5432** (Session mode) | postgresql+asyncpg:// |
| Alembic migrations (DATABASE_URL_SYNC) | **5432** (Session mode) | postgresql:// |

Supabase pooler host: ws-0-<region>.pooler.supabase.com

### 4. Migrations run in Render's Pre-Deploy Command — not in the Docker CMD
ender.yaml is configured with preDeployCommand: "alembic upgrade head".  
This means a failed migration cancels the deploy **before** traffic is switched — no crash-loop.  
Do **not** re-add lembic upgrade head to the Dockerfile CMD.

---

## ⎋ License

Distributed under the **MIT License**. See `LICENSE` for details.
