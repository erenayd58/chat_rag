# /Users/murseltasgin/projects/chat_rag/README.md
# Advanced RAG System with Conversational Context

A modular, production-ready Retrieval-Augmented Generation (RAG) system with sophisticated ingestion and query pipelines.

## Features

- **Semantic Chunking**: Context-preserving document chunking with overlap
- **Contextual RAG**: Document metadata enrichment and context awareness
- **Query Understanding**: Automatic query clarification and expansion
- **Hybrid Retrieval**: Combines vector search (semantic) with BM25 (keyword)
- **Intelligent Reranking**: LLM-based or Cross-Encoder reranking for optimal results
- **Conversation Tracking**: Multi-turn conversation support with reference resolution
- **Smart Search Strategy**: Automatically selects optimal retrieval method
- **Multiple LLM Providers**: Support for Azure OpenAI and Ollama
- **Multiple Vector DBs**: Support for ChromaDB and FAISS
- **Knowledge Base Management**: Create and manage multiple knowledge bases
- **Document Tracking**: Automatic tracking to avoid re-processing documents
- **Web Interface**: Modern browser-based UI for document management and chat
- **CLI Interface**: Command-line chat application with automatic ingestion
- **Modular Architecture**: Pluggable components following SOLID principles

## Architecture

```
chat_rag/
├── config/                  # Configuration management
│   ├── __init__.py
│   └── settings.py         # Central config from .env
├── core/                   # Core data models and exceptions
│   ├── __init__.py
│   ├── models.py          # Data models (DocumentChunk, RetrievalResult, etc.)
│   └── exceptions.py      # Custom exceptions
├── components/            # Pluggable components
│   ├── chunker/          # Text chunking strategies
│   │   ├── base.py       # Base chunker interface
│   │   └── semantic_chunker.py
│   ├── embedding/        # Embedding models
│   │   ├── base.py       # Base embedding interface
│   │   └── sentence_transformer_embedding.py
│   ├── vectordb/         # Vector database providers
│   │   ├── base.py       # Base vectordb interface
│   │   ├── chroma_vectordb.py  # ChromaDB implementation
│   │   └── faiss_vectordb.py   # FAISS implementation
│   ├── llm/              # LLM providers
│   │   ├── base.py       # Base LLM interface
│   │   ├── azure_openai_llm.py  # Azure OpenAI implementation
│   │   └── ollama_llm.py        # Ollama implementation
│   ├── retriever/        # Retrieval strategies
│   │   ├── base.py       # Base retriever interface
│   │   └── hybrid_retriever.py
│   ├── query_processor/  # Query understanding and enhancement
│   │   └── query_enhancer.py
│   ├── reranker/         # Result reranking
│   │   ├── base.py      # Base reranker interface
│   │   ├── reranker.py  # LLM-based reranker
│   │   └── cross_encoder_reranker.py  # Cross-encoder reranker
│   ├── conversation/     # Conversation management
│   │   └── conversation_manager.py
│   ├── document_processor/ # Document preprocessing
│   │   └── document_processor.py
│   ├── parsers/            # Document parsers
│   │   ├── base.py        # Base parser interface
│   │   ├── pdf_parser.py  # PDF parsing
│   │   ├── docx_parser.py # DOCX parsing
│   │   ├── markdown_parser.py  # Markdown parsing
│   │   ├── text_parser.py # Plain text parsing
│   │   ├── image_parser.py # Image OCR parsing
│   │   └── parser_factory.py  # Parser factory
│   └── contextual_enhancer/ # Contextual enrichment
│       └── contextual_enhancer.py
│   └── knowledgebase/    # Knowledge base management
│       └── manager.py    # Multi-KB manager
├── pipeline/             # Main orchestration
│   ├── __init__.py
│   └── rag_pipeline.py   # Main RAG pipeline
├── utils/                # Utilities
│   ├── logger.py         # Logging utilities
│   └── document_tracker.py # Document ingestion tracking
├── main_new.py           # CLI chat application
├── app.py                # Web application (Flask)
├── requirements.txt      # Dependencies
└── env.example          # Environment variables template
```

## The model chain

Three model roles, one OpenRouter key, each role configured on its own:

| Stage | Model (demo) | Configuration | Runs |
|---|---|---|---|
| Deep Analysis / Agentic chunking — proposer + verifier | `qwen/qwen3-30b-a3b-instruct-2507` | `DEEP_ANALYSIS_MODEL`, `DEEP_ANALYSIS_ENDPOINT`, `DEEP_ANALYSIS_API_KEY_ENV`, `DEEP_ANALYSIS_VERIFY` | at upload only, through `amsc.deep_pipeline` |
| Embedding — document chunks and questions, one space | `qwen/qwen3-embedding-8b` | `EMBEDDING_PROVIDER=openrouter`, `EMBEDDING_MODEL`, `EMBEDDING_ENDPOINT`, `EMBEDDING_API_KEY_ENV` | at upload (chunks) and per question (query vector) |
| Answer — reads the assembled context, cites sources | `minimax/minimax-m2.7` | `ANSWER_PROVIDER=openrouter`, `ANSWER_MODEL`, `ANSWER_ENDPOINT`, `ANSWER_API_KEY_ENV` | per question |
| Answer fallback (local, offline) | Ollama `qwen2.5:3b` | `ANSWER_FALLBACK_PROVIDER=ollama`, `ANSWER_FALLBACK_MODEL` | only when the primary answer call fails |

`RETRIEVAL_PROFILE=hybrid_rrf` is the final retrieval profile: the stored
Qwen3 vectors (cosine) and the frozen deterministic BM25 (Turkish diacritic
fold) each rank a candidate pool of 50, reciprocal-rank fusion (k = 60)
merges them with chunk-id tie-breaking, and the top hits are assembled into a
labelled context (`[S1]`, `[S2]`, …; de-duplicated, same-section neighbours
allowed, `CONTEXT_MAX_TOKENS` budget) that the answer model must cite.
Standard and Deep Analysis documents go through exactly this path — only
their chunk partition differs. No ingest-time model runs during a question.

Each knowledge base's vector store carries an `embedding_index.json`
manifest (provider, model, dimension, fingerprint). When the configured
embedding changes, the knowledge base reports **re-index required**: dense
retrieval is switched off (keyword results only, said so in the chat), new
uploads are refused with 409 until the store is rebuilt, and **Settings →
Embedding index → Re-index** re-embeds every stored chunk with the current
model. Vectors from two models are never compared.

Query responses carry the observability the Lab and the chat notices use:
retrieval mode and hit counts (dense / BM25 / fused), selected context and
its token estimate, embedding model and fingerprint, answer provider and
model, whether the fallback answered, and per-stage latency. Prompts and
keys are never stored.

## Demo mode (product + Agentic Chunking Viewer)

The proof of concept has two faces: this product (how it is used) and the
chunk repository's **Viewer v2** (what the chunking technology does
underneath — Sunum / Sorgu / Debug / Benchmark). They stay separate servers;
one script starts both for a presentation.

```powershell
.\start-demo.ps1      # start both, wait until each answers, open the product
.\stop-demo.ps1       # stop what start-demo started
```

| | Address | Server |
|---|---|---|
| Product (chat_rag) | http://127.0.0.1:5005 | `venv\Scripts\python.exe app.py` (`FLASK_PORT`, reloader off) |
| Viewer (chunk, Viewer v2) | http://127.0.0.1:8765 | `py -3.11 -m amsc.viewer_server --viewer artifacts/viewer-v2/index.html` in the chunk repo |

The chunk repository is expected next to this one (`..\chunk`); override with
`-ChunkPath` or `CHUNK_REPO`. The launcher checks readiness over HTTP
(`/api/health` on both), recognises servers that are already running instead
of starting a second copy, refuses a port held by something else, writes the
servers' output to `.demo\logs\` and the started process ids to
`.demo\state.json` (both git-ignored). `stop-demo.ps1` stops only processes it
can identify as those servers; `-All` extends that to a product/viewer server
on the demo ports that it did not start. Nothing from `.env` is printed; the
Viewer's chat gets `OPENROUTER_API_KEY` from the environment or `.env`
(without it the Viewer runs BM25-only, no answers — use `-Lexical` to force
that). Options: `-NoBrowser`, `-OpenViewer` (second tab), `-ProductPort`,
`-ViewerPort`, `-TimeoutSeconds`.

The two windows share one state. `GET /api/demo/workspace` returns this
console's knowledge bases, their documents and chunk counts as a read-only
snapshot; the Viewer's server reads it (`--console-url`, which `start-demo.ps1`
points back here) and serves it to its own page. So a knowledge base created
here, or a document ingested into it, appears in the Viewer's workspace strip
on its next refresh — there is no second copy of that state to keep in step,
and the browser never has to reach a second origin. The snapshot carries names,
counts and ingest metadata only: no absolute paths and no full file hashes.

**A document uploaded here becomes a document you can analyse over there.**
The Viewer reads one shape — a packaged Deep Analysis tree pinned to the
canonical it was chunked from — and an ingest already produces every expensive
input that tree needs, so nothing is computed twice:

* the canonical is the one the chunker normalised for *this* ingest, so the
  PDF is never parsed again (a document ingested before this existed has its
  canonical recovered from the parser's own cache instead);
* a **Deep Analysis** upload hands over its whole run — deep rows, Standard
  rows, selection audit, verifier verdicts, proposer audit — so **no second
  proposer or verifier call is ever made**;
* a **Standard** upload has no run to reuse, so the Deep side of the
  comparison is the *deterministic* quality contract (`use_llm=False`): zero
  provider calls, zero cost, and labelled as such rather than passed off as a
  model-backed run.

`components/viewer/analysis.py` does the packaging on a background worker, so
no HTTP call waits on it: an upload records the ingest and returns, and the
Viewer's refresh (`?prepare=1`) only *queues* what is missing. Each document's
state — `missing` / `pending` / `running` / `ready` / `failed` — travels with
it in the workspace snapshot, so the Viewer can say "Viewer analizi
hazırlanıyor…" and open it when it is done. A build interrupted by a restart
is picked up again from disk. Deleting a document here deletes its analysis;
the chunk repository's frozen benchmark trees are never reachable from this
path. Everything lives under `artifacts/viewer-live/` (git-ignored) and is
regenerable from an ingest.

These documents are a **live workspace category**, not benchmark data. They
have no gold query set, so no Hit@k or MRR is computed for them — the Viewer
says so rather than inventing numbers — and they never enter the frozen
benchmark tables or the cross-document contract table.

Inside the product, **Tools → Agentic Chunking Viewer** (sidebar, with a
live/offline dot) and the card at the top of **Lab** open the Viewer in a new
tab. The address comes from `VIEWER_URL` (default `http://127.0.0.1:8765/`;
empty hides the link).

Presentation order: **1.** chat_rag — a knowledge base and its documents;
**2.** upload a document with **Deep Analysis** (status and quality summary
under the chunking badge, *Details* for before/after); **3.** Chat — an answer
with sources; **4.** Agentic Chunking Viewer; **5.** Sunum (the four methods
side by side) → Debug (why each boundary) → Benchmark; then back to the
product.

## Running with Docker

One container runs the whole application. There is no separate database,
queue or model service: Ollama stays on the host, and everything else runs
in-process.

### Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine with Compose v2
- Ollama on the host **only if you want generated answers**. Uploading,
  parsing, structure-first chunking, structural QA and BM25 search all work
  with Ollama stopped; generation then returns an explanatory error instead of
  taking the application down.

### Start

```bash
docker compose up --build
```

To have the QA report name the commit the image was built from -- the image
does not ship `.git`, so it otherwise reports it as unknown:

```bash
CHAT_RAG_GIT_SHA=$(git rev-parse HEAD) docker compose up --build
# PowerShell: $env:CHAT_RAG_GIT_SHA = (git rev-parse HEAD); docker compose up --build
```

Then open <http://localhost:5005>. The app lands on Knowledge Bases;
Chat is at `/chat` and the technical tools (retrieval quality review,
parser output, chunk browser) are under `/lab`.

### Stop

```bash
docker compose down
```

### Configuration

`.env.docker` holds the container's settings and contains no secrets. Put
anything private in `.env.docker.local`, which is git-ignored and overrides
it. The local `.env` is deliberately not used by the container: it points at
`localhost`, which inside a container means the container itself.

Ollama is reached at `http://host.docker.internal:11434`. That address lives
in `.env.docker`, not in the code; the compose file maps the name explicitly
so it also works on plain Linux Docker.

### Where the data lives

Everything the container persists is under `./.docker-data`, which is a
different place from the paths a local checkout uses. Running the container
never reads or writes your local `chroma_db/`, `.knowledge_bases.json`,
`.ingested_documents.json` or `.cache/`.

```
.docker-data/
  state/      knowledge_bases.json, ingested_documents.json, gold_set.json
  chroma/     one vector store per knowledge base
  faiss/      the same, for knowledge bases using the FAISS provider
  cache/      the parser's canonical-unit cache
  logs/       application logs
  artifacts/  evaluation runs and QA reports written by the CLI
```

Frozen gold sets under `artifacts/gold/` are inputs, not state: they travel
inside the image and are never written to.

### Reset the container's data

Stop the container first, then delete the one directory:

```bash
docker compose down
rm -rf ./.docker-data          # PowerShell: Remove-Item -Recurse -Force .docker-data
```

This removes only the container's knowledge bases, stores and logs. Your local
development data is untouched.

### The CLI, inside the container

The same commands, no separate image:

```bash
docker compose exec app python -m cli inspect --kb <name>
docker compose exec app python -m cli search  --kb <name> --query "..."
docker compose exec app python -m cli qa      --kb <name>
docker compose exec app python -m cli report  --kb <name> --gold artifacts/gold/<set>.json
docker compose exec app python -m cli eval    --kb <name> --gold artifacts/gold/<set>.json
```

Reports and runs land in `./.docker-data/artifacts/` on the host.

### Health

`GET /api/health` answers from the Flask app alone -- it loads no model, parses
nothing and does not touch the vector store. That is what the container's
healthcheck calls.

```bash
docker compose ps          # STATUS shows (healthy)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Ollama installed locally if using Ollama LLM provider

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd chat_rag
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```bash
python setup_nltk.py
```

5. **Configure environment**
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your configuration
# For Azure OpenAI: Set AZURE_ENDPOINT, AZURE_API_KEY, AZURE_DEPLOYMENT
# For Ollama: Set LLM_PROVIDER=ollama and OLLAMA_MODEL
```

### Default demo profile

The low-cost profile validated on the KKB documents:

```env
CHUNKER_TYPE=structure_first
RETRIEVAL_PROFILE=bm25_only
```

Structure-first chunking lets document structure decide chunk boundaries
(a chunk opens at every heading and section change, oversized units split at
table row / list item / sentence seams) and BM25-only retrieval loads no
embedding model at all: no dense vectors are computed or stored in this
profile. The legacy and V4 chunkers and the `legacy` / `benchmark_aligned`
retrieval profiles remain selectable for comparison.

**First upload of a PDF is slow.** Parsing runs layout inference over every
logical page, which is a few seconds per page on CPU -- an 85-page report takes
roughly ten minutes, and essentially all of it is layout model inference rather
than anything in this repository. The resulting canonical units are cached on
disk under `.cache/canonical-units/`, keyed by the PDF content hash, so
re-ingesting the same document afterwards takes well under a second. For a
demo, upload the document once beforehand. Deleting the cache directory is
safe; it is regenerated on the next ingest. Set `STRUCTURED_PARSER_CACHE` to
move it elsewhere.

### LLM Provider Setup

**Option 1: Azure OpenAI (Cloud-based)**
```env
LLM_PROVIDER=azure
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_KEY=your-api-key
AZURE_DEPLOYMENT=gpt-4o
```

**Option 2: Ollama (Local, Free)**
```bash
# Install Ollama first from https://ollama.ai
# Pull a model
ollama pull llama2

# Configure in .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434
```

See [Ollama Guide](docs/OLLAMA_GUIDE.md) for more details.

## Configuration

All configuration is centralized in `config/settings.py` and reads from `.env` file. See `env.example` for all available options.

### Key Configuration Options

```env
# LLM Provider (azure or ollama)
LLM_PROVIDER=azure

# Azure OpenAI (when LLM_PROVIDER=azure)
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_KEY=your-api-key
AZURE_DEPLOYMENT=gpt-4o

# Ollama (when LLM_PROVIDER=ollama)
OLLAMA_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Database Provider (chroma or faiss)
VECTOR_DB_PROVIDER=chroma
VECTOR_DB_PATH=./chroma_db

# Chunking
CHUNKER_TYPE=legacy  # legacy or v4
CHUNK_SIZE=512
CHUNK_OVERLAP=128
MIN_CHUNK_SIZE=50

# Retrieval
RETRIEVAL_PROFILE=legacy  # legacy or benchmark_aligned
DEFAULT_TOP_K=5
VECTOR_WEIGHT=0.7
BM25_WEIGHT=0.3

# Conversation
ENABLE_CONVERSATION=true
MAX_CONVERSATION_HISTORY=10

# Reranker Configuration
RERANKER_TYPE=cross_encoder  # Options: 'llm' or 'cross_encoder'
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Document Input
DOCUMENTS_INPUT_PATH=./documents
DOCUMENTS_RECURSIVE=true

# Logging
LOG_LEVEL=INFO
LOG_TOKEN_USAGE=true
```

For complete configuration options, see `env.example`.

### Frozen V4/A4 chunking

`CHUNKER_TYPE=legacy` preserves the existing `SemanticChunker`. Setting
`CHUNKER_TYPE=v4`, or selecting `v4` while creating a knowledge base in the
web UI, uses the Phase-5 AMSC V4/A4 implementation. The dependency is pinned to
`erenayd58/chunk` commit
`1e7f7186c13729c739ccb3170da0892f7350cb27`; the integration rejects any V4
config whose semantic hash differs from `f29f805deee9189c`. V4 does not accept
runtime chunk-size or threshold parameters.

The current parsers return flat text. The normalization adapter therefore maps
blank-line-delimited parser blocks to ordered canonical paragraphs and does not
guess headings, pages, tables, lists, or visuals. If a parser supplies structured
unit metadata, the same adapter preserves those fields directly.

To run the minimal product demo:

1. Install `requirements.txt` and start `python app.py`.
2. Create one knowledge base with chunker `legacy` and another with `v4`.
3. Upload a document to either knowledge base and ask a question from the chat.
4. Open `/documents` to inspect the stored chunks and retrieval results.
5. To compare the same query, select each knowledge base in turn in Retrieval
   Experimentation and run the identical query.

The first V4 ingestion may download `intfloat/multilingual-e5-base`; subsequent
boundary embeddings use `.cache/boundary-embeddings`.

### Retrieval profiles

`RETRIEVAL_PROFILE=legacy` preserves the existing chat_rag retrieval behavior.
`RETRIEVAL_PROFILE=benchmark_aligned` selects the Phase 4/5 profile pinned at
commit `1e7f7186c13729c739ccb3170da0892f7350cb27`: multilingual E5 role prefixes,
normalized deterministic long-text pooling, Unicode BM25, and equal-weight RRF
with a 100-result pool and `k=60`. Query expansion, contextualization, and
reranking are disabled in this profile. Its E5 model is loaded with
`local_files_only=true`, matching the frozen benchmark configuration.

Indexes are profile-specific because the embedding models and dimensions differ.
Use a new vector-database path/collection and re-ingest documents when changing
profiles; do not point `benchmark_aligned` at an index created by `legacy`.

## Quick Start

### 1. Installation (5 minutes)

```bash
# Clone and navigate
git clone <repository-url>
cd chat_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup NLTK data
python setup_nltk.py

# Configure environment
cp env.example .env
# Edit .env with your credentials
```

### 2. Choose Your Interface

**Option A: CLI Chat (Simple)**
```bash
# Add documents to ./documents folder
# Start CLI application
python main_new.py
```

**Option B: Web Application (Full Features)**
```bash
# Start web server
python app.py
# Open browser: http://localhost:5005
```

### 3. Start Chatting

- CLI: Type questions directly in the terminal
- Web: Use the browser interface to chat and manage documents

See detailed usage sections below for more information.

## Usage

### CLI Chat Application

The command-line interface provides an interactive chat experience with automatic document ingestion.

**Start the CLI application:**
```bash
python main_new.py
```

**What happens when you start:**
1. Configuration is loaded from `.env` file
2. RAG pipeline is initialized with your settings
3. Documents are automatically scanned from `DOCUMENTS_INPUT_PATH` (default: `./documents`)
4. New documents are ingested (already processed documents are skipped)
5. Interactive chat session begins

**Available CLI Commands:**
- Type your question and press Enter to chat
- `help` - Show available commands
- `stats` - Display document statistics (total documents, chunks, size, etc.)
- `clear` - Clear conversation history
- `exit` or `quit` - Exit the application

**Example CLI Session:**
```bash
$ python main_new.py

================================================================================
  RAG CONVERSATIONAL CHAT
  Retrieval-Augmented Generation with Document Ingestion
================================================================================

⚙️  Loading configuration...
✓ Configuration loaded

🚀 Initializing RAG pipeline...
✓ Pipeline initialized

================================================================================
DOCUMENT INGESTION
================================================================================

📂 Scanning for documents in: ./documents
   Found 2 new document(s)
   Skipping 0 already ingested document(s)

📥 Ingesting new documents...

[1/2] Processing: report.pdf
  ✓ Success: 15 chunks created

[2/2] Processing: notes.pdf
  ✓ Success: 22 chunks created

✓ Successfully ingested 2 new document(s)

📊 DOCUMENT STATISTICS
================================================================================
Total Documents: 2
Total Chunks: 37
Total Size: 2.45 MB

================================================================================
💬 CHAT MODE
================================================================================

You: What is the main topic?
Assistant: The main topic covers project documentation and requirements...

📚 Show sources? (y/n): y

📄 Sources:
1. report.pdf
   Section: Introduction
   Relevance Score: 0.856
   Preview: The document discusses...

You: 
```

### Web Application

The web application provides a modern browser-based interface with advanced features.

**Start the web application:**
```bash
python app.py
```

**Access the application:**
Open your browser to: `http://localhost:5005`

**Web Application Features:**
- 🎨 Modern UI with gradient design
- 💬 Real-time conversational chat with multi-turn support
- 📚 Source citations with relevance scores
- 📊 Document and knowledge base statistics
- 📁 Document management (upload, view, delete)
- 🔍 Chunk browsing and editing
- 🗄️ Multiple knowledge base support
- 🗑️ Clear conversation history
- 📱 Fully responsive design

**Important Notes:**
- The web app runs on port **5005** (not 5000)
- Documents can be managed through the web interface at `/documents`
- Multiple knowledge bases can be created and managed
- Each knowledge base can have its own vector database, embedding model, and chunker configuration

### Knowledge Base Management

The system supports multiple knowledge bases, each with its own configuration:

**Creating a Knowledge Base (via Web UI):**
1. Click "➕ New KB" button in the web interface
2. Configure:
   - Name: Descriptive name for the KB
   - Vector DB Provider: chroma or faiss
   - Embedding Model: Model name for embeddings
   - Chunker Config: Chunking parameters
   - Vector DB Path: Storage location (optional)

**Using Knowledge Bases:**
- Each KB has a unique ID
- Documents are ingested into specific KBs
- Queries can target specific KBs or use the default
- KBs can be managed through the web interface

**Programmatic KB Management:**
```python
from components.knowledgebase.manager import KnowledgeBaseManager

kb_manager = KnowledgeBaseManager()

# Create a new KB
kb = kb_manager.create(
    name="Technical Documentation",
    vector_db_provider="faiss",
    embedding_model_name="all-MiniLM-L6-v2"
)

# List all KBs
all_kbs = kb_manager.list()

# Get a specific KB
kb_config = kb_manager.get(kb_id="abc12345")

# Update a KB
kb_manager.update(kb_id="abc12345", updates={"name": "Updated Name"})

# Delete a KB
kb_manager.delete(kb_id="abc12345")
```

### Document Ingestion

**Automatic Ingestion (CLI):**
- Documents in `./documents` folder are automatically ingested on startup
- Already processed documents are skipped (tracked in `.ingested_documents.json`)

**Manual Ingestion (Web UI):**
- Open a knowledge base and use **Upload Document**
- Choose the chunking mode per document: **Standard** or **Deep Analysis**
- Documents are processed and indexed automatically

**Chunking modes (chosen at upload, never at query time):**

| Mode | What runs | When the model is unavailable |
|---|---|---|
| Standard | The frozen structure-first walk (`amsc.structural_chunker`). Fast, deterministic, no model. | — |
| Deep Analysis | `amsc.deep_pipeline.chunk_document(mode="deep")`: the same structural walk, a backend LLM **proposer** (one bounded prompt per section that still has a choice), the deterministic **quality selector** (never worse than Standard on any smell type), the double-order **verifier** (a change is kept only when it wins in both orders) and the quality measurement. | The ingest still completes on the deterministic quality contract and the document is labelled with the pipeline status — never passed off as Standard. |

Deep Analysis statuses, as recorded on the document and shown under the
chunking badge: `ok` (quality checks passed), `deterministic` (LLM not
requested), `fallback_no_provider` (model or key not configured),
`fallback_provider_error` (every model call failed), `degraded` (some calls
failed; those sections kept their deterministic result). The document's
**Details** row shows quality before → after, model/verifier usage and the
structural checks (hard token cap, coverage).

Configuration is backend-only (`DEEP_ANALYSIS_MODEL`, `DEEP_ANALYSIS_ENDPOINT`,
`DEEP_ANALYSIS_API_KEY_ENV`, `DEEP_ANALYSIS_VERIFY`, …; see `env.example`).
Only the *name* of the key variable is configured; the key is read at request
time by the provider and never stored, logged or written to provenance. Chat
reads the chunks that were indexed at upload; no ingest model runs during a
question.

**Programmatic Ingestion:**
```python
from pipeline import RAGPipeline
from config import Settings

settings = Settings()
pipeline = RAGPipeline(settings=settings)

# Ingest from directory
results = pipeline.ingest_documents_from_directory(
    directory_path="./my_documents",
    recursive=True
)

# Ingest single file
chunks = pipeline.ingest_document_from_file("./documents/report.pdf")
```

### Basic Usage

```python
from pipeline import RAGPipeline
from config import Settings

# Initialize
settings = Settings()
rag_pipeline = RAGPipeline(settings=settings)

# Ingest documents from default directory (set in .env: DOCUMENTS_INPUT_PATH)
results = rag_pipeline.ingest_documents_from_directory()

# Or specify a custom directory
results = rag_pipeline.ingest_documents_from_directory(
    directory_path="./my_documents",
    recursive=True,
    file_pattern="*.pdf"  # Optional: filter by file type
)

# Ingest a single document
chunks = rag_pipeline.ingest_document_from_file("./documents/report.pdf")

# Retrieve relevant information
results, metadata = rag_pipeline.retrieve(
    query="What is John Smith's role?",
    top_k=5
)

# Format results for LLM
context = rag_pipeline.get_retrieval_context(results)
print(context)
```

### Conversational Usage

```python
# Turn 1
results1, _ = rag_pipeline.retrieve("What is John Smith's role?")
assistant_response = "John Smith is a Senior Software Engineer."
rag_pipeline.add_assistant_response(assistant_response)

# Turn 2 - Ambiguous reference resolved automatically
results2, _ = rag_pipeline.retrieve("How old is he?")
# System automatically clarifies to "How old is John Smith?"

# View conversation history
print(rag_pipeline.get_conversation_summary())
```

### Custom Components

You can replace any component with your own implementation:

```python
from components.llm import BaseLLM
from components.embedding import BaseEmbedding
from components.reranker import CrossEncoderReranker

# Use cross-encoder reranker for better performance
cross_encoder = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
rag_pipeline = RAGPipeline(reranker=cross_encoder, settings=settings)

# Custom LLM
class MyCustomLLM(BaseLLM):
    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        # Your implementation
        pass
    
    def get_name(self):
        return "MyCustomLLM"
    
    def get_model_name(self):
        return "custom-model"

# Use custom component
custom_llm = MyCustomLLM()
rag_pipeline = RAGPipeline(llm_model=custom_llm, settings=settings)
```

## Adding New Components

### Adding a New LLM Provider

1. Create a new file: `components/llm/my_llm.py`
2. Implement `BaseLLM` interface
3. Import in `components/llm/__init__.py`
4. Use in pipeline initialization

```python
from components.llm.base import BaseLLM

class MyLLM(BaseLLM):
    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        # Implementation
        pass
    
    def get_name(self):
        return "MyLLM"
    
    def get_model_name(self):
        return "my-model-v1"
```

### Adding a New Vector Database

1. Create: `components/vectordb/my_vectordb.py`
2. Implement `BaseVectorDB` interface
3. Import in `components/vectordb/__init__.py`

## Design Principles

This codebase follows these key principles:

1. **Abstraction**: No hardcoded technology-specific code in main components
2. **Separation of Concerns**: Data access, business logic, and presentation are separated
3. **Strategy Pattern**: Algorithms are pluggable (retrieval, chunking, etc.)
4. **Interface Segregation**: Components depend only on methods they use
5. **Modularity**: Code is organized in small, focused modules (< 500 lines per file)
6. **Centralized Configuration**: All config through central settings module
7. **Resilience**: Exception handling and graceful degradation
8. **Extensibility**: Easy to add new components without changing existing code

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run end-to-end tests
python -m pytest tests/e2e/

# Run example with cross-encoder reranker
python examples/cross_encoder_reranker_example.py
```

## Reranking Strategies

The system supports two reranking strategies:

1. **LLM Reranker**: Uses language model for relevance assessment (flexible but slower)
2. **Cross-Encoder Reranker**: Uses specialized cross-encoder model (fast and accurate)

See [Reranker Guide](docs/RERANKER_GUIDE.md) for detailed comparison and usage.

**Quick Start with Cross-Encoder:**
```python
# Set in .env
RERANKER_TYPE=cross_encoder

# Or in code
from components.reranker import CrossEncoderReranker
reranker = CrossEncoderReranker()
pipeline = RAGPipeline(reranker=reranker)
```

## Logging and Metrics

The system includes:
- Standard logging for debugging
- Token usage tracking
- Performance metrics
- Input/output logging for LLM calls

## Health Checks

For API deployments, health check endpoints verify:
- LLM connectivity
- Vector database status
- Embedding model availability

## Contributing

1. Follow the existing code structure
2. Keep files under 500-600 lines
3. Keep functions/methods under 20-30 lines
4. Add unit tests for new components
5. Update documentation

## License

[Your License]

## Support

For issues and questions, please open a GitHub issue.

