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
- Navigate to `/documents` page
- Upload documents directly through the web interface
- Documents are processed and indexed automatically

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

