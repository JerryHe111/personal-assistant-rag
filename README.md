# Personal RAG Assistant

A step-by-step RAG (Retrieval-Augmented Generation) learning project — from minimal prototype to production-ready personal life assistant.

See [ROADMAP.md](ROADMAP.md) for the full learning plan (中文 + English).

## Quick Start

### 1. Install dependencies

```bash
cd personal_rag_assistant
python -m venv venv
source venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

### 2. Set up API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Add your documents

Put `.txt`, `.md`, or `.pdf` files into `data/raw/`. Sample files are included.

### 4. Run

```bash
python -m src.app
```

### 5. Interactive commands

```
You: What are Jerry's career goals?     ← ask any question
You: /chunks                            ← view current chunks
You: /chunks compare                    ← compare recursive vs semantic chunking
You: /cost                              ← view accumulated API cost & latency
You: quit                               ← exit (prints session cost summary)
```

### 6. Debug chunking (standalone)

```bash
python -m src.chunking.preview_chunks                 # view chunks
python -m src.chunking.preview_chunks --compare        # recursive vs semantic side-by-side
python -m src.chunking.preview_chunks --strategy semantic
```

## Project Structure

```
personal_rag_assistant/
├── data/
│   └── raw/                   # Your source documents (.txt, .md, .pdf)
├── src/
│   ├── config.py              # Global configuration (models, chunk params, etc.)
│   ├── app.py                 # Main entry point — interactive CLI
│   ├── ingest/                # Document loaders
│   │   ├── load_text.py       #   .txt files
│   │   ├── load_markdown.py   #   .md files (extracts title as metadata)
│   │   └── load_pdf.py        #   .pdf files (per-page, with page number metadata)
│   ├── chunking/
│   │   ├── chunk_text.py      #   Recursive + Semantic chunking strategies
│   │   └── preview_chunks.py  #   Debug tool: view/compare chunk results
│   ├── embeddings/
│   │   └── embedding_model.py #   OpenAI embedding wrapper
│   ├── storage/
│   │   └── vector_store.py    #   ChromaDB operations
│   ├── generation/
│   │   ├── prompt_builder.py  #   RAG prompt construction
│   │   └── llm_client.py      #   LLM call with streaming
│   └── utils/
│       └── cost_tracker.py    #   API cost & latency monitoring
├── requirements.txt
├── .env.example
├── ROADMAP.md                 # Full learning roadmap (12 phases)
└── README.md
```

## Learning Phases

| Phase | Topic | Status |
|-------|-------|--------|
| 0 | Concept Map | ✅ |
| 1 | Minimal RAG | ✅ |
| 2 | Markdown/TXT Ingestion + Chunking | ✅ |
| 3 | PDF Parsing + Cost Monitor | ✅ Current |
| 4 | CSV/Table Support | ⬜ |
| 5 | Image Support (OCR/Vision) | ⬜ |
| 6 | Vector DB Persistence | ⬜ |
| 7 | Hybrid Search (Vector + BM25) | ⬜ |
| 8 | Reranking | ⬜ |
| 9 | Query Rewrite | ⬜ |
| 10 | Prompt Construction & Generation | ⬜ |
| 11 | Evaluation & Debug Tools | ⬜ |
| 12 | Full Personal Assistant | ⬜ |

## Tech Stack

| Component | Current | Production Alternative |
|-----------|---------|----------------------|
| Vector DB | ChromaDB (in-memory) | Qdrant / Pinecone |
| Embedding | OpenAI text-embedding-3-small | BGE / Cohere |
| LLM | GPT-4o-mini (streaming) | GPT-4o / Claude |
| PDF Parser | PyMuPDF | LlamaParse / Unstructured |
| Chunking | RecursiveCharacterTextSplitter + SemanticChunker | Custom |
