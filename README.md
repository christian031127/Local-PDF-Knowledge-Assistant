# Local PDF Knowledge Assistant

A terminal-based Knowledge Base Assistant that allows you to query information from PDF
documents using local Large Language Models (LLMs) via Ollama. The application implements a
Retrieval-Augmented Generation (RAG) pipeline that runs entirely on your local hardware —
no cloud APIs required.

## Architecture (RAG Pipeline)

```
┌──────────────┐    ┌──────────────────────────┐    ┌─────────────┐
│  PyPDFLoader │───▶│ RecursiveCharacterText   │───▶│  nomic-embed│
│  (Extract)   │    │ Splitter     (Chunking)   │    │  -text      │
└──────────────┘    └──────────────────────────┘    └──────┬──────┘
                                                          │
                                                          ▼
┌──────────────┐    ┌──────────────────────────┐    ┌─────────────┐
│   Llama 3    │◀───│  RAG Chain               │◀───│  ChromaDB   │
│  (Generate)  │    │  (Prompt + Retrieval)    │    │  (Search)   │
└──────────────┘    └──────────────────────────┘    └─────────────┘
```

## Prerequisites

- **Python 3.10+** (check with `python --version`)
- **Ollama** — local LLM runtime

## Setup Instructions

### 1. Install Ollama

Download and install Ollama from: https://ollama.com/download

After installation, verify it is running:

```bash
ollama --version
```

### 2. Download Required Models

Pull the two models needed by the application:

```bash
# LLM for text generation
ollama pull llama3

# Embedding model for semantic search
ollama pull nomic-embed-text
```

Verify the models are available:

```bash
ollama list
```

You should see both `llama3` and `nomic-embed-text` in the list.

### 3. Clone / Set Up the Project

```bash
cd mi_technologies_hw
```

### 4. Create a Python Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
# source venv/bin/activate
```

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure the Environment

Copy the example environment file and edit it to match your setup:

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # macOS/Linux
```

Open `.env` in a text editor and set the path to your PDF document:

```env
PDF_PATH=./document.pdf          # Your PDF file path
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
EMBEDDING_MODEL=nomic-embed-text
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CHROMA_PERSIST_DIR=./chroma_db
```

## Usage

### Quick Start

Place your PDF file in the project directory and run:

```bash
python knowledge_assistant.py
```

### Specify PDF Path via Command Line

```bash
python knowledge_assistant.py path/to/your/document.pdf
```

This overrides the `PDF_PATH` set in `.env`.

### Rebuild the Vector Store

If you change the PDF, the chunk size, or the embedding model, rebuild the vector store:

```bash
python knowledge_assistant.py --rebuild
```

### Interactive Q&A Session

Once started, the assistant displays its configuration and waits for your questions:

```
============================================================
  Local PDF Knowledge Assistant
============================================================
  PDF:              ./document.pdf
  LLM Model:        llama3
  Embedding Model:  nomic-embed-text
  Ollama URL:       http://localhost:11434
  Chunk Size:       1000
  Chunk Overlap:    200
  Vector Store:     ./chroma_db
============================================================
[INFO] Loading PDF: ./document.pdf
[INFO] Loaded 5 pages
[INFO] Created 14 text chunks (size=1000, overlap=200)
[INFO] Creating embeddings with model: nomic-embed-text
[INFO] Storing vectors in ChromaDB at: ./chroma_db
[INFO] Vector store created with 14 embedded documents
[INFO] Setting up RAG chain with LLM: llama3

============================================================
  Ready! Type your questions below.
  Type 'exit', 'quit', or press Ctrl+C to stop.
============================================================

[QUESTION] What is the main topic of this document?

[THINKING] Searching document and generating answer...

[ANSWER] The document discusses...
```

- Type `exit`, `quit`, or press `Ctrl+C` to stop.
- Pressing Enter on an empty line also exits.

## Configuration Reference

| Variable             | Default                  | Description                        |
|----------------------|--------------------------|------------------------------------|
| `PDF_PATH`           | `./document.pdf`         | Path to the PDF file               |
| `OLLAMA_BASE_URL`    | `http://localhost:11434` | Ollama server URL                  |
| `OLLAMA_MODEL`       | `llama3`                 | Ollama model for text generation   |
| `EMBEDDING_MODEL`    | `nomic-embed-text`       | Ollama model for embeddings        |
| `CHUNK_SIZE`         | `1000`                   | Max characters per text chunk      |
| `CHUNK_OVERLAP`      | `200`                    | Overlapping characters per chunk   |
| `CHROMA_PERSIST_DIR` | `./chroma_db`            | ChromaDB persistence directory     |

## Troubleshooting

| Symptom                                   | Solution                                         |
|-------------------------------------------|--------------------------------------------------|
| `ConnectionError` or `httpx.ConnectError` | Make sure Ollama is running (`ollama serve`)     |
| Model not found                           | Run `ollama pull <model>` to download models     |
| PDF file not found                        | Check `PDF_PATH` in `.env` or pass via CLI       |
| Empty or poor quality answers             | Try different `CHUNK_SIZE` values (e.g. 500)      |
| Slow responses                            | First run downloads models; subsequent runs faster|
| Import errors after `pip install`         | Ensure virtual environment is activated          |

## Project Structure

```
mi_technologies_hw/
├── knowledge_assistant.py   # Entry point — orchestrates the RAG pipeline
├── config.py                # Configuration loader (.env + CLI args)
├── pdf_loader.py            # PDF loading & chunking (PyPDFLoader + splitter)
├── vector_store.py          # ChromaDB operations (create / load / search)
├── rag_chain.py             # RAG chain assembly (retriever → prompt → LLM)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment configuration template
├── .env                     # Your local configuration (create from .env.example)
├── README.md                # This file
├── chroma_db/               # ChromaDB vector store (auto-generated)
└── venv/                    # Python virtual environment (created by you)
```

### Module Overview

| Module                 | Responsibility                                                    |
|------------------------|-------------------------------------------------------------------|
| `knowledge_assistant.py` | Entry point: loads config, wires modules, runs interactive Q&A  |
| `config.py`            | Reads `.env` file and CLI arguments, provides `AppConfig`         |
| `pdf_loader.py`        | Loads PDF via `PyPDFLoader`, splits with `RecursiveCharacterTextSplitter` |
| `vector_store.py`      | Creates/loads `ChromaDB` vector store with Ollama embeddings      |
| `rag_chain.py`         | Builds the LangChain RAG chain (retriever + prompt + LLM)         |