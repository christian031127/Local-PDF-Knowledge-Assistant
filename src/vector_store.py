"""Vector store module for the Local PDF Knowledge Assistant.

Handles creation, loading, and retrieval of ChromaDB vector stores
backed by Ollama embedding models.
"""

from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


def create_vector_store(
    chunks: list,
    embedding_model: str,
    base_url: str,
    persist_dir: str,
) -> Chroma:
    """Create a new ChromaDB vector store from document chunks.

    Args:
        chunks: List of LangChain Document objects.
        embedding_model: Name of the Ollama embedding model (e.g. nomic-embed-text).
        base_url: Ollama server URL.
        persist_dir: Directory where ChromaDB data will be persisted.

    Returns:
        A Chroma vector store instance.
    """
    print(f"[INFO] Creating embeddings with model: {embedding_model}")

    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=base_url,
    )

    print(f"[INFO] Storing vectors in ChromaDB at: {persist_dir}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    count = vectorstore._collection.count()
    print(f"[INFO] Vector store created with {count} embedded documents")
    return vectorstore


def load_vector_store(
    embedding_model: str,
    base_url: str,
    persist_dir: str,
) -> Chroma:
    """Load an existing ChromaDB vector store from disk.

    Args:
        embedding_model: Name of the Ollama embedding model.
        base_url: Ollama server URL.
        persist_dir: Directory containing persisted ChromaDB data.

    Returns:
        A Chroma vector store instance.
    """
    print(f"[INFO] Loading existing vector store from: {persist_dir}")

    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=base_url,
    )

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    count = vectorstore._collection.count()
    print(f"[INFO] Vector store loaded with {count} embedded documents")
    return vectorstore


def get_or_create_vectorstore(
    chunks: list,
    embedding_model: str,
    base_url: str,
    persist_dir: str,
    rebuild: bool,
) -> Chroma:
    """Get an existing vector store or create a new one if none exists.

    If persist_dir contains ChromaDB data and rebuild is not requested,
    the existing store is reused to avoid re-indexing the PDF.

    Args:
        chunks: List of LangChain Document objects (used only for new stores).
        embedding_model: Name of the Ollama embedding model.
        base_url: Ollama server URL.
        persist_dir: Directory for ChromaDB data.
        rebuild: If True, force creation of a new vector store.

    Returns:
        A Chroma vector store instance.
    """
    store_exists = (
        Path(persist_dir).exists()
        and any(Path(persist_dir).iterdir())
    )

    if store_exists and not rebuild:
        return load_vector_store(embedding_model, base_url, persist_dir)
    else:
        if rebuild:
            print("[INFO] Rebuild requested, creating fresh vector store...")
        return create_vector_store(
            chunks, embedding_model, base_url, persist_dir
        )