#!/usr/bin/env python3
"""
Local PDF Knowledge Assistant

Terminal-based Knowledge Base Assistant that answers questions from a PDF
document using a local RAG pipeline (Ollama + LangChain + ChromaDB).

Pipeline: PyPDFLoader → RecursiveCharacterTextSplitter → ChromaDB → Llama 3

Usage:
    python knowledge_assistant.py [pdf_path] [--rebuild]
"""

from config import load_config
from pdf_loader import load_and_chunk_pdf
from rag_chain import build_rag_chain
from vector_store import get_or_create_vectorstore


def _print_banner(config) -> None:
    """Print the application banner with current configuration."""
    print("=" * 60)
    print("  Local PDF Knowledge Assistant")
    print("=" * 60)
    print(f"  PDF:              {config.pdf_path}")
    print(f"  LLM Model:        {config.ollama_model}")
    print(f"  Embedding Model:  {config.embedding_model}")
    print(f"  Ollama URL:       {config.ollama_base_url}")
    print(f"  Chunk Size:       {config.chunk_size}")
    print(f"  Chunk Overlap:    {config.chunk_overlap}")
    print(f"  Vector Store:     {config.chroma_persist_dir}")
    print("=" * 60)


def _print_ready() -> None:
    """Print the interactive session header."""
    print("\n" + "=" * 60)
    print("  Ready! Type your questions below.")
    print("  Type 'exit', 'quit', or press Ctrl+C to stop.")
    print("=" * 60 + "\n")


def interactive_loop(rag_chain) -> None:
    """Run the interactive Q&A loop in the terminal.

    Reads user questions from stdin, invokes the RAG chain,
    and prints the generated answer. Exits on 'exit', 'quit',
    empty input, or Ctrl+C.

    Args:
        rag_chain: The compiled RAG chain (Runnable).
    """
    _print_ready()

    while True:
        try:
            question = input("\n[QUESTION] ").strip()

            if question.lower() in ("exit", "quit", ""):
                print("[INFO] Goodbye!")
                break

            print("\n[THINKING] Searching document and generating answer...\n")
            answer = rag_chain.invoke(question)
            print(f"[ANSWER] {answer}")

        except KeyboardInterrupt:
            print("\n\n[INFO] Goodbye!")
            break
        except Exception as exc:
            print(f"\n[ERROR] An unexpected error occurred: {exc}")
            print("[INFO] Please try again or check that Ollama is running.")


def main() -> None:
    """Main entry point — orchestrates the full RAG pipeline.

    1. Load configuration (.env + CLI args)
    2. Load PDF and split into chunks
    3. Create or load the ChromaDB vector store
    4. Build the RAG chain (retriever + LLM)
    5. Start the interactive terminal Q&A loop
    """
    config = load_config()
    _print_banner(config)

    # Step 1 & 2: PDF → Chunks
    chunks = load_and_chunk_pdf(
        config.pdf_path, config.chunk_size, config.chunk_overlap
    )

    # Step 3 & 4: Embedding → Vector Store
    vectorstore = get_or_create_vectorstore(
        chunks=chunks,
        embedding_model=config.embedding_model,
        base_url=config.ollama_base_url,
        persist_dir=config.chroma_persist_dir,
        rebuild=config.rebuild,
    )

    # Step 5: Build RAG Chain
    print(f"[INFO] Setting up RAG chain with LLM: {config.ollama_model}")
    rag_chain = build_rag_chain(
        vectorstore=vectorstore,
        llm_model=config.ollama_model,
        base_url=config.ollama_base_url,
    )

    # Step 6: Interactive Q&A
    interactive_loop(rag_chain)


if __name__ == "__main__":
    main()