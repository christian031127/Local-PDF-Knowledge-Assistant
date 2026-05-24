"""PDF loading and chunking module.

Provides functions to load a PDF document with PyPDFLoader and
split it into overlapping text chunks using RecursiveCharacterTextSplitter.
"""

import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def load_and_chunk_pdf(pdf_path: str, chunk_size: int, chunk_overlap: int) -> list:
    """Load a PDF file with PyPDFLoader and split it into overlapping chunks.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        A list of LangChain Document objects (chunks).
    """
    print(f"[INFO] Loading PDF: {pdf_path}")

    if not Path(pdf_path).exists():
        print(f"[ERROR] PDF file not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.lower().endswith(".pdf"):
        print(f"[ERROR] The file does not appear to be a PDF: {pdf_path}")
        sys.exit(1)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(
        f"[INFO] Created {len(chunks)} text chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )

    return chunks
