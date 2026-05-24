"""Configuration module for the Local PDF Knowledge Assistant.

Handles .env loading, CLI argument parsing, and provides a unified
configuration object for all other modules.
"""

import argparse
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    """Unified application configuration.

    Values are resolved in the following priority order:
    1. CLI arguments (if provided)
    2. Environment variables (.env file)
    3. Hard-coded defaults
    """

    pdf_path: str = "./document.pdf"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chroma_persist_dir: str = "./chroma_db"
    rebuild: bool = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace containing CLI-supplied values.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Local PDF Knowledge Assistant - "
            "Query PDF documents using local LLMs via Ollama"
        )
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=None,
        help="Path to the PDF document (overrides PDF_PATH in .env)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the vector store even if it already exists",
    )
    return parser.parse_args()


def load_config() -> AppConfig:
    """Load configuration from .env and CLI arguments.

    CLI arguments take precedence over environment variables.
    Environment variables take precedence over defaults.

    Returns:
        Fully resolved AppConfig instance.
    """
    args = parse_args()

    return AppConfig(
        pdf_path=args.pdf_path or os.getenv("PDF_PATH", "./document.pdf"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        rebuild=args.rebuild,
    )
