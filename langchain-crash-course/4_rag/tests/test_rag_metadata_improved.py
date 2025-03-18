"""
Test module for the improved RAG implementation with metadata handling.
Tests functionality and logging behavior.
"""

import pytest
from pathlib import Path
from typing import Generator, List
import shutil
import logging
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document

# Import the modules to test
from utils.logger import RAGLogger
import sys
import os

# Add the parent directory to sys.path to import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _2b_rag_basic_metadata_improved import (
    load_documents,
    create_text_chunks,
    initialize_vector_store,
    main,
)

# Setup test logger
logger = RAGLogger.get_logger(__name__)


@pytest.fixture
def test_directories() -> Generator[tuple[Path, Path, Path], None, None]:
    """
    Fixture to set up and tear down test directories.
    """
    # Setup test directories
    current_dir = Path(__file__).parent.resolve()
    test_books_dir = current_dir / "test_books"
    test_db_dir = current_dir / "test_db"
    test_persistent_dir = test_db_dir / "test_chroma_db"

    # Create test directories
    test_books_dir.mkdir(exist_ok=True)
    test_db_dir.mkdir(exist_ok=True)

    # Create a test text file
    test_file = test_books_dir / "test_book.txt"
    test_file.write_text("This is a test book content.\nIt has multiple lines.\n")

    yield test_books_dir, test_db_dir, test_persistent_dir

    # Cleanup after tests
    shutil.rmtree(test_books_dir)
    shutil.rmtree(test_db_dir)


def test_load_documents(test_directories: tuple[Path, Path, Path]) -> None:
    """
    Test document loading functionality.
    """
    test_books_dir, _, _ = test_directories
    logger.info("Testing document loading...")

    documents = load_documents(test_books_dir)

    assert len(documents) > 0
    assert isinstance(documents[0], Document)
    assert "test_book.txt" in documents[0].metadata["source"]

    logger.info("Document loading test completed successfully")


def test_create_text_chunks() -> None:
    """
    Test text chunking functionality.
    """
    logger.info("Testing text chunking...")

    # Create a test document
    test_doc = Document(
        page_content="This is a test content.\n" * 10, metadata={"source": "test.txt"}
    )

    chunks = create_text_chunks([test_doc], chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all(len(chunk.page_content) <= 50 for chunk in chunks)

    logger.info("Text chunking test completed successfully")


def test_initialize_vector_store_existing(
    test_directories: tuple[Path, Path, Path],
) -> None:
    """
    Test vector store initialization when store already exists.
    """
    test_books_dir, _, test_persistent_dir = test_directories
    logger.info("Testing vector store initialization with existing store...")

    # Create a mock existing directory
    test_persistent_dir.mkdir(parents=True, exist_ok=True)

    result = initialize_vector_store(
        books_dir=test_books_dir, persistent_directory=test_persistent_dir
    )

    assert result is None
    assert test_persistent_dir.exists()

    logger.info(
        "Vector store initialization test (existing store) completed successfully"
    )


def test_main_flow() -> None:
    """
    Test the main application flow.
    """
    logger.info("Testing main application flow...")

    # Execute main function
    main()

    # Verify log file exists and contains expected entries
    current_dir = Path(__file__).parent.parent.resolve()
    log_file = current_dir / "logs" / "rag_application.log"

    assert log_file.exists()
    log_content = log_file.read_text()

    # Check for expected log messages
    assert "Starting RAG Metadata Application" in log_content
    assert "Vector store already exists" in log_content
    assert "Using existing vector store" in log_content

    logger.info("Main application flow test completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
