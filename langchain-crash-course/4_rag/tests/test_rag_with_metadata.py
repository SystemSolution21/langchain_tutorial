# test_rag_with_metadata.py
"""
Test module for the RAG implementation with metadata handling.
Tests functionality and logging behavior.
"""

# Import necessary libraries
import logging
import shutil
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

# Import pytest
import pytest

# Import langchain modules
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document

# Import the modules to test
from rag_with_metadata import (
    create_text_chunks,
    initialize_vector_store,
    load_documents,
    main,
)
from utils.logger import RAGLogger

# Setup test logger
logger: logging.Logger = RAGLogger.get_logger(module_name=__name__)


@pytest.fixture
def rag_module_path_setup():
    """Fixture to add the rag module path to sys.path for test execution."""
    rag_module_path = Path(__file__).parent.parent.resolve().as_posix()
    if rag_module_path not in sys.path:
        sys.path.insert(0, rag_module_path)
        yield
        sys.path.remove(rag_module_path)
    else:
        yield


@pytest.fixture
def test_directories() -> Generator[tuple[Path, Path, Path], None, None]:
    """
    Fixture to set up and tear down test directories for isolated testing.
    """
    # Setup test directories
    current_dir: Path = Path(__file__).parent.resolve()
    test_books_dir: Path = current_dir / "test_books"
    test_db_dir: Path = current_dir / "test_db"
    test_persistent_dir: Path = test_db_dir / "test_chroma_db"

    # Create test directories
    test_books_dir.mkdir(exist_ok=True)
    test_db_dir.mkdir(exist_ok=True)

    # Create a test text file
    test_file: Path = test_books_dir / "test_book.txt"
    test_file.write_text(data="This is a test book content.\nIt has multiple lines.\n")

    yield test_books_dir, test_db_dir, test_persistent_dir

    # Cleanup after tests
    shutil.rmtree(test_books_dir, ignore_errors=True)
    shutil.rmtree(test_db_dir, ignore_errors=True)


def test_load_documents(
    test_directories: tuple[Path, Path, Path], rag_module_path_setup
) -> None:
    """
    Test document loading functionality, including handling of empty files.
    """
    test_books_dir, _, _ = test_directories
    # Create an empty file to test handling of empty documents
    (test_books_dir / "empty_book.txt").touch()
    logger.info("Testing document loading...")

    documents = load_documents(test_books_dir)

    # Both test_book.txt and empty_book.txt should be loaded.
    assert len(documents) == 2
    assert all(isinstance(doc, Document) for doc in documents)

    loaded_sources = {doc.metadata["source"] for doc in documents}
    assert loaded_sources == {"test_book.txt", "empty_book.txt"}

    logger.info("Document loading test completed successfully")


def test_create_text_chunks(rag_module_path_setup) -> None:
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
    # The splitter might create chunks slightly larger than chunk_size if it can't
    # find a separator. A better check is that it creates more than one chunk.
    assert len(chunks) > 1

    logger.info("Text chunking test completed successfully")


@patch("rag_with_metadata.Chroma.from_documents")
@patch("rag_with_metadata.OllamaEmbeddings")
def test_initialize_vector_store_new(
    mock_ollama_embeddings: MagicMock,
    mock_chroma_from_documents: MagicMock,
    test_directories: tuple[Path, Path, Path],
    rag_module_path_setup,
) -> None:
    """
    Test vector store initialization when no store exists, using mocks.
    This test avoids actual embedding creation and database I/O, preventing
    slow and flaky tests and silencing external library warnings.
    """
    test_books_dir, _, test_persistent_dir = test_directories
    logger.info("Testing vector store initialization for a new store (with mocks)...")

    # Ensure the directory does not exist
    if test_persistent_dir.exists():
        shutil.rmtree(test_persistent_dir)

    # Configure mocks
    mock_db = MagicMock(spec=Chroma)
    mock_chroma_from_documents.return_value = mock_db
    mock_embeddings_instance = MagicMock()
    mock_ollama_embeddings.return_value = mock_embeddings_instance

    # Call the function to test
    result = initialize_vector_store(
        books_dir=test_books_dir, persistent_directory=test_persistent_dir
    )

    # --- Assertions ---
    # 1. Check that the function returns the mocked DB object
    assert result is mock_db

    # 2. Verify that OllamaEmbeddings was instantiated correctly
    mock_ollama_embeddings.assert_called_once_with(model="nomic-embed-text:latest")

    # 3. Verify that Chroma.from_documents was called once
    mock_chroma_from_documents.assert_called_once()

    # 4. Inspect the arguments passed to Chroma.from_documents
    call_args, call_kwargs = mock_chroma_from_documents.call_args
    assert "documents" in call_kwargs
    assert len(call_kwargs["documents"]) > 0  # Check that chunks were passed
    assert call_kwargs["embedding"] is mock_embeddings_instance
    assert call_kwargs["persist_directory"] == str(test_persistent_dir)

    logger.info(
        "Vector store initialization test (new store, mocked) completed successfully"
    )


def test_initialize_vector_store_existing(
    test_directories: tuple[Path, Path, Path],
    rag_module_path_setup,
) -> None:
    """
    Test vector store initialization when store already exists.
    """
    test_books_dir, _, test_persistent_dir = test_directories
    logger.info("Testing vector store initialization with existing store...")

    # Create a mock existing directory
    test_persistent_dir.mkdir(parents=True, exist_ok=True)
    # Add a dummy file to simulate an existing store
    (test_persistent_dir / "dummy.file").touch()

    result = initialize_vector_store(
        books_dir=test_books_dir, persistent_directory=test_persistent_dir
    )

    assert result is None
    assert test_persistent_dir.exists()

    logger.info(
        "Vector store initialization test (existing store) completed successfully"
    )


@pytest.mark.parametrize(
    "init_return, dir_exists, expected_log",
    [
        ("new_store", False, "Vector store initialization completed successfully"),
        (None, True, "Using existing vector store - no initialization needed"),
        (None, False, "Vector store initialization failed"),
    ],
)
@patch("rag_with_metadata.initialize_vector_store")
@patch("rag_with_metadata.persistent_directory")
def test_main_flow(
    mock_persistent_dir: MagicMock,
    mock_initialize_vector_store: MagicMock,
    init_return: str,
    dir_exists: bool,
    expected_log: str,
    caplog: pytest.LogCaptureFixture,
    rag_module_path_setup,
) -> None:
    """
    Test the main application flow under different conditions.
    - A new store is created.
    - An existing store is used.
    - Initialization fails.
    """
    logger.info(
        f"Testing main flow with: init_return={init_return}, dir_exists={dir_exists}"
    )

    # Configure mocks based on test case
    mock_persistent_dir.exists.return_value = dir_exists
    if init_return == "new_store":
        mock_initialize_vector_store.return_value = MagicMock(spec=Chroma)
    else:
        mock_initialize_vector_store.return_value = None

    # Capture log output
    with caplog.at_level(logging.INFO):
        main()

    # Verify behavior
    mock_initialize_vector_store.assert_called_once()
    assert expected_log in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
