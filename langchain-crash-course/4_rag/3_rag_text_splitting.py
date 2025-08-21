# 3_rag_text_splitting.py

import sys
from logging import Logger
from pathlib import Path
from typing import List

# Import langchain modules
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents.base import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama.embeddings import OllamaEmbeddings

# Import logger
from utils.logger import RAGLogger

# Module path
module_path: Path = Path(__file__).resolve()

# Set logger
logger: Logger = RAGLogger.get_logger(module_name=module_path.name)

# Define directories and paths
current_dir: Path = Path(__file__).parent.resolve()
books_dir: Path = current_dir / "books"
file_path: Path = books_dir / "romeo_and_juliet.txt"
db_dir: Path = current_dir / "db"

logger.info(msg=f"Books Directory: {books_dir}")
logger.info(msg=f"Database Directory: {db_dir}")

# Create db directory if it doesn't exist
try:
    db_dir.mkdir(exist_ok=True)
    logger.info(msg="Database directory created/verified successfully")
except Exception as e:
    logger.error(msg=f"Error creating database directory: {str(object=e)}")
    sys.exit(1)


# Define embeddings model
embeddings: OllamaEmbeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Start logging
logger.info(msg="=" * 50)
logger.info(msg="Starting RAG Text Splitting Application")
logger.info(msg="=" * 50)


# Create vector store
def create_vector_store(
    documents: List[Document],
    store_name: str,
) -> None:
    persistent_directory: Path = db_dir / store_name
    if Path.exists(self=persistent_directory):
        logger.info(
            msg=f"Vector store '{store_name}' already exists. No need to initialize."
        )
        return

    logger.info(msg=f"Creating vector store '{store_name}'...")
    try:
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(object=persistent_directory),
        )
        logger.info(msg=f"Vector store '{store_name}' created successfully.")
    except Exception as e:
        logger.error(
            msg=f"Unexpected error initializing vector store!: {str(object=e)}"
        )


# Character-base splitting
def character_splitter(documents: list[Document]) -> None:
    """
    Splits text into chunks based on a specified number of characters.

    Args:
        documents: List of documents to split
        store_name: Name of the vector store
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    Returns:
        None
    Usages:
        Useful for consistent chunk sizes regardless of content structure.
    """
    logger.info(msg="---- Using character-based splitting ----")
    char_txt_splitter: CharacterTextSplitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    char_docs: List[Document] = char_txt_splitter.split_documents(documents=documents)
    create_vector_store(
        documents=char_docs,
        store_name="chroma_db_character",
    )


# Sentence-based splitting
def sentence_splitter(documents: list[Document]) -> None:
    """
    Splits text into chunks based on sentence boundaries.

    Args:
        documents: List of documents to split
        store_name: Name of the vector store
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    Returns:
        None
    Usages:
        Ideal for maintaining semantic coherence within chunks.
    """
    logger.info(msg="---- Using sentence-based splitting ----")
    sent_splitter: SentenceTransformersTokenTextSplitter = (
        SentenceTransformersTokenTextSplitter(chunk_size=1000)
    )
    sent_docs: List[Document] = sent_splitter.split_documents(documents=documents)
    create_vector_store(
        documents=sent_docs,
        store_name="chroma_db_sentence",
    )


# Token-based splitting
def token_splitter(documents: list[Document]) -> None:
    """Splits text into chunks based on tokens (words or sub-words), using tokenizers like GPT-2.

    Args:
        documents: List of documents to split
        store_name: Name of the vector store
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    Returns:
        None
    Usages:
        Useful for transformer models with strict token limits.
    """
    logger.info(msg="---- Using token-based splitting ----")
    token_splitter: TokenTextSplitter = TokenTextSplitter(
        chunk_size=512, chunk_overlap=0
    )
    token_docs: List[Document] = token_splitter.split_documents(documents=documents)
    create_vector_store(
        documents=token_docs,
        store_name="chroma_db_token",
    )


# Recursive character-based splitting
def recursive_character_splitter(documents: list[Document]) -> None:
    """Split text at natural boundaries (sentences, paragraphs) within character limit.

    Args:
        documents: List of documents to split
        store_name: Name of the vector store
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    Returns:
        None
    Usages:
        Balances between maintaining coherence and adhering to character limits.
    """
    logger.info(msg="---- Using recursive character-based splitting ----")
    recursive_char_splitter: RecursiveCharacterTextSplitter = (
        RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    )
    recursive_char_docs: List[Document] = recursive_char_splitter.split_documents(
        documents=documents
    )
    create_vector_store(
        documents=recursive_char_docs,
        store_name="chroma_db_recursive_character",
    )


# Custom text splitting
class CustomTextSplitter(TextSplitter):
    """Custom splitting logic based on specific requirements."""

    def split_text(self, text: str) -> List[str]:
        return text.split(sep="\n\n")  # split by paragraphs


# Custom-based splitter
def custom_splitter(documents: list[Document]) -> None:
    """Custom text splitter based on specific requirements.

    Args:
        documents: List of documents to split
        store_name: Name of the vector store
    Returns:
        None
    Usages:
        Useful when default splitters don't meet specific requirements.
    """
    logger.info(msg="---- Using custom-based splitting ----")
    custom_splitter: CustomTextSplitter = CustomTextSplitter()
    custom_docs: List[Document] = custom_splitter.split_documents(documents=documents)
    create_vector_store(
        documents=custom_docs,
        store_name="chroma_db_custom",
    )


# Query vector store
def query_vector_store(store_name: str, query: str) -> None:
    """Query the vector store for relevant documents.

    Args:
        store_name: Name of the vector store
        query: Query string
    Returns:
        None
    """
    persistent_directory: Path = db_dir / store_name
    if not Path.exists(self=persistent_directory):
        logger.info(
            msg=f"Vector store '{store_name}' does not exist. No need to query."
        )
        return

    try:
        logger.info(msg=f"Querying vector store '{store_name}'...")
        db: Chroma = Chroma(
            persist_directory=str(object=persistent_directory),
            embedding_function=embeddings,
        )

        retriever: VectorStoreRetriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.1},
        )

        relevant_docs: list[Document] = retriever.invoke(input=query)
        if relevant_docs:
            logger.info(
                msg=f"For vector store '{store_name}', found {len(relevant_docs)} relevant documents"
            )
        else:
            logger.info(msg="No relevant documents found")
            return

        for i, doc in enumerate(relevant_docs, start=1):
            print(f"\n--- Relevant Document {i} ---")
            print(f"Document:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

    except Exception as e:
        logger.error(msg=f"Unexpected error querying vector store: {str(object=e)}")


def main() -> None:
    """Main function to set up and initialize the vector store."""
    try:
        # Check file exists
        if not Path.exists(self=file_path):
            logger.error(msg=f"The file {file_path} does not exist")
            sys.exit(1)

        # logging file path
        logger.info(msg=f"File Path: {file_path}")

        # Load text content from file
        text_loader: TextLoader = TextLoader(file_path=file_path, encoding="utf-8")
        documents: List[Document] = text_loader.load()

    except Exception as e:
        logger.error(msg=f"Failed to load documents: {str(object=e)}")
        sys.exit(1)

    # Character-based splitting
    character_splitter(documents=documents)

    # Sentence-based splitting
    sentence_splitter(documents=documents)

    # Token-based splitting
    token_splitter(documents=documents)

    # Recursive character-based splitting
    recursive_character_splitter(documents=documents)

    # Custom-based splitting
    custom_splitter(documents=documents)


# Main entry point
if __name__ == "__main__":
    # Setup and initialize vector stores
    main()

    # User query
    query: str = "How did Juliet die?"

    # Query vector store
    # Character-based
    query_vector_store(
        store_name="chroma_db_character",
        query=query,
    )
    # Sentence-based
    query_vector_store(
        store_name="chroma_db_sentence",
        query=query,
    )
    # Token-based
    query_vector_store(
        store_name="chroma_db_token",
        query=query,
    )
    # Recursive character-based
    query_vector_store(
        store_name="chroma_db_recursive_character",
        query=query,
    )
    # Custom-based
    query_vector_store(
        store_name="chroma_db_custom",
        query=query,
    )
