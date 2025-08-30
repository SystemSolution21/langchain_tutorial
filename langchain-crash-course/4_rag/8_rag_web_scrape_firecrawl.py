# 8_rag_web_scrape_firecrawl.py

"""
A RAG (Retrieval-Augmented Generation) application that scrapes a website using FireCrawl,
chunks the content, creates a vector store, and allows for similarity searches.
"""

# Import standard libraries
import os
import sys
from logging import Logger
from pathlib import Path

# Import environment variables
from dotenv import load_dotenv

# Import langchain modules
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import FireCrawlLoader
from langchain_core.documents.base import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama.embeddings import OllamaEmbeddings

# Import custom logger
from utils.logger import RAGLogger

# Load environment variables
load_dotenv()

# Check FIRECRAWL_API_KEY
api_key: str | None = os.getenv(key="FIRECRAWL_API_KEY")
if not api_key:
    raise ValueError("FIRECRAWL_API_KEY is not set. Check your .env file.")
    sys.exit(1)

# Check URL to scrape
url: str | None = os.getenv(key="URL")
if not url:
    raise ValueError("URL to scrape is not set. Check your .env file.")
    sys.exit(1)

# Module path
module_path: Path = Path(__file__).resolve()

# Set logger
logger: Logger = RAGLogger.get_logger(module_name=module_path.name)

# Log application startup
logger.info(msg="=" * 50)
logger.info(msg="Starting RAG Web Scraping FireCrawl Application")
logger.info(msg="=" * 50)

# Define directories and paths
current_dir: Path = Path(__file__).parent.resolve()
db_dir: Path = current_dir / "db"
store_name: str = "chroma_db_web_scrape_firecrawl"
persistent_directory: Path = db_dir / store_name

# Define embeddings models
ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
)


# Create vector store
def create_vector_store() -> Chroma | None:
    """Create a vector store from the crawled website.

    Args:
        None
    Returns:
        Chroma | None
    """
    # Check vector store exists
    if persistent_directory.exists():
        logger.info(
            msg=f"Vector store '{store_name}' already exists. No need to initialize."
        )
        db: Chroma = Chroma(
            persist_directory=str(object=persistent_directory),
            embedding_function=ollama_embeddings,
        )
        return db

    # Create vector store
    try:
        # Crawl website using FireCrawl
        logger.info(msg=f"Crawling website {url}...")
        loader = FireCrawlLoader(
            url=str(url),
            api_key=api_key,
            mode="scrape",
        )

        docs: list[Document] = loader.load()
        logger.info(msg=f"Successfully crawled website {url}")

        # Convert metadata to string
        for doc in docs:
            for key, value in doc.metadata.items():
                if isinstance(value, list):
                    doc.metadata[key] = ", ".join(map(str, value))
                elif isinstance(value, dict):
                    doc.metadata[key] = ", ".join(map(str, value.values()))
        logger.info(msg="Successfully converted metadata to string.")

        # Slipt the crawled content into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_docs: list[Document] = text_splitter.split_documents(documents=docs)
        logger.info(msg=f"Successfully split documents into chunks: {len(split_docs)}")
        print(f"Sample chunk:\n{split_docs[0].page_content}\n")

        # Initialize vector store
        logger.info(msg=f"Creating vector store '{store_name}'...")
        db: Chroma = Chroma.from_documents(
            documents=split_docs,
            embedding=ollama_embeddings,
            persist_directory=str(object=persistent_directory),
        )
        logger.info(msg=f"Successfully created vector store '{store_name}'.")
        return db

    except Exception as e:
        logger.error(
            msg=f"Unexpected error initializing vector store!: {str(object=e)}"
        )
        return None


# Query vector store
def query_vector_store(query: str) -> None:
    """Query the vector store for relevant documents.

    Args:
        query: Query string
    Returns:
        None
    """
    logger.info(msg=f"Querying vector store '{store_name}'...")
    try:
        # Create vector store
        db: Chroma | None = create_vector_store()
        if db:
            retriever: VectorStoreRetriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3},
            )

            # Retrieve relevant documents based on the query
            relevant_docs: list[Document] = retriever.invoke(input=query)
            if relevant_docs:
                logger.info(msg="Relevant documents retrieved successfully.")

                # Display the retrieved documents with MetaData
                for i, doc in enumerate(iterable=relevant_docs, start=1):
                    print(f"\n--- Relevant Document {i} ---")
                    print(f"Document:\n{doc.page_content}\n")
                    if doc.metadata:
                        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

    except Exception as e:
        logger.error(msg=f"Unexpected error creating vector store: {str(object=e)}")
        return None


def main() -> None:
    """
    Main function to run the RAG application.

    This function continuously prompts the user for a query, retrieves relevant
    documents from the vector store, and displays them. The loop can be
    exited by typing 'exit', or by sending a KeyboardInterrupt (Ctrl+C) or
    EOF (Ctrl+D).
    """
    print(
        "\nStart RAG Web Scraping FireCrawl Application... Type 'exit' to end the conversation."
    )

    while True:
        try:
            # User query
            query: str = input("You: ").strip()

            if not query:
                continue

            if query.lower() == "exit":
                logger.info(msg="User exited conversation")
                print("Exiting...")
                break

            # Retrieve relevant documents based on the query
            query_vector_store(query=query)

        except (KeyboardInterrupt, EOFError):
            logger.info(msg="Keyboard interrupt or EOF error")
            print("Exiting...")
            break

        except Exception as e:
            logger.error(msg=f"Unexpected error: {e}")
            print("Exiting...")
            break


# Main entry point
if __name__ == "__main__":
    main()
