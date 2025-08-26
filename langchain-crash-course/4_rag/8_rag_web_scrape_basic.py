"""
A basic RAG (Retrieval-Augmented Generation) application that scrapes a website,
chunks the content, creates a vector store, and allows for similarity searches.
"""
# 8_rag_web_scrape_basic.py

# Import standard libraries
import sys
from logging import Logger
from pathlib import Path
from typing import List

# Import environment variables
from dotenv import load_dotenv

# Import langchain modules
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents.base import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama.embeddings import OllamaEmbeddings

# Import custom logger
from utils.logger import RAGLogger

# Load environment variables
load_dotenv()

# Module path
module_path: Path = Path(__file__).resolve()

# Set logger
logger: Logger = RAGLogger.get_logger(module_name=module_path.name)

# Log application startup
logger.info(msg="=" * 50)
logger.info(msg="Starting RAG Web Scraping Basic Application")
logger.info(msg="=" * 50)

# Define directories and paths
current_dir: Path = Path(__file__).parent.resolve()
db_dir: Path = current_dir / "db"
store_name: str = "chroma_db_web_scrape_basic"
persistent_directory: Path = db_dir / store_name

# Define embeddings models
ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
)


# Define web scraper and Load documents from the web
try:
    # Setting url to scrape
    url: list[str] = ["https://www.apple.com/"]

    # Define web scraper
    web_scraper = WebBaseLoader(web_path=url)

    # Load documents from the web
    logger.info(msg=f"Loading documents from {url}...")
    documents: list[Document] = web_scraper.load()
    logger.info(msg=f"Loaded {len(documents)} documents from {url} successfully.")

    # Chunking Web content using Character text splitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs: List[Document] = text_splitter.split_documents(documents=documents)
    logger.info(msg=f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

except Exception as e:
    logger.error(msg=f"Error loading documents from {url}: {str(object=e)}")
    logger.critical(msg="Failed to load documents, exiting application.")
    sys.exit(1)


# Create vector store
if not persistent_directory.exists():
    logger.info(msg=f"Creating vector store '{store_name}'...")
    db: Chroma = Chroma.from_documents(
        documents=docs,
        embedding=ollama_embeddings,
        persist_directory=str(persistent_directory),
    )
    logger.info(msg=f"Vector store '{store_name}' created successfully.")
else:
    logger.error(
        msg=f"Vector store '{store_name}' already exists. No need to initialize."
    )
    db: Chroma = Chroma(
        persist_directory=str(persistent_directory),
        embedding_function=ollama_embeddings,
    )
    logger.info(msg=f"Vector store '{store_name}' loaded successfully.")

# Define retriever
retriever: VectorStoreRetriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)


def main() -> None:
    """
    Main function to run the RAG application.

    This function continuously prompts the user for a query, retrieves relevant
    documents from the vector store, and displays them. The loop can be
    exited by typing 'exit', or by sending a KeyboardInterrupt (Ctrl+C) or
    EOF (Ctrl+D).
    """
    print(
        "\nStart retrieving relevant documents... Type 'exit' to end the conversation."
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
            logger.info(msg="Retrieving relevant documents...")
            relevant_docs: list[Document] = retriever.invoke(input=query)

            # Display the retrieved documents with MetaData
            print("\n--- Relevant Document ---")
            for i, doc in enumerate(relevant_docs, start=1):
                print(f"Document {i}:\n{doc.page_content}\n")
                if doc.metadata:
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
            logger.info(msg="Relevant documents retrieved successfully")

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
