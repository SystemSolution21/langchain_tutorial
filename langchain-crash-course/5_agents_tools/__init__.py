import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ollama_llm: str | None = os.getenv(key="OLLAMA_LLM", default="gemma3:4b")
ollama_embeddings_model: str | None = os.getenv(
    key="OLLAMA_EMBEDDINGS_MODEL", default="nomic-embed-text:latest"
)

# Define books and database directories paths
current_dir: Path = Path(__file__).parent.resolve()
books_dir: Path = current_dir / "books"
db_dir: Path = current_dir / "db"
store_name: str = "chroma_db_with_metadata"
persistent_directory: Path = db_dir / store_name
