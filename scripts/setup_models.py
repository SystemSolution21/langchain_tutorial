"""
Setup script to pre-download required AI models for the LangChain tutorial.

This script downloads and caches the following models:
1. HuggingFace Embeddings: BAAI/bge-large-en-v1.5 (~1.34 GB)

Run this script once before using the advanced RAG features to avoid
runtime downloads and ensure offline capability.

Usage:
    python scripts/setup_models.py
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_huggingface_embeddings(
    model_name: str = "BAAI/bge-large-en-v1.5", cache_dir: Optional[Path] = None
) -> bool:
    """
    Download and cache HuggingFace embedding model.

    Args:
        model_name: Name of the model to download
        cache_dir: Optional custom cache directory

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\n{'=' * 70}")
        print(f"Downloading HuggingFace Embedding Model: {model_name}")
        print(f"{'=' * 70}")
        print("Size: ~1.34 GB")
        print("This may take 5-15 minutes depending on your internet speed...")
        print(f"{'=' * 70}\n")

        from sentence_transformers import SentenceTransformer

        # Download the model
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        else:
            model = SentenceTransformer(model_name)

        # Verify the model works
        test_embedding = model.encode("Test sentence")

        print("\n✅ SUCCESS: Model downloaded and cached successfully!")
        print(f"   Model: {model_name}")
        print(f"   Dimensions: {len(test_embedding)}")

        if cache_dir:
            print(f"   Cache location: {cache_dir}")
        else:
            default_cache = Path.home() / ".cache" / "huggingface" / "hub"
            print(f"   Cache location: {default_cache}")

        return True

    except ImportError as e:
        print(f"\n❌ ERROR: Required package not installed: {e}")
        print("   Please install: pip install sentence-transformers")
        return False

    except Exception as e:
        print(f"\n❌ ERROR: Failed to download model: {e}")
        return False


def check_disk_space(required_gb: float = 2.0) -> bool:
    """
    Check if sufficient disk space is available.

    Args:
        required_gb: Required space in GB

    Returns:
        bool: True if sufficient space available
    """
    try:
        import shutil

        cache_dir = Path.home() / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        total, used, free = shutil.disk_usage(cache_dir)
        free_gb = free / (1024**3)

        print("\nDisk Space Check:")
        print(f"  Available: {free_gb:.2f} GB")
        print(f"  Required: {required_gb:.2f} GB")

        if free_gb < required_gb:
            print("  ⚠️  WARNING: Low disk space!")
            return False
        else:
            print("  ✅ Sufficient space available")
            return True

    except Exception as e:
        print(f"  ⚠️  Could not check disk space: {e}")
        return True  # Continue anyway


def main() -> None:
    """Main setup function."""
    print("\n" + "=" * 70)
    print("LangChain Tutorial - Model Setup Script")
    print("=" * 70)

    # Check disk space
    if not check_disk_space(required_gb=2.0):
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != "y":
            print("Setup cancelled.")
            sys.exit(1)

    # Download HuggingFace embeddings
    success = download_huggingface_embeddings()

    if success:
        print("\n" + "=" * 70)
        print("✅ All models downloaded successfully!")
        print("=" * 70)
        print("\nYou can now run the advanced RAG scripts without internet connection.")
        print("The models will be loaded from cache.\n")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ Model setup failed!")
        print("=" * 70)
        print("\nPlease check the error messages above and try again.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
