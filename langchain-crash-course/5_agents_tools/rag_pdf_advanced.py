"""
This script demonstrates an advanced RAG (Retrieval-Augmented Generation)
system with PDF documents that handles tables, images, OCR, and complex layouts.
It includes the following steps:
1. Loading PDF documents with advanced parsing (tables, images, OCR)
2. Table-aware text splitting to preserve structure
3. Creating embeddings optimized for structured data
4. Initializing and persisting a Chroma vector store with enhanced metadata
The script is designed to be idempotent and handles complex PDF content.
"""

# Import standard libraries
import io
import json
import re
from logging import Logger
from pathlib import Path
from typing import List, Literal, Optional, Tuple

# Import necessary libraries
import cv2
import fitz
import numpy as np
import pytesseract

# Import langchain modules
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from PIL import Image

# Import custom modules
from util.logger import ReActAgentLogger

# Module path
module_path: Path = Path(__file__).resolve()

# Set up logger
logger: Logger = ReActAgentLogger.get_logger(module_name=module_path.name)

# Define PDFs and database directories paths
current_dir: Path = Path(__file__).parent.resolve()
pdfs_dir: Path = current_dir / "pdfs"
db_dir: Path = current_dir / "db"
store_name: str = "chroma_db_pdf_advanced"
persistent_directory: Path = db_dir / store_name


def load_pdf_documents_advanced(pdfs_dir: Path) -> List[Document]:
    """Load PDFs with advanced parsing for tables, images, charts, and complex layouts."""
    documents: List[Document] = []
    failed_files: List[Tuple[Path, str]] = []

    try:
        pdf_files: List[Path] = list(pdfs_dir.glob(pattern="*.pdf"))
        if not pdf_files:
            logger.warning(msg=f"No '.pdf' files found in {pdfs_dir}")
            return documents

        for file_path in pdf_files:
            try:
                # Fallback to PyPDFLoader if UnstructuredPDFLoader fails
                try:
                    loader = PyPDFLoader(file_path=str(file_path))
                    docs: List[Document] = loader.load()
                    logger.info(f"Loaded with PyPDFLoader: {file_path}")
                except Exception as fallback_error:
                    logger.warning(
                        f"PyPDFLoader also failed for {file_path}: {fallback_error}"
                    )
                    continue

                # Add OCR processing for images
                ocr_docs: List[Document] = extract_text_with_ocr(
                    pdf_path=str(file_path)
                )

                # Add chart and graph processing
                chart_docs: List[Document] = extract_charts_and_graphs(str(file_path))

                # Combine all content types
                all_docs: List[Document] = docs + ocr_docs + chart_docs

                for doc in all_docs:
                    doc.metadata.update(
                        {"source": file_path.name, "processing_type": "advanced"}
                    )
                    documents.append(doc)

                logger.info(
                    msg=f"Successfully loaded PDF with advanced processing: {file_path} "
                    f"(Text: {len(docs)}, OCR: {len(ocr_docs)}, Charts: {len(chart_docs)})"
                )

            except Exception as e:
                error_msg: str = f"Error loading {file_path}: {str(object=e)}"
                failed_files.append((file_path, str(object=e)))
                logger.error(msg=error_msg)
                continue

        if failed_files:
            logger.warning(
                msg=f"Failed to load {len(failed_files)} files. Check individual error messages above."
            )

        logger.info(
            msg=f"Successfully loaded {len(documents)} documents from {len(pdf_files) - len(failed_files)} PDF files"
        )

    except Exception as e:
        logger.error(msg=f"Error accessing directory {pdfs_dir}: {str(object=e)}")

    return documents


def extract_text_with_ocr(pdf_path: str) -> List[Document]:
    """Extract text including OCR from images in PDF."""
    documents = []

    try:
        doc = fitz.open(pdf_path)

        for page_num in range(doc.page_count):
            page = doc[page_num]
            ocr_text = ""

            # Extract images and OCR them
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_pil = Image.open(io.BytesIO(img_data))
                        extracted_text = pytesseract.image_to_string(img_pil)
                        if extracted_text.strip():
                            ocr_text += f"\n[Image {img_index} OCR]: {extracted_text}"

                    pix = None

                except Exception as e:
                    logger.warning(
                        f"Error processing image {img_index} on page {page_num}: {e}"
                    )
                    continue

            if ocr_text.strip():
                documents.append(
                    Document(
                        page_content=ocr_text,
                        metadata={
                            "source": Path(pdf_path).name,
                            "page": page_num,
                            "content_type": "ocr_images",
                        },
                    )
                )

        doc.close()

    except Exception as e:
        logger.error(f"Error during OCR processing of {pdf_path}: {e}")

    return documents


class TableAwareTextSplitter:
    """Text splitter that preserves table structure and handles complex layouts."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents while preserving table structure."""
        chunks = []

        for doc in documents:
            # Detect tables and handle them specially
            if self._contains_table(doc.page_content):
                table_chunks = self._extract_table_chunks(doc)
                chunks.extend(table_chunks)
            else:
                # Regular text splitting
                regular_chunks = self.base_splitter.split_documents([doc])
                chunks.extend(regular_chunks)

        return chunks

    def _contains_table(self, text: str) -> bool:
        """Detect if text contains tabular data."""
        patterns = [
            r"\|.*\|.*\|",  # Pipe-separated
            r"\t.*\t.*\t",  # Tab-separated
            r"^\s*\w+\s+\w+\s+\w+\s*$",  # Space-separated columns
            r"Table \d+",  # Table labels
            r"^\s*\d+\.\d+\s+\d+\.\d+",  # Numeric data in columns
        ]
        return any(re.search(pattern, text, re.MULTILINE) for pattern in patterns)

    def _extract_table_chunks(self, doc: Document) -> List[Document]:
        """Extract table chunks while preserving structure."""
        content = doc.page_content

        # Split by table boundaries but keep tables intact
        table_pattern = r"(Table \d+.*?)(?=Table \d+|\Z)"
        tables = re.findall(table_pattern, content, re.DOTALL)

        chunks = []
        if tables:
            for i, table in enumerate(tables):
                if len(table.strip()) > 50:  # Only process substantial tables
                    chunk_doc = Document(
                        page_content=f"TABLE DATA:\n{table.strip()}",
                        metadata={
                            **doc.metadata,
                            "content_type": "table",
                            "table_index": i,
                        },
                    )
                    chunks.append(chunk_doc)
        else:
            # Fallback: treat as regular table content
            chunk_doc = Document(
                page_content=f"STRUCTURED DATA:\n{content}",
                metadata={**doc.metadata, "content_type": "structured"},
            )
            chunks.append(chunk_doc)

        return chunks


def create_advanced_text_chunks(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Create text chunks with table-aware splitting."""
    splitter = TableAwareTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    # Enhance chunks with content type context
    enhanced_chunks = create_table_specific_chunks(chunks)

    logger.info(
        msg=f"Split {len(documents)} documents into {len(enhanced_chunks)} chunks"
    )
    return enhanced_chunks


def create_multimodal_embeddings() -> HuggingFaceEmbeddings:
    """Create embeddings optimized for structured and multimodal content."""
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info(
        msg="Advanced embeddings created successfully with BAAI/bge-large-en-v1.5 (1024 dimensions)"
    )
    return embeddings


def create_table_specific_chunks(documents: List[Document]) -> List[Document]:
    """Enhance chunks with content-type specific context."""
    enhanced_chunks = []

    for doc in documents:
        content_type = doc.metadata.get("content_type", "text")

        # Add context based on content type
        if content_type == "table" or "table" in doc.page_content.lower():
            enhanced_content = f"TABLE DATA: {doc.page_content}"
        elif content_type == "ocr_images":
            enhanced_content = f"IMAGE TEXT: {doc.page_content}"
        elif content_type == "chart_graph":
            chart_type = doc.metadata.get("chart_type", "unknown")
            enhanced_content = (
                f"CHART/GRAPH DATA ({chart_type.upper()}): {doc.page_content}"
            )
        elif "chart" in doc.page_content.lower() or "graph" in doc.page_content.lower():
            enhanced_content = f"CHART/GRAPH DATA: {doc.page_content}"
        else:
            enhanced_content = doc.page_content

        doc.page_content = enhanced_content
        enhanced_chunks.append(doc)

    return enhanced_chunks


def create_vector_store(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persistent_directory: Path,
) -> Optional[Chroma]:
    """Create and persist a Chroma vector store from document chunks."""
    try:
        # Ensure the directory exists
        persistent_directory.mkdir(parents=True, exist_ok=True)

        # Create the vector store
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persistent_directory),
        )

        logger.info(
            msg=f"Vector store created and persisted at {persistent_directory} with {len(chunks)} chunks"
        )
        return db

    except Exception as e:
        logger.error(msg=f"Error creating vector store: {str(e)}")
        return None


def initialize_advanced_pdf_vector_store(
    pdfs_dir: Path, persistent_directory: Path
) -> None | Chroma | Literal["EXISTS"]:
    """Initialize an advanced vector store from PDF documents with complex content."""
    logger.info(
        msg="======== Starting Advanced PDF RAG With Multimodal Content ========"
    )
    logger.info(msg=f"PDFs Directory: {pdfs_dir}")
    logger.info(msg=f"Persistent Directory: {persistent_directory}")

    # Check if vector store already exists
    if persistent_directory.exists():
        logger.info(msg="Advanced vector store already exists. No need to initialize.")
        return "EXISTS"  # Return a sentinel value instead of None

    # Check if PDFs directory exists
    if not pdfs_dir.exists():
        logger.error(
            msg=f"The directory '{pdfs_dir}' does not exist. Please check the path."
        )
        return None

    logger.info(msg="Initializing new advanced vector store...")

    # Load PDF documents with advanced processing
    documents: List[Document] = load_pdf_documents_advanced(pdfs_dir=pdfs_dir)
    if not documents:
        logger.error(msg="No PDF documents loaded. Cannot create vector store.")
        return None

    # Create advanced text chunks
    chunks: List[Document] = create_advanced_text_chunks(
        documents=documents, chunk_size=1000, chunk_overlap=200
    )

    # Create multimodal embeddings
    embeddings: HuggingFaceEmbeddings = create_multimodal_embeddings()

    # Create and persist vector store
    db: Optional[Chroma] = create_vector_store(
        chunks=chunks, embeddings=embeddings, persistent_directory=persistent_directory
    )

    # Save PDF metadata for future change detection
    if db is not None:
        save_pdf_metadata(pdfs_dir=pdfs_dir, persistent_directory=persistent_directory)

    return db


def extract_charts_and_graphs(pdf_path: str) -> List[Document]:
    """Extract and analyze charts, graphs, and visual data from PDF."""
    documents = []

    try:
        doc = fitz.open(pdf_path)

        for page_num in range(doc.page_count):
            page = doc[page_num]

            # Extract images that might be charts/graphs
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_pil = Image.open(io.BytesIO(img_data))

                        # Convert to numpy array for analysis
                        img_array = np.array(img_pil)

                        # Analyze if image contains chart/graph
                        chart_analysis = analyze_chart_content(img_array, img_pil)

                        if chart_analysis["is_chart"]:
                            # Extract chart data and description
                            chart_description = generate_chart_description(
                                img_pil, chart_analysis
                            )

                            documents.append(
                                Document(
                                    page_content=chart_description,
                                    metadata={
                                        "source": Path(pdf_path).name,
                                        "page": page_num,
                                        "content_type": "chart_graph",
                                        "chart_type": chart_analysis["chart_type"],
                                        "image_index": img_index,
                                    },
                                )
                            )

                    pix = None

                except Exception as e:
                    logger.warning(
                        f"Error processing chart/graph {img_index} on page {page_num}: {e}"
                    )
                    continue

        doc.close()

    except Exception as e:
        logger.error(f"Error during chart/graph extraction from {pdf_path}: {e}")

    return documents


def analyze_chart_content(img_array: np.ndarray, img_pil: Image.Image) -> dict:
    """Analyze image to determine if it's a chart/graph and extract features."""
    analysis = {
        "is_chart": False,
        "chart_type": "unknown",
        "has_axes": False,
        "has_legend": False,
        "color_count": 0,
        "text_regions": [],
    }

    try:
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Detect lines (potential axes, grid lines)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        # Count unique colors (charts often have distinct colors)
        unique_colors = len(
            np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0)
        )
        analysis["color_count"] = unique_colors

        # Detect if it's likely a chart based on features
        has_lines = lines is not None and len(lines) > 5
        has_varied_colors = unique_colors > 10 and unique_colors < 50

        # OCR to detect chart-related text
        ocr_text = pytesseract.image_to_string(img_pil).lower()
        chart_keywords = [
            "chart",
            "graph",
            "plot",
            "axis",
            "legend",
            "%",
            "data",
            "figure",
            "table",
            "year",
            "month",
            "value",
            "total",
        ]
        has_chart_text = any(keyword in ocr_text for keyword in chart_keywords)

        # Determine if it's a chart
        analysis["is_chart"] = has_lines and (has_varied_colors or has_chart_text)

        if analysis["is_chart"]:
            # Determine chart type based on features
            analysis["chart_type"] = determine_chart_type(img_array, ocr_text)
            analysis["has_axes"] = detect_axes(gray)
            analysis["has_legend"] = detect_legend(ocr_text)
            analysis["text_regions"] = extract_text_regions(img_pil)

    except Exception as e:
        logger.warning(f"Error analyzing chart content: {e}")

    return analysis


def determine_chart_type(img_array: np.ndarray, ocr_text: str) -> str:
    """Determine the type of chart based on visual and text features."""

    # Text-based detection
    if any(word in ocr_text for word in ["pie", "slice", "%"]):
        return "pie_chart"
    elif any(word in ocr_text for word in ["bar", "column"]):
        return "bar_chart"
    elif any(word in ocr_text for word in ["line", "trend", "time"]):
        return "line_chart"
    elif any(word in ocr_text for word in ["scatter", "correlation"]):
        return "scatter_plot"
    elif any(word in ocr_text for word in ["histogram", "distribution"]):
        return "histogram"

    # Visual-based detection (simplified)
    gray = (
        cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        if len(img_array.shape) == 3
        else img_array
    )

    # Detect circular shapes (pie charts)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0
    )
    if circles is not None:
        return "pie_chart"

    # Detect rectangular patterns (bar charts)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangular_contours = [c for c in contours if cv2.contourArea(c) > 100]
    if len(rectangular_contours) > 3:
        return "bar_chart"

    return "generic_chart"


def detect_axes(gray_img: np.ndarray) -> bool:
    """Detect if image has chart axes."""
    try:
        # Look for long straight lines that could be axes
        edges = cv2.Canny(gray_img, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10
        )

        if lines is not None:
            # Normalize the lines array to shape (N, 4) so each entry is [x1, y1, x2, y2]
            arr = np.asarray(lines)
            try:
                if arr.ndim == 3:
                    arr = arr.reshape(-1, 4)
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr.reshape(-1, 4)
            except Exception:
                arr = arr.flatten()
                if arr.size % 4 == 0:
                    arr = arr.reshape(-1, 4)
                else:
                    # Fallback: cannot interpret lines, assume no axes
                    return False

            # Check for perpendicular lines (typical of axes)
            horizontal_lines = []
            vertical_lines = []

            for coords in arr:
                # Ensure we have four numeric values
                coords = np.asarray(coords).reshape(-1)[:4]
                x1, y1, x2, y2 = (
                    float(coords[0]),
                    float(coords[1]),
                    float(coords[2]),
                    float(coords[3]),
                )
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                if abs(angle) < 10 or abs(angle - 180) < 10:  # Horizontal
                    horizontal_lines.append((x1, y1, x2, y2))
                elif abs(angle - 90) < 10 or abs(angle + 90) < 10:  # Vertical
                    vertical_lines.append((x1, y1, x2, y2))

            return len(horizontal_lines) > 0 and len(vertical_lines) > 0
    except Exception:
        pass

    return False


def detect_legend(ocr_text: str) -> bool:
    """Detect if chart has a legend based on text."""
    legend_indicators = ["legend", "key", "series", "category"]
    return any(indicator in ocr_text.lower() for indicator in legend_indicators)


def extract_text_regions(img_pil: Image.Image) -> List[str]:
    """Extract text regions from chart image."""
    try:
        # Use pytesseract to get detailed text information
        data = pytesseract.image_to_data(img_pil, output_type=pytesseract.Output.DICT)

        text_regions = []
        for i, text in enumerate(data["text"]):
            if text.strip() and int(data["conf"][i]) > 30:  # Confidence threshold
                text_regions.append(text.strip())

        return text_regions
    except Exception:
        return []


def generate_chart_description(img_pil: Image.Image, chart_analysis: dict) -> str:
    """Generate a comprehensive description of the chart/graph."""

    # Extract all text from the image
    ocr_text = pytesseract.image_to_string(img_pil)

    # Build description
    description_parts = []

    # Chart type and basic info
    chart_type = chart_analysis.get("chart_type", "chart")
    description_parts.append(f"CHART TYPE: {chart_type.replace('_', ' ').title()}")

    # Extract numerical data
    numbers = re.findall(r"\d+\.?\d*", ocr_text)
    if numbers:
        description_parts.append(
            f"NUMERICAL DATA: {', '.join(numbers[:10])}"
        )  # Limit to first 10 numbers

    # Extract text labels and titles
    text_lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
    if text_lines:
        # Likely title (usually first significant text)
        potential_title = next((line for line in text_lines if len(line) > 3), "")
        if potential_title:
            description_parts.append(f"TITLE: {potential_title}")

        # Other labels
        labels = [
            line for line in text_lines[1:] if len(line) > 1 and not line.isdigit()
        ]
        if labels:
            description_parts.append(
                f"LABELS: {', '.join(labels[:5])}"
            )  # Limit to first 5 labels

    # Chart features
    features = []
    if chart_analysis.get("has_axes"):
        features.append("has axes")
    if chart_analysis.get("has_legend"):
        features.append("has legend")
    if chart_analysis.get("color_count", 0) > 5:
        features.append("multiple colors")

    if features:
        description_parts.append(f"FEATURES: {', '.join(features)}")

    # Full OCR text for context
    if ocr_text.strip():
        description_parts.append(f"EXTRACTED TEXT: {ocr_text.strip()}")

    return "\n".join(description_parts)


def create_chart_specific_chunks(documents: List[Document]) -> List[Document]:
    """Create specialized chunks for charts and graphs."""
    enhanced_chunks = []

    for doc in documents:
        content = doc.page_content
        content_type = doc.metadata.get("content_type", "text")

        if content_type == "chart_graph":
            chart_type = doc.metadata.get("chart_type", "unknown")
            enhanced_content = f"CHART/GRAPH DATA ({chart_type.upper()}):\n{content}"
            doc.page_content = enhanced_content

        enhanced_chunks.append(doc)

    return enhanced_chunks


def get_pdf_metadata_file(persistent_directory: Path) -> Path:
    """Get the path to the PDF metadata tracking file."""
    return persistent_directory.parent / f"{persistent_directory.name}_metadata.json"


def save_pdf_metadata(pdfs_dir: Path, persistent_directory: Path) -> None:
    """Save metadata about processed PDFs for change detection."""
    metadata_file = get_pdf_metadata_file(persistent_directory)

    pdf_files = list(pdfs_dir.glob(pattern="*.pdf"))
    metadata = {
        "pdf_files": {
            str(pdf.name): {
                "size": pdf.stat().st_size,
                "modified_time": pdf.stat().st_mtime,
            }
            for pdf in pdf_files
        }
    }

    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata for {len(pdf_files)} PDF files")


def load_pdf_metadata(persistent_directory: Path) -> dict:
    """Load previously saved PDF metadata."""
    metadata_file = get_pdf_metadata_file(persistent_directory)

    if not metadata_file.exists():
        return {"pdf_files": {}}

    with open(metadata_file, "r") as f:
        return json.load(f)


def detect_pdf_changes(
    pdfs_dir: Path, persistent_directory: Path
) -> Tuple[List[str], List[str]]:
    """
    Detect changes in PDF files.

    Returns:
        Tuple of (added_files, deleted_files)
    """
    old_metadata = load_pdf_metadata(persistent_directory)
    old_files = set(old_metadata.get("pdf_files", {}).keys())

    current_pdf_files = list(pdfs_dir.glob(pattern="*.pdf"))
    current_files = {pdf.name for pdf in current_pdf_files}

    # Detect additions and deletions
    added_files = list(current_files - old_files)
    deleted_files = list(old_files - current_files)

    # Detect modifications (size or time changed)
    modified_files = []
    for pdf in current_pdf_files:
        if pdf.name in old_files:
            old_info = old_metadata["pdf_files"][pdf.name]
            current_size = pdf.stat().st_size
            current_mtime = pdf.stat().st_mtime

            if (
                old_info["size"] != current_size
                or abs(old_info["modified_time"] - current_mtime) > 1
            ):
                modified_files.append(pdf.name)

    # Treat modified files as additions (need reprocessing)
    added_files.extend(modified_files)

    return added_files, deleted_files


def update_vector_store(
    pdfs_dir: Path,
    persistent_directory: Path,
    embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> Optional[Chroma]:
    """
    Update vector store when PDFs are added, modified, or deleted.

    Args:
        pdfs_dir: Directory containing PDF files
        persistent_directory: Directory where vector store is persisted
        embeddings: Embeddings model to use (creates new if None)

    Returns:
        Updated Chroma vector store or None if update failed
    """
    logger.info("Checking for PDF changes...")

    added_files, deleted_files = detect_pdf_changes(pdfs_dir, persistent_directory)

    if not added_files and not deleted_files:
        logger.info("No PDF changes detected. Vector store is up to date.")
        return None

    logger.info(
        f"Detected changes - Added/Modified: {len(added_files)}, Deleted: {len(deleted_files)}"
    )
    if added_files:
        logger.info(f"Added/Modified files: {added_files}")
    if deleted_files:
        logger.info(f"Deleted files: {deleted_files}")

    # Create embeddings if not provided
    if embeddings is None:
        embeddings = create_multimodal_embeddings()

    # Load existing vector store
    try:
        db = Chroma(
            persist_directory=str(persistent_directory),
            embedding_function=embeddings,
        )
        logger.info("Loaded existing vector store for update")
    except Exception as e:
        logger.error(f"Error loading vector store for update: {e}")
        return None

    # Handle deleted files
    if deleted_files:
        for deleted_file in deleted_files:
            try:
                # Delete documents from this source
                db.delete(where={"source": deleted_file})
                logger.info(f"Removed documents from deleted file: {deleted_file}")
            except Exception as e:
                logger.warning(f"Error removing documents for {deleted_file}: {e}")

    # Handle added/modified files
    if added_files:
        new_documents = []
        for added_file in added_files:
            file_path = pdfs_dir / added_file
            if file_path.exists():
                try:
                    # Load and process the new/modified PDF
                    loader = PyPDFLoader(file_path=str(file_path))
                    docs = loader.load()

                    # Add OCR and chart processing
                    ocr_docs = extract_text_with_ocr(pdf_path=str(file_path))
                    chart_docs = extract_charts_and_graphs(str(file_path))

                    all_docs = docs + ocr_docs + chart_docs

                    for doc in all_docs:
                        doc.metadata.update(
                            {"source": added_file, "processing_type": "advanced"}
                        )
                        new_documents.append(doc)

                    logger.info(f"Processed new/modified file: {added_file}")

                except Exception as e:
                    logger.error(f"Error processing {added_file}: {e}")

        if new_documents:
            # Create chunks from new documents
            chunks = create_advanced_text_chunks(
                documents=new_documents, chunk_size=1000, chunk_overlap=200
            )

            # Add to existing vector store
            try:
                db.add_documents(documents=chunks)
                logger.info(f"Added {len(chunks)} new chunks to vector store")
            except Exception as e:
                logger.error(f"Error adding new documents to vector store: {e}")
                return None

    # Save updated metadata
    save_pdf_metadata(pdfs_dir, persistent_directory)

    logger.info("Vector store update complete")
    return db


def generate_sample_questions(db: Chroma, num_questions: int = 5) -> List[str]:
    """
    Generate sample questions based on PDF content.

    Args:
        db: Chroma vector store
        num_questions: Number of sample questions to generate

    Returns:
        List of sample questions
    """
    try:
        # Get a sample of documents from the vector store
        sample_docs = db.similarity_search("", k=20)

        if not sample_docs:
            return ["What information is available in the documents?"]

        # Extract key topics and entities from documents
        keywords = set()

        for doc in sample_docs:
            # Extract potential topics (capitalized phrases, technical terms)
            # Look for common patterns in technical documents
            patterns = [
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",  # Capitalized phrases
                r"\b(neural network|transformer|attention|embedding|model|algorithm|learning)\b",
                r"\b(table|chart|figure|graph)\s+\d+",  # References to visuals
            ]

            for pattern in patterns:
                matches = re.findall(pattern, doc.page_content)
                for match in matches:
                    if isinstance(match, tuple):
                        match = (
                            match[0] if match[0] else match[1] if len(match) > 1 else ""
                        )
                    if match and len(match) > 3:
                        keywords.add(match)

        # Generate questions based on extracted keywords
        questions = []
        question_templates = [
            "What is {}?",
            "Explain about {}",
            "How does {} work?",
            "What are the key features of {}?",
            "Describe the {} architecture",
        ]

        keyword_list = list(keywords)[
            : num_questions * 2
        ]  # Get more keywords than needed

        for i, keyword in enumerate(keyword_list):
            if len(questions) >= num_questions:
                break
            template = question_templates[i % len(question_templates)]
            questions.append(template.format(keyword))

        # Add some generic questions if we don't have enough
        generic_questions = [
            "What are the main topics covered in the documents?",
            "Summarize the key concepts from the PDFs",
            "What technical details are provided?",
        ]

        while len(questions) < num_questions:
            questions.append(generic_questions[len(questions) % len(generic_questions)])

        return questions[:num_questions]

    except Exception as e:
        logger.warning(f"Error generating sample questions: {e}")
        return [
            "What information is available in the documents?",
            "Summarize the main topics",
            "What are the key concepts?",
        ]


def main() -> None:
    """Main function for standalone execution of the advanced PDF RAG initialization."""
    result: None | Chroma | Literal["EXISTS"] = initialize_advanced_pdf_vector_store(
        pdfs_dir=pdfs_dir, persistent_directory=persistent_directory
    )

    if result == "EXISTS":
        logger.info(
            msg="Advanced PDF Vector store already exists. Standalone execution complete."
        )
    elif result is not None:
        logger.info(msg="Advanced PDF Vector store initialization complete.")
    else:
        logger.error(msg="Advanced PDF Vector store initialization failed.")


# Main entry point
if __name__ == "__main__":
    main()
