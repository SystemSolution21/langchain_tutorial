# 5_agents_tools

This directory contains examples of LangChain agents and tools, including basic agents, RAG (Retrieval-Augmented Generation) systems, and advanced PDF processing.

## 📋 Contents

### Basic Agents

- **agent_tools_basic.py** - Minimal agent with a single tool (Current Time)
- **agent_react_chat.py** - ReAct agent with chat capabilities
- **agent_react_rag_context.py** - Agent with RAG context retrieval

### RAG Systems

- **rag.py** - Basic RAG implementation
- **rag_pdf.py** - RAG with PDF document loading
- **rag_pdf_advanced.py** - Advanced RAG with multimodal PDF processing

### Advanced RAG Agents

- **agent_react_rag_pdf.py** - Agent with basic PDF RAG
- **agent_react_rag_pdf_advanced.py** - Agent with advanced PDF RAG (tables, charts, OCR)

### Tools

- **tools/tool_basetool.py** - Custom tool using BaseTool class
- **tools/tool_constructor.py** - Tool creation using constructors
- **tools/tool_decorator.py** - Tool creation using decorators

---

## 🚀 Quick Start

### 1. Setup Environment

Create a `.env` file in the project root:

```bash
# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key
OPENAI_LLM=gpt-4o-mini

# Ollama (local LLM)
OLLAMA_LLM=llama3.2:3b
```

### 2. Pre-download Models (for Advanced RAG)

**Recommended for production use:**

```bash
# From project root
python scripts/setup_models.py
```

This downloads the HuggingFace embedding model (~1.34 GB) used by `rag_pdf_advanced.py`.

**Skip this if:**

- Just experimenting (auto-downloads on first use)
- You have fast internet and don't mind waiting 5-15 minutes

### 3. Install System Dependencies (for Advanced PDF)

**For OCR and image processing in `rag_pdf_advanced.py`:**

**Windows:**

```bash
# Download Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR
```

**macOS:**

```bash
brew install tesseract
```

**Linux:**

```bash
sudo apt-get install tesseract-ocr
```

---

## 📖 Usage Examples

### Basic Agent

```bash
python agent_tools_basic.py
```

Simple agent with time tool. Good starting point.

### Basic PDF RAG

```bash
# 1. Initialize vector store (first time only)
python rag_pdf.py

# 2. Run agent with PDF knowledge
python agent_react_rag_pdf.py
```

### Advanced PDF RAG (Tables, Charts, OCR)

```bash
# 1. Pre-download models (recommended)
python ../../scripts/setup_models.py

# 2. Add PDFs to pdfs/ directory
cp your_document.pdf pdfs/

# 3. Initialize vector store
python rag_pdf_advanced.py

# 4. Run advanced agent
python agent_react_rag_pdf_advanced.py
```

---

## 🔧 Advanced RAG Features

The `rag_pdf_advanced.py` module provides:

### 1. **Multi-Loader Strategy**

- **Primary**: `UnstructuredPDFLoader` (advanced table/layout parsing)
- **Fallback**: `PyPDFLoader` (basic text extraction)

### 2. **Multimodal Content Extraction**

- **Text**: Standard text extraction
- **Tables**: Preserves table structure
- **Images**: OCR text extraction from embedded images
- **Charts/Graphs**: Visual data analysis and description

### 3. **Smart Text Splitting**

- Table-aware chunking (preserves table integrity)
- Content-type specific processing
- Metadata enrichment

### 4. **Advanced Embeddings**

- Model: `BAAI/bge-large-en-v1.5` (1024 dimensions)
- Optimized for structured/multimodal content
- Normalized embeddings for better similarity search

### 5. **Incremental Updates**

- Detects PDF changes (added/modified/deleted)
- Updates vector store incrementally
- Metadata tracking for change detection

---

## 📁 Directory Structure

```langcchain-crash-course/
5_agents_tools/
├── pdfs/                    # Place your PDF files here
├── db/                      # Vector store databases (auto-created)
│   ├── chroma_db_pdf/
│   └── chroma_db_pdf_advanced/
├── logs/                    # Application logs
├── tools/                   # Custom tool implementations
└── util/                    # Utilities (logger, etc.)
```

---

## 🐛 Troubleshooting

### Model Download Issues

**Problem**: "Model not found" or slow first run

**Solution**:

```bash
# Pre-download the model
python scripts/setup_models.py

# Or manually:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
```

### Tesseract OCR Not Found

**Problem**: `TesseractNotFoundError`

**Solution**:

- Install Tesseract (see setup instructions above)
- Add to system PATH
- Restart terminal/IDE

### UnstructuredPDFLoader Fails

**Problem**: `ImportError` or parsing errors

**Solution**:

- The system automatically falls back to `PyPDFLoader`
- Check logs for details
- Ensure `unstructured` package is installed: `pip install unstructured`

### Out of Memory

**Problem**: Large PDFs cause memory issues

**Solution**:

- Reduce `chunk_size` in text splitting
- Process PDFs in smaller batches
- Use smaller embedding model (see alternatives below)

### Alternative Embedding Models

If `BAAI/bge-large-en-v1.5` is too large:

```python
# In rag_pdf_advanced.py, replace with:
"BAAI/bge-small-en-v1.5"              # 133 MB, 384 dims
"BAAI/bge-base-en-v1.5"               # 438 MB, 768 dims
"sentence-transformers/all-MiniLM-L6-v2"  # 90 MB, 384 dims
```

---

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Unstructured Library](https://unstructured-io.github.io/unstructured/)
- [HuggingFace Embeddings](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [Chroma Vector Store](https://docs.trychroma.com/)
