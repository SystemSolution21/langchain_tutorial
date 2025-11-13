# LangChain Tutorial - Setup Guide

## Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd langchain_tutorial

# Install packages
uv sync
# OR
pip install -e .
```

### 2. Configure Environment

Create `.env` file in project root:

```bash
# Required for OpenAI models (optional)
OPENAI_API_KEY=sk-...
OPENAI_LLM=gpt-4o-mini

# Required for Ollama (local, recommended)
OLLAMA_LLM=llama3.2:3b

# Optional: Other providers
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
TAVILY_API_KEY=...
```

### 3. Pre-download AI Models (Recommended)

**For production or offline use:**

```bash
python scripts/setup_models.py
```

This downloads:
- HuggingFace Embeddings: `BAAI/bge-large-en-v1.5` (~1.34 GB)

**Benefits:**
- ✅ No 5-15 minute wait on first run
- ✅ Works offline
- ✅ Predictable deployment

**Skip if:**
- Just experimenting (auto-downloads on first use)
- Fast internet available

---

## Advanced PDF Processing Setup

### Required for `rag_pdf_advanced.py`

#### Install Tesseract OCR

**Windows:**
1. Download: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to: `C:\Program Files\Tesseract-OCR`
3. Add to PATH
4. Restart terminal

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

#### Verify Installation

```bash
tesseract --version
```

---

## Testing Your Setup

### Test Basic Agent

```bash
cd langchain-crash-course/5_agents_tools
python agent_tools_basic.py
```

### Test PDF RAG (Basic)

```bash
cd langchain-crash-course/5_agents_tools
python rag_pdf.py
python agent_react_rag_pdf.py
```

### Test Advanced PDF RAG

```bash
cd langchain-crash-course/5_agents_tools

# 1. Pre-download models (if not done)
python ../../scripts/setup_models.py

# 2. Add a PDF to test
cp /path/to/your.pdf pdfs/

# 3. Initialize vector store
python rag_pdf_advanced.py

# 4. Run agent
python agent_react_rag_pdf_advanced.py
```

---

## Troubleshooting

### "Model not found" Error

```bash
# Manually download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
```

### "Tesseract not found" Error

- Ensure Tesseract is installed
- Check it's in your PATH: `tesseract --version`
- Restart terminal/IDE after installation

### "UnstructuredPDFLoader failed" Warning

- This is normal - system falls back to PyPDFLoader
- Check logs for details
- Ensure `unstructured` package is installed

### Out of Memory

Use a smaller embedding model in `rag_pdf_advanced.py`:

```python
# Replace in create_multimodal_embeddings():
model_name = "BAAI/bge-small-en-v1.5"  # 133 MB instead of 1.34 GB
```

---

## Docker Setup (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync

# Pre-download models
COPY scripts/setup_models.py scripts/
RUN python scripts/setup_models.py

# Copy application
COPY . .

CMD ["python", "langchain-crash-course/5_agents_tools/agent_react_rag_pdf_advanced.py"]
```

Build and run:

```bash
docker build -t langchain-tutorial .
docker run -it --env-file .env langchain-tutorial
```

---

## Next Steps

1. ✅ Complete setup steps above
2. 📖 Read `langchain-crash-course/5_agents_tools/README.md`
3. 🚀 Try the examples in order:
   - Basic agent → Basic RAG → Advanced RAG
4. 🔧 Customize for your use case

---

## Support

- **Documentation**: See README files in each directory
- **Issues**: Check troubleshooting sections
- **Examples**: All scripts include inline documentation

