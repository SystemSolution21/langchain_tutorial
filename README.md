# Langchain Tutorial

## langchain-crash-course

This repository contains a crash course on Langchain, a framework for developing applications using large language models (LLMs). The course covers various aspects of working with LLMs, including chat models, prompt templates, chains, retrieval augmented generation (RAG), agents, and tools. It also includes examples of using different LLM providers and Ollama models.

## Course Outline

1. Setup Environment
2. Chat Models
3. Prompt Templates
4. Chains
5. RAG
6. Agents & Tools

### Prerequisites

- Python 3.10 or 3.11
- Poetry or uv package manager

## Setup Instructions

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Pre-download AI Models (Recommended)

For production use or offline environments, pre-download the required AI models:

```bash
python scripts/setup_models.py
```

This will download:

- **HuggingFace Embeddings**: `BAAI/bge-large-en-v1.5` (~1.34 GB)
- Used by advanced RAG features in `5_agents_tools/rag_pdf_advanced.py`

**Why pre-download?**

- ✅ Faster first run (no 5-15 minute wait)
- ✅ Works offline after initial download
- ✅ Predictable deployment
- ✅ Avoids network timeouts in restricted environments

**Skip this step if:**

- You're just experimenting (models auto-download on first use)
- You have fast, reliable internet
- You don't mind the initial wait

### 3. Configure Environment Variables

Create a `.env` file in the root directory with your API keys:

```bash
# OpenAI (optional)
OPENAI_API_KEY=your_openai_key_here

# Google (optional)
GOOGLE_API_KEY=your_google_key_here

# Anthropic (optional)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Tavily (for web search, optional)
TAVILY_API_KEY=your_tavily_key_here
```

### 4. Install System Dependencies (for Advanced PDF Processing)

**For OCR and image processing:**

**Windows:**

```bash
# Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
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

### 1_chat_models

- 1_chat_model_basic.py
- 2_chat_model_basic_conversation.py
- 3_chat_model_alternatives.py
- 4_chat_model_conversation_with_user.py

### 2_prompt_templates

- prompt_template.py

### 3_chains

- 1_chains_basic.py
- 2_chains_runnable_sequence.py
- 3_chains_extended.py
- 4_chains_parallel.py
- 5_chains_branching.py

### 4_rag

- 1a_rag_basic.py
- 1b_rag_basic_query.py
- 2a_rag_basic_metadata.py
- 2a_rag_basic_metadata_robust.py
- 2b_rag_basic_metadata.py
- 2b_rag_basic_metadata_robust.py
- 3_rag_text_splitting.py
- 4_rag_multi_model_embeddings.py
- 5_rag_retriever_search_types.py
- 6_rag_question_answering.py
- 7_rag_llm_conversation.py
- 8_rag_web_scrape_basic.py
- 8_rag_web_scrape_firecrawl.py
- rag_with_metadata.py

### 5_agents_tools

- agent_tools_basic.py
- agent_react_chat.py
- agent_react_rag_context.py
- rag.py
- tools/tool_basetool.py
- tools/tool_constructor.py
- tools/tool_decorator.py
