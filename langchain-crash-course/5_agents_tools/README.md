# 5_agents_tools

## agent_tools_basic.py

This script demonstrates a minimal LangChain agent that can use a single
tool – *Current Time* – to answer user queries.  
It supports two LLM back‑ends:

* **OpenAI** – configured via ``OPENAI_API_KEY`` and ``OPENAI_LLM`` in a
  ``.env`` file.
* **Ollama** – configured via ``OLLAMA_LLM`` in the same file.

The agent is built with the ReAct prompt style (Reason → Action → Observation)
and is executed in a simple REPL loop.  The script logs key events using a
custom `RAGLogger`.

## Usage

1. Create a ``.env`` file in the same directory with the required keys.

   ```bash
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_LLM=gpt-4o-mini
   OLLAMA_LLM=gemma3:4b
    ```

2. Run the script:

   ```bash
   python agent_tools_basic.py
    ```
