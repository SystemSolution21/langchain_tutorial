# 3_chains_extended.py

# Import standard libraries
import os
from typing import Any

# Import third-party libraries
from dotenv import load_dotenv

# Import langchain modules
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableSerializable,
)
from langchain_ollama import ChatOllama

# Load Environment Variables
load_dotenv()

# Create Chat Model
model: str = os.getenv(key="OLLAMA_LLM", default="llama3.2:latest")
llm = ChatOllama(model=model)

# Create Chat Prompt Template
chat_prom_temp = ChatPromptTemplate(
    messages=[
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes"),
    ]
)

# Convert output to uppercase
uppercase_output: RunnableLambda[Any, Any] = RunnableLambda(func=lambda x: x.upper())


def count_words(x: str) -> str:
    """Count the number of words in a string."""
    return f"Word Count: {len(x.split())}\n{x}"


# Count output words
words_count: RunnableLambda[Any, Any] = RunnableLambda(func=lambda x: count_words(x))

# Create LangChain Pipeline
chain: RunnableSerializable[dict[str, str | int], Any] = (
    chat_prom_temp | llm | StrOutputParser() | uppercase_output | words_count
)

# Run Chain
response: Any = chain.invoke(input={"topic": "lawyer", "joke_count": 3})
print(response)
