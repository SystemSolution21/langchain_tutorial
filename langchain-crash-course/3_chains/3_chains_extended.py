from typing import Callable, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableSequence,
    RunnableSerializable,
)
from langchain.schema.language_model import LanguageModelInput
from langchain_ollama import ChatOllama
from dotenv import load_dotenv


# Load Environment Variables
load_dotenv()

# Create Chat Model
model: list[str] = [
    "llama3.2:3b",  # For simple, quick tasks
    "gemma3:4b",  # For balanced performance
    "openthinker:7b",  # For better reasoning with moderate resources
    "deepseek-r1:14b",  # For complex analysis and best quality
]
llm = ChatOllama(model=model[0])

# Create Chat Prompt Template
chat_prom_temp = ChatPromptTemplate(
    messages=[
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes"),
    ]
)

# Define additional processing steps using RunnableLambda
uppercase_output: RunnableLambda[Callable[[LanguageModelInput], Any], Any] = (
    RunnableLambda(func=lambda x: x.upper())
)

words_count: RunnableLambda[Callable[[LanguageModelInput], Any], Any] = RunnableLambda(
    func=lambda x: f"Word Count: {len(x.split())}\n{x}"
)

# Create combined Chain using LangChain Expression Language
chain: RunnableSerializable[dict[str, str | int], Any] = (
    chat_prom_temp | llm | StrOutputParser() | uppercase_output | words_count
)

# Run Chain
response: Any = chain.invoke(input={"topic": "lawyer", "joke_count": 3})
print(response)
