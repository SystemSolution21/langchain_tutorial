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
ollama_model = ChatOllama(model="llama3.2:3b")

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

words_counted: RunnableLambda[Callable[[LanguageModelInput], Any], Any] = (
    RunnableLambda(func=lambda x: f"Word Count: {len(x.split())}\n{x}")
)


# Create combined Chain using LangChain Expression Language
chain = (
    chat_prom_temp | ollama_model | StrOutputParser() | uppercase_output | words_counted
)

# Run Chain
response: Any = chain.invoke(input={"topic": "lawyer", "joke_count": 3})
print(response)
