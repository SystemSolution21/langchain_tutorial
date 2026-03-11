# 2_chains_runnable_sequence.py

# Import standard libraries
import os
from typing import Any

# Import third-party libraries
from dotenv import load_dotenv

# Import langchain modules
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_ollama import ChatOllama

# Load Environment Variables
load_dotenv()

model: str = os.getenv(key="OLLAMA_LLM", default="llama3.2:latest")

# Create Chat Model
llm = ChatOllama(model=model)

# Set Chat Prompt Template
chat_prom_temp = ChatPromptTemplate(
    messages=[
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes"),
    ]
)

# Create RunnableLambda for prompt format
format_prompt: RunnableLambda[dict[str, str | int], PromptValue] = RunnableLambda(
    func=lambda x: chat_prom_temp.format_prompt(**x)
)


def invoke_model(prompt: PromptValue) -> BaseMessage:
    """Invoke the model with the given prompt."""
    return llm.invoke(input=prompt.to_messages())


# Create RunnableLambda for model invoke
invoke_llm: RunnableLambda[PromptValue, BaseMessage] = RunnableLambda(func=invoke_model)

# Create RunnableLambda for output parse
parse_output: RunnableLambda[BaseMessage, str] = RunnableLambda(
    func=lambda x: x.content
)

# Create RunnableSequence Chain
chain: RunnableSequence[dict[str, str | int], str] = RunnableSequence(
    first=format_prompt, middle=[invoke_llm], last=parse_output
)

# Run the Chain
response: Any = chain.invoke(input={"topic": "cats", "joke_count": 3})

# Output
print(response)
