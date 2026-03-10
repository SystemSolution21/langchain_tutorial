from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_ollama import ChatOllama

# Load Environment Variables
load_dotenv()

# Create Chat Model
llm = ChatOllama(model="llama3.2:latest")

# Set Chat Prompt Template
chat_prom_temp = ChatPromptTemplate(
    messages=[
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes"),
    ]
)

# Create RunnableLambda for prompt format, model invoke and output parse
format_prompt = RunnableLambda(func=lambda x: chat_prom_temp.format_prompt(**x))


def invoke_model(prompt: PromptValue) -> BaseMessage:
    return llm.invoke(input=prompt.to_messages())


invoke_llm = RunnableLambda(func=invoke_model)

parse_output = RunnableLambda(func=lambda x: x.content)

# Create RunnableSequence Chain
chain = RunnableSequence(first=format_prompt, middle=[invoke_llm], last=parse_output)

# Run the Chain
response: Any = chain.invoke(input={"topic": "cats", "joke_count": 3})

# Output
print(response)
