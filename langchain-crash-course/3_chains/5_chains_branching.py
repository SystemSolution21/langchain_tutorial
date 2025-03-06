"""
This module implements a sentiment analysis chain using LangChain and local LLMs.
It uses Ollama as the local LLM provider with models like deepseek and llama.
"""

from typing import Callable, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableSerializable, Runnable
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Create Chat Models using local LLM provider
model: ChatOllama = ChatOllama(model="deepseek-r1:14b")
# Alternative model configuration:
# model: ChatOllama = ChatOllama(model="llama3.2:3b")

# Define chat prompt template for positive feedback
positive_feedback_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "You are a helpful assistant."),
        ("human", "Generate a thank note for this positive feedback: {feedback}."),
    ]
)

# Define chat prompt template for negative feedback
negative_feedback_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "You are a helpful assistant."),
        ("human", "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

# Define chat prompt template for neutral feedback
neutral_feedback_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

# Define chat prompt template for escalate feedback
escalate_feedback_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to human agent: {feedback}.",
        ),
    ]
)

# Define chat prompt template for feedback classification
classification_feedback_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Classify the sentiment of this feedback as positive, negative, neutral or escalate: {feedback}.",
        ),
    ]
)


runnable_branches: RunnableBranch[
    tuple[Runnable[Any, bool], Runnable[Any, Any]], Runnable[Any, Any]
] = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template
        | model
        | StrOutputParser(),  # Positive feedback chain
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template
        | model
        | StrOutputParser(),  # Negative feedback chain
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser(),  # Neutral feedback chain
    ),
    escalate_feedback_template | model | StrOutputParser(),  # Default
)


# Create Classification Chain
classification_chain = classification_feedback_template | model | StrOutputParser()

# Combine classification and response generation into one chain
chain = classification_chain | runnable_branches

# Run the chain with an example review
# Good review: "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review: "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review: "The product is okay. It works as expected but nothing exceptional."
# Default: "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = (
    "The product is terrible. It broke after just one use and the quality is very poor."
)
result: Runnable[Any, Any] = chain.invoke(input={"feedback": review})
# Output Result
print(result)
