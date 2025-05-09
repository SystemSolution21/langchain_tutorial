from typing import Callable, Any, TypedDict, Dict, cast
from langchain_core.prompt_values import PromptValue
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableParallel,
    RunnableSerializable,
)
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

# Set Prompt Template
chat_prompt_template: ChatPromptTemplate = ChatPromptTemplate(
    messages=[
        ("system", "Your are an expert product reviewer."),
        ("human", "List the main features of the product {product_name}."),
    ]
)


# Define pros analysis step
def analyze_pros(features) -> PromptValue:
    chat_prompt_template: ChatPromptTemplate = ChatPromptTemplate(
        messages=[
            ("system", "Your are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the pros of these features.",
            ),
        ]
    )
    return chat_prompt_template.format_prompt(features=features)


# Define cons analysis step
def analyze_cons(features) -> PromptValue:
    chat_prompt_template: ChatPromptTemplate = ChatPromptTemplate(
        messages=[
            ("system", "Your are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the cons of these features.",
            ),
        ]
    )
    return chat_prompt_template.format_prompt(features=features)


# Simplify pros branches chain with LangChain Expression Language (LCEL)
pros_branch_chain: RunnableSerializable[Any, str] = (
    RunnableLambda(func=lambda x: analyze_pros(features=x)) | llm | StrOutputParser()
)

# Simplify cons branches chain with LangChain Expression Language (LCEL)
cons_branch_chain: RunnableSerializable[Any, str] = (
    RunnableLambda(func=lambda x: analyze_cons(features=x)) | llm | StrOutputParser()
)


# Define the expected structure of the branches
class BranchesDict(TypedDict):
    pros: str
    cons: str


# Define the input structure
class InputDict(TypedDict):
    branches: BranchesDict


# Combine pros and cons branches with RunnableLambda
combine_pros_cons_branch = RunnableLambda(
    func=lambda x: pros_cons_branch(
        pros=cast(InputDict, x)["branches"]["pros"],
        cons=cast(InputDict, x)["branches"]["cons"],
    )
)


# Return pros and cons into final review
def pros_cons_branch(pros: str, cons: str) -> str:
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


# Create combined chain using LangChain Expression Language (LCEL)
chain: RunnableSerializable[dict[str, str], str] = (
    chat_prompt_template
    | llm
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | combine_pros_cons_branch
)


# Run the chain
result: str = chain.invoke(input={"product_name": "MacBook Pro"})

# Output
print(result)
