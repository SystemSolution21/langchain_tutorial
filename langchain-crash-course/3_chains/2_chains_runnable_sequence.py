from typing import Callable, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.language_models import LanguageModelInput
from langchain_ollama import ChatOllama

# Load Environment Variables
load_dotenv()

# Create Chat Model
llm = ChatOllama(model="llama3.2:3b")

# Set Chat Prompt Template
chat_prom_temp = ChatPromptTemplate(
    messages=[
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes"),
    ]
)

# Create RunnableLambda for prompt format, model invoke and output parse
format_prompt: RunnableLambda[Callable[[LanguageModelInput], Any], Any] = (
    RunnableLambda(func=lambda x: chat_prom_temp.format_prompt(**x))
)

invoke_llm: RunnableLambda[Callable[[LanguageModelInput], Any], Any] = RunnableLambda(
    func=lambda x: llm.invoke(input=x.to_messages())
)

parse_output: RunnableLambda[Callable[[LanguageModelInput], Any], Any] = RunnableLambda(
    func=lambda x: x.content
)

# Create RunnableSequence Chain
chain = RunnableSequence(first=format_prompt, middle=[invoke_llm], last=parse_output)

# Run the Chain
response: Any = chain.invoke(input={"topic": "cats", "joke_count": 3})

# Output
print(response)
