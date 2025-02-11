from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Create Chat Ollama Model
ollama_model = ChatOllama(model="llama3.2:3b")

# Set Chat Prompt Template
chat_prom_temp = ChatPromptTemplate(
    messages=[
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes"),
    ]
)

# Create RunnableLambda
format_prompt = RunnableLambda(func=lambda x: chat_prom_temp.format_prompt(**x))
invoke_model = RunnableLambda(func=lambda x: ollama_model.invoke(input=x.to_messages()))  # type: ignore
parse_output = RunnableLambda(func=lambda x: x.content)  # type: ignore

# Create RunnableSequence Chain
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the Chain
response = chain.invoke(input={"topic": "cats", "joke_count": 3})

# Output
print(response)
