from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama
from langchain.schema.runnable import RunnableSerializable
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Create ChatOllama Model
model = ChatOllama(model="llama3.2:3b", temperature=0.8, num_predict=256)

# Set Chat Prompt Template
chat_prom_temp: ChatPromptTemplate = ChatPromptTemplate(
    messages=[
        ("system", "Your are comedian who tells jokes about {topic}."),
        ("user", "Tell me {joke_count} jokes."),
    ]
)

# Create Chain using LangChain Expression Language (LCEL)
chain: RunnableSerializable[dict[str, str | int], str] = (
    chat_prom_temp | model | StrOutputParser()
)
# chain = chat_prom_temp | model

response: str = chain.invoke(input={"topic": "Python Programming", "joke_count": 3})

print(response)
