from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableSerializable
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Load Environment Variables
load_dotenv()

# Create Chat Model
llm = ChatOllama(model="llama3.2:latest", temperature=0.8, num_predict=256)

# Set Chat Prompt Template
chat_prom_temp: ChatPromptTemplate = ChatPromptTemplate(
    messages=[
        ("system", "Your are comedian who tells jokes about {topic}."),
        ("user", "Tell me {joke_count} jokes."),
    ]
)

# Create Chain using LangChain Expression Language (LCEL)
chain: RunnableSerializable[dict[str, str | int], str] = (
    chat_prom_temp | llm | StrOutputParser()
)

response: str = chain.invoke(input={"topic": "Python Programming", "joke_count": 3})

print(response)
