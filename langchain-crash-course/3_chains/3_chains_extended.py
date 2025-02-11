from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
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


# Convert to Upper Case
def convert_to_upper_case(input) -> str:
    return input.upper()


# Count words
def count_words(input):
    return f"Word Count: {len(input.split())}\n {input}"


# Define additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(func=lambda x: convert_to_upper_case(input=x))
words_counted = RunnableLambda(func=lambda x: count_words(input=x))

# Create combined Chain using LangChain Expression Language
chain = (
    chat_prom_temp | ollama_model | StrOutputParser() | uppercase_output | count_words
)

# Run Chain
response = chain.invoke(input={"topic": "lawyer", "joke_count": 3})
print(response)
