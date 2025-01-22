# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/


from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Invoke model with input prompt
result = model.invoke(input="Divide 81 by 9")
print(result.content)
