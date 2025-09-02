from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatGroq(model = "gemma2-9b-it")
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Russia?"})
print(result)
