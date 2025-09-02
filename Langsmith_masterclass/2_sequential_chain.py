from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "Sequential LLM App"


prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatGroq(model = "gemma2-9b-it", temperature = 0.7)

model2 = ChatGroq(model = "meta-llama/llama-4-maverick-17b-128e-instruct", temperature = 0.5)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

#Add Tags and metadata 
config = {
    "run_name": "Sequential chain",
    "tags": ["llm app" ,"report parser", "summarization"],
    "metadata": {"model1": "gemaa2", "model_temperatre": 0.7, "parser": "stroutputparser"}
}

result = chain.invoke({'topic': 'Quantum Computing'}, config = config)

print(result)

