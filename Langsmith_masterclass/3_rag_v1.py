# pip install -U langchain langchain-groq langchain-community faiss-cpu pypdf python-dotenv

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()  
os.environ["LANGCHAIN_PROJECT"] = "RAG Chatbot"

PDF_PATH = "islr.pdf"  # <-- change to your PDF filename

#Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page

#Split documents into Chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

#Create Embedding Model
model_kwargs = {'device': 'cpu'} # Use 'cuda' if a GPU is available
encode_kwargs = {'normalize_embeddings': False} # Set to True for normalized embeddings

embed_model = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs=model_kwargs,
                                    encode_kwargs=encode_kwargs)
vs = FAISS.from_documents(splits, embed_model)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

#Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

#Create Chain
llm = ChatGroq(model = "gemma2-9b-it", temperature = 0.7)

def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

#Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)
