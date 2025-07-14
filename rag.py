
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings
from llms import assistant_llm
import os

# Load and split document
loader = TextLoader("company_doc.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Vector DB
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
db = FAISS.from_documents(docs, embeddings)

# Multi-query Retriever (LLM generates rephrased versions of the user's question)
base_retriever = db.as_retriever(search_kwargs={"k": 4})
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=assistant_llm,
)

# Build RAG chain (no memory)
qa_chain = RetrievalQA.from_chain_type(
    llm=assistant_llm,
    chain_type="stuff",
    retriever=multi_query_retriever,
    return_source_documents=True
)
