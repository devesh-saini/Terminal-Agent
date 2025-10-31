# Data -> Embeddings -> Store -> Retrieve

import requests
import chromadb
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Loading Documents.

pythonDocs = PyPDFLoader("./PDFs/pythonDocumentation.pdf")
docs = pythonDocs.load()


# Splitting Docs.

textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )

splittedDoc = textSplitter.split_documents(docs)


# Creating Embeddings and storing splittedDoc in chromadb.

embeddings = OllamaEmbeddings(model="mistral:instruct")

vector1 = embeddings.embed_query(splittedDoc[0].page_content)
vector2 = embeddings.embed_query(splittedDoc[1].page_content)

vectorDb = Chroma(
            collection_name="Python-Documentation",
            embedding_function=embeddings,
            persist_directory="./docVectorDb/"
        )

vectorDb.add_documents(documents=splittedDoc)


