# Data -> Embeddings -> Store -> Retrieve

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

#vector1 = embeddings.embed_query(splittedDoc[0].page_content)
#vector2 = embeddings.embed_query(splittedDoc[1].page_content)

vectorDb = Chroma(
            collection_name="Python-Documentation",
            embedding_function=embeddings,
            persist_directory="./docVectorDb/"
        )

vectorDb.add_documents(documents=splittedDoc)


# Creating tools.

@tool
def retrieveContext(query: str):
    retrievedInfo = vectorDb.similarity_search_with_score(query, k=2)
    return retrievedInfo

tools = [retrieveContext]
prompt = (
            """
            You are Alfred from the movie 'The Batman' that was released in 
            2022. You have a tool that helps in retrieving information about
            Python programming language.
            Your task is to help with any queries that the user has. Treat
            the user like they are Bruce Wayne to you.
            """
        )

model = "mistral:instruct"

Alfred = create_agent(model, tools, system_prompt=prompt)

Query = input("=>")

for event in Alfred.stream(
        {"messages": [{"role": "user", "content": Query}]},
        stream_mode="values"
        ):
    event["messages"][-1].pretty_print()
