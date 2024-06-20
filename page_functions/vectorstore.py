import os
import shutil

from langchain_chroma import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# llm = ChatCohere(model="command-r")


def create_vectorstore(urls):
    vectorestore_folder = "vectorstore"
    loader = WebBaseLoader(
        web_paths=urls,
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    if os.path.isdir(vectorestore_folder):
        shutil.rmtree(vectorestore_folder)
        os.makedirs(vectorestore_folder)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=CohereEmbeddings(),
        persist_directory=vectorestore_folder,
    )
    return vectorestore_folder
