from langchain_community.embeddings.ollama import OllamaEmbeddings
import argparse
import os, shutil
from tqdm import tqdm
# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from config import input_variables
from datasets import load_dataset
from embedding import create_embedding

# calling the main function
main_class = create_embedding(input_variables)

# create embeddings
data = main_class.create_embeddings()

import pandas as pd
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

# Load embeddings and metadata from CSV
data = pd.read_csv('embeddings.csv')

# Initialize ChromaDB client
client = Chroma(
    chroma_db_impl="duckdb+parquet",
    persist_directory="path_to_persist_directory"
)

# Create or connect to a collection
collection_name = "gpt-4o-research-agent"
if collection_name not in client.list_collections():
    collection = client.create_collection(name=collection_name)
else:
    collection = client.get_collection(name=collection_name)

# Prepare data for ChromaDB
documents = data['content'].tolist()
metadatas = data[['title', 'content', 'arxiv_id', 'references']].to_dict(orient="records")
ids = data['id'].tolist()
embeddings = data['embeddings'].apply(eval).tolist()  # Convert string representation of list back to list

# Add to ChromaDB
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
    embeddings=embeddings
)

