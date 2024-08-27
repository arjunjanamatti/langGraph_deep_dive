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
import pandas as pd
from chromadb.config import Settings


class populate_db:
    def __init__(self,input_variables) -> None:
        self.CHROMA_PATH = input_variables.variables.CHROMA_PATH
        self.DATA_PATH = input_variables.variables.DATA_PATH
        self.DATASET_URL = input_variables.variables.DATASET_URL
        self.EMBEDDING_model = input_variables.variables.EMBEDDING_MODEL
        self.CHUNK_SIZE = input_variables.variables.CHUNK_SIZE
        self.CHUNK_OVERLAP = input_variables.variables.CHUNK_OVERLAP
        self.COLLECTION_NAME = input_variables.variables.COLLECTION_NAME

    def get_embeddings(self):
        # calling the main function
        main_class = create_embedding(input_variables)
        # create embeddings
        data = main_class.create_embeddings()
        return data
    
    def initialize_db(self):
        # Initialize ChromaDB client
        client = Chroma(
            chroma_db_impl="duckdb+parquet",
            persist_directory = input_variables.variables.CHROMA_PATH
        )

        return client
    
    def create_collection(self):
        # get the client
        client = self.initialize_db()   
        # Create or connect to a collection
        collection_name = self.COLLECTION_NAME
        if collection_name not in client.list_collections():
            collection = client.create_collection(name=collection_name)
        else:
            collection = client.get_collection(name=collection_name) 
        return collection

    def connect_and_add(self):
        # get the client
        collection = self.create_collection()

        # get the data with embeddings
        data = self.get_embeddings()

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






# # Initialize ChromaDB client
# client = Chroma(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory="path_to_persist_directory"
# )

# # Create or connect to a collection
# collection_name = "gpt-4o-research-agent"
# if collection_name not in client.list_collections():
#     collection = client.create_collection(name=collection_name)
# else:
#     collection = client.get_collection(name=collection_name)

# # Prepare data for ChromaDB
# documents = data['content'].tolist()
# metadatas = data[['title', 'content', 'arxiv_id', 'references']].to_dict(orient="records")
# ids = data['id'].tolist()
# embeddings = data['embeddings'].apply(eval).tolist()  # Convert string representation of list back to list

# # Add to ChromaDB
# collection.add(
#     documents=documents,
#     metadatas=metadatas,
#     ids=ids,
#     embeddings=embeddings
# )

