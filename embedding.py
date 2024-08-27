from langchain_community.embeddings.ollama import OllamaEmbeddings
import argparse
import os, shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from config import input_variables
from datasets import load_dataset

class Embedder:
    def __init__(self, input_variables):
        self.embedder = OllamaEmbeddings(model=input_variables.variables.EMBEDDING_MODEL)

    def embed_single_text(self, text):
        return self.embedder._embed([text])[0]

class create_embedding:
    def __init__(self,input_variables) -> None:
        # define the variables from config file
        self.CHROMA_PATH = input_variables.variables.CHROMA_PATH
        self.DATA_PATH = input_variables.variables.DATA_PATH
        self.DATASET_URL = input_variables.variables.DATASET_URL
        self.EMBEDDING_model = input_variables.variables.EMBEDDING_MODEL
        self.CHUNK_SIZE = input_variables.variables.CHUNK_SIZE
        self.CHUNK_OVERLAP = input_variables.variables.CHUNK_OVERLAP

    def get_embedding_function(self):
        embeddings = OllamaEmbeddings(model = self.EMBEDDING_model)
        return embeddings
    
    def load_documents(self):
        dataset = load_dataset(self.DATASET_URL, split="train")
        return dataset
    
    def create_embeddings(self):
        # Initialize the embedder
        embedder = self.get_embedding_function()

        # get dataset
        dataset = self.load_documents()

        # Convert dataset to pandas dataframe
        data = dataset.to_pandas().iloc[:1000]
        
        # Function to embed a single text
        def embed_single_text(text):
            return embedder._embed([text])[0]

        # Function to embed texts with multiprocessing
        def embed_texts(texts):
            with Pool(cpu_count()) as pool:
                embeddings = list(tqdm(pool.imap(embed_single_text, texts), total=len(texts), desc="Embedding texts"))
            return embeddings

        # create embeddings
        data['embeddings'] = embed_texts(data['content'].tolist())

        return data

# calling the main function
main_class = create_embedding(input_variables)

# create embeddings
data = main_class.create_embeddings()

print(data.head(1))