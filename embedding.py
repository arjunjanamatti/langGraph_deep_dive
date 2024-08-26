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

class main:
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
    
    def split_documents(self, documents: list[Document]):
        # # First, use semantic chunking
        # semantic_splitter = SemanticTextSplitter(
        #     chunk_size = self.CHUNK_SIZE,
        #     chunk_overlap = self.CHUNK_OVERLAP,
        #     length_function = len,
        #     is_separator_regex = False,
        # )

        semantic_chunker = SemanticChunker(self.get_embedding_function())
        semantic_chunks = semantic_chunker.split_documents(documents)

        # Then, use recursive chunking on the semantic chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.CHUNK_SIZE,
            chunk_overlap = self.CHUNK_OVERLAP,
            length_function = len,
            is_separator_regex = False,
        )
        final_chunks = []
        for chunk in semantic_chunks:
            final_chunks.extend(text_splitter.split_documents([chunk]))

        return final_chunks

# calling the main function
main_class = main(input_variables)

# loading the dataset
dataset = main_class.load_documents()

print(dataset)