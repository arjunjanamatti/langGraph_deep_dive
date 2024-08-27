from langchain_community.embeddings.ollama import OllamaEmbeddings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import input_variables
from datasets import load_dataset
import time

class create_embedding:
    def __init__(self, input_variables) -> None:
        # define the variables from config file
        self.CHROMA_PATH = input_variables.variables.CHROMA_PATH
        self.DATA_PATH = input_variables.variables.DATA_PATH
        self.DATASET_URL = input_variables.variables.DATASET_URL
        self.EMBEDDING_model = input_variables.variables.EMBEDDING_MODEL
        self.CHUNK_SIZE = input_variables.variables.CHUNK_SIZE
        self.CHUNK_OVERLAP = input_variables.variables.CHUNK_OVERLAP

    def get_embedding_function(self):
        embeddings = OllamaEmbeddings(model=self.EMBEDDING_model)
        return embeddings

    def load_documents(self):
        dataset = load_dataset(self.DATASET_URL, split="train")
        return dataset

    def embed_single_text(self, embedder, text):
        retries = 3
        for attempt in range(retries):
            try:
                return embedder._embed([text])[0]
            except Exception as e:
                print(f"Error embedding text: {e}. Retrying {attempt + 1}/{retries}...")
                time.sleep(2)
        raise ValueError("Failed to generate embedding after retries")

    def embed_texts(self, embedder, texts):
        embeddings = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.embed_single_text, embedder, text): text for text in texts}
            for future in tqdm(as_completed(futures), total=len(texts), desc="Embedding texts"):
                try:
                    embeddings.append(future.result())
                except Exception as e:
                    print(f"Error in thread: {e}")
        return embeddings

    def create_embeddings(self):
        # Initialize the embedder
        embedder = self.get_embedding_function()
        # get dataset
        dataset = self.load_documents()
        # Convert dataset to pandas dataframe
        data = dataset.to_pandas().iloc[:1000]
        # create embeddings with multithreading
        data['embeddings'] = self.embed_texts(embedder, data['content'].tolist())
        return data

# # calling the main function
# main_class = create_embedding(input_variables)

# # create embeddings
# data = main_class.create_embeddings()

# print(data.head(1))