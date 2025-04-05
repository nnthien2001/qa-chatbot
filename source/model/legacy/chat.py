import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Step 1: Setup directories and models
txt_file_dir = './llm-qa/data/results/thoi-su'
index_storage_path = './chroma_db_news'
collection_name = 'thoi_su'
embedding_model = 'llama3.1:latest'
llm_model = 'llama3.1:latest'
timeout = 120

# Step 2: Load text files and URLs
txt_files = os.listdir(txt_file_dir)
with open('./llm-qa/data/urls/thoi-su.txt', 'r') as f:
    url_list = f.read().splitlines()

# Check for missing files between txt files and URL list
overlaps = set([i for i in range(1, len(txt_files) + 1) if f"url_{i:03}.txt" in txt_files])
missing = sorted(set([i for i in range(1, len(txt_files) + 1)]) - overlaps)
print(f"Missing: {missing}")

# Step 3: Initialize index and storage
Settings.embed_model = OllamaEmbedding(embedding_model)
Settings.llm = Ollama(model=llm_model, timeout=timeout)

# Load documents from the directory
documents = SimpleDirectoryReader(txt_file_dir).load_data()

# Create a persistent Chroma client and collection in the database
db = chromadb.PersistentClient(index_storage_path)
collection = db.get_or_create_collection(collection_name)

# Assign Chroma as vector store and create an index from documents
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context, show_progress=True)

# Step 4: Create a query engine
query_engine = index.as_query_engine()

# Step 5: Implement chatbot interaction loop
def chatbot():
    print("Chatbot is ready! Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        # Query the index with user input
        response = query_engine.query(user_input)
        
        # Print the response from the query engine
        print(f"Chatbot: {response.response}")

# Run the chatbot function
if __name__ == "__main__":
    chatbot()