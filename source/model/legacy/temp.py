<<<<<<< HEAD
import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.prompts import ChatPromptTemplate, ChatMessage, MessageRole
from typing import List, Dict, Any

# Step 1: Setup directories and models
txt_file_dir = './llm-qa/data/results/thoi-su'
index_storage_path = './chroma_db_news'
collection_name = 'thoi_su'
embedding_model = 'llama3.1:latest'
llm_model = 'llama3.1:latest'
evaluator_model = 'llama3.1:latest'
timeout = 120

# Step 2: Load text files and URLs
txt_files = os.listdir(txt_file_dir)
with open('./llm-qa/data/urls/thoi-su.txt', 'r') as f:
    url_list = f.read().splitlines()

# Check for missing files between txt files and URL list
overlaps = set([i for i in range(1, len(txt_files)+1) if f"url_{i:03}.txt" in txt_files])
missing = sorted(set([i for i in range(1, len(txt_files)+1)]) - overlaps)
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

# Step 4: Create a chat memory buffer to maintain conversation history
memory_buffer = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Step 5: Define a custom prompt template for structured responses with references and follow-up questions
chat_prompt_template = ChatPromptTemplate([
    ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant. Provide a response, references, and 3 follow-up questions."),
    ChatMessage(role=MessageRole.USER, content="User asked: {user_input}."),
    ChatMessage(role=MessageRole.SYSTEM, content="Response: {response}\nReferences: {references}\nFollow-up Questions:\n1. {q1}\n2. {q2}\n3. {q3}")
])

# Step 6: Create a chat engine from the index with memory and custom prompts
chat_engine = index.as_chat_engine(memory=memory_buffer, text_qa_template=chat_prompt_template)

# Step 7: Initialize the evaluator LLM
evaluator_llm = Ollama(model=evaluator_model, timeout=timeout)

# Step 8: Define a custom response class for the evaluator
class EvaluatorResponse:
    def __init__(self, verdict: bool, reason: str):
        self.verdict = verdict
        self.reason = reason

# Step 9: Function to evaluate the chatbot's response using concatenated source texts
def evaluate_response(user_input: str, chatbot_response: str, source_texts: List[str]) -> EvaluatorResponse:
    concatenated_sources = "\n\n".join(source_texts)
    prompt = f"""
    You are an evaluator responsible for assessing whether the chatbot's response is grounded in its source texts.
    
    User Input: {user_input}
    
    Chatbot Response: {chatbot_response}
    
    Source Texts: {concatenated_sources}
    
    Provide a verdict (True if grounded, False if not) and a reason for your assessment.
    Format your response as follows:
    Verdict: [True/False]
    Reason: [Your explanation]
    """
    
    evaluation_result = evaluator_llm.complete(prompt)
    
    # Parse the structured response
    lines = evaluation_result.text.strip().split('\n')
    verdict = lines[0].split(': ')[1].lower() == 'true'
    reason = lines[1].split(': ')[1]
    
    return EvaluatorResponse(verdict, reason)

# Step 10: Implement chatbot interaction loop with all features
def chatbot():
    print("Chatbot is ready! Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        # Chat with user input while maintaining context
        response = chat_engine.chat(user_input)
        
        # Extract main response, references, and follow-up questions
        main_response = response.response.split("References:")[0].strip()
        references = response.response.split("References:")[1].split("Follow-up Questions:")[0].strip()
        follow_up_questions = response.response.split("Follow-up Questions:")[1].strip().split("\n")
        
        # Print the main response
        print(f"Chatbot: {main_response}")
        
        # Print references
        print("\nReferences:")
        print(references)
        
        # Print follow-up questions
        print("\nFollow-up Questions:")
        for q in follow_up_questions:
            print(q)
        
        # Collect all source texts for evaluation
        source_texts = [node.text for node in response.source_nodes]
        
        # Evaluate the response
        evaluation = evaluate_response(user_input, main_response, source_texts)
        print("\nEvaluation:")
        print(f"Verdict: {'Grounded' if evaluation.verdict else 'Not Grounded'}")
        print(f"Reason: {evaluation.reason}")
        
        print("\n" + "-"*50 + "\n")

# Run the chatbot function
if __name__ == "__main__":
=======
import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.prompts import ChatPromptTemplate, ChatMessage, MessageRole
from typing import List, Dict, Any

# Step 1: Setup directories and models
txt_file_dir = './llm-qa/data/results/thoi-su'
index_storage_path = './chroma_db_news'
collection_name = 'thoi_su'
embedding_model = 'llama3.1:latest'
llm_model = 'llama3.1:latest'
evaluator_model = 'llama3.1:latest'
timeout = 120

# Step 2: Load text files and URLs
txt_files = os.listdir(txt_file_dir)
with open('./llm-qa/data/urls/thoi-su.txt', 'r') as f:
    url_list = f.read().splitlines()

# Check for missing files between txt files and URL list
overlaps = set([i for i in range(1, len(txt_files)+1) if f"url_{i:03}.txt" in txt_files])
missing = sorted(set([i for i in range(1, len(txt_files)+1)]) - overlaps)
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

# Step 4: Create a chat memory buffer to maintain conversation history
memory_buffer = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Step 5: Define a custom prompt template for structured responses with references and follow-up questions
chat_prompt_template = ChatPromptTemplate([
    ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant. Provide a response, references, and 3 follow-up questions."),
    ChatMessage(role=MessageRole.USER, content="User asked: {user_input}."),
    ChatMessage(role=MessageRole.SYSTEM, content="Response: {response}\nReferences: {references}\nFollow-up Questions:\n1. {q1}\n2. {q2}\n3. {q3}")
])

# Step 6: Create a chat engine from the index with memory and custom prompts
chat_engine = index.as_chat_engine(memory=memory_buffer, text_qa_template=chat_prompt_template)

# Step 7: Initialize the evaluator LLM
evaluator_llm = Ollama(model=evaluator_model, timeout=timeout)

# Step 8: Define a custom response class for the evaluator
class EvaluatorResponse:
    def __init__(self, verdict: bool, reason: str):
        self.verdict = verdict
        self.reason = reason

# Step 9: Function to evaluate the chatbot's response using concatenated source texts
def evaluate_response(user_input: str, chatbot_response: str, source_texts: List[str]) -> EvaluatorResponse:
    concatenated_sources = "\n\n".join(source_texts)
    prompt = f"""
    You are an evaluator responsible for assessing whether the chatbot's response is grounded in its source texts.
    
    User Input: {user_input}
    
    Chatbot Response: {chatbot_response}
    
    Source Texts: {concatenated_sources}
    
    Provide a verdict (True if grounded, False if not) and a reason for your assessment.
    Format your response as follows:
    Verdict: [True/False]
    Reason: [Your explanation]
    """
    
    evaluation_result = evaluator_llm.complete(prompt)
    
    # Parse the structured response
    lines = evaluation_result.text.strip().split('\n')
    verdict = lines[0].split(': ')[1].lower() == 'true'
    reason = lines[1].split(': ')[1]
    
    return EvaluatorResponse(verdict, reason)

# Step 10: Implement chatbot interaction loop with all features
def chatbot():
    print("Chatbot is ready! Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        # Chat with user input while maintaining context
        response = chat_engine.chat(user_input)
        
        # Extract main response, references, and follow-up questions
        main_response = response.response.split("References:")[0].strip()
        references = response.response.split("References:")[1].split("Follow-up Questions:")[0].strip()
        follow_up_questions = response.response.split("Follow-up Questions:")[1].strip().split("\n")
        
        # Print the main response
        print(f"Chatbot: {main_response}")
        
        # Print references
        print("\nReferences:")
        print(references)
        
        # Print follow-up questions
        print("\nFollow-up Questions:")
        for q in follow_up_questions:
            print(q)
        
        # Collect all source texts for evaluation
        source_texts = [node.text for node in response.source_nodes]
        
        # Evaluate the response
        evaluation = evaluate_response(user_input, main_response, source_texts)
        print("\nEvaluation:")
        print(f"Verdict: {'Grounded' if evaluation.verdict else 'Not Grounded'}")
        print(f"Reason: {evaluation.reason}")
        
        print("\n" + "-"*50 + "\n")

# Run the chatbot function
if __name__ == "__main__":
>>>>>>> 8bca85a659065dcca55cd508bd6b7211260d4a21
    chatbot()