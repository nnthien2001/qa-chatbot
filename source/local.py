# %%
!ollama list

# %%
import os
import json
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage, MessageRole, PromptTemplate
from typing import List, Dict, Any
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, CitationQueryEngine
from llama_index.core.types import BaseModel
from pydantic import BaseModel
from llama_index.core import Document
from llama_index.vector_stores.chroma import ChromaVectorStore

# %%
with open('configs/gcp.env', 'r') as json_file:
    config = json.load(json_file)

# %%
print(config)

# %%
doc_path = config['data_dir'] + "/" + config['doc_folder'] + "/" + config["collection"]
print(doc_path)

# %%
# Step 2: Load text files and URLs
doc_files = os.listdir(doc_path)
print(f"Number of doc_files: {len(doc_files)}")

# %%
index_path = config['data_dir'] + "/" + config['index_folder'] + "/" + config["collection"] + ".txt"
print(index_path)

# %%
with open(index_path, 'r') as f:
    url_list = f.read().splitlines()

# %%
# Check for missing files between txt files and URL list
overlaps = set([i for i in range(1, len(doc_files)+1) if f"url_{i:03}.txt" in doc_files])
missing = sorted(set([i for i in range(1, len(doc_files)+1)]) - overlaps)
print(f"Missing: {missing}")

# %%
timeout = int(config['timeout'])
print(f"Timeout: {timeout}")

# %%
# Step 3: Initialize index and storage
Settings.embed_model = OllamaEmbedding(config["embedding_model"], timeout=timeout)
Settings.llm = Ollama(model=config["llm_model"], timeout=timeout)

# %%
# Load documents from the directory
documents = SimpleDirectoryReader(doc_path).load_data()

# %%
# Create a persistent Chroma client and collection in the database
storage_path = config['data_dir'] + "/" + config['storage_folder'] + "/" + config["collection"]
print(storage_path)
db = chromadb.PersistentClient(storage_path)

# %%
collection_name = config["collection"].replace("-", "_")
print(collection_name)

# %%
try:
    # Attempt to delete the existing collection
    db.delete_collection(name=collection_name)
    print(f"Collection '{collection_name}' has been deleted.")
except ValueError as e:
    # Handle case where collection does not exist
    if "does not exist" in str(e):
        print(f"No existing collection named '{collection_name}' to delete.")
    else:
        raise e  # Re-raise unexpected errors

# %%
collection = db.create_collection(collection_name)

# %%
# Assign Chroma as vector store and create an index from documents
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# %%
doc_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)

# %%
doc_retriever = doc_index.as_retriever(
    similarity_top_k=config["top_docs"],
    verbose=True
)

# %%

response_prompt_template = PromptTemplate(
    "Câu hỏi gốc: {query_str}\n"
    "Ngữ cảnh đi kèm: {orig_query}\n"
    "Tài liệu tham khảo:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Trả lời câu hỏi gốc dưới đây một cách chi tiết, biết ngữ cảnh đi kèm của câu hỏi được đính vào phía dưới đây cũng như các tài liệu tham khảo để trả lời câu hỏi được cung cấp sẵn. Câu trả lời phải thật rõ ràng, cụ thể và đầy đủ.\n"
    "Câu trả lời bằng tiếng Việt: "
)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    text_qa_template=response_prompt_template,
    structured_answer_filtering=True,
    verbose=True
)


# %%

query_prompt_template = PromptTemplate(
    "Câu truy vấn của người dùng: {query_str}\n"
    "Cuộc trò chuyện quá khứ như sau:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Dựa trên dữ liệu cuộc trò chuyện, hãy viết lại câu truy vấn của người dùng dưới dạng một câu hỏi duy nhất có ngữ cảnh sao cho không cần biết cuộc trò chuyện trước đó vẫn đọc hiểu câu truy vấn. Nếu cuộc trò chuyện quá khứ không đủ để làm rõ câu truy vấn, giữ lại y nguyên câu truy vấn. Câu truy vấn phải được viết lại thành một câu hỏi duy nhất mà không kèm câu trả lời.\n"
    "Câu hỏi bằng tiếng Việt: "
)

query_synthesizer = get_response_synthesizer(
    response_mode="refine",
    text_qa_template=query_prompt_template,
    # structured_answer_filtering=True,
)


# %%
question_prompt_template = PromptTemplate("""\
Với ngữ liệu tham khảo dưới đây:
---------------------
{context_str}
---------------------
Dựa trên những ngữ liệu tham khảo này, hãy đưa ra chỉ những câu hỏi mà không kèm câu trả lời cho câu truy vấn dưới đây.
Mỗi câu hỏi phải được viết dưới dạng câu hỏi hoàn chỉnh và cụ thể nhất có thể. Câu truy vấn:
{query_str}
Chỉ đưa ra các câu hỏi bằng tiếng Việt:
"""
)

# %%
# Step 10: Implement chatbot interaction loop with all features
def chatbot():
    print("Chatbot is ready! Type 'exit' to end the chat.")
    past_query = ""
    conversation_iter = 0
    mem_index = VectorStoreIndex([], show_progress=False)
    
    while True:
        mem_retriever = mem_index.as_retriever(
            similarity_top_k=config["top_buffer"],
            verbose=True,
        )
        
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        mem_nodes = mem_retriever.retrieve(user_input)
        query_input = query_synthesizer.synthesize(user_input, mem_nodes)
        
        if len(query_input.source_nodes) == 0:
            query_inp = user_input
        else:
            query_inp = query_input.response
            
        print("Query: ", query_inp)
        doc_nodes = doc_retriever.retrieve(query_inp)
        
        chatbot_response = response_synthesizer.synthesize(query_inp, doc_nodes, original_query=user_input)
        response = chatbot_response.response
        print("Chatbot: ", response)
        
        query = f"Người dùng: {user_input}\n" + f"Cụ thể hơn: {user_input}\n" + f"Chatbot: {response}\n"
        concat_query = "Dữ liệu cuộc trò chuyện quá khứ\n." + past_query + "\nLượt kế:\n" + query
        
        # print("-"*20 + "References" + "-"*20)
        # print(concat_query)            
        # print("-"*20 + "End References" + "-"*20)
        
        doc = Document(text=concat_query, doc_id=f"turn_{conversation_iter}")
        mem_index.insert(doc)
        
        adjusted_ques_num = 2 * config["max_questions"] + 1
        dg = RagDatasetGenerator.from_documents([doc], text_question_template=question_prompt_template,
                                                num_questions_per_chunk=adjusted_ques_num)
        questions = dg.generate_questions_from_nodes().to_pandas()
        followups = questions.loc[questions['query'].str.endswith('?'), "query"].values
        
        print("Follow-ups", followups)
        
        past_query = query
        conversation_iter += 1

# %%
import nest_asyncio
nest_asyncio.apply()

# %%
chatbot()

# %%


# %%



