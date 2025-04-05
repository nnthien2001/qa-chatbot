import os
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
)
from llama_index.vector_stores import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.output_parsers import StructuredOutputParser, ResponseSchema
from llama_index.prompts.chat_prompts import ChatPromptTemplate, ChatMessage, MessageRole
from typing import List

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
expected_files = [f"url_{i:03}.txt" for i in range(1, len(url_list) + 1)]
overlaps = set(txt_files) & set(expected_files)
missing_files = sorted(set(expected_files) - set(txt_files))
print(f"Missing files: {missing_files}")

# Step 3: Initialize index and storage
# Create service context with embedding and LLM
embed_model = OllamaEmbedding(model=embedding_model, timeout=timeout)
llm = Ollama(model=llm_model, timeout=timeout)
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

# Load documents from the directory
documents = SimpleDirectoryReader(txt_file_dir).load_data()

# Create a persistent Chroma client and collection in the database
db = chromadb.PersistentClient(path=index_storage_path)
collection = db.get_or_create_collection(name=collection_name)

# Assign Chroma as vector store and create an index from documents
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context,
    show_progress=True,
)

# Step 4: Create a chat memory buffer to maintain conversation history
memory_buffer = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Step 5: Define the response schema and structured output parser
response_schemas = [
    ResponseSchema(
        name="response",
        description="A detailed answer to the user's question.",
    ),
    ResponseSchema(
        name="references",
        description="A list of references used to generate the answer.",
    ),
    ResponseSchema(
        name="follow_up_questions",
        description="A list of 3 relevant follow-up questions for the user.",
    ),
]

output_parser = StructuredOutputParser(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Create a custom prompt template for structured responses
chat_prompt_template = ChatPromptTemplate(
    messages=[
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant. Answer the user's question using the provided format.",
        ),
        ChatMessage(role=MessageRole.USER, content="{user_input}"),
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Please provide your response in the following format:\n"
                f"{format_instructions}"
            ),
        ),
    ]
)

# Step 6: Create a chat engine from the index with memory and custom prompts
chat_engine = index.as_chat_engine(
    memory=memory_buffer,
    chat_prompt_template=chat_prompt_template,
    output_parser=output_parser,
    service_context=service_context,
)

# Step 7: Initialize the evaluator LLM
evaluator_llm = Ollama(model=evaluator_model, timeout=timeout)

# Step 8: Define a custom response class for the evaluator
class EvaluatorResponse:
    def __init__(self, verdict: bool, reason: str):
        self.verdict = verdict
        self.reason = reason

# Step 9: Function to evaluate the chatbot's response using concatenated source texts
def evaluate_response(
    user_input: str, chatbot_response: str, source_texts: List[str]
) -> EvaluatorResponse:
    concatenated_sources = "\n\n".join(source_texts[:5])  # Limit to avoid exceeding context
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

    # Generate the evaluation result
    evaluation_result = evaluator_llm.predict(prompt)

    # Parse the structured response
    verdict = False
    reason = ""
    for line in evaluation_result.strip().split('\n'):
        if line.lower().startswith('verdict:'):
            verdict_text = line.split(':', 1)[1].strip().lower()
            verdict = 'true' in verdict_text
        elif line.lower().startswith('reason:'):
            reason = line.split(':', 1)[1].strip()
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

        # The response is already parsed into a structured format
        structured_response = response.response

        # Extract main response, references, and follow-up questions
        main_response = structured_response.get('response', '')
        references = structured_response.get('references', '')
        follow_up_questions = structured_response.get('follow_up_questions', [])

        # Print the main response
        print(f"Chatbot: {main_response}")

        # Print references if available
        if references:
            print("\nReferences:")
            print(references)

        # Print follow-up questions if available
        if follow_up_questions:
            print("\nFollow-up Questions:")
            for q in follow_up_questions:
                print(f"- {q}")

        # Collect all source texts for evaluation
        source_texts = []
        if hasattr(response, 'source_nodes'):
            source_texts = [node.get_text() for node in response.source_nodes]

        # Evaluate the response
        evaluation = evaluate_response(user_input, main_response, source_texts)
        print("\nEvaluation:")
        print(f"Verdict: {'Grounded' if evaluation.verdict else 'Not Grounded'}")
        print(f"Reason: {evaluation.reason}")

        print("\n" + "-" * 50 + "\n")

# Run the chatbot function
if __name__ == "__main__":
    chatbot()
