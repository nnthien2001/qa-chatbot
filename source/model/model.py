import json
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from llama_index.core import get_response_synthesizer, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from typing import Dict, Any, List, Tuple
import nest_asyncio
import streamlit as st
nest_asyncio.apply()

@st.cache_resource(show_spinner='Tải cấu hình chatbot...')
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a JSON file and returns its content as a dictionary.

    This function is a wrapper around the built-in `json` module's
    `load` function. It takes a path to a JSON file as input and
    returns the content of the file as a dictionary.

    The function is cached using Streamlit's `@st.cache_resource`
    decorator. This means that the file will only be loaded once,
    and the result will be stored in memory. Subsequent calls to
    this function will return the cached result instead of loading
    the file again.

    The `show_spinner` argument is set to a string that will be
    displayed as a spinner while the function is running. In this
    case, the string is 'T i cấu hình chatbot...'.

    Args:
        config_path (str): The path to the JSON file.

    Returns:
        Dict[str, Any]: The configuration dictionary.
    """
    with open(config_path, 'r') as json_file:
        # Open the file in read mode and assign it to the variable
        # `json_file`.
        config = json.load(json_file)
        # Load the JSON file using the `json` module's `load` function.
        # The result is a dictionary.
    return config
    # Return the configuration dictionary.

def initialize_settings(config: Dict[str, Any]) -> None:
    """
    Initialize the settings for the embedding and LLM models.

    This function sets the embedding model and LLM model settings
    based on the configuration dictionary. It also sets the timeout
    value for the models.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
    """
    timeout = int(config['timeout'])
    # Set the timeout value for the models.
    Settings.embed_model = OllamaEmbedding(
        config["embedding_model"],
        timeout=timeout
    )
    # Initialize the embedding model with the specified model name
    # and timeout value.
    Settings.llm = Ollama(
        model=config["llm_model"],
        timeout=timeout
    )
    # Initialize the LLM model with the specified model name and
    # timeout value.

def load_documents(doc_path: str) -> List[Document]:
    """Load documents from the specified directory."""
    return SimpleDirectoryReader(doc_path).load_data()

def create_chroma_collection(config: Dict[str, Any]) -> chromadb.PersistentClient:
    """Create a Chroma collection, deleting any existing collection with the same name."""
    storage_path = config['data_dir'] + "/" + config['storage_folder'] + "/" + config["collection"]
    db = chromadb.PersistentClient(storage_path)
    collection_name = config["collection"].replace("-", "_")
    try:
        db.delete_collection(name=collection_name)
    except ValueError as e:
        if "does not exist" not in str(e):
            raise e
    collection = db.create_collection(collection_name,
                                      metadata={"hnsw:space": config["embedding_space"]})
    return collection

def create_vector_store_and_index(documents: List[Document], collection: chromadb.PersistentClient) -> VectorStoreIndex:
    """Create a vector store and index from the documents."""
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    doc_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    return doc_index

def create_prompt_templates() -> Tuple[PromptTemplate, PromptTemplate, PromptTemplate]:
    """Create prompt templates for response and query synthesis."""
    response_prompt_template = PromptTemplate(
        "Câu hỏi gốc: {orig_query}\n"
        "Ngữ cảnh đi kèm: {query_str}\n"
        "Tài liệu tham khảo:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Trả lời câu hỏi gốc dưới đây một cách chi tiết, biết ngữ cảnh đi kèm của câu hỏi được đính vào phía dưới đây cũng như các tài liệu tham khảo để trả lời câu hỏi được cung cấp sẵn. Câu trả lời phải thật rõ ràng, cụ thể và đầy đủ.\n"
        "Câu trả lời bằng tiếng Việt: "
    )
    query_prompt_template = PromptTemplate(
        "Câu truy vấn của người dùng: {query_str}\n"
        "Cuộc trò chuyện quá khứ như sau:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Dựa trên dữ liệu cuộc trò chuyện, hãy viết lại câu truy vấn của người dùng dưới dạng một câu hỏi duy nhất có ngữ cảnh sao cho không cần biết cuộc trò chuyện trước đó vẫn đọc hiểu câu truy vấn. Nếu cuộc trò chuyện quá khứ không đủ để làm rõ câu truy vấn, giữ lại y nguyên câu truy vấn. Câu truy vấn phải được viết lại thành một câu hỏi duy nhất mà không kèm câu trả lời.\n"
        "Câu hỏi bằng tiếng Việt: "
    )
    question_prompt_template = PromptTemplate(
        "Với ngữ liệu tham khảo dưới đây:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Dựa trên những ngữ liệu tham khảo này, hãy đưa ra chỉ những câu hỏi mà không kèm câu trả lời cho câu truy vấn dưới đây.\n"
        "Mỗi câu hỏi phải được viết dưới dạng câu hỏi hoàn chỉnh và cụ thể nhất có thể. Câu truy vấn:\n"
        "{query_str}\n"
        "Chỉ đưa ra các câu hỏi bằng tiếng Việt:\n"
    )
    return response_prompt_template, query_prompt_template, question_prompt_template

def create_synthesizers(
    response_prompt_template: PromptTemplate, query_prompt_template: PromptTemplate
) -> Tuple[Any, Any]:
    """
    This function creates two synthesizers: a response synthesizer and a query synthesizer.

    The response synthesizer is used to generate a detailed response to a user's query.
    It takes the user's query and some documents as input, and generates a response that is
    relevant to the user's query and grounded in the documents. The response synthesizer
    uses a tree-structured neural network to generate the response, which is designed to
    capture the hierarchical structure of language.

    The query synthesizer is used to refine the user's query to make it more specific and clear.
    It takes the user's query and some documents as input, and generates a refined query that
    is more specific and clear than the original query. The query synthesizer uses a different
    neural network architecture than the response synthesizer, which is designed to refine
    the user's query without generating a response.

    Args:
        response_prompt_template (PromptTemplate): The prompt template for response synthesis.
            The prompt template is a string that contains placeholders for the user's query and
            the documents. The placeholders are replaced with the actual query and documents when
            the synthesizer is called.
        query_prompt_template (PromptTemplate): The prompt template for query synthesis.
            The prompt template is a string that contains placeholders for the user's query and
            the documents. The placeholders are replaced with the actual query and documents when
            the synthesizer is called.

    Returns:
        Tuple[Any, Any]: A tuple containing the response synthesizer and query synthesizer.
            The response synthesizer is an instance of the ResponseSynthesizer class, which is
            defined in the llama_index library. The query synthesizer is an instance of the
            QuerySynthesizer class, which is also defined in the llama_index library.
    """
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        text_qa_template=response_prompt_template,
        structured_answer_filtering=True,
        verbose=True
    )
    query_synthesizer = get_response_synthesizer(
        response_mode="refine",
        text_qa_template=query_prompt_template,
    )
    return response_synthesizer, query_synthesizer

def init(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize the chatbot with the given configuration.

    This function initializes the chatbot with the given configuration, loads the
    data from the specified directory, creates a vector store and index from the
    documents, and creates a retriever for the index. It also creates a response
    synthesizer and query synthesizer using the templates defined in the
    `create_prompt_templates` function. Finally, it initializes the memory buffer
    using the `init_memory` function.

    The memory buffer is initialized using the `init_memory` function, which
    creates an empty list of documents and no embeddings. This means that the memory
    buffer does not contain any information initially, and it will be populated with
    documents and embeddings as the conversation progresses.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing the initialized components.
    """
    # Initialize the chatbot with the given configuration
    cached_buffer = cached_init(config)

    # Initialize the memory buffer
    cached_buffer["mem_index"] = init_memory(config)

    # Return the initialized components
    return cached_buffer

@st.cache_resource(show_spinner='Khởi động chatbot...')
def cached_init(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize the chatbot with the given configuration.

    This function is cached using Streamlit's caching mechanism, so it will only be
    executed once. It initializes the chatbot with the given configuration, loads the
    data from the specified directory, creates a vector store and index from the
    documents, and creates a retriever for the index. It also creates a response
    synthesizer and query synthesizer using the templates defined in the
    `create_prompt_templates` function.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing the initialized components.
    """

    # Initialize settings from the configuration dictionary
    initialize_settings(config)

    # Load the documents from the specified directory
    doc_path = config['data_dir'] + "/" + config['doc_folder'] + "/" + config["collection"]
    documents = load_documents(doc_path)

    # Create a vector store and index from the documents
    collection = create_chroma_collection(config)
    doc_index = create_vector_store_and_index(documents, collection)

    # Create a retriever for the index
    doc_retriever = doc_index.as_retriever(similarity_top_k=config["top_docs"], verbose=True)

    # Create prompt templates for response and query synthesis
    response_prompt_template, query_prompt_template, question_prompt_template = create_prompt_templates()

    # Create response and query synthesizers using the templates
    response_synthesizer, query_synthesizer = create_synthesizers(response_prompt_template, query_prompt_template)

    # Initialize the conversation buffer
    past_query = ""
    conversation_iter = 0
    buffer = {
        "doc_retriever": doc_retriever,
        "past_query": past_query,
        "conversation_iter": conversation_iter,
        "response_synthesizer": response_synthesizer,
        "query_synthesizer": query_synthesizer,
        "question_prompt_template": question_prompt_template
    }
    return buffer

def init_memory(config: Dict[str, Any]) -> VectorStoreIndex:
    """Initialize the memory index with the given configuration.

    This function initializes an empty memory index with the given configuration.
    The memory index is a VectorStoreIndex, which is a data structure that stores a
    list of documents and their corresponding embeddings. The embeddings are used
    to compute similarities between the documents and the user's input.

    The memory index is initialized with an empty list of documents and no
    embeddings. This means that the memory index does not contain any information
    initially, and it will be populated with documents and embeddings as the
    conversation progresses.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        VectorStoreIndex: The initialized memory index.
    """
    # Create an empty list of documents
    documents = []

    # Create a VectorStoreIndex with the empty list of documents
    mem_index = VectorStoreIndex(documents, show_progress=False)

    # Return the initialized memory index
    return mem_index

def retrieve_memory_nodes(user_input: str, buffer: Dict[str, Any], config: Dict[str, Any]) -> List[Document]:
    """
    Retrieve memory nodes from the memory index.

    This function uses the memory index to retrieve the top documents that are similar to
    the user's input. The retrieved documents are then returned as a list of Document objects.

    Args:
        user_input (str): The user's input.
        buffer (Dict[str, Any]): The conversation buffer.
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        List[Document]: The retrieved memory nodes.
    """
    # Use the memory index to retrieve the top documents that are similar to the user's input
    mem_retriever = buffer["mem_index"].as_retriever(similarity_top_k=config["top_buffer"], verbose=True)
    # Retrieve the documents using the memory retriever
    mem_nodes = mem_retriever.retrieve(user_input)
    # Return the retrieved memory nodes
    return mem_nodes

def synthesize_query(
    user_input: str, mem_nodes: List[Document], buffer: Dict[str, Any]
) -> str:
    """
    Synthesize the query using the query synthesizer.

    This function takes the user's input, the retrieved memory nodes, and the
    conversation buffer as input. It uses the query synthesizer to synthesize
    the query from the user's input and the memory nodes. If the memory nodes are
    empty, it returns the user's input as the query. Otherwise, it returns the
    synthesized query.

    Args:
        user_input (str): The user's input.
        mem_nodes (List[Document]): The retrieved memory nodes.
        buffer (Dict[str, Any]): The conversation buffer.

    Returns:
        str: The synthesized query.
    """
    query_input = buffer["query_synthesizer"].synthesize(user_input, mem_nodes)
    return user_input if len(query_input.source_nodes) == 0 else query_input.response

def retrieve_document_nodes(query_inp: str, buffer: Dict[str, Any]) -> List[Document]:
    """
    Retrieve document nodes from the document retriever.

    This function takes the synthesized query and the conversation buffer as input,
    and uses the document retriever to retrieve a list of Document objects from the
    document index. The retrieved documents are then returned.

    Args:
        query_inp (str): The synthesized query.
        buffer (Dict[str, Any]): The conversation buffer.

    Returns:
        List[Document]: The retrieved document nodes.
    """
    return buffer["doc_retriever"].retrieve(query_inp)

def synthesize_response(
    query_inp: str, doc_nodes: List[Document], user_input: str, buffer: Dict[str, Any]
) -> Tuple[str, List[Document]]:
    """
    Synthesize the response using the response synthesizer and return the response along with source nodes.

    Args:
        query_inp (str): The synthesized query.
        doc_nodes (List[Document]): The retrieved document nodes.
        user_input (str): The user's input.
        buffer (Dict[str, Any]): The conversation buffer.

    Returns:
        Tuple[str, List[Document]]: The synthesized response and the source nodes.
    """
    # Use the response synthesizer to synthesize the response from the query and the document nodes
    chatbot_response = buffer["response_synthesizer"].synthesize(query_inp, doc_nodes, original_query=user_input)
    # Return the synthesized response and the source nodes
    return chatbot_response.response, chatbot_response.source_nodes

def update_memory_index(user_input: str, response: str, buffer: Dict[str, Any]) -> None:
    """
    Update the memory index with the new conversation turn.

    This function takes the user's input, the chatbot's response, and the conversation
    buffer as input, and updates the memory index with the new conversation turn.

    The function first creates a new Document object with the concatenated query string
    and the document ID. The document ID is the turn number of the conversation, which
    is retrieved from the conversation buffer.

    The concatenated query string is created by concatenating the following strings:

    1. "Người dùng: " followed by the user's input
    2. "Cụ thể hơn: " followed by the user's input (why is this needed?)
    3. "Chatbot: " followed by the chatbot's response
    4. The past query string from the conversation buffer
    5. A newline character to separate the past query string from the new query string

    The concatenated query string is then used to create a new Document object, with
    the document ID set to the turn number of the conversation. The Document object is
    then inserted into the memory index using the insert method of the memory index.

    Args:
        user_input (str): The user's input.
        response (str): The chatbot's response.
        buffer (Dict[str, Any]): The conversation buffer.
    """
    query = f"Người dùng: {user_input}\nCụ thể hơn: {user_input}\nChatbot: {response}\n"
    concat_query = f"Dữ liệu cuộc trò chuyện quá khứ\n.{buffer['past_query']}\nLượt kế:\n{query}"
    doc = Document(text=concat_query, doc_id=f"turn_{buffer['conversation_iter']}")
    buffer["mem_index"].insert(doc)

def generate_followup_questions(doc: Document, buffer: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    """
    Generate follow-up questions from the document.

    This function takes a Document object, the conversation buffer, and the configuration
    dictionary as input. It generates follow-up questions from the document using the
    RagDatasetGenerator.

    The function first creates a RagDatasetGenerator from the document and the
    question prompt template from the conversation buffer. The question prompt template
    is a PromptTemplate object that defines the format of the follow-up questions to be
    generated. The format of the follow-up questions is as follows:

    "Câu hỏi gốc: {orig_query}\n
    Ngữ cảnh đi kèm: {query_str}\n
    Tài liệu tham khảo:\n
    ---------------------\n
    {context_str}\n
    ---------------------\n
    Dựa trên dữ liệu cuộc trò chuyện, hãy viết lại câu truy vấn của người dùng dưới dạng một câu hỏi duy nhất có ngữ cảnh sao cho không cần biết cuộc trò chuyện trước đó vẫn đọc hiểu câu truy vấn. Nếu cuộc trò chuyện quá khứ không đủ để làm rõ câu truy vấn, giữ lại y nguyên câu truy vấn. Câu truy vấn phải được viết lại thành một câu hỏi duy nhất mà không kèm câu trả lời.\n
    Câu hỏi bằng tiếng Việt: "

    The RagDatasetGenerator is created with the document, the question prompt template,
    and the number of questions to generate per chunk. The number of questions to generate
    per chunk is set to twice the maximum number of questions plus one, which is the default
    value of the max_questions configuration parameter.

    The function then generates questions from the document using the
    generate_questions_from_nodes method. The generated questions are then converted to a
    Pandas DataFrame.

    Finally, the function returns the follow-up questions as a list of strings. The
    follow-up questions are the questions that end with a question mark.

    Args:
        doc (Document): The document to generate follow-up questions from.
        buffer (Dict[str, Any]): The conversation buffer.
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        List[str]: The follow-up questions as a list of strings.
    """
    # Create a RagDatasetGenerator from the document and the question prompt template
    adjusted_ques_num = 2 * config["max_questions"] + 1
    dg = RagDatasetGenerator.from_documents([doc], text_question_template=buffer["question_prompt_template"], num_questions_per_chunk=adjusted_ques_num)
    
    # Generate questions from the document
    questions = dg.generate_questions_from_nodes().to_pandas()
    
    # Return the follow-up questions as a list of strings
    # The follow-up questions are the questions that end with a question mark
    return questions.loc[questions['query'].str.endswith('?'), "query"].values

def chat(config, user_input: str, buffer: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate a response to the user input and update the buffer."""
    # Retrieve memory nodes
    mem_nodes = retrieve_memory_nodes(user_input, buffer, config)

    # Synthesize query
    query_inp = synthesize_query(user_input, mem_nodes, buffer)

    # Retrieve document nodes
    doc_nodes = retrieve_document_nodes(query_inp, buffer)

    # Synthesize response
    response, source_nodes = synthesize_response(query_inp, doc_nodes, user_input, buffer)

    # Update memory index
    update_memory_index(user_input, response, buffer)

    # Generate follow-up questions
    query = f"Người dùng: {user_input}\nCụ thể hơn: {user_input}\nChatbot: {response}\n"
    doc = Document(text=query, doc_id=f"turn_{buffer['conversation_iter']}")
    followups = generate_followup_questions(doc, buffer, config)

    # Update buffer
    buffer["past_query"] = query
    buffer["conversation_iter"] += 1

    # Return response and updated buffer
    response_dict = {
        "response": response,
        "source_nodes": source_nodes,
        "additional_nodes": doc_nodes,
        "buffer_nodes": mem_nodes,
        "processed_query": query_inp,
        "followup_questions": followups
    }
    return response_dict, buffer

if __name__ == "__main__":
    config_path = 'configs/gcp.env'
    config = load_config(config_path)
    buffer = init(config)
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response, buffer = chat(user_input, buffer)
        print("Chatbot: ", response["response"])
