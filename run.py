from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
import time

# Initialize models
llm = ChatOllama(model="llama3", temperature=0)
ollama_nomic_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load and split documents
loader = PyPDFium2Loader(file_path="LLM.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
splits = text_splitter.split_documents(data)

# Set up vector store
collection_name = "Ashwin"
persist_directory = f"./{collection_name}"
vectorstore = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=ollama_nomic_embeddings)

# Add documents to vector store in batches
batch_size_limit = 4096
batch_splits = []
current_batch_size = 0

for split in splits:
    split_size = len(split.page_content)
    if current_batch_size + split_size > batch_size_limit:
        vectorstore.add_documents(documents=batch_splits)
        batch_splits = [split]
        current_batch_size = split_size
    else:
        batch_splits.append(split)
        current_batch_size += split_size

if batch_splits:
    vectorstore.add_documents(documents=batch_splits)

# Set up retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Limit to top 3 documents

# Define prompts
system_prompt = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions: {input}
"""

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create chains
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Chat loop
chat_history = []

while True:
    question = input("Enter your question (or type 'quit' to exit): ").strip()
    if question.lower() in ["quit", "exit"]:
        break

    start_time = time.time()

    # Invoke RAG chain
    ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
    answer = ai_msg["answer"]

    end_time = time.time()
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("**********************************************************************")

    # Update chat history
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer)
    ])