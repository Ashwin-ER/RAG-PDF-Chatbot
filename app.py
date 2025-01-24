from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="llama3", temperature=0)
ollama_nomic_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Explicitly import only the required loader
from langchain_community.document_loaders import PyPDFium2Loader

loader = PyPDFium2Loader(file_path="LLM.pdf")
data = loader.load()


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
splits = text_splitter.split_documents(data)

from langchain_chroma import Chroma

batch_size_limit = 4096
current_batch_size = 0
batch_splits = []
all_splits = []
collection_name = "Ashwin"
persist_directory = f"./{collection_name}"

vectorstore = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=ollama_nomic_embeddings)

for split in splits:
    split_size = len(split.page_content)
    if current_batch_size + split_size > batch_size_limit:
        vectorstore.add_documents(documents=batch_splits)
        all_splits.extend(batch_splits)
        batch_splits = [split]
        current_batch_size = split_size
    else:
        batch_splits.append(split)
        current_batch_size += split_size

if batch_splits:
    vectorstore.add_documents(documents=batch_splits)
    all_splits.extend(batch_splits)

vectorstore = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=ollama_nomic_embeddings)
retriever = vectorstore.as_retriever()

system_prompt = (
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate

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
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain_core.messages import AIMessage, HumanMessage

chat_history = []

while True:
    question = input("Enter your question...")

    if question in ["quit", "exit"]:
        break

    ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
    print(f"Question: {question}")
    print(f"Answer: {ai_msg['answer']}")
    print("**********************************************************************")

    chat_history.extend([
        HumanMessage(content=question), 
        AIMessage(content=ai_msg["answer"])
    ])