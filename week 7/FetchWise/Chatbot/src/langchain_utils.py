from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from src.chroma_utils import get_vectorstore
from dotenv import load_dotenv
from pathlib import Path
from langchain_groq import ChatGroq

# Load environment variables deterministically from Chatbot/.env
_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=str(_env_path), override=False)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Setting up prompts
contextualize_q_system_prompt = os.getenv("CONTEXTUALIZE_Q_PROMPT", 
    "Given a chat history and the latest user question " 
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question in detail. If the context does not contain the answer, or if no context is provided, answer using your own general knowledge."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def get_rag_chain(model="llama-3.3-70b-versatile"):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    llm = ChatGroq(
        model=model,
        api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)    
    return rag_chain