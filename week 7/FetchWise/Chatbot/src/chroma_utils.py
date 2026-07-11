from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
import os
import gc
import logging
import threading

import sys

# Set up logging to console (stdout) for visibility in Render/Production
logging.basicConfig(stream=sys.stdout, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Lazy-init the embedding function + Chroma vector store.
# This avoids loading HuggingFace models at import time (which can break/startup-slow in deployments).
_vectorstore = None
_vectorstore_lock = threading.Lock()

def _get_persist_dir() -> str:
    # Since this file is in src/, go up one level to Chatbot/, then into data/chroma_db
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_root, "data", "chroma_db")

def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    with _vectorstore_lock:
        if _vectorstore is not None:
            return _vectorstore

        persist_dir = _get_persist_dir()
        logging.info(f"Initializing Chroma persist directory: {persist_dir}")
        os.makedirs(persist_dir, exist_ok=True)

        embedding_function = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=1)
        _vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_function)
        return _vectorstore

from typing import List
from langchain_core.documents import Document

# Initialize text splitter (lightweight; safe to do eagerly)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)

# Document loading and splitting
def load_and_split_document(file_path: str) -> List[Document]:
    logging.info(f"Attempting to load and split document from: {file_path}")
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        logging.info("Using PyPDFLoader for PDF file")
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        logging.info("Using Docx2txtLoader for DOCX file")
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
        logging.info("Using UnstructuredHTMLLoader for HTML file")
    else:
        logging.error(f"Unsupported file type: {file_path}")
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    logging.info(f"Loaded {len(documents)} document(s) from {file_path}")
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(split_docs)} chunks")
    return split_docs

# Indexing documents
def index_document_to_chroma(file_path: str, file_id: str) -> bool:
    try:
        vectorstore = get_vectorstore()
        logging.info(f"Starting indexing process for file: {file_path} with file_id: {file_id}")
        splits = load_and_split_document(file_path)
        logging.info(f"Loaded and split {len(splits)} document chunks")

        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
            logging.debug(f"Added metadata file_id={file_id} to chunk: {split.page_content[:50]}...")

        logging.info(f"Adding {len(splits)} documents to Chroma vector store in memory-safe batches")
        batch_size = 25  # strictly limit batch size to preserve 512MB RAM constraint
        total_batches = (len(splits) + batch_size - 1) // batch_size
        
        for i in range(0, len(splits), batch_size):
            batch_splits = splits[i : i + batch_size]
            logging.info(f"Indexing batch {i // batch_size + 1} of {total_batches}")
            vectorstore.add_documents(batch_splits)
            gc.collect()  # force clear memory buffers after each ML embedding execution
            
        logging.info(f"Successfully indexed {len(splits)} documents to Chroma")
        gc.collect() # Release memory after full indexing sequence
        logging.info(f"Chroma collection count after indexing: {vectorstore._collection.count()}")
        return True
    except Exception as e:
        logging.error(f"Error indexing document {file_path}: {str(e)}")
        return False

# Deleting documents
def delete_doc_from_chroma(file_id: str):
    try:
        vectorstore = get_vectorstore()
        logging.info(f"Attempting to delete documents with file_id: {file_id}")
        docs = vectorstore.get(where={"file_id": file_id})
        logging.info(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")

        vectorstore._collection.delete(where={"file_id": file_id})
        logging.info(f"Deleted all instances with file_id {file_id}")
        logging.info(f"Chroma collection count after deletion: {vectorstore._collection.count()}")

        return True
    except Exception as e:
        logging.error(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False