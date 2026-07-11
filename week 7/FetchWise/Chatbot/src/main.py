from fastapi import FastAPI, File, UploadFile, HTTPException
from src.pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from src.langchain_utils import get_rag_chain
from src.db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from src.chroma_utils import index_document_to_chroma, delete_doc_from_chroma, get_vectorstore
import os
import gc
import uuid
import logging
import shutil

import sys

# Set up logging to console (stdout)
logging.basicConfig(stream=sys.stdout, 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI()

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"status": "ok", "message": "FetchWise API is running."}

# Eager Loading: Load models on startup so they are ready for the first request
@app.on_event("startup")
async def startup_event():
    logging.info("Starting up service: Pre-loading AI models and Vector Store...")
    try:
        from src.db_utils import initialize_database
        initialize_database()
        get_vectorstore()
        logging.info("AI models and Vector Store loaded successfully!")
    except Exception as e:
        logging.error(f"Error during startup: {str(e)}")

# Chat endpoint
@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model}")

    try:
        chat_history = get_chat_history(session_id)
        rag_chain = get_rag_chain(query_input.model)
        answer = rag_chain.invoke({
            "input": query_input.question,
            "chat_history": chat_history
        })['answer']

        insert_application_logs(session_id, query_input.question, answer, query_input.model)
        logging.info(f"Session ID: {session_id}, AI Response: {answer}")
        return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)
    except Exception as e:
        logging.error(f"Session ID: {session_id}, Error during chat: {str(e)}")
        raise HTTPException(status_code=503, detail=f"AI service temporarily unavailable: {str(e)}")

# Document upload endpoint
@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")

    # Save the file to the data/documents directory
    documents_dir = "data/documents"
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)

    temp_file_path = os.path.join(documents_dir, f"temp_{file.filename}")

    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        import sqlite3
        try:
            file_id = insert_document_record(file.filename)
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' already exists.")
        
        success = index_document_to_chroma(temp_file_path, file_id)

        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        gc.collect() # Force memory cleanup after heavy indexing

# List documents endpoint
@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()

# Delete document endpoint
@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}

if __name__ == "__main__":
    import uvicorn
    # When run directly, start the uvicorn server
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)