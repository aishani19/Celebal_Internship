import sqlite3
from datetime import datetime
import os
import logging
import uuid
from pathlib import Path

# Create database in the data directory
_db_path = Path(__file__).resolve().parents[1] / "data" / "fetchwise.db"
_db_path.parent.mkdir(parents=True, exist_ok=True)

# Set up logging to app.log in the project root
logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), "..", "..", "app.log"), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection():
    """Establishes a connection to SQLite database and returns the connection object."""
    try:
        conn = sqlite3.connect(_db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to SQLite: {str(e)}")
        raise

def initialize_database():
    """Ensures necessary tables exist in SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create application_logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS application_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_query TEXT,
            gpt_response TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON application_logs(session_id)')
    logging.info("application_logs table and indexes ensured.")

    # Create document_store table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_store (
            id TEXT PRIMARY KEY,
            filename TEXT UNIQUE,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    logging.info("document_store table and indexes ensured.")
    
    conn.commit()
    conn.close()
    logging.info("Database initialization complete.")

def insert_application_logs(session_id, user_query, gpt_response, model):
    """Inserts a log entry into the application_logs table in SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO application_logs (session_id, user_query, gpt_response, model, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, user_query, gpt_response, model, datetime.utcnow()))
    conn.commit()
    conn.close()
    logging.info("Log inserted successfully.")

def get_chat_history(session_id):
    """Retrieves chat history for a session from SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT user_query, gpt_response FROM application_logs 
        WHERE session_id = ? ORDER BY created_at ASC
    ''', (session_id,))
    logs = cursor.fetchall()
    conn.close()
    
    messages = []
    for log in logs:
        messages.extend([
            {"role": "human", "content": log["user_query"]},
            {"role": "ai", "content": log["gpt_response"]}
        ])
    return messages

def insert_document_record(filename):
    """Inserts a document record into the document_store table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    file_id = str(uuid.uuid4())
    try:
        cursor.execute('''
            INSERT INTO document_store (id, filename, upload_timestamp)
            VALUES (?, ?, ?)
        ''', (file_id, filename, datetime.utcnow()))
        conn.commit()
    except sqlite3.IntegrityError:
        logging.error(f"Document with filename {filename} already exists.")
        raise
    finally:
        conn.close()
    return file_id

def delete_document_record(file_id):
    """Deletes a document record from the document_store table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    logging.info(f"Delete result for file_id {file_id}: deleted_count={deleted_count}")
    return deleted_count > 0
    
def get_all_documents():
    """Retrieves all document records from document_store, sorted by upload_timestamp."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    documents = cursor.fetchall()
    conn.close()
    
    return [{"id": doc["id"], "filename": doc["filename"], "upload_timestamp": doc["upload_timestamp"]}
            for doc in documents]