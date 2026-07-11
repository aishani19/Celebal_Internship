from dotenv import load_dotenv
import os
import requests
import streamlit as st

# Load .env deterministically relative to Chatbot/ directory
from pathlib import Path
_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=str(_env_path), override=False)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Ensure API_URL has a scheme (crucial for internal Render networking)
if API_URL and not API_URL.startswith(("http://", "https://")):
    # If Render provides a hostport (e.g., host:10000), just prepend http://
    if ":" in API_URL:
        API_URL = f"http://{API_URL}"
    else:
        # Fallback for local development or simple hostnames
        API_URL = f"http://{API_URL}:8000"

def get_api_response(question, session_id, model="llama-3.3-70b-versatile"):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {"question": question, "model": model}
    if session_id:
        data["session_id"] = session_id

    try:
        response = requests.post(f"{API_URL}/chat", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def upload_document(file):
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_URL}/upload-doc", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to upload file. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while uploading the file: {str(e)}")
        return None

def list_documents():
    try:
        response = requests.get(f"{API_URL}/list-docs")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch document list. Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching the document list: {str(e)}")
        return []

def delete_document(file_id):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {"file_id": file_id}

    try:
        response = requests.post(f"{API_URL}/delete-doc", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to delete document. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while deleting the document: {str(e)}")
        return None