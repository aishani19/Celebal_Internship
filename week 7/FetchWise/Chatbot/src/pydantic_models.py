from pydantic import BaseModel, Field
from datetime import datetime

class QueryInput(BaseModel):
    question: str
    session_id: str | None = Field(default=None)
    # Groq model name (e.g. "llama3-70b-8192", "mixtral-8x7b-32768", etc.)
    model: str = Field(default="llama-3.3-70b-versatile")

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: str

class DocumentInfo(BaseModel):
    id: str   # changed from int → str
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: str   # changed from int → str
