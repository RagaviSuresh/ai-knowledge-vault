from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class DocumentBase(BaseModel):
    filename: str
    original_filename: str
    file_type: str
    file_size: int

class DocumentCreate(DocumentBase):
    storage_path: str
    extracted_text: Optional[str] = None
    status: str = "processing"

class DocumentUpdate(BaseModel):
    status: Optional[str] = None
    ai_summary: Optional[str] = None
    ai_keywords: Optional[str] = None
    extracted_text: Optional[str] = None
    embedding: Optional[str] = None
    category: Optional[str] = None
    document_type: Optional[str] = None
    ocr_used: Optional[int] = None
    ocr_confidence: Optional[float] = None
    topics: Optional[str] = None
    keywords: Optional[str] = None
    entities: Optional[str] = None
    document_structure: Optional[str] = None
    skills: Optional[str] = None
    organizations: Optional[str] = None
    technologies: Optional[str] = None

class DocumentResponse(DocumentBase):
    id: int
    upload_date: datetime
    status: str
    ai_summary: Optional[Optional[str]] = None
    ai_keywords: Optional[Optional[str]] = None
    category: Optional[Optional[str]] = None
    document_type: Optional[Optional[str]] = None
    embedding: Optional[Optional[str]] = None
    ocr_used: Optional[int] = None
    ocr_confidence: Optional[float] = None
    topics: Optional[Optional[str]] = None
    keywords: Optional[Optional[str]] = None
    entities: Optional[Optional[str]] = None
    document_structure: Optional[Optional[str]] = None
    skills: Optional[Optional[str]] = None
    organizations: Optional[Optional[str]] = None
    technologies: Optional[Optional[str]] = None
    semantic_score: Optional[float] = None
    metadata_score: Optional[float] = None
    keyword_score: Optional[float] = None
    intent_bonus: Optional[float] = None
    final_score: Optional[float] = None

    class Config:
        from_attributes = True

class MessageModel(BaseModel):
    sender: str
    text: str

class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[int]] = None  # Optional filter by document IDs
    history: Optional[List[MessageModel]] = None  # Optional conversation history

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[DocumentResponse]
    relevant_passages: List[str]
    confidence_score: Optional[float] = None
    page_numbers: Optional[List[int]] = None
