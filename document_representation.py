import os
import json
from typing import List, Dict, Any, Optional

class ChunkRepresentation:
    def __init__(self, text: str, page: int, doc_id: int):
        self.text = text
        self.page = page
        self.doc_id = doc_id

class DocumentRepresentation:
    def __init__(
        self,
        doc_id: int,
        filename: str,
        filepath: str,
        page_count: int,
        extracted_text: str,
        category: Optional[str] = None,
        document_type: Optional[str] = None,
        ai_keywords: Optional[str] = None,
        document_structure: Optional[str] = None,
        keywords: Optional[str] = None,
        topics: Optional[str] = None,
        skills: Optional[str] = None,
        technologies: Optional[str] = None,
        chunks: Optional[List[ChunkRepresentation]] = None,
        db_doc_instance: Any = None
    ):
        self.doc_id = doc_id
        self.filename = filename
        self.filepath = filepath
        self.page_count = page_count
        self.extracted_text = extracted_text
        self.category = category or "General"
        self.document_type = document_type or "Notes"
        self.ai_keywords = ai_keywords or ""
        self.document_structure = document_structure or "{}"
        self.keywords = keywords or "[]"
        self.topics = topics or "[]"
        self.skills = skills or "[]"
        self.technologies = technologies or "[]"
        self.chunks = chunks or []
        self.db_doc_instance = db_doc_instance # Backreference to SQLAlchemy model if needed

def chunk_text(text: str, max_chars: int = 500, overlap_chars: int = 100) -> List[str]:
    """
    Groups lines of text into chunks of at most max_chars, with overlap_chars overlap.
    Consolidated chunking helper to ensure consistent sliding window logic.
    """
    if not text:
        return []
        
    lines = [line.strip() for line in text.replace('\r', '').split('\n') if line.strip()]
    chunks = []
    
    current_lines = []
    current_len = 0
    
    for line in lines:
        if current_len + len(line) + (1 if current_len > 0 else 0) > max_chars and current_lines:
            chunks.append(" ".join(current_lines))
            
            overlap_lines = []
            overlap_len = 0
            for prev_line in reversed(current_lines):
                if overlap_len + len(prev_line) + (1 if overlap_len > 0 else 0) <= overlap_chars:
                    overlap_lines.insert(0, prev_line)
                    overlap_len += len(prev_line) + 1
                else:
                    break
            current_lines = overlap_lines
            current_len = overlap_len
            
        current_lines.append(line)
        current_len += len(line) + (1 if current_len > 0 else 0)
        
    if current_lines:
        chunks.append(" ".join(current_lines))
        
    return chunks

_representation_cache = {}

def from_db_model(doc: Any) -> DocumentRepresentation:
    """
    Maps a database Document model to a DocumentRepresentation,
    parsing PDF pages with PyMuPDF to extract clean page-by-page chunks if possible.
    Uses a global memory cache to optimize retrieval times.
    """
    doc_id = getattr(doc, "id", 0)
    if doc_id in _representation_cache:
        rep = _representation_cache[doc_id]
        # Update the SQLAlchemy model reference in case of session refresh
        rep.db_doc_instance = doc
        return rep

    filename = getattr(doc, "original_filename", "") or getattr(doc, "filename", "") or ""
    filepath = getattr(doc, "storage_path", "") or ""
    extracted_text = getattr(doc, "extracted_text", "") or ""
    category = getattr(doc, "category", "") or "General"
    document_type = getattr(doc, "document_type", "") or "Notes"
    ai_keywords = getattr(doc, "ai_keywords", "") or ""
    document_structure = getattr(doc, "document_structure", "") or "{}"
    keywords = getattr(doc, "keywords", "") or "[]"
    topics = getattr(doc, "topics", "") or "[]"
    skills = getattr(doc, "skills", "") or "[]"
    technologies = getattr(doc, "technologies", "") or "[]"

    chunks = []
    max_chars = 500
    overlap_chars = 100
    page_count = 1
    
    # Try page-by-page extraction with our PyMuPDF parser if it's a PDF and exists
    if filepath and filepath.lower().endswith(".pdf") and os.path.exists(filepath):
        from .pdf_parser import parse_pdf
        parse_res = parse_pdf(filepath)
        if parse_res.get("success", False):
            page_count = parse_res.get("page_count", 1)
            for page_data in parse_res.get("pages", []):
                p_num = page_data.get("page_number", 1)
                p_text = page_data.get("text", "")
                if not p_text.strip():
                    continue
                p_chunks = chunk_text(p_text, max_chars, overlap_chars)
                for txt in p_chunks:
                    chunks.append(ChunkRepresentation(text=txt, page=p_num, doc_id=doc_id))
                    
    # Fallback to extracted_text if no chunks were extracted (or not a PDF)
    if not chunks and extracted_text:
        text_chunks = chunk_text(extracted_text, max_chars, overlap_chars)
        for txt in text_chunks:
            chunks.append(ChunkRepresentation(text=txt, page=1, doc_id=doc_id))
            
    # Deduplicate chunks to keep only unique text
    seen_texts = set()
    unique_chunks = []
    for c in chunks:
        clean_txt = c.text.strip().lower()
        if clean_txt not in seen_texts:
            seen_texts.add(clean_txt)
            unique_chunks.append(c)

    rep = DocumentRepresentation(
        doc_id=doc_id,
        filename=filename,
        filepath=filepath,
        page_count=page_count,
        extracted_text=extracted_text,
        category=category,
        document_type=document_type,
        ai_keywords=ai_keywords,
        document_structure=document_structure,
        keywords=keywords,
        topics=topics,
        skills=skills,
        technologies=technologies,
        chunks=unique_chunks,
        db_doc_instance=doc
    )
    if doc_id > 0:
        _representation_cache[doc_id] = rep
    return rep
