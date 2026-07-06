from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
import os
import shutil
from typing import List

import models
import schemas
import crud
import ai
from database import engine, get_db

# Initialize SQLite database tables
models.Base.metadata.create_all(bind=engine)

# Migration: Add category, document_type, ocr_used, and ocr_confidence columns if they don't exist
def run_migrations():
    inspector = inspect(engine)
    if inspector.has_table("documents"):
        columns = [col['name'] for col in inspector.get_columns('documents')]
        with engine.begin() as conn:
            if 'category' not in columns:
                print("Migration: Adding category column to documents table...")
                conn.execute(text("ALTER TABLE documents ADD COLUMN category TEXT"))
            if 'document_type' not in columns:
                print("Migration: Adding document_type column to documents table...")
                conn.execute(text("ALTER TABLE documents ADD COLUMN document_type TEXT"))
            if 'ocr_used' not in columns:
                print("Migration: Adding ocr_used column to documents table...")
                conn.execute(text("ALTER TABLE documents ADD COLUMN ocr_used INTEGER"))
            if 'ocr_confidence' not in columns:
                print("Migration: Adding ocr_confidence column to documents table...")
                conn.execute(text("ALTER TABLE documents ADD COLUMN ocr_confidence REAL"))
            if 'topics' not in columns:
                print("Migration: Adding topics column to documents table...")
                conn.execute(text("ALTER TABLE documents ADD COLUMN topics TEXT"))
            if 'keywords' not in columns:
                print("Migration: Adding keywords column to documents table...")
                conn.execute(text("ALTER TABLE documents ADD COLUMN keywords TEXT"))
            if 'entities' not in columns:
                print("Migration: Adding entities column to documents table...")
                conn.execute(text("ALTER TABLE documents ADD COLUMN entities TEXT"))
            if 'document_structure' not in columns:
                print("Migration: Adding document_structure column to documents table...")
                conn.execute(text("ALTER TABLE documents ADD COLUMN document_structure TEXT"))
            if 'skills' not in columns:
                print("Migration: Adding skills column to documents table...")
                conn.execute(text("ALTER TABLE documents ADD COLUMN skills TEXT"))
            if 'organizations' not in columns:
                print("Migration: Adding organizations column to documents table...")
                conn.execute(text("ALTER TABLE documents ADD COLUMN organizations TEXT"))
            if 'technologies' not in columns:
                print("Migration: Adding technologies column to documents table...")
                conn.execute(text("ALTER TABLE documents ADD COLUMN technologies TEXT"))

run_migrations()

# Ensure uploads directory exists
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="AI Personal Knowledge Vault API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_uploaded_document(doc_id: int, file_path: str, file_type: str, db: Session):
    """Background task to extract text, classify, summarize, and generate tags for a document."""
    try:
        # Get original filename from DB
        db_doc = crud.get_document(db, doc_id)
        original_filename = db_doc.original_filename if db_doc else ""

        # Step 1: Extract Text and OCR metadata
        content, ocr_used, ocr_confidence = ai.extract_text_from_file(file_path, file_type)
        
        # Step 2: Decoupled Metadata Extraction Layer (spaCy)
        import json
        metadata = ai.extract_metadata_spacy(content)
        entities = metadata["entities"]
        struct = metadata["document_structure"]
        keywords = metadata["keywords"]
        topics = metadata["topics"]
        skills = metadata["skills"]
        organizations = metadata["organizations"]
        technologies = metadata["technologies"]
        
        # Step 3: Classification Layer using the metadata and content snippet (independent of spaCy objects)
        category, doc_type = ai.classify_document(content, metadata)
        
        # Step 4: Generate Summary, Tags (including category/type), and Vector Embedding
        summary, tags_list = ai.generate_summary_and_tags(content, category, doc_type)
        
        # If low-confidence OCR, add tag and prefix to summary
        if ocr_used and ocr_confidence < 70.0:
            if "Low-Confidence OCR" not in tags_list:
                tags_list.append("Low-Confidence OCR")
            summary = f"[Low-Confidence OCR] {summary}"
            
        tags_str = ",".join(tags_list)
        embedding_str = ai.generate_embedding(content)
        
        # Step 5: Update Document in DB
        doc_update = schemas.DocumentUpdate(
            status="ready",
            ai_summary=summary,
            ai_keywords=tags_str,
            extracted_text=content,
            embedding=embedding_str,
            category=category,
            document_type=doc_type,
            ocr_used=1 if ocr_used else 0,
            ocr_confidence=ocr_confidence,
            topics=json.dumps(topics),
            keywords=json.dumps(keywords),
            entities=json.dumps(entities),
            document_structure=json.dumps(struct),
            skills=json.dumps(skills),
            organizations=json.dumps(organizations),
            technologies=json.dumps(technologies)
        )
        crud.update_document(db, doc_id, doc_update)
        
        # Hook up new retrieval indexing dynamically
        try:
            from ai import USE_NEW_RETRIEVAL, get_new_retrieval_engine
            if USE_NEW_RETRIEVAL:
                engine = get_new_retrieval_engine()
                engine.index_documents([file_path])
                print(f"[NEW RETRIEVAL] Successfully indexed uploaded document: {file_path}")
        except Exception as e:
            print(f"[NEW RETRIEVAL] Indexing failed for uploaded document {file_path}: {e}")
            
    except Exception as e:

        print(f"Failed to process document {doc_id}: {e}")
        doc_update = schemas.DocumentUpdate(status="failed")
        crud.update_document(db, doc_id, doc_update)

@app.post("/api/upload", response_model=schemas.DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Save the file locally using a safe disk filename
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    if not safe_filename:
        safe_filename = "unnamed_document"
        
    storage_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    # Resolve name collision
    base, extension = os.path.splitext(safe_filename)
    counter = 1
    while os.path.exists(storage_path):
        safe_filename = f"{base}_{counter}{extension}"
        storage_path = os.path.join(UPLOAD_DIR, safe_filename)
        counter += 1

    try:
        with open(storage_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file on server: {str(e)}"
        )

    # Get file size and mimetype
    file_size = os.path.getsize(storage_path)
    file_type = file.content_type or "application/octet-stream"

    # Create document record in database
    doc_create = schemas.DocumentCreate(
        filename=safe_filename,
        original_filename=file.filename,  # Keep the raw user-provided name
        storage_path=storage_path,
        file_type=file_type,
        file_size=file_size,
        status="processing"
    )
    db_doc = crud.create_document(db, doc_create)

    # Queue background task for text extraction and analysis
    background_tasks.add_task(
        process_uploaded_document, 
        db_doc.id, 
        storage_path, 
        file_type, 
        db
    )

    return db_doc

@app.get("/api/documents", response_model=List[schemas.DocumentResponse])
def get_all_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_documents(db, skip=skip, limit=limit)

@app.get("/api/documents/{doc_id}", response_model=schemas.DocumentResponse)
def get_document_details(doc_id: int, db: Session = Depends(get_db)):
    doc = crud.get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@app.delete("/api/documents/{doc_id}", response_model=schemas.DocumentResponse)
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = crud.get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if os.path.exists(doc.storage_path):
        try:
            os.remove(doc.storage_path)
        except Exception as e:
            print(f"Warning: Failed to delete file on disk at {doc.storage_path}: {e}")
            
    deleted_doc = crud.delete_document(db, doc_id)
    return deleted_doc

@app.post("/api/query")
def query_knowledge_vault(payload: schemas.QueryRequest, db: Session = Depends(get_db)):
    from fastapi.responses import JSONResponse
    import traceback
    try:
        # Defensive validation of inputs
        if payload is None:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "stage": "validation",
                    "message": "Invalid request payload."
                }
            )
            
        query_text = payload.query.strip() if payload.query else ""
        if not query_text:
            return schemas.QueryResponse(
                answer="I couldn't find a highly relevant document matching your query.",
                source_documents=[],
                relevant_passages=[],
                confidence_score=0.0,
                page_numbers=[]
            )
            
        query_obj = db.query(models.Document).filter(models.Document.status == "ready")
        if payload.document_ids:
            query_obj = query_obj.filter(models.Document.id.in_(payload.document_ids))
            
        documents = query_obj.all()
        
        result = ai.query_vault(query_text, documents, history=payload.history)
        
        # Validate result structure defensively
        if not result or not isinstance(result, dict):
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "stage": "answer_generation",
                    "message": "Invalid answer generation result."
                }
            )
            
        source_responses = []
        source_docs = result.get("source_documents")
        if source_docs is not None and isinstance(source_docs, list):
            for doc in source_docs:
                if doc is not None:
                    try:
                        source_responses.append(schemas.DocumentResponse.model_validate(doc))
                    except Exception:
                        pass
                        
        return schemas.QueryResponse(
            answer=result.get("answer") or "I couldn't find a highly relevant document matching your query.",
            source_documents=source_responses,
            relevant_passages=result.get("relevant_passages") or [],
            confidence_score=result.get("confidence_score") or 0.0,
            page_numbers=result.get("page_numbers") or []
        )
    except Exception as e:
        print(f"CRITICAL BACKEND ERROR: {str(e)}")
        print(traceback.format_exc())
        stage = getattr(e, "stage", "unknown")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "stage": stage,
                "message": f"An internal error occurred: {str(e)}"
            }
        )

@app.get("/api/stats")
def get_vault_statistics(db: Session = Depends(get_db)):
    """Retrieve summarized statistics for dashboard widgets."""
    docs = db.query(models.Document).all()
    
    total_docs = len(docs)
    total_size = sum(doc.file_size for doc in docs)
    processing_count = sum(1 for doc in docs if doc.status == "processing")
    ready_count = sum(1 for doc in docs if doc.status == "ready")
    failed_count = sum(1 for doc in docs if doc.status == "failed")
    
    all_categories = {}
    for doc in docs:
        if doc.category:
            all_categories[doc.category] = all_categories.get(doc.category, 0) + 1
        elif doc.ai_keywords:
            for t in doc.ai_keywords.split(','):
                cleaned_tag = t.strip()
                if cleaned_tag:
                    all_categories[cleaned_tag] = all_categories.get(cleaned_tag, 0) + 1
                    
    sorted_categories = sorted(all_categories.items(), key=lambda x: x[1], reverse=True)[:5]
    top_tags = [{"tag": cat, "count": count} for cat, count in sorted_categories]
    
    type_distribution = {}
    for doc in docs:
        ext = os.path.splitext(doc.filename)[1].lower() or ".unknown"
        type_distribution[ext] = type_distribution.get(ext, 0) + 1
        
    return {
        "total_documents": total_docs,
        "total_size_bytes": total_size,
        "status_breakdown": {
            "processing": processing_count,
            "ready": ready_count,
            "failed": failed_count
        },
        "top_tags": top_tags,
        "type_distribution": type_distribution
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
