from sqlalchemy.orm import Session
import models
import schemas

def get_document(db: Session, document_id: int):
    return db.query(models.Document).filter(models.Document.id == document_id).first()

def get_documents(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Document).order_by(models.Document.upload_date.desc()).offset(skip).limit(limit).all()

def create_document(db: Session, document: schemas.DocumentCreate):
    db_document = models.Document(
        filename=document.filename,
        original_filename=document.original_filename,
        storage_path=document.storage_path,
        file_type=document.file_type,
        file_size=document.file_size,
        extracted_text=document.extracted_text,
        status=document.status
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def update_document(db: Session, document_id: int, document_update: schemas.DocumentUpdate):
    db_document = get_document(db, document_id)
    if not db_document:
        return None
    
    update_data = document_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_document, key, value)
        
    db.commit()
    db.refresh(db_document)
    return db_document

def delete_document(db: Session, document_id: int):
    db_document = get_document(db, document_id)
    if db_document:
        db.delete(db_document)
        db.commit()
        return db_document
    return None
