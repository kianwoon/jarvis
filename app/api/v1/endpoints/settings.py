from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.core.db import SessionLocal, Settings as SettingsModel
from app.core.llm_settings_cache import reload_llm_settings
from app.core.vector_db_settings_cache import reload_vector_db_settings
from app.core.embedding_settings_cache import reload_embedding_settings
from app.core.iceberg_settings_cache import reload_iceberg_settings
from typing import Any, Dict
from pydantic import BaseModel

router = APIRouter()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class SettingsUpdate(BaseModel):
    settings: Dict[str, Any]

@router.get("/{category}")
def get_settings(category: str, db: Session = Depends(get_db)):
    settings_row = db.query(SettingsModel).filter(SettingsModel.category == category).first()
    if not settings_row:
        raise HTTPException(status_code=404, detail="Settings not found")
    return {"category": category, "settings": settings_row.settings}

@router.put("/{category}")
def update_settings(category: str, update: SettingsUpdate, db: Session = Depends(get_db)):
    settings_row = db.query(SettingsModel).filter(SettingsModel.category == category).first()
    if settings_row:
        settings_row.settings = update.settings
    else:
        settings_row = SettingsModel(category=category, settings=update.settings)
        db.add(settings_row)
    db.commit()
    db.refresh(settings_row)
    # If updating LLM settings, reload cache
    if category == 'llm':
        reload_llm_settings()
    # If updating storage settings, reload all related caches
    if category == 'storage':
        reload_vector_db_settings()
        reload_embedding_settings()
        reload_iceberg_settings()
    return {"category": category, "settings": settings_row.settings} 