"""
IDC (Intelligent Document Comparison) Reference Manager Service

Manages reference documents with extraction, storage, and retrieval capabilities.
Follows Jarvis patterns - uses settings classes, database sessions, and Redis caching.
"""

import logging
import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from uuid import uuid4
import os

from app.core.config import get_settings
from app.core.db import get_db_session, IDCReferenceDocument, IDCTemplate
from app.core.redis_client import get_redis_client
from app.services.idc_extraction_service import IDCExtractionService

logger = logging.getLogger(__name__)

@dataclass
class ReferenceDocumentInfo:
    document_id: str
    name: str
    document_type: str
    category: str
    original_filename: str
    file_hash: str
    file_size_bytes: int
    extracted_markdown: str
    extraction_metadata: Dict[str, Any]
    recommended_extraction_modes: List[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

@dataclass
class ReferenceUploadResult:
    document_id: str
    extraction_preview: str
    recommended_extraction_modes: List[str]
    extraction_confidence: float
    processing_time_ms: int
    model_used: str
    
class IDCReferenceManager:
    """
    Reference document manager following Jarvis patterns:
    - Uses PostgreSQL for persistent storage
    - Redis for caching frequently accessed references
    - Integrates with IDC extraction service
    - NO hardcoded values, uses settings classes
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = get_redis_client()
        self.extraction_service = IDCExtractionService()
        
        # Cache settings
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.cache_prefix = "idc_reference"
        
    def _calculate_file_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        return hashlib.sha256(content).hexdigest()
    
    def _get_cache_key(self, document_id: str) -> str:
        """Get Redis cache key for reference document"""
        return f"{self.cache_prefix}:{document_id}"
    
    async def upload_reference_document(
        self,
        file_content: bytes,
        name: str,
        document_type: str,
        category: Optional[str] = None,
        extraction_model: Optional[str] = None,
        recommended_modes: Optional[List[str]] = None,
        created_by: Optional[str] = None
    ) -> ReferenceUploadResult:
        """
        Upload and process reference document
        """
        start_time = time.time()
        
        # Generate document ID and calculate hash
        document_id = str(uuid4())
        file_hash = self._calculate_file_hash(file_content)
        file_size_bytes = len(file_content)
        
        # Check if document already exists
        existing_doc = await self._get_document_by_hash(file_hash)
        if existing_doc:
            logger.info(f"Document with hash {file_hash} already exists: {existing_doc.document_id}")
            return ReferenceUploadResult(
                document_id=existing_doc.document_id,
                extraction_preview=existing_doc.extracted_markdown[:500],
                recommended_extraction_modes=existing_doc.recommended_extraction_modes or ["paragraph", "sentence"],
                extraction_confidence=existing_doc.extraction_confidence or 0.8,
                processing_time_ms=0,  # No processing needed
                model_used="cached"
            )
        
        # Extract content to markdown
        try:
            # Convert bytes to string for processing
            document_text = file_content.decode('utf-8', errors='ignore')
            
            extracted_markdown, extraction_metadata = await self.extraction_service.extract_document_to_markdown(
                document_content=document_text,
                document_type=document_type,
                extraction_model=extraction_model
            )
            
            # Determine recommended extraction modes based on document type and content
            if not recommended_modes:
                recommended_modes = self._determine_recommended_modes(document_type, document_text, extracted_markdown)
            
            # Save to database
            with get_db_session() as db:
                reference_doc = IDCReferenceDocument(
                    document_id=document_id,
                    name=name,
                    document_type=document_type,
                    category=category,
                    original_filename=name,  # Use provided name as filename
                    file_hash=file_hash,
                    file_size_bytes=file_size_bytes,
                    extracted_markdown=extracted_markdown,
                    extraction_metadata=extraction_metadata,
                    extraction_model=extraction_model,
                    extraction_confidence=extraction_metadata.get("confidence_score", 0.8),
                    recommended_extraction_modes=recommended_modes,
                    processing_time_ms=extraction_metadata.get("processing_time_ms"),
                    created_by=created_by,
                    is_active=True
                )
                
                db.add(reference_doc)
                db.commit()
                db.refresh(reference_doc)
                
                logger.info(f"Reference document saved: {document_id}")
            
            # Cache the reference document
            await self._cache_reference_document(document_id, reference_doc)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return ReferenceUploadResult(
                document_id=document_id,
                extraction_preview=extracted_markdown[:500] + "..." if len(extracted_markdown) > 500 else extracted_markdown,
                recommended_extraction_modes=recommended_modes,
                extraction_confidence=extraction_metadata.get("confidence_score", 0.8),
                processing_time_ms=processing_time_ms,
                model_used=extraction_metadata.get("model_used", "unknown")
            )
            
        except Exception as e:
            logger.error(f"Failed to process reference document: {str(e)}")
            raise Exception(f"Reference document processing failed: {str(e)}")
    
    async def get_reference_document(self, document_id: str) -> Optional[ReferenceDocumentInfo]:
        """
        Get reference document by ID (with caching)
        """
        # Try cache first
        cached_doc = await self._get_cached_reference_document(document_id)
        if cached_doc:
            return cached_doc
        
        # Get from database
        try:
            with get_db_session() as db:
                reference_doc = db.query(IDCReferenceDocument).filter(
                    IDCReferenceDocument.document_id == document_id,
                    IDCReferenceDocument.is_active == True
                ).first()
                
                if reference_doc:
                    doc_info = self._convert_to_info(reference_doc)
                    
                    # Cache for future use
                    await self._cache_reference_document(document_id, reference_doc)
                    
                    return doc_info
                
        except Exception as e:
            logger.error(f"Failed to get reference document {document_id}: {str(e)}")
            
        return None
    
    async def get_all_reference_documents(
        self,
        document_type: Optional[str] = None,
        category: Optional[str] = None,
        active_only: bool = True
    ) -> List[ReferenceDocumentInfo]:
        """
        Get all reference documents with optional filtering
        """
        try:
            with get_db_session() as db:
                query = db.query(IDCReferenceDocument)
                
                if active_only:
                    query = query.filter(IDCReferenceDocument.is_active == True)
                
                if document_type:
                    query = query.filter(IDCReferenceDocument.document_type == document_type)
                
                if category:
                    query = query.filter(IDCReferenceDocument.category == category)
                
                reference_docs = query.order_by(IDCReferenceDocument.created_at.desc()).all()
                
                return [self._convert_to_info(doc) for doc in reference_docs]
                
        except Exception as e:
            logger.error(f"Failed to get reference documents: {str(e)}")
            return []
    
    async def update_reference_document(
        self,
        document_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update reference document metadata
        """
        try:
            with get_db_session() as db:
                reference_doc = db.query(IDCReferenceDocument).filter(
                    IDCReferenceDocument.document_id == document_id
                ).first()
                
                if not reference_doc:
                    return False
                
                # Update allowed fields
                updateable_fields = [
                    'name', 'document_type', 'category', 
                    'recommended_extraction_modes', 'is_active', 'extraction_model',
                    'extracted_markdown'
                ]
                
                for field, value in updates.items():
                    if field in updateable_fields and hasattr(reference_doc, field):
                        setattr(reference_doc, field, value)
                
                reference_doc.updated_at = datetime.utcnow()
                
                db.commit()
                
                # Invalidate cache
                await self._invalidate_cache(document_id)
                
                logger.info(f"Reference document updated: {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update reference document {document_id}: {str(e)}")
            return False
    
    async def update_reference_document_with_reextraction(
        self,
        document_id: str,
        updates: Dict[str, Any],
        new_extraction_model: str
    ) -> bool:
        """
        Update reference document with re-extraction using new model
        """
        try:
            with get_db_session() as db:
                reference_doc = db.query(IDCReferenceDocument).filter(
                    IDCReferenceDocument.document_id == document_id
                ).first()
                
                if not reference_doc:
                    return False
                
                # Get the original file content (we need to store this or re-extract from existing markdown)
                # For now, we'll re-extract from the existing markdown as a demonstration
                # In production, you might want to store the original file content
                
                try:
                    # Re-extract content with new model
                    extracted_markdown, extraction_metadata = await self.extraction_service.extract_document_to_markdown(
                        document_content=reference_doc.extracted_markdown,  # Using existing markdown as source
                        document_type=updates.get('document_type', reference_doc.document_type),
                        extraction_model=new_extraction_model
                    )
                    
                    # Update the document with new content and metadata
                    reference_doc.extracted_markdown = extracted_markdown
                    reference_doc.extraction_metadata = extraction_metadata
                    reference_doc.extraction_model = new_extraction_model
                    reference_doc.extraction_confidence = extraction_metadata.get("confidence_score", 0.8)
                    reference_doc.processing_time_ms = extraction_metadata.get("processing_time_ms")
                    
                    # Update other metadata fields
                    updateable_fields = [
                        'name', 'document_type', 'category', 
                        'recommended_extraction_modes', 'is_active'
                    ]
                    
                    for field, value in updates.items():
                        if field in updateable_fields and hasattr(reference_doc, field):
                            setattr(reference_doc, field, value)
                    
                    reference_doc.updated_at = datetime.utcnow()
                    reference_doc.version = (reference_doc.version or 1) + 1
                    
                    db.commit()
                    
                    # Invalidate cache
                    await self._invalidate_cache(document_id)
                    
                    logger.info(f"Reference document updated with re-extraction: {document_id}")
                    return True
                    
                except Exception as extraction_error:
                    logger.error(f"Re-extraction failed for {document_id}: {str(extraction_error)}")
                    # Fall back to metadata-only update
                    updateable_fields = [
                        'name', 'document_type', 'category', 
                        'recommended_extraction_modes', 'is_active', 'extraction_model'
                    ]
                    
                    for field, value in updates.items():
                        if field in updateable_fields and hasattr(reference_doc, field):
                            setattr(reference_doc, field, value)
                    
                    reference_doc.updated_at = datetime.utcnow()
                    
                    db.commit()
                    
                    # Invalidate cache
                    await self._invalidate_cache(document_id)
                    
                    logger.warning(f"Reference document updated with metadata only (re-extraction failed): {document_id}")
                    return True
                
        except Exception as e:
            logger.error(f"Failed to update reference document with re-extraction {document_id}: {str(e)}")
            return False
    
    async def delete_reference_document(self, document_id: str) -> bool:
        """
        Soft delete reference document (mark as inactive)
        """
        try:
            with get_db_session() as db:
                reference_doc = db.query(IDCReferenceDocument).filter(
                    IDCReferenceDocument.document_id == document_id
                ).first()
                
                if not reference_doc:
                    return False
                
                reference_doc.is_active = False
                reference_doc.updated_at = datetime.utcnow()
                
                db.commit()
                
                # Invalidate cache
                await self._invalidate_cache(document_id)
                
                logger.info(f"Reference document deleted: {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete reference document {document_id}: {str(e)}")
            return False
    
    async def get_reference_content(self, document_id: str) -> Optional[str]:
        """
        Get extracted markdown content for validation
        """
        reference_doc = await self.get_reference_document(document_id)
        if reference_doc:
            return reference_doc.extracted_markdown
        return None
    
    async def create_template(
        self,
        name: str,
        description: str,
        template_type: str,
        reference_document_id: str,
        default_extraction_mode: str,
        validation_config: Dict[str, Any],
        created_by: Optional[str] = None
    ) -> str:
        """
        Create validation template from reference document
        """
        try:
            template_id = str(uuid4())
            
            with get_db_session() as db:
                template = IDCTemplate(
                    template_id=template_id,
                    name=name,
                    description=description,
                    template_type=template_type,
                    reference_document_id=reference_document_id,
                    default_extraction_mode=default_extraction_mode,
                    validation_config=validation_config,
                    created_by=created_by,
                    is_public=False  # Default to private
                )
                
                db.add(template)
                db.commit()
                
                logger.info(f"Template created: {template_id}")
                return template_id
                
        except Exception as e:
            logger.error(f"Failed to create template: {str(e)}")
            raise Exception(f"Template creation failed: {str(e)}")
    
    async def get_templates(
        self,
        template_type: Optional[str] = None,
        reference_document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get validation templates
        """
        try:
            with get_db_session() as db:
                query = db.query(IDCTemplate)
                
                if template_type:
                    query = query.filter(IDCTemplate.template_type == template_type)
                
                if reference_document_id:
                    query = query.filter(IDCTemplate.reference_document_id == reference_document_id)
                
                templates = query.order_by(IDCTemplate.created_at.desc()).all()
                
                return [
                    {
                        "template_id": template.template_id,
                        "name": template.name,
                        "description": template.description,
                        "template_type": template.template_type,
                        "reference_document_id": template.reference_document_id,
                        "default_extraction_mode": template.default_extraction_mode,
                        "validation_config": template.validation_config,
                        "usage_count": template.usage_count,
                        "success_rate": template.success_rate,
                        "is_public": template.is_public,
                        "created_by": template.created_by,
                        "created_at": template.created_at.isoformat()
                    }
                    for template in templates
                ]
                
        except Exception as e:
            logger.error(f"Failed to get templates: {str(e)}")
            return []
    
    def _determine_recommended_modes(
        self,
        document_type: str,
        original_content: str,
        extracted_markdown: str
    ) -> List[str]:
        """
        Determine recommended extraction modes based on document analysis
        """
        recommendations = []
        
        # Analyze content structure
        lines = extracted_markdown.split('\n')
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        
        # Check for Q&A patterns
        qa_indicators = ['q:', 'a:', 'question', 'answer', '?']
        qa_count = sum(1 for line in lines if any(indicator in line.lower() for indicator in qa_indicators))
        
        # Check for structured content
        header_count = sum(1 for line in lines if line.strip().startswith('#'))
        
        # Determine recommendations based on analysis
        if document_type.lower() in ['exam', 'test', 'quiz', 'assessment']:
            recommendations = ['qa_pairs', 'paragraph', 'sentence']
        elif document_type.lower() in ['contract', 'legal', 'policy', 'compliance']:
            recommendations = ['paragraph', 'sentence', 'section']
        elif document_type.lower() in ['resume', 'cv', 'profile']:
            recommendations = ['sentence', 'paragraph', 'section']
        elif qa_count > len(lines) * 0.2:  # >20% of lines have Q&A indicators
            recommendations = ['qa_pairs', 'paragraph']
        elif header_count > 5:  # Well-structured document
            recommendations = ['section', 'paragraph', 'sentence']
        elif avg_line_length > 100:  # Long lines suggest paragraph structure
            recommendations = ['paragraph', 'sentence']
        else:
            # Default recommendations
            recommendations = ['paragraph', 'sentence', 'section']
        
        # Ensure we don't recommend duplicate modes
        return list(dict.fromkeys(recommendations))
    
    async def _get_document_by_hash(self, file_hash: str) -> Optional[IDCReferenceDocument]:
        """Get document by file hash to prevent duplicates"""
        try:
            with get_db_session() as db:
                return db.query(IDCReferenceDocument).filter(
                    IDCReferenceDocument.file_hash == file_hash,
                    IDCReferenceDocument.is_active == True
                ).first()
        except Exception as e:
            logger.error(f"Failed to check document hash: {str(e)}")
            return None
    
    def _convert_to_info(self, reference_doc: IDCReferenceDocument) -> ReferenceDocumentInfo:
        """Convert database model to info dataclass"""
        return ReferenceDocumentInfo(
            document_id=reference_doc.document_id,
            name=reference_doc.name,
            document_type=reference_doc.document_type,
            category=reference_doc.category or "",
            original_filename=reference_doc.original_filename or "",
            file_hash=reference_doc.file_hash,
            file_size_bytes=reference_doc.file_size_bytes or 0,
            extracted_markdown=reference_doc.extracted_markdown,
            extraction_metadata=reference_doc.extraction_metadata or {},
            recommended_extraction_modes=reference_doc.recommended_extraction_modes or [],
            is_active=reference_doc.is_active,
            created_at=reference_doc.created_at,
            updated_at=reference_doc.updated_at
        )
    
    async def _cache_reference_document(self, document_id: str, reference_doc: IDCReferenceDocument):
        """Cache reference document in Redis"""
        if self.redis_client:
            try:
                cache_key = self._get_cache_key(document_id)
                doc_info = self._convert_to_info(reference_doc)
                
                # Convert to JSON-serializable format
                cache_data = asdict(doc_info)
                cache_data['created_at'] = cache_data['created_at'].isoformat()
                cache_data['updated_at'] = cache_data['updated_at'].isoformat()
                
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(cache_data)
                )
                
            except Exception as e:
                logger.warning(f"Failed to cache reference document: {e}")
    
    async def _get_cached_reference_document(self, document_id: str) -> Optional[ReferenceDocumentInfo]:
        """Get cached reference document from Redis"""
        if self.redis_client:
            try:
                cache_key = self._get_cache_key(document_id)
                cached = self.redis_client.get(cache_key)
                
                if cached:
                    cache_data = json.loads(cached)
                    
                    # Convert datetime strings back to datetime objects
                    cache_data['created_at'] = datetime.fromisoformat(cache_data['created_at'])
                    cache_data['updated_at'] = datetime.fromisoformat(cache_data['updated_at'])
                    
                    return ReferenceDocumentInfo(**cache_data)
                    
            except Exception as e:
                logger.warning(f"Failed to get cached reference document: {e}")
        
        return None
    
    async def _invalidate_cache(self, document_id: str):
        """Invalidate cached reference document"""
        if self.redis_client:
            try:
                cache_key = self._get_cache_key(document_id)
                self.redis_client.delete(cache_key)
            except Exception as e:
                logger.warning(f"Failed to invalidate cache: {e}")