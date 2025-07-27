"""
Knowledge Graph API Endpoints

Independent API for knowledge graph document ingestion and management.
Provides endpoints for processing documents into Neo4j knowledge graphs
with progress tracking and entity/relationship visualization.
"""

import asyncio
import logging
import json
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.services.knowledge_graph_service import get_knowledge_graph_service
from app.document_handlers.graph_processor import get_graph_document_processor, GraphProcessingResult
from app.core.temp_document_manager import TempDocumentManager
from app.services.neo4j_service import get_neo4j_service
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for request/response validation
class GraphIngestionRequest(BaseModel):
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for grouping")
    extract_entities: bool = Field(default=True, description="Extract entities from document")
    extract_relationships: bool = Field(default=True, description="Extract relationships between entities")
    store_in_neo4j: bool = Field(default=True, description="Store results in Neo4j database")
    preview_only: bool = Field(default=False, description="Generate preview without storing")

class GraphIngestionResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    processing_result: Optional[Dict[str, Any]] = None
    preview_data: Optional[Dict[str, Any]] = None
    message: str
    error: Optional[str] = None

class EntityResponse(BaseModel):
    id: str
    name: str
    type: str
    confidence: float
    properties: Dict[str, Any]
    document_id: str

class RelationshipResponse(BaseModel):
    id: str
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    properties: Dict[str, Any]
    document_id: str

class GraphStatsResponse(BaseModel):
    total_entities: int
    total_relationships: int
    entity_types: Dict[str, int]
    relationship_types: Dict[str, int]
    documents_processed: int
    last_updated: str

class GraphQueryRequest(BaseModel):
    query: str = Field(..., description="Cypher query or natural language query")
    query_type: str = Field(default="cypher", description="Type of query: 'cypher' or 'natural'")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results to return")

@router.post("/ingest", response_model=GraphIngestionResponse)
async def ingest_document_to_graph(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
    extract_entities: bool = Form(default=True),
    extract_relationships: bool = Form(default=True),
    store_in_neo4j: bool = Form(default=True),
    preview_only: bool = Form(default=False)
):
    """
    Ingest a document into the knowledge graph with entity and relationship extraction.
    
    - **file**: Document file (PDF, DOCX, XLSX, PPTX, TXT)
    - **conversation_id**: Optional conversation ID for grouping documents
    - **extract_entities**: Whether to extract entities from the document
    - **extract_relationships**: Whether to extract relationships between entities
    - **store_in_neo4j**: Whether to store results in Neo4j database
    - **preview_only**: Generate preview without actually storing data
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        allowed_extensions = {'.pdf', '.docx', '.xlsx', '.pptx', '.txt'}
        file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        if f'.{file_ext}' not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Check file size (50MB limit)
        max_size_mb = 50
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)"
            )
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Process document for graph extraction
        temp_doc_manager = TempDocumentManager()
        
        # Save temporary file first
        import tempfile
        import os
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
            tmp_file.write(file_content)
            temp_file_path = tmp_file.name
        
        try:
            # Extract chunks using existing document handlers
            chunks = await temp_doc_manager._extract_document_content(temp_file_path, file.filename)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No content could be extracted from document")
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        # Process for knowledge graph
        graph_processor = get_graph_document_processor()
        
        if preview_only:
            # Generate preview without storing
            preview_data = await graph_processor.preview_graph_extraction(chunks, max_chunks=5)
            
            return GraphIngestionResponse(
                success=True,
                document_id=document_id,
                filename=file.filename,
                preview_data=preview_data,
                message=f"Preview generated for '{file.filename}' - {preview_data.get('total_entities_found', 0)} entities, {preview_data.get('total_relationships_found', 0)} relationships found"
            )
        else:
            # Full processing
            graph_result = await graph_processor.process_document_for_graph(
                chunks=chunks,
                document_id=document_id,
                store_in_neo4j=store_in_neo4j
            )
            
            return GraphIngestionResponse(
                success=graph_result.success,
                document_id=document_id,
                filename=file.filename,
                processing_result={
                    'total_chunks': graph_result.total_chunks,
                    'processed_chunks': graph_result.processed_chunks,
                    'total_entities': graph_result.total_entities,
                    'total_relationships': graph_result.total_relationships,
                    'processing_time_ms': graph_result.processing_time_ms,
                    'errors': graph_result.errors
                },
                message=f"Successfully processed '{file.filename}' - {graph_result.total_entities} entities, {graph_result.total_relationships} relationships" if graph_result.success else "Processing completed with errors"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph ingestion failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@router.post("/ingest-with-progress")
async def ingest_document_with_progress(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
    extract_entities: bool = Form(default=True),
    extract_relationships: bool = Form(default=True),
    store_in_neo4j: bool = Form(default=True)
):
    """
    Ingest a document into the knowledge graph with real-time progress updates via SSE.
    """
    
    # Read file content before starting streaming response
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_content = await file.read()
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    document_id = str(uuid.uuid4())
    
    async def progress_generator():
        try:
            # Initial progress
            yield f"data: {json.dumps({'progress': 0, 'step': 'Starting document processing', 'status': 'processing'})}\n\n"
            
            # Document parsing
            yield f"data: {json.dumps({'progress': 20, 'step': 'Parsing document content', 'status': 'processing'})}\n\n"
            
            temp_doc_manager = TempDocumentManager()
            
            # Save temporary file
            import tempfile
            import os
            
            # Get file extension
            file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                tmp_file.write(file_content)
                temp_file_path = tmp_file.name
            
            try:
                chunks = await temp_doc_manager._extract_document_content(temp_file_path, file.filename)
                
                if not chunks:
                    yield f"data: {json.dumps({'progress': 0, 'step': 'No content could be extracted', 'status': 'error', 'error': 'Document parsing failed'})}\n\n"
                    return
                    
            finally:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            yield f"data: {json.dumps({'progress': 40, 'step': f'Extracted {len(chunks)} chunks', 'status': 'processing'})}\n\n"
            
            # Entity extraction
            if extract_entities:
                yield f"data: {json.dumps({'progress': 60, 'step': 'Extracting entities and relationships', 'status': 'processing'})}\n\n"
            
            # Graph processing
            graph_processor = get_graph_document_processor()
            graph_result = await graph_processor.process_document_for_graph(
                chunks=chunks,
                document_id=document_id,
                store_in_neo4j=store_in_neo4j
            )
            
            if store_in_neo4j:
                yield f"data: {json.dumps({'progress': 80, 'step': 'Storing in Neo4j knowledge graph', 'status': 'processing'})}\n\n"
            
            # Final result
            if graph_result.success:
                result_data = {
                    'document_id': document_id,
                    'filename': file.filename,
                    'total_entities': graph_result.total_entities,
                    'total_relationships': graph_result.total_relationships,
                    'processing_time_ms': graph_result.processing_time_ms
                }
                yield f"data: {json.dumps({'progress': 100, 'step': 'Knowledge graph ingestion complete', 'status': 'success', 'result': result_data})}\n\n"
            else:
                yield f"data: {json.dumps({'progress': 0, 'step': 'Graph processing failed', 'status': 'error', 'errors': graph_result.errors})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'progress': 0, 'step': 'Ingestion failed', 'status': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        progress_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/entities/{document_id}")
async def get_document_entities(document_id: str, limit: int = 100):
    """
    Get all entities extracted from a specific document.
    """
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        entities = neo4j_service.find_entities(
            properties={'document_id': document_id},
            limit=limit
        )
        
        # Format entities for response
        formatted_entities = []
        for entity in entities:
            formatted_entities.append({
                'id': entity.get('id', 'unknown'),
                'name': entity.get('name', 'unknown'),
                'type': entity.get('type', 'unknown'),
                'confidence': entity.get('confidence', 0.0),
                'original_text': entity.get('original_text', ''),
                'chunk_id': entity.get('chunk_id', ''),
                'created_at': entity.get('created_at', '')
            })
        
        return {
            'document_id': document_id,
            'entities': formatted_entities,
            'total_count': len(formatted_entities)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get entities for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve entities: {str(e)}")

@router.get("/relationships/{document_id}")
async def get_document_relationships(document_id: str, limit: int = 100):
    """
    Get all relationships extracted from a specific document.
    """
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        # Use custom query to get relationships for the document
        cypher_query = """
        MATCH (a)-[r]->(b)
        WHERE r.document_id = $document_id
        RETURN a.name as source_name, type(r) as relationship_type, b.name as target_name,
               r.confidence as confidence, r.context as context, r.chunk_id as chunk_id,
               r.created_at as created_at
        LIMIT $limit
        """
        
        results = neo4j_service.execute_cypher(
            cypher_query, 
            {'document_id': document_id, 'limit': limit}
        )
        
        # Format relationships for response
        formatted_relationships = []
        for result in results:
            formatted_relationships.append({
                'source_entity': result.get('source_name', 'unknown'),
                'target_entity': result.get('target_name', 'unknown'),
                'relationship_type': result.get('relationship_type', 'unknown'),
                'confidence': result.get('confidence', 0.0),
                'context': result.get('context', ''),
                'chunk_id': result.get('chunk_id', ''),
                'created_at': result.get('created_at', '')
            })
        
        return {
            'document_id': document_id,
            'relationships': formatted_relationships,
            'total_count': len(formatted_relationships)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get relationships for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve relationships: {str(e)}")

@router.get("/stats", response_model=GraphStatsResponse)
async def get_knowledge_graph_stats():
    """
    Get overall statistics about the knowledge graph.
    """
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        # Get database info
        db_info = neo4j_service.get_database_info()
        
        # Get entity type distribution
        entity_types_query = """
        MATCH (n)
        RETURN labels(n)[0] as entity_type, count(n) as count
        ORDER BY count DESC
        """
        entity_results = neo4j_service.execute_cypher(entity_types_query)
        entity_types = {result['entity_type']: result['count'] for result in entity_results}
        
        # Get relationship type distribution
        relationship_types_query = """
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(r) as count
        ORDER BY count DESC
        """
        rel_results = neo4j_service.execute_cypher(relationship_types_query)
        relationship_types = {result['relationship_type']: result['count'] for result in rel_results}
        
        # Get document count
        doc_count_query = """
        MATCH (n)
        WHERE n.document_id IS NOT NULL
        RETURN count(DISTINCT n.document_id) as doc_count
        """
        doc_results = neo4j_service.execute_cypher(doc_count_query)
        documents_processed = doc_results[0]['doc_count'] if doc_results else 0
        
        return GraphStatsResponse(
            total_entities=db_info.get('node_count', 0),
            total_relationships=db_info.get('relationship_count', 0),
            entity_types=entity_types,
            relationship_types=relationship_types,
            documents_processed=documents_processed,
            last_updated=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get knowledge graph stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")

@router.post("/query")
async def query_knowledge_graph(request: GraphQueryRequest):
    """
    Query the knowledge graph using Cypher or natural language.
    """
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        if request.query_type == "cypher":
            # Execute raw Cypher query
            results = neo4j_service.execute_cypher(request.query)
            limited_results = results[:request.limit]
            
            return {
                'query': request.query,
                'query_type': 'cypher',
                'results': limited_results,
                'total_count': len(results),
                'returned_count': len(limited_results)
            }
        else:
            # Natural language query - would need LLM integration
            # For now, return a simple response
            return {
                'query': request.query,
                'query_type': 'natural',
                'results': [],
                'total_count': 0,
                'returned_count': 0,
                'message': 'Natural language queries not yet implemented'
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to query knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.delete("/document/{document_id}")
async def delete_document_from_graph(document_id: str):
    """
    Delete all entities and relationships for a specific document from the knowledge graph.
    """
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        # Delete all nodes and relationships for the document
        delete_query = """
        MATCH (n {document_id: $document_id})
        DETACH DELETE n
        """
        
        neo4j_service.execute_cypher(delete_query, {'document_id': document_id})
        
        return {
            'success': True,
            'message': f'Successfully deleted all graph data for document {document_id}',
            'document_id': document_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id} from graph: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@router.get("/health")
async def knowledge_graph_health():
    """Health check for knowledge graph services."""
    try:
        kg_service = get_knowledge_graph_service()
        neo4j_service = get_neo4j_service()
        config = get_knowledge_graph_settings()
        
        # Test Neo4j connection
        neo4j_status = neo4j_service.test_connection()
        
        return {
            "status": "healthy" if neo4j_status['success'] else "degraded",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "knowledge_graph_extraction": "available",
                "neo4j_database": "available" if neo4j_status['success'] else "unavailable",
                "graph_processor": "available"
            },
            "config": {
                "neo4j_enabled": config.get('neo4j', {}).get('enabled', False),
                "extraction_enabled": config.get('extraction', {}).get('enabled', True)
            },
            "neo4j_info": neo4j_status.get('database_info', {}) if neo4j_status['success'] else None
        }
    
    except Exception as e:
        logger.error(f"Knowledge graph health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")