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
            # Full processing with progressive storage
            graph_result = await graph_processor.process_document_for_graph(
                chunks=chunks,
                document_id=document_id,
                store_in_neo4j=store_in_neo4j,
                progressive_storage=True
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
            
            # Graph processing with progressive storage
            graph_processor = get_graph_document_processor()
            graph_result = await graph_processor.process_document_for_graph(
                chunks=chunks,
                document_id=document_id,
                store_in_neo4j=store_in_neo4j,
                progressive_storage=True
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

@router.get("/entities")
async def get_all_entities(limit: int = 1000):
    """
    Get all entities from the knowledge graph.
    """
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        # Get all entities without document filtering
        query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
        RETURN n.id as id, n.name as name, 
               COALESCE(n.type, labels(n)[0]) as type, 
               COALESCE(n.confidence, 0.5) as confidence,
               COALESCE(n.original_text, '') as original_text,
               COALESCE(n.chunk_id, '') as chunk_id,
               COALESCE(n.created_at, '') as created_at,
               COALESCE(n.document_id, '') as document_id
        ORDER BY n.name
        LIMIT $limit
        """
        
        entities = neo4j_service.execute_cypher(query, {'limit': limit})
        
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
                'created_at': entity.get('created_at', ''),
                'document_id': entity.get('document_id', '')
            })
        
        return {
            'entities': formatted_entities,
            'total_count': len(formatted_entities)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get all entities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve entities: {str(e)}")

@router.get("/entities/{document_id}")
async def get_document_entities(document_id: str, limit: int = 100):
    """
    Get all entities extracted from a specific document.
    """
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        # Handle special case for "all" documents
        if document_id.lower() == "all":
            return await get_all_entities(max(limit, 1000))  # Ensure we get all entities
        
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

@router.get("/relationships")
async def get_all_relationships(limit: int = 1000):
    """
    Get all relationships from the knowledge graph.
    """
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        # Get all relationships without document filtering
        cypher_query = """
        MATCH (a)-[r]->(b)
        RETURN a.id as source_entity_id, a.name as source_name, 
               type(r) as relationship_type, 
               b.id as target_entity_id, b.name as target_name,
               COALESCE(r.confidence, 0.5) as confidence, 
               COALESCE(r.context, '') as context, 
               COALESCE(r.chunk_id, '') as chunk_id,
               COALESCE(r.created_at, '') as created_at, 
               COALESCE(r.document_id, '') as document_id
        ORDER BY relationship_type, source_name, target_name
        LIMIT $limit
        """
        
        results = neo4j_service.execute_cypher(cypher_query, {'limit': limit})
        
        # Format relationships for response
        formatted_relationships = []
        for result in results:
            formatted_relationships.append({
                'id': f"{result.get('source_entity_id', 'unknown')}_{result.get('target_entity_id', 'unknown')}_{result.get('relationship_type', 'unknown')}",
                'source_entity': result.get('source_entity_id', result.get('source_name', 'unknown')),
                'target_entity': result.get('target_entity_id', result.get('target_name', 'unknown')),
                'source_name': result.get('source_name', 'unknown'),
                'target_name': result.get('target_name', 'unknown'),
                'relationship_type': result.get('relationship_type', 'unknown'),
                'confidence': result.get('confidence', 0.0),
                'context': result.get('context', ''),
                'chunk_id': result.get('chunk_id', ''),
                'created_at': result.get('created_at', ''),
                'document_id': result.get('document_id', '')
            })
        
        return {
            'relationships': formatted_relationships,
            'total_count': len(formatted_relationships)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get all relationships: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve relationships: {str(e)}")

@router.get("/relationships/{document_id}")
async def get_document_relationships(document_id: str, limit: int = 100):
    """
    Get all relationships extracted from a specific document.
    """
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        # Handle special case for "all" documents
        if document_id.lower() == "all":
            return await get_all_relationships(max(limit, 1000))  # Ensure we get all relationships
        
        # Use custom query to get relationships for the document
        # Return both IDs and names for proper frontend linking
        cypher_query = """
        MATCH (a)-[r]->(b)
        WHERE r.document_id = $document_id
        RETURN a.id as source_entity_id, a.name as source_name, 
               type(r) as relationship_type, 
               b.id as target_entity_id, b.name as target_name,
               r.confidence as confidence, r.context as context, r.chunk_id as chunk_id,
               r.created_at as created_at, r.document_id as document_id
        LIMIT $limit
        """
        
        results = neo4j_service.execute_cypher(
            cypher_query, 
            {'document_id': document_id, 'limit': limit}
        )
        
        # Format relationships for response using entity IDs for proper frontend linking
        formatted_relationships = []
        for result in results:
            formatted_relationships.append({
                'id': f"{result.get('source_entity_id', 'unknown')}_{result.get('target_entity_id', 'unknown')}_{result.get('relationship_type', 'unknown')}",
                'source_entity': result.get('source_entity_id', result.get('source_name', 'unknown')),  # Use ID, fallback to name
                'target_entity': result.get('target_entity_id', result.get('target_name', 'unknown')),  # Use ID, fallback to name
                'source_name': result.get('source_name', 'unknown'),  # Keep names for display
                'target_name': result.get('target_name', 'unknown'),  # Keep names for display
                'relationship_type': result.get('relationship_type', 'unknown'),
                'confidence': result.get('confidence', 0.0),
                'context': result.get('context', ''),
                'chunk_id': result.get('chunk_id', ''),
                'created_at': result.get('created_at', ''),
                'document_id': result.get('document_id', document_id)
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
        
        # Get entity type distribution using the type property instead of labels
        entity_types_query = """
        MATCH (n)
        RETURN COALESCE(n.type, labels(n)[0]) as entity_type, count(n) as count
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

@router.get("/debug/{document_id}")
async def debug_document_data(document_id: str):
    """Debug endpoint to inspect raw Neo4j data for a specific document"""
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        # Get raw entity data
        entities_query = """
        MATCH (n) 
        WHERE n.document_id = $document_id
        RETURN n.id as id, n.name as name, n.type as type, n.confidence as confidence,
               n.original_text as original_text, n.chunk_id as chunk_id, labels(n) as labels
        """
        raw_entities = neo4j_service.execute_cypher(entities_query, {'document_id': document_id})
        
        # Get raw relationship data
        relationships_query = """
        MATCH (a)-[r]->(b)
        WHERE r.document_id = $document_id
        RETURN a.id as source_id, a.name as source_name, 
               type(r) as relationship_type,
               b.id as target_id, b.name as target_name,
               r.confidence as confidence, r.context as context,
               r.chunk_id as chunk_id, r.document_id as document_id
        """
        raw_relationships = neo4j_service.execute_cypher(relationships_query, {'document_id': document_id})
        
        # Check for orphaned entities (entities without relationships)
        orphaned_query = """
        MATCH (n)
        WHERE n.document_id = $document_id 
        AND NOT (n)-[]-()
        RETURN n.id as id, n.name as name, n.type as type
        """
        orphaned_entities = neo4j_service.execute_cypher(orphaned_query, {'document_id': document_id})
        
        # Get document statistics
        stats_query = """
        MATCH (n)
        WHERE n.document_id = $document_id
        RETURN count(n) as total_entities
        """
        entity_stats = neo4j_service.execute_cypher(stats_query, {'document_id': document_id})
        total_entities = entity_stats[0]['total_entities'] if entity_stats else 0
        
        rel_stats_query = """
        MATCH ()-[r]->()
        WHERE r.document_id = $document_id
        RETURN count(r) as total_relationships
        """
        rel_stats = neo4j_service.execute_cypher(rel_stats_query, {'document_id': document_id})
        total_relationships = rel_stats[0]['total_relationships'] if rel_stats else 0
        
        return {
            'document_id': document_id,
            'summary': {
                'total_entities_in_neo4j': total_entities,
                'total_relationships_in_neo4j': total_relationships,
                'orphaned_entities': len(orphaned_entities),
                'data_exists': total_entities > 0
            },
            'raw_entities': raw_entities,
            'raw_relationships': raw_relationships,
            'orphaned_entities': orphaned_entities,
            'analysis': {
                'has_entities': len(raw_entities) > 0,
                'has_relationships': len(raw_relationships) > 0,
                'entities_without_connections': len(orphaned_entities),
                'relationship_coverage': f"{len(raw_relationships)}/{len(raw_entities)} entities have relationships" if raw_entities else "No entities found"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to debug document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@router.post("/deduplicate")
async def deduplicate_entities():
    """Remove duplicate entities with the same ID, keeping the highest confidence one."""
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        result = neo4j_service.deduplicate_entities()
        
        if result['success']:
            return {
                'success': True,
                'message': result['message'],
                'total_duplicates_removed': result['total_duplicates_removed'],
                'entities_cleaned': result['entities_cleaned']
            }
        else:
            raise HTTPException(status_code=500, detail=result['error'])
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deduplicate entities: {e}")
        raise HTTPException(status_code=500, detail=f"Deduplication failed: {str(e)}")

@router.get("/isolated-nodes")
async def get_isolated_nodes():
    """Get truly isolated nodes with comprehensive debugging information."""
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        # Method 1: Neo4j Service method (deduplication-aware)
        isolated_method1 = neo4j_service.get_truly_isolated_nodes()
        
        # Method 2: Simple isolation check (like anti-silo service)
        simple_isolation_query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
        WITH n, [(n)-[r]-(other) | r] as relationships
        WHERE size(relationships) = 0
        RETURN n.id as id, n.name as name, n.type as type, labels(n) as labels
        ORDER BY n.name
        """
        isolated_method2 = neo4j_service.execute_cypher(simple_isolation_query)
        
        # Method 3: Direct relationship count check
        direct_count_query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as rel_count
        WHERE rel_count = 0
        RETURN n.id as id, n.name as name, n.type as type, labels(n) as labels, rel_count
        ORDER BY n.name
        """
        isolated_method3 = neo4j_service.execute_cypher(direct_count_query)
        
        # Method 4: Check for nodes with no incoming or outgoing relationships
        no_relationships_query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
        AND NOT (n)--()
        RETURN n.id as id, n.name as name, n.type as type, labels(n) as labels
        ORDER BY n.name
        """
        isolated_method4 = neo4j_service.execute_cypher(no_relationships_query)
        
        # Compare results
        method1_ids = {node['id'] for node in isolated_method1}
        method2_ids = {node['id'] for node in isolated_method2}
        method3_ids = {node['id'] for node in isolated_method3}
        method4_ids = {node['id'] for node in isolated_method4}
        
        # Find discrepancies
        all_isolated_ids = method1_ids | method2_ids | method3_ids | method4_ids
        
        discrepancy_analysis = {}
        for node_id in all_isolated_ids:
            discrepancy_analysis[node_id] = {
                'in_method1': node_id in method1_ids,
                'in_method2': node_id in method2_ids,
                'in_method3': node_id in method3_ids,
                'in_method4': node_id in method4_ids,
                'node_info': next((node for node in isolated_method2 if node['id'] == node_id), 
                                 next((node for node in isolated_method1 if node['id'] == node_id), None))
            }
        
        return {
            'success': True,
            'isolation_methods': {
                'method1_deduplication_aware': {
                    'count': len(isolated_method1),
                    'nodes': isolated_method1[:10]  # Limit for readability
                },
                'method2_simple_check': {
                    'count': len(isolated_method2),
                    'nodes': isolated_method2[:10]
                },
                'method3_direct_count': {
                    'count': len(isolated_method3),
                    'nodes': isolated_method3[:10]
                },
                'method4_no_relationships': {
                    'count': len(isolated_method4),
                    'nodes': isolated_method4[:10]
                }
            },
            'summary': {
                'method1_count': len(isolated_method1),
                'method2_count': len(isolated_method2),
                'method3_count': len(isolated_method3),
                'method4_count': len(isolated_method4),
                'total_unique_isolated': len(all_isolated_ids),
                'methods_agree': len(method1_ids) == len(method2_ids) == len(method3_ids) == len(method4_ids) and method1_ids == method2_ids == method3_ids == method4_ids
            },
            'discrepancy_analysis': discrepancy_analysis,
            'message': f'Found isolated nodes: Method1={len(isolated_method1)}, Method2={len(isolated_method2)}, Method3={len(isolated_method3)}, Method4={len(isolated_method4)}'
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get isolated nodes: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.post("/force-nuclear-anti-silo")
async def force_nuclear_anti_silo():
    """Manually trigger nuclear anti-silo processing to eliminate any remaining isolated nodes."""
    try:
        from app.services.knowledge_graph_service import get_knowledge_graph_service
        
        neo4j_service = get_neo4j_service()
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        kg_service = get_knowledge_graph_service()
        
        # Check isolated nodes before processing
        isolated_before = neo4j_service.get_truly_isolated_nodes()
        
        # Force nuclear anti-silo processing
        connections_made = await kg_service._nuclear_anti_silo_elimination(neo4j_service)
        
        # Check isolated nodes after processing
        isolated_after = neo4j_service.get_truly_isolated_nodes()
        
        return {
            'success': True,
            'processing_result': {
                'isolated_before': len(isolated_before),
                'isolated_after': len(isolated_after),
                'connections_made': connections_made,
                'nodes_eliminated': len(isolated_before) - len(isolated_after),
                'elimination_success': len(isolated_after) == 0
            },
            'remaining_isolated_nodes': isolated_after,
            'message': f'Nuclear anti-silo processing complete. Eliminated {len(isolated_before) - len(isolated_after)} isolated nodes with {connections_made} new connections.'
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run nuclear anti-silo: {e}")
        raise HTTPException(status_code=500, detail=f"Nuclear processing failed: {str(e)}")

@router.get("/frontend-validation")
async def validate_frontend_data():
    """Validate that frontend data has proper entity-relationship mapping."""
    try:
        neo4j_service = get_neo4j_service()
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        # Get entities (same as frontend receives)
        entities_query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
        RETURN n.id as id, n.name as name, 
               COALESCE(n.type, labels(n)[0]) as type, 
               COALESCE(n.confidence, 0.5) as confidence
        ORDER BY n.name
        LIMIT 1000
        """
        entities = neo4j_service.execute_cypher(entities_query)
        
        # Get relationships (same as frontend receives)
        relationships_query = """
        MATCH (a)-[r]->(b)
        RETURN a.id as source_entity_id, a.name as source_name, 
               type(r) as relationship_type, 
               b.id as target_entity_id, b.name as target_name,
               COALESCE(r.confidence, 0.5) as confidence
        ORDER BY relationship_type, source_name, target_name
        LIMIT 1000
        """
        relationships = neo4j_service.execute_cypher(relationships_query)
        
        # Analyze connectivity like frontend would
        entity_ids = {e['id'] for e in entities}
        entity_names = {e['name'] for e in entities}
        
        # Check relationship validity
        valid_relationships = []
        invalid_relationships = []
        entities_with_connections = set()
        
        for rel in relationships:
            source_id = rel['source_entity_id']
            target_id = rel['target_entity_id']
            
            source_valid = source_id in entity_ids
            target_valid = target_id in entity_ids
            
            if source_valid and target_valid:
                valid_relationships.append(rel)
                entities_with_connections.add(source_id)
                entities_with_connections.add(target_id)
            else:
                invalid_relationships.append({
                    'relationship': rel,
                    'source_valid': source_valid,
                    'target_valid': target_valid
                })
        
        # Find entities without connections
        isolated_entities = []
        for entity in entities:
            if entity['id'] not in entities_with_connections:
                isolated_entities.append(entity)
        
        # Analyze connection distribution
        connection_count = {}
        for rel in valid_relationships:
            source_id = rel['source_entity_id']
            target_id = rel['target_entity_id']
            connection_count[source_id] = connection_count.get(source_id, 0) + 1
            connection_count[target_id] = connection_count.get(target_id, 0) + 1
        
        # Find weakly connected entities (1-2 connections)
        weakly_connected = []
        for entity in entities:
            count = connection_count.get(entity['id'], 0)
            if 1 <= count <= 2:
                weakly_connected.append({
                    'entity': entity,
                    'connection_count': count
                })
        
        return {
            'success': True,
            'entity_count': len(entities),
            'relationship_count': len(relationships),
            'valid_relationships': len(valid_relationships),
            'invalid_relationships': len(invalid_relationships),
            'isolated_entities': {
                'count': len(isolated_entities),
                'entities': isolated_entities
            },
            'weakly_connected': {
                'count': len(weakly_connected),
                'entities': weakly_connected[:10]  # Top 10 weakly connected
            },
            'connectivity_stats': {
                'fully_connected': len(entities_with_connections),
                'average_connections': sum(connection_count.values()) / len(connection_count) if connection_count else 0,
                'max_connections': max(connection_count.values()) if connection_count else 0,
                'min_connections': min(connection_count.values()) if connection_count else 0
            },
            'sample_invalid': invalid_relationships[:5] if invalid_relationships else []
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate frontend data: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/connectivity-analysis")
async def analyze_graph_connectivity():
    """Analyze knowledge graph connectivity and component structure."""
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        # Get basic graph stats
        basic_stats_query = """
        MATCH (n) 
        RETURN count(n) as total_nodes
        """
        basic_result = neo4j_service.execute_cypher(basic_stats_query)
        total_nodes = basic_result[0]['total_nodes'] if basic_result else 0
        
        # Get relationship stats
        rel_stats_query = """
        MATCH ()-[r]->()
        RETURN count(r) as total_relationships
        """
        rel_result = neo4j_service.execute_cypher(rel_stats_query)
        total_relationships = rel_result[0]['total_relationships'] if rel_result else 0
        
        # Find weakly connected components (treating relationships as undirected)
        components_query = """
        CALL gds.graph.project('connectivity-graph', '*', '*', {undirectedRelationshipTypes: ['*']})
        YIELD graphName, nodeCount, relationshipCount
        CALL gds.wcc.stream('connectivity-graph')
        YIELD nodeId, componentId
        WITH gds.util.asNode(nodeId) as node, componentId
        WITH componentId, collect({id: node.id, name: node.name, type: node.type}) as nodes
        WITH componentId, nodes, size(nodes) as componentSize
        ORDER BY componentSize DESC
        RETURN componentId, componentSize, nodes[0..5] as sampleNodes
        """
        
        # Try using built-in graph algorithms if available, otherwise use custom logic
        try:
            components_result = neo4j_service.execute_cypher(components_query)
            
            # Clean up the projected graph
            cleanup_query = "CALL gds.graph.drop('connectivity-graph') YIELD graphName"
            neo4j_service.execute_cypher(cleanup_query)
            
        except Exception as e:
            logger.warning(f"GDS not available, using custom connectivity analysis: {e}")
            # Fallback to custom connectivity analysis
            components_result = []
        
        # If GDS failed, use simple connectivity analysis
        if not components_result:
            # Find nodes with most connections (hub analysis)
            hub_analysis_query = """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) as degree
            WHERE degree > 0
            RETURN n.id as id, n.name as name, n.type as type, degree
            ORDER BY degree DESC
            LIMIT 10
            """
            hub_nodes = neo4j_service.execute_cypher(hub_analysis_query)
            
            # Count isolated nodes
            isolated_query = """
            MATCH (n)
            WHERE NOT (n)-[]-()
            RETURN count(n) as isolated_count
            """
            isolated_result = neo4j_service.execute_cypher(isolated_query)
            isolated_count = isolated_result[0]['isolated_count'] if isolated_result else 0
            
            # Relationship type distribution
            rel_type_query = """
            MATCH ()-[r]->()
            RETURN type(r) as relationship_type, count(r) as count
            ORDER BY count DESC
            """
            rel_types = neo4j_service.execute_cypher(rel_type_query)
            
            return {
                'total_nodes': total_nodes,
                'total_relationships': total_relationships,
                'isolated_nodes': isolated_count,
                'connected_nodes': total_nodes - isolated_count,
                'connectivity_ratio': (total_nodes - isolated_count) / total_nodes if total_nodes > 0 else 0,
                'average_degree': (total_relationships * 2) / total_nodes if total_nodes > 0 else 0,
                'hub_nodes': hub_nodes,
                'relationship_types': rel_types,
                'components_analysis': 'GDS not available - using basic connectivity metrics',
                'recommendations': [
                    f"Graph has {isolated_count} isolated nodes ({isolated_count/total_nodes*100:.1f}%)" if total_nodes > 0 else "No nodes in graph",
                    f"Average node degree: {(total_relationships * 2) / total_nodes:.2f}" if total_nodes > 0 else "No relationships",
                    "Consider improving relationship extraction for better connectivity" if isolated_count > total_nodes * 0.3 else "Good connectivity"
                ]
            }
        
        # If GDS worked, process component results
        total_components = len(components_result)
        largest_component = components_result[0] if components_result else None
        small_components = [c for c in components_result if c['componentSize'] <= 3]
        
        return {
            'total_nodes': total_nodes,
            'total_relationships': total_relationships,
            'total_components': total_components,
            'largest_component_size': largest_component['componentSize'] if largest_component else 0,
            'small_components': len(small_components),
            'connectivity_ratio': 1 - (len(small_components) / total_nodes) if total_nodes > 0 else 0,
            'average_component_size': total_nodes / total_components if total_components > 0 else 0,
            'components_summary': components_result[:5],  # Top 5 largest components
            'recommendations': [
                f"Graph has {total_components} connected components",
                f"Largest component contains {largest_component['componentSize'] if largest_component else 0} nodes",
                f"{len(small_components)} small components (≤3 nodes) may need better relationships",
                "Good connectivity" if total_components < total_nodes * 0.1 else "Consider improving relationship extraction"
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze graph connectivity: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/quality-assessment")
async def assess_knowledge_graph_quality():
    """Comprehensive quality assessment of the knowledge graph."""
    try:
        from app.services.quality_assessment_service import get_quality_assessment_service
        quality_service = get_quality_assessment_service()
        
        metrics = await quality_service.assess_graph_quality()
        
        return {
            "overall_quality_score": metrics.overall_quality_score,
            "component_scores": {
                "connectivity": metrics.connectivity_score,
                "relationship_quality": metrics.relationship_quality_score,
                "entity_quality": metrics.entity_quality_score
            },
            "basic_metrics": {
                "total_entities": metrics.total_entities,
                "total_relationships": metrics.total_relationships,
                "isolated_entities": metrics.isolated_entities,
                "connectivity_ratio": metrics.connectivity_ratio,
                "average_degree": metrics.average_degree
            },
            "relationship_quality": {
                "generic_ratio": metrics.generic_relationship_ratio,
                "semantic_ratio": metrics.semantic_relationship_ratio,
                "high_confidence_count": metrics.high_confidence_relationships,
                "low_confidence_count": metrics.low_confidence_relationships
            },
            "entity_metrics": {
                "type_distribution": metrics.entity_type_distribution,
                "entities_with_attributes": metrics.entities_with_attributes,
                "cross_document_entities": metrics.cross_document_entities
            },
            "quality_issues": {
                "naming_issues": metrics.potential_naming_issues,
                "questionable_relationships": metrics.questionable_relationships,
                "classification_errors": metrics.classification_errors
            },
            "assessment_timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

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
                "extraction_enabled": config.get('extraction', {}).get('enabled', True),
                "llm_enhancement_enabled": config.get('extraction', {}).get('enable_llm_enhancement', False),
                "cross_document_linking": config.get('extraction', {}).get('enable_cross_document_linking', False),
                "multi_chunk_processing": config.get('extraction', {}).get('enable_multi_chunk_relationships', False)
            },
            "neo4j_info": neo4j_status.get('database_info', {}) if neo4j_status['success'] else None
        }
    
    except Exception as e:
        logger.error(f"Knowledge graph health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@router.post("/global-anti-silo-cleanup")
async def run_global_anti_silo_cleanup():
    """
    Run comprehensive global anti-silo analysis to eliminate isolated nodes.
    
    This endpoint performs a system-wide analysis to find and link entities
    that should be connected across documents, reducing silo nodes.
    """
    try:
        logger.info("🌍 Starting global anti-silo cleanup via API request")
        
        kg_service = get_knowledge_graph_service()
        
        # Run the global analysis
        results = await kg_service.run_global_anti_silo_analysis()
        
        return {
            "success": True,
            "message": "Global anti-silo analysis completed successfully",
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "silo_nodes_analyzed": results.get('entities_analyzed', 0),
                "cross_document_links_created": results.get('cross_document_links_created', 0),
                "potential_silos_found": results.get('silo_nodes_found', 0),
                "errors_encountered": len(results.get('errors', []))
            }
        }
        
    except Exception as e:
        logger.error(f"Global anti-silo cleanup failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Global anti-silo cleanup failed: {str(e)}"
        )

@router.get("/silo-analysis")
async def get_silo_analysis():
    """
    Analyze current silo nodes in the knowledge graph.
    
    Returns information about entities that have few or no connections,
    helping identify potential data silos.
    """
    try:
        neo4j_service = get_neo4j_service()
        
        # Query for silo nodes (entities with 2 or fewer connections)
        silo_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-(connected)
        WITH n, count(DISTINCT connected) as connection_count
        WHERE connection_count <= 2
        RETURN n.id as entity_id, n.name as name, n.type as type, 
               n.document_id as document_id, connection_count,
               n.confidence as confidence
        ORDER BY connection_count ASC, n.name ASC
        """
        
        silo_nodes = neo4j_service.execute_cypher(silo_query)
        
        # Group by connection count for analysis
        analysis = {
            'isolated_nodes': [],      # 0 connections
            'weakly_connected': [],    # 1-2 connections  
            'total_silos': len(silo_nodes),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        for node in silo_nodes:
            node_info = {
                'entity_id': node['entity_id'],
                'name': node['name'],
                'type': node['type'],
                'document_id': node['document_id'],
                'connections': node['connection_count'],
                'confidence': node.get('confidence', 0.0)
            }
            
            if node['connection_count'] == 0:
                analysis['isolated_nodes'].append(node_info)
            else:
                analysis['weakly_connected'].append(node_info)
        
        # Add summary statistics
        analysis['summary'] = {
            'total_entities_analyzed': len(silo_nodes),
            'isolated_count': len(analysis['isolated_nodes']),
            'weakly_connected_count': len(analysis['weakly_connected']),
            'silo_percentage': round((len(silo_nodes) / max(1, len(silo_nodes))) * 100, 2)
        }
        
        return {
            "success": True,
            "analysis": analysis,
            "message": f"Found {len(silo_nodes)} potential silo nodes"
        }
        
    except Exception as e:
        logger.error(f"Silo analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Silo analysis failed: {str(e)}"
        )

@router.get("/debug/entity-types")
async def debug_entity_types():
    """Debug endpoint to verify entity types in Neo4j database"""
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service not available")
        
        # Query to get entity type distribution
        type_distribution_query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
        WITH n.type as entity_type, count(n) as count, collect(n.name)[0..5] as sample_names
        RETURN entity_type, count, sample_names
        ORDER BY count DESC
        """
        
        type_results = neo4j_service.execute_cypher(type_distribution_query)
        
        # Query to get recent entities with their types
        recent_entities_query = """
        MATCH (n)
        WHERE n.name IS NOT NULL AND n.created_at IS NOT NULL
        RETURN n.name as name, n.type as type, n.created_at as created_at, 
               n.confidence as confidence, n.document_id as document_id
        ORDER BY n.created_at DESC
        LIMIT 20
        """
        
        recent_results = neo4j_service.execute_cypher(recent_entities_query)
        
        # Query to check for null types
        null_type_query = """
        MATCH (n)
        WHERE n.name IS NOT NULL AND (n.type IS NULL OR n.type = '')
        RETURN count(n) as null_type_count, collect(n.name)[0..10] as sample_null_entities
        """
        
        null_results = neo4j_service.execute_cypher(null_type_query)
        
        return {
            "success": True,
            "entity_type_distribution": type_results,
            "recent_entities": recent_results,
            "null_type_analysis": null_results[0] if null_results else {"null_type_count": 0, "sample_null_entities": []},
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Debug entity types failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Debug entity types failed: {str(e)}"
        )