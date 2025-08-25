"""
Chunk Management Service for universal chunk editing operations.
Handles chunk retrieval, editing, and re-embedding for both documents and memories.
"""

import logging
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, DatabaseError
from sqlalchemy import and_, func, text, or_, desc

from app.models.notebook_models import (
    ChunkResponse, ChunkListResponse, ChunkEditHistory, ChunkOperationResponse,
    BulkChunkOperationResponse, ContentType
)
from app.core.db import get_db
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings

# Milvus imports
from pymilvus import connections, Collection, utility
import requests

logger = logging.getLogger(__name__)

class ChunkManagementService:
    """
    Service for managing chunk operations across documents and memories.
    Provides universal chunk editing, re-embedding, and history tracking.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def get_chunks_for_document(
        self,
        db: Session,
        collection_name: str,
        document_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> ChunkListResponse:
        """
        Get all chunks for a specific document or memory.
        
        Args:
            db: Database session
            collection_name: Milvus collection name
            document_id: Document or memory ID
            page: Page number for pagination
            page_size: Number of chunks per page
            
        Returns:
            List of chunks with pagination info
        """
        try:
            self.logger.info(f"Retrieving chunks for document {document_id} from collection {collection_name}")
            
            # Get vector database settings
            vector_settings = get_vector_db_settings()
            if not vector_settings:
                raise ValueError("Vector database settings not found")
            
            milvus_uri = vector_settings.get('uri')
            milvus_token = vector_settings.get('token')
            
            # Connect to Milvus
            connections.connect(uri=milvus_uri, token=milvus_token)
            
            if not utility.has_collection(collection_name):
                raise ValueError(f"Collection {collection_name} not found")
            
            collection = Collection(collection_name)
            collection.load()
            
            # Calculate offset
            offset = (page - 1) * page_size
            
            # Query chunks for this document
            search_expr = f'doc_id == "{document_id}"'
            
            # Get total count
            count_result = collection.query(
                expr=search_expr,
                output_fields=["id"],
                limit=16384  # Max limit
            )
            total_count = len(count_result)
            
            # Get paginated results
            results = collection.query(
                expr=search_expr,
                output_fields=["id", "content", "vector", "doc_id", "source", "page", 
                              "doc_type", "uploaded_at", "section", "author"],
                offset=offset,
                limit=page_size
            )
            
            # Determine content type from knowledge_graph_documents
            content_type_query = text("""
                SELECT content_type, edited_chunks_count 
                FROM knowledge_graph_documents 
                WHERE document_id = :doc_id
            """)
            
            kg_result = db.execute(content_type_query, {'doc_id': document_id}).fetchone()
            content_type = ContentType.DOCUMENT
            edited_chunks_count = 0
            
            if kg_result:
                content_type = ContentType(kg_result.content_type or 'document')
                edited_chunks_count = kg_result.edited_chunks_count or 0
            
            # Get edit history for all chunks
            chunk_ids = [result['id'] for result in results]
            edit_history = await self._get_edit_history_for_chunks(db, chunk_ids)
            
            # Build chunk responses
            chunks = []
            for result in results:
                chunk_id = result['id']
                chunk_edit_history = edit_history.get(chunk_id, [])
                last_edited = None
                if chunk_edit_history:
                    last_edited = max(edit['edited_at'] for edit in chunk_edit_history)
                
                chunks.append(ChunkResponse(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content_type=content_type,
                    content=result['content'],
                    vector=result.get('vector'),
                    metadata={
                        'source': result.get('source', ''),
                        'page': result.get('page', 0),
                        'doc_type': result.get('doc_type', ''),
                        'uploaded_at': result.get('uploaded_at', ''),
                        'section': result.get('section', ''),
                        'author': result.get('author', '')
                    },
                    edit_history=[ChunkEditHistory(**edit) for edit in chunk_edit_history],
                    last_edited=datetime.fromisoformat(last_edited) if last_edited else None
                ))
            
            return ChunkListResponse(
                chunks=chunks,
                total_count=total_count,
                document_id=document_id,
                content_type=content_type,
                edited_chunks_count=edited_chunks_count
            )
            
        except Exception as e:
            self.logger.error(f"Error retrieving chunks for document {document_id}: {str(e)}")
            raise
    
    async def get_chunk_by_id(
        self,
        db: Session,
        collection_name: str,
        chunk_id: str
    ) -> ChunkResponse:
        """
        Get a specific chunk by ID.
        
        Args:
            db: Database session
            collection_name: Milvus collection name
            chunk_id: Chunk ID to retrieve
            
        Returns:
            Chunk details with edit history
        """
        try:
            self.logger.info(f"Retrieving chunk {chunk_id} from collection {collection_name}")
            
            # Get vector database settings
            vector_settings = get_vector_db_settings()
            if not vector_settings:
                raise ValueError("Vector database settings not found")
            
            milvus_uri = vector_settings.get('uri')
            milvus_token = vector_settings.get('token')
            
            # Connect to Milvus
            connections.connect(uri=milvus_uri, token=milvus_token)
            
            if not utility.has_collection(collection_name):
                raise ValueError(f"Collection {collection_name} not found")
            
            collection = Collection(collection_name)
            collection.load()
            
            # Query specific chunk
            results = collection.query(
                expr=f'id == "{chunk_id}"',
                output_fields=["id", "content", "vector", "doc_id", "source", "page",
                              "doc_type", "uploaded_at", "section", "author"],
                limit=1
            )
            
            if not results:
                raise ValueError(f"Chunk {chunk_id} not found")
            
            result = results[0]
            document_id = result['doc_id']
            
            # Determine content type
            content_type_query = text("""
                SELECT content_type 
                FROM knowledge_graph_documents 
                WHERE document_id = :doc_id
            """)
            
            kg_result = db.execute(content_type_query, {'doc_id': document_id}).fetchone()
            content_type = ContentType.DOCUMENT
            if kg_result:
                content_type = ContentType(kg_result.content_type or 'document')
            
            # Get edit history
            edit_history = await self._get_edit_history_for_chunks(db, [chunk_id])
            chunk_edit_history = edit_history.get(chunk_id, [])
            last_edited = None
            if chunk_edit_history:
                last_edited = max(edit['edited_at'] for edit in chunk_edit_history)
            
            return ChunkResponse(
                chunk_id=chunk_id,
                document_id=document_id,
                content_type=content_type,
                content=result['content'],
                vector=result.get('vector'),
                metadata={
                    'source': result.get('source', ''),
                    'page': result.get('page', 0),
                    'doc_type': result.get('doc_type', ''),
                    'uploaded_at': result.get('uploaded_at', ''),
                    'section': result.get('section', ''),
                    'author': result.get('author', '')
                },
                edit_history=[ChunkEditHistory(**edit) for edit in chunk_edit_history],
                last_edited=datetime.fromisoformat(last_edited) if last_edited else None
            )
            
        except Exception as e:
            self.logger.error(f"Error retrieving chunk {chunk_id}: {str(e)}")
            raise
    
    async def update_chunk(
        self,
        db: Session,
        collection_name: str,
        chunk_id: str,
        new_content: str,
        re_embed: bool = True,
        user_id: Optional[str] = None,
        edit_metadata: Optional[Dict[str, Any]] = None
    ) -> ChunkOperationResponse:
        """
        Update a chunk's content and optionally re-embed it.
        
        Args:
            db: Database session
            collection_name: Milvus collection name
            chunk_id: Chunk ID to update
            new_content: New content for the chunk
            re_embed: Whether to generate new embedding
            user_id: User making the edit
            edit_metadata: Additional metadata for the edit
            
        Returns:
            Operation result
        """
        try:
            self.logger.info(f"Updating chunk {chunk_id} in collection {collection_name}")
            
            # First, get the current chunk to store original content
            current_chunk = await self.get_chunk_by_id(db, collection_name, chunk_id)
            original_content = current_chunk.content
            
            # Get vector database settings
            vector_settings = get_vector_db_settings()
            if not vector_settings:
                raise ValueError("Vector database settings not found")
            
            milvus_uri = vector_settings.get('uri')
            milvus_token = vector_settings.get('token')
            
            # Connect to Milvus
            connections.connect(uri=milvus_uri, token=milvus_token)
            
            if not utility.has_collection(collection_name):
                raise ValueError(f"Collection {collection_name} not found")
            
            collection = Collection(collection_name)
            collection.load()
            
            # Generate new embedding if requested
            new_vector = None
            if re_embed:
                new_vector = await self._generate_embedding(new_content)
            else:
                # Keep the existing vector
                new_vector = current_chunk.vector
            
            # Delete the old chunk
            delete_expr = f'id == "{chunk_id}"'
            collection.delete(expr=delete_expr)
            
            # Insert updated chunk
            insert_data = [
                [chunk_id],  # id
                [new_vector],  # vector
                [new_content],  # content
                [current_chunk.metadata.get('source', '')],  # source
                [current_chunk.metadata.get('page', 0)],  # page
                [current_chunk.metadata.get('doc_type', '')],  # doc_type
                [current_chunk.metadata.get('uploaded_at', '')],  # uploaded_at
                [current_chunk.metadata.get('section', '')],  # section
                [current_chunk.metadata.get('author', '')],  # author
                [''],  # hash (will be recalculated)
                [current_chunk.document_id],  # doc_id
                [''],  # bm25_tokens
                [0],  # bm25_term_count
                [0],  # bm25_unique_terms
                [''],  # bm25_top_terms
                [current_chunk.metadata.get('uploaded_at', '')],  # creation_date
                [datetime.now().isoformat()],  # last_modified_date
            ]
            
            collection.insert(insert_data)
            collection.flush()
            
            # Record the edit in audit table
            await self._record_chunk_edit(
                db=db,
                chunk_id=chunk_id,
                document_id=current_chunk.document_id,
                content_type=current_chunk.content_type,
                original_content=original_content,
                edited_content=new_content,
                user_id=user_id,
                re_embedded=re_embed,
                metadata=edit_metadata
            )
            
            db.commit()
            
            self.logger.info(f"Successfully updated chunk {chunk_id}")
            
            return ChunkOperationResponse(
                success=True,
                chunk_id=chunk_id,
                message=f"Chunk updated successfully{'and re-embedded' if re_embed else ''}",
                re_embedded=re_embed
            )
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error updating chunk {chunk_id}: {str(e)}")
            return ChunkOperationResponse(
                success=False,
                chunk_id=chunk_id,
                message=f"Failed to update chunk: {str(e)}",
                re_embedded=False
            )
    
    async def bulk_re_embed_chunks(
        self,
        db: Session,
        collection_name: str,
        chunk_ids: List[str],
        user_id: Optional[str] = None
    ) -> BulkChunkOperationResponse:
        """
        Re-embed multiple chunks in bulk.
        
        Args:
            db: Database session
            collection_name: Milvus collection name
            chunk_ids: List of chunk IDs to re-embed
            user_id: User performing the operation
            
        Returns:
            Bulk operation results
        """
        try:
            self.logger.info(f"Bulk re-embedding {len(chunk_ids)} chunks in collection {collection_name}")
            
            operation_details = []
            successful_operations = 0
            failed_operations = 0
            
            for chunk_id in chunk_ids:
                try:
                    # Get current chunk
                    current_chunk = await self.get_chunk_by_id(db, collection_name, chunk_id)
                    
                    # Re-embed with same content
                    result = await self.update_chunk(
                        db=db,
                        collection_name=collection_name,
                        chunk_id=chunk_id,
                        new_content=current_chunk.content,
                        re_embed=True,
                        user_id=user_id,
                        edit_metadata={'bulk_re_embed': True}
                    )
                    
                    operation_details.append(result)
                    
                    if result.success:
                        successful_operations += 1
                    else:
                        failed_operations += 1
                        
                except Exception as e:
                    self.logger.error(f"Error re-embedding chunk {chunk_id}: {str(e)}")
                    operation_details.append(ChunkOperationResponse(
                        success=False,
                        chunk_id=chunk_id,
                        message=f"Failed to re-embed: {str(e)}",
                        re_embedded=False
                    ))
                    failed_operations += 1
            
            overall_success = failed_operations == 0
            
            return BulkChunkOperationResponse(
                success=overall_success,
                total_requested=len(chunk_ids),
                successful_operations=successful_operations,
                failed_operations=failed_operations,
                operation_details=operation_details,
                message=f"Re-embedded {successful_operations}/{len(chunk_ids)} chunks successfully"
            )
            
        except Exception as e:
            self.logger.error(f"Error in bulk re-embedding: {str(e)}")
            raise
    
    async def _get_edit_history_for_chunks(
        self,
        db: Session,
        chunk_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get edit history for multiple chunks.
        
        Args:
            db: Database session
            chunk_ids: List of chunk IDs
            
        Returns:
            Dictionary mapping chunk_id to list of edits
        """
        if not chunk_ids:
            return {}
        
        chunk_ids_str = "', '".join(chunk_ids)
        query = text(f"""
            SELECT chunk_id, id, original_content, edited_content, 
                   edited_by, edited_at, re_embedded, metadata
            FROM chunk_edits 
            WHERE chunk_id IN ('{chunk_ids_str}')
            ORDER BY edited_at DESC
        """)
        
        results = db.execute(query).fetchall()
        
        edit_history = {}
        for result in results:
            chunk_id = result.chunk_id
            if chunk_id not in edit_history:
                edit_history[chunk_id] = []
            
            edit_history[chunk_id].append({
                'id': str(result.id),
                'chunk_id': result.chunk_id,
                'original_content': result.original_content,
                'edited_content': result.edited_content,
                'edited_by': result.edited_by,
                'edited_at': result.edited_at.isoformat(),
                're_embedded': result.re_embedded,
                'metadata': result.metadata
            })
        
        return edit_history
    
    async def _record_chunk_edit(
        self,
        db: Session,
        chunk_id: str,
        document_id: str,
        content_type: ContentType,
        original_content: str,
        edited_content: str,
        user_id: Optional[str],
        re_embedded: bool,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """
        Record a chunk edit in the audit table.
        
        Args:
            db: Database session
            chunk_id: Chunk ID that was edited
            document_id: Parent document/memory ID
            content_type: Type of content (document or memory)
            original_content: Original chunk content
            edited_content: New chunk content
            user_id: User who made the edit
            re_embedded: Whether the chunk was re-embedded
            metadata: Additional edit metadata
        """
        insert_query = text("""
            INSERT INTO chunk_edits 
            (chunk_id, document_id, content_type, original_content, edited_content, 
             edited_by, re_embedded, metadata)
            VALUES (:chunk_id, :document_id, :content_type, :original_content, 
                    :edited_content, :edited_by, :re_embedded, :metadata)
        """)
        
        db.execute(insert_query, {
            'chunk_id': chunk_id,
            'document_id': document_id,
            'content_type': content_type.value,
            'original_content': original_content,
            'edited_content': edited_content,
            'edited_by': user_id,
            're_embedded': re_embedded,
            'metadata': json.dumps(metadata) if metadata else None
        })
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using the configured embedding service.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embedding_settings = get_embedding_settings()
        if not embedding_settings:
            raise ValueError("Embedding settings not found")
        
        endpoint = embedding_settings.get('embedding_endpoint')
        if not endpoint:
            raise ValueError("Embedding endpoint not configured")
        
        payload = {"texts": [text.lower().strip()]}
        
        try:
            response = requests.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            embedding_data = response.json()["embeddings"][0]
            return embedding_data
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise