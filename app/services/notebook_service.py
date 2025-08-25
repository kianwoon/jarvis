"""
Notebook service for managing notebook CRUD operations and business logic.
Handles database operations for notebooks, documents, and conversations.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, DatabaseError
from sqlalchemy import and_, func, text, or_, desc

from app.models.notebook_models import (
    NotebookResponse, NotebookDetailResponse, NotebookListResponse,
    NotebookDocumentResponse, NotebookConversationResponse, NotebookStatsResponse,
    DocumentType
)

logger = logging.getLogger(__name__)

class NotebookService:
    """
    Service for managing notebook operations with proper error handling and validation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def create_notebook(
        self,
        db: Session,
        name: str,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        source_filter: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> NotebookResponse:
        """
        Create a new notebook.
        
        Args:
            db: Database session
            name: Notebook name
            description: Optional description
            user_id: Optional user ID
            source_filter: Optional filtering configuration
            metadata: Optional metadata
            
        Returns:
            Created notebook response
            
        Raises:
            ValueError: If validation fails
            IntegrityError: If database constraints are violated
            DatabaseError: If database operation fails
        """
        try:
            self.logger.info(f"Creating notebook: {name}")
            
            # Validate input
            if not name or not name.strip():
                raise ValueError("Notebook name cannot be empty")
            
            # Create notebook record
            notebook_id = str(uuid.uuid4())
            now = datetime.now()
            
            insert_query = text("""
                INSERT INTO notebooks (id, name, description, user_id, source_filter, metadata, created_at, updated_at)
                VALUES (:id, :name, :description, :user_id, :source_filter, :metadata, :created_at, :updated_at)
                RETURNING id, name, description, user_id, source_filter, metadata, created_at, updated_at
            """)
            
            result = db.execute(insert_query, {
                'id': notebook_id,
                'name': name.strip(),
                'description': description,
                'user_id': user_id,
                'source_filter': source_filter,
                'metadata': metadata,
                'created_at': now,
                'updated_at': now
            })
            
            row = result.fetchone()
            db.commit()
            
            self.logger.info(f"Successfully created notebook {notebook_id}")
            
            return NotebookResponse(
                id=str(row.id),
                name=row.name,
                description=row.description,
                user_id=row.user_id,
                source_filter=json.loads(row.source_filter) if row.source_filter and isinstance(row.source_filter, str) else row.source_filter,
                metadata=json.loads(row.metadata) if row.metadata and isinstance(row.metadata, str) else row.metadata,
                created_at=row.created_at,
                updated_at=row.updated_at,
                document_count=0,
                conversation_count=0
            )
            
        except IntegrityError as e:
            db.rollback()
            self.logger.error(f"Integrity error creating notebook: {str(e)}")
            raise
        except DatabaseError as e:
            db.rollback()
            self.logger.error(f"Database error creating notebook: {str(e)}")
            raise
        except Exception as e:
            db.rollback()
            self.logger.error(f"Unexpected error creating notebook: {str(e)}")
            raise
    
    async def list_notebooks(
        self,
        db: Session,
        page: int = 1,
        page_size: int = 20,
        user_id: Optional[str] = None,
        search: Optional[str] = None
    ) -> NotebookListResponse:
        """
        List notebooks with pagination and filtering.
        
        Args:
            db: Database session
            page: Page number (1-based)
            page_size: Items per page
            user_id: Filter by user ID
            search: Search in name or description
            
        Returns:
            Paginated notebook list
        """
        try:
            self.logger.info(f"Listing notebooks: page={page}, size={page_size}")
            
            offset = (page - 1) * page_size
            where_conditions = []
            params = {}
            
            if user_id:
                where_conditions.append("user_id = :user_id")
                params['user_id'] = user_id
            
            if search and search.strip():
                where_conditions.append("(name ILIKE :search OR description ILIKE :search)")
                params['search'] = f"%{search.strip()}%"
            
            where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            # Count query
            count_query = text(f"""
                SELECT COUNT(*) as total
                FROM notebooks
                {where_clause}
            """)
            
            count_result = db.execute(count_query, params)
            total_count = count_result.scalar()
            
            # Data query with document and conversation counts
            data_query = text(f"""
                SELECT 
                    n.id, n.name, n.description, n.user_id, n.source_filter, n.metadata,
                    n.created_at, n.updated_at,
                    COALESCE(doc_counts.doc_count, 0) as document_count,
                    COALESCE(conv_counts.conv_count, 0) as conversation_count
                FROM notebooks n
                LEFT JOIN (
                    SELECT notebook_id, COUNT(*) as doc_count
                    FROM notebook_documents
                    GROUP BY notebook_id
                ) doc_counts ON n.id = doc_counts.notebook_id
                LEFT JOIN (
                    SELECT notebook_id, COUNT(*) as conv_count
                    FROM notebook_conversations
                    GROUP BY notebook_id
                ) conv_counts ON n.id = conv_counts.notebook_id
                {where_clause}
                ORDER BY n.updated_at DESC
                LIMIT :limit OFFSET :offset
            """)
            
            params.update({
                'limit': page_size,
                'offset': offset
            })
            
            result = db.execute(data_query, params)
            rows = result.fetchall()
            
            notebooks = []
            for row in rows:
                notebooks.append(NotebookResponse(
                    id=str(row.id),
                    name=row.name,
                    description=row.description,
                    user_id=row.user_id,
                    source_filter=json.loads(row.source_filter) if row.source_filter and isinstance(row.source_filter, str) else row.source_filter,
                    metadata=json.loads(row.metadata) if row.metadata and isinstance(row.metadata, str) else row.metadata,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    document_count=row.document_count,
                    conversation_count=row.conversation_count
                ))
            
            return NotebookListResponse(
                notebooks=notebooks,
                total_count=total_count,
                page=page,
                page_size=page_size
            )
            
        except DatabaseError as e:
            self.logger.error(f"Database error listing notebooks: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error listing notebooks: {str(e)}")
            raise
    
    async def get_notebook_detail(
        self,
        db: Session,
        notebook_id: str
    ) -> Optional[NotebookDetailResponse]:
        """
        Get detailed notebook information including documents and conversations.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            
        Returns:
            Detailed notebook information or None if not found
        """
        try:
            self.logger.info(f"Getting notebook detail: {notebook_id}")
            
            # Get notebook basic info
            notebook_query = text("""
                SELECT id, name, description, user_id, source_filter, metadata, created_at, updated_at
                FROM notebooks
                WHERE id = :notebook_id
            """)
            
            notebook_result = db.execute(notebook_query, {'notebook_id': notebook_id})
            notebook_row = notebook_result.fetchone()
            
            if not notebook_row:
                return None
            
            # Get documents
            documents_query = text("""
                SELECT id, notebook_id, document_id, document_name, document_type, 
                       milvus_collection, added_at, metadata
                FROM notebook_documents
                WHERE notebook_id = :notebook_id
                ORDER BY added_at DESC
            """)
            
            documents_result = db.execute(documents_query, {'notebook_id': notebook_id})
            document_rows = documents_result.fetchall()
            
            documents = []
            for row in document_rows:
                documents.append(NotebookDocumentResponse(
                    id=str(row.id),
                    notebook_id=str(row.notebook_id),
                    document_id=row.document_id,
                    document_name=row.document_name,
                    document_type=DocumentType(row.document_type) if row.document_type else None,
                    milvus_collection=row.milvus_collection,
                    added_at=row.added_at,
                    metadata=json.loads(row.metadata) if row.metadata and isinstance(row.metadata, str) else row.metadata
                ))
            
            # Get conversations
            conversations_query = text("""
                SELECT id, notebook_id, conversation_id, started_at, last_activity
                FROM notebook_conversations
                WHERE notebook_id = :notebook_id
                ORDER BY last_activity DESC
            """)
            
            conversations_result = db.execute(conversations_query, {'notebook_id': notebook_id})
            conversation_rows = conversations_result.fetchall()
            
            conversations = []
            for row in conversation_rows:
                conversations.append(NotebookConversationResponse(
                    id=str(row.id),
                    notebook_id=str(row.notebook_id),
                    conversation_id=row.conversation_id,
                    started_at=row.started_at,
                    last_activity=row.last_activity
                ))
            
            return NotebookDetailResponse(
                id=str(notebook_row.id),
                name=notebook_row.name,
                description=notebook_row.description,
                user_id=notebook_row.user_id,
                source_filter=notebook_row.source_filter,
                metadata=notebook_row.metadata,
                created_at=notebook_row.created_at,
                updated_at=notebook_row.updated_at,
                document_count=len(documents),
                conversation_count=len(conversations),
                documents=documents,
                conversations=conversations
            )
            
        except DatabaseError as e:
            self.logger.error(f"Database error getting notebook detail: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error getting notebook detail: {str(e)}")
            raise
    
    async def update_notebook(
        self,
        db: Session,
        notebook_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        source_filter: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[NotebookResponse]:
        """
        Update notebook information.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            name: Updated name (optional)
            description: Updated description (optional)
            user_id: Updated user ID (optional)
            source_filter: Updated source filter (optional)
            metadata: Updated metadata (optional)
            
        Returns:
            Updated notebook response or None if not found
        """
        try:
            self.logger.info(f"Updating notebook: {notebook_id}")
            
            # Build update fields
            update_fields = []
            params = {'notebook_id': notebook_id, 'updated_at': datetime.now()}
            
            if name is not None:
                if not name.strip():
                    raise ValueError("Notebook name cannot be empty")
                update_fields.append("name = :name")
                params['name'] = name.strip()
            
            if description is not None:
                update_fields.append("description = :description")
                params['description'] = description
            
            if user_id is not None:
                update_fields.append("user_id = :user_id")
                params['user_id'] = user_id
            
            if source_filter is not None:
                update_fields.append("source_filter = :source_filter")
                params['source_filter'] = source_filter
            
            if metadata is not None:
                update_fields.append("metadata = :metadata")
                params['metadata'] = metadata
            
            if not update_fields:
                # No updates requested, return current notebook
                return await self._get_notebook_basic(db, notebook_id)
            
            update_fields.append("updated_at = :updated_at")
            
            update_query = text(f"""
                UPDATE notebooks
                SET {', '.join(update_fields)}
                WHERE id = :notebook_id
                RETURNING id, name, description, user_id, source_filter, metadata, created_at, updated_at
            """)
            
            result = db.execute(update_query, params)
            row = result.fetchone()
            
            if not row:
                return None
            
            db.commit()
            
            # Get counts
            counts = await self._get_notebook_counts(db, notebook_id)
            
            self.logger.info(f"Successfully updated notebook {notebook_id}")
            
            return NotebookResponse(
                id=str(row.id),
                name=row.name,
                description=row.description,
                user_id=row.user_id,
                source_filter=json.loads(row.source_filter) if row.source_filter and isinstance(row.source_filter, str) else row.source_filter,
                metadata=json.loads(row.metadata) if row.metadata and isinstance(row.metadata, str) else row.metadata,
                created_at=row.created_at,
                updated_at=row.updated_at,
                document_count=counts['document_count'],
                conversation_count=counts['conversation_count']
            )
            
        except IntegrityError as e:
            db.rollback()
            self.logger.error(f"Integrity error updating notebook: {str(e)}")
            raise
        except DatabaseError as e:
            db.rollback()
            self.logger.error(f"Database error updating notebook: {str(e)}")
            raise
        except Exception as e:
            db.rollback()
            self.logger.error(f"Unexpected error updating notebook: {str(e)}")
            raise
    
    async def delete_notebook(
        self,
        db: Session,
        notebook_id: str
    ) -> bool:
        """
        Delete a notebook and all associated data.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            self.logger.info(f"Deleting notebook: {notebook_id}")
            
            # Delete notebook (cascade will handle related records)
            delete_query = text("""
                DELETE FROM notebooks
                WHERE id = :notebook_id
            """)
            
            result = db.execute(delete_query, {'notebook_id': notebook_id})
            
            if result.rowcount == 0:
                return False
            
            db.commit()
            
            self.logger.info(f"Successfully deleted notebook {notebook_id}")
            return True
            
        except DatabaseError as e:
            db.rollback()
            self.logger.error(f"Database error deleting notebook: {str(e)}")
            raise
        except Exception as e:
            db.rollback()
            self.logger.error(f"Unexpected error deleting notebook: {str(e)}")
            raise
    
    async def add_document_to_notebook(
        self,
        db: Session,
        notebook_id: str,
        document_id: str,
        document_name: Optional[str] = None,
        document_type: Optional[str] = None,
        milvus_collection: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[NotebookDocumentResponse]:
        """
        Add a document to a notebook.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            document_id: Document ID to add
            document_name: Document display name
            document_type: Document type
            milvus_collection: Milvus collection name
            metadata: Document metadata
            
        Returns:
            Added document response or None if notebook not found
        """
        try:
            self.logger.info(f"Adding document {document_id} to notebook {notebook_id}")
            
            # Verify notebook exists
            if not await self.notebook_exists(db, notebook_id):
                return None
            
            # Insert document record
            doc_id = str(uuid.uuid4())
            now = datetime.now()
            
            insert_query = text("""
                INSERT INTO notebook_documents 
                (id, notebook_id, document_id, document_name, document_type, milvus_collection, added_at, metadata)
                VALUES (:id, :notebook_id, :document_id, :document_name, :document_type, :milvus_collection, :added_at, :metadata)
                RETURNING id, notebook_id, document_id, document_name, document_type, milvus_collection, added_at, metadata
            """)
            
            result = db.execute(insert_query, {
                'id': doc_id,
                'notebook_id': notebook_id,
                'document_id': document_id,
                'document_name': document_name,
                'document_type': document_type,
                'milvus_collection': milvus_collection,
                'added_at': now,
                'metadata': json.dumps(metadata) if metadata else None
            })
            
            row = result.fetchone()
            db.commit()
            
            self.logger.info(f"Successfully added document to notebook {notebook_id}")
            
            return NotebookDocumentResponse(
                id=str(row.id),
                notebook_id=str(row.notebook_id),
                document_id=row.document_id,
                document_name=row.document_name,
                document_type=DocumentType(row.document_type) if row.document_type else None,
                milvus_collection=row.milvus_collection,
                added_at=row.added_at,
                metadata=json.loads(row.metadata) if row.metadata and isinstance(row.metadata, str) else row.metadata
            )
            
        except IntegrityError as e:
            db.rollback()
            self.logger.error(f"Integrity error adding document: {str(e)}")
            raise
        except DatabaseError as e:
            db.rollback()
            self.logger.error(f"Database error adding document: {str(e)}")
            raise
        except Exception as e:
            db.rollback()
            self.logger.error(f"Unexpected error adding document: {str(e)}")
            raise
    
    async def remove_document_from_notebook(
        self,
        db: Session,
        notebook_id: str,
        document_id: str
    ) -> bool:
        """
        Remove a document from a notebook.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            document_id: Document ID to remove
            
        Returns:
            True if removed, False if not found
        """
        try:
            self.logger.info(f"Removing document {document_id} from notebook {notebook_id}")
            
            delete_query = text("""
                DELETE FROM notebook_documents
                WHERE notebook_id = :notebook_id AND document_id = :document_id
            """)
            
            result = db.execute(delete_query, {
                'notebook_id': notebook_id,
                'document_id': document_id
            })
            
            if result.rowcount == 0:
                return False
            
            db.commit()
            
            self.logger.info(f"Successfully removed document from notebook {notebook_id}")
            return True
            
        except DatabaseError as e:
            db.rollback()
            self.logger.error(f"Database error removing document: {str(e)}")
            raise
        except Exception as e:
            db.rollback()
            self.logger.error(f"Unexpected error removing document: {str(e)}")
            raise
    
    async def start_conversation(
        self,
        db: Session,
        notebook_id: str,
        conversation_id: str
    ) -> Optional[NotebookConversationResponse]:
        """
        Start a new conversation with the notebook.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            conversation_id: Conversation ID
            
        Returns:
            Conversation response or None if notebook not found
        """
        try:
            self.logger.info(f"Starting conversation {conversation_id} for notebook {notebook_id}")
            
            # Verify notebook exists
            if not await self.notebook_exists(db, notebook_id):
                return None
            
            # Insert or update conversation record
            conv_id = str(uuid.uuid4())
            now = datetime.now()
            
            insert_query = text("""
                INSERT INTO notebook_conversations (id, notebook_id, conversation_id, started_at, last_activity)
                VALUES (:id, :notebook_id, :conversation_id, :started_at, :last_activity)
                ON CONFLICT (notebook_id, conversation_id) DO UPDATE SET
                    last_activity = :last_activity
                RETURNING id, notebook_id, conversation_id, started_at, last_activity
            """)
            
            result = db.execute(insert_query, {
                'id': conv_id,
                'notebook_id': notebook_id,
                'conversation_id': conversation_id,
                'started_at': now,
                'last_activity': now
            })
            
            row = result.fetchone()
            db.commit()
            
            self.logger.info(f"Successfully started conversation for notebook {notebook_id}")
            
            return NotebookConversationResponse(
                id=str(row.id),
                notebook_id=str(row.notebook_id),
                conversation_id=row.conversation_id,
                started_at=row.started_at,
                last_activity=row.last_activity
            )
            
        except IntegrityError as e:
            db.rollback()
            self.logger.error(f"Integrity error starting conversation: {str(e)}")
            raise
        except DatabaseError as e:
            db.rollback()
            self.logger.error(f"Database error starting conversation: {str(e)}")
            raise
        except Exception as e:
            db.rollback()
            self.logger.error(f"Unexpected error starting conversation: {str(e)}")
            raise
    
    async def get_notebook_stats(
        self,
        db: Session,
        notebook_id: str
    ) -> Optional[NotebookStatsResponse]:
        """
        Get comprehensive statistics for a notebook.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            
        Returns:
            Notebook statistics or None if not found
        """
        try:
            self.logger.info(f"Getting stats for notebook {notebook_id}")
            
            # Get notebook basic info and stats
            stats_query = text("""
                SELECT 
                    n.id, n.created_at,
                    COALESCE(doc_stats.total_documents, 0) as total_documents,
                    COALESCE(doc_stats.documents_by_type, '{}') as documents_by_type,
                    COALESCE(doc_stats.collections_used, '[]') as collections_used,
                    COALESCE(conv_stats.total_conversations, 0) as total_conversations,
                    COALESCE(conv_stats.active_conversations, 0) as active_conversations,
                    conv_stats.last_activity
                FROM notebooks n
                LEFT JOIN (
                    SELECT 
                        notebook_id,
                        COUNT(*) as total_documents,
                        json_object_agg(
                            COALESCE(document_type, 'unknown'), 
                            count
                        ) as documents_by_type,
                        json_agg(DISTINCT milvus_collection) as collections_used
                    FROM (
                        SELECT 
                            notebook_id,
                            document_type,
                            milvus_collection,
                            COUNT(*) as count
                        FROM notebook_documents
                        GROUP BY notebook_id, document_type, milvus_collection
                    ) doc_counts
                    GROUP BY notebook_id
                ) doc_stats ON n.id = doc_stats.notebook_id
                LEFT JOIN (
                    SELECT 
                        notebook_id,
                        COUNT(*) as total_conversations,
                        COUNT(CASE WHEN last_activity > NOW() - INTERVAL '24 hours' THEN 1 END) as active_conversations,
                        MAX(last_activity) as last_activity
                    FROM notebook_conversations
                    GROUP BY notebook_id
                ) conv_stats ON n.id = conv_stats.notebook_id
                WHERE n.id = :notebook_id
            """)
            
            result = db.execute(stats_query, {'notebook_id': notebook_id})
            row = result.fetchone()
            
            if not row:
                return None
            
            # Clean up collections list (remove nulls)
            collections_used = [col for col in (row.collections_used or []) if col is not None]
            
            return NotebookStatsResponse(
                notebook_id=str(row.id),
                total_documents=row.total_documents,
                documents_by_type=row.documents_by_type or {},
                total_conversations=row.total_conversations,
                active_conversations=row.active_conversations,
                collections_used=collections_used,
                last_activity=row.last_activity,
                created_at=row.created_at
            )
            
        except DatabaseError as e:
            self.logger.error(f"Database error getting stats: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error getting stats: {str(e)}")
            raise
    
    async def notebook_exists(self, db: Session, notebook_id: str) -> bool:
        """Check if notebook exists."""
        try:
            query = text("SELECT 1 FROM notebooks WHERE id = :notebook_id")
            result = db.execute(query, {'notebook_id': notebook_id})
            return result.fetchone() is not None
        except Exception:
            return False
    
    async def _get_notebook_basic(self, db: Session, notebook_id: str) -> Optional[NotebookResponse]:
        """Get basic notebook info with counts."""
        try:
            query = text("""
                SELECT id, name, description, user_id, source_filter, metadata, created_at, updated_at
                FROM notebooks
                WHERE id = :notebook_id
            """)
            
            result = db.execute(query, {'notebook_id': notebook_id})
            row = result.fetchone()
            
            if not row:
                return None
            
            counts = await self._get_notebook_counts(db, notebook_id)
            
            return NotebookResponse(
                id=str(row.id),
                name=row.name,
                description=row.description,
                user_id=row.user_id,
                source_filter=json.loads(row.source_filter) if row.source_filter and isinstance(row.source_filter, str) else row.source_filter,
                metadata=json.loads(row.metadata) if row.metadata and isinstance(row.metadata, str) else row.metadata,
                created_at=row.created_at,
                updated_at=row.updated_at,
                document_count=counts['document_count'],
                conversation_count=counts['conversation_count']
            )
        except Exception:
            return None
    
    async def _get_notebook_counts(self, db: Session, notebook_id: str) -> Dict[str, int]:
        """Get document and conversation counts for a notebook."""
        try:
            counts_query = text("""
                SELECT 
                    COALESCE(doc_count, 0) as document_count,
                    COALESCE(conv_count, 0) as conversation_count
                FROM (
                    SELECT 
                        :notebook_id as id,
                        (SELECT COUNT(*) FROM notebook_documents WHERE notebook_id = :notebook_id) as doc_count,
                        (SELECT COUNT(*) FROM notebook_conversations WHERE notebook_id = :notebook_id) as conv_count
                ) counts
            """)
            
            result = db.execute(counts_query, {'notebook_id': notebook_id})
            row = result.fetchone()
            
            return {
                'document_count': row.document_count if row else 0,
                'conversation_count': row.conversation_count if row else 0
            }
        except Exception:
            return {'document_count': 0, 'conversation_count': 0}
    
    # Memory Management Methods
    
    async def create_memory(
        self,
        db: Session,
        notebook_id: str,
        name: str,
        content: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'MemoryResponse':
        """
        Create a new memory for a notebook.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            name: Memory name
            content: Memory content
            description: Optional description
            metadata: Optional metadata
            
        Returns:
            Created memory response
            
        Raises:
            ValueError: If validation fails
            IntegrityError: If database constraints are violated
            DatabaseError: If database operation fails
        """
        try:
            self.logger.info(f"Creating memory '{name}' for notebook {notebook_id}")
            
            # Validate input
            if not name or not name.strip():
                raise ValueError("Memory name cannot be empty")
            if not content or not content.strip():
                raise ValueError("Memory content cannot be empty")
            
            # Verify notebook exists
            if not await self.notebook_exists(db, notebook_id):
                raise ValueError(f"Notebook {notebook_id} not found")
            
            # Create memory record
            memory_id = str(uuid.uuid4())
            now = datetime.now()
            
            # Import here to avoid circular imports
            from app.models.notebook_models import MemoryResponse
            
            # Process content: chunk, embed, store in Milvus
            chunk_count, target_collection = await self._process_memory_content(
                db=db,
                memory_id=memory_id,
                content=content.strip(),
                metadata=metadata or {}
            )
            
            # Use the safe synchronized memory creation function
            sync_result = db.execute(text("""
                SELECT success, message, memory_id FROM create_memory_synchronized(
                    :notebook_id, :memory_id, :name, :description, :content, 
                    :milvus_collection, :chunk_count, :metadata
                )
            """), {
                'notebook_id': notebook_id,
                'memory_id': memory_id,
                'name': name.strip(),
                'description': description,
                'content': content.strip(),
                'milvus_collection': target_collection,
                'chunk_count': chunk_count,
                'metadata': json.dumps(metadata) if metadata else None
            })
            
            sync_row = sync_result.fetchone()
            if not sync_row.success:
                raise ValueError(f"Failed to create synchronized memory: {sync_row.message}")
            
            # Get the created memory record
            result = db.execute(text("""
                SELECT id, notebook_id, memory_id, name, description, content, 
                       milvus_collection, chunk_count, created_at, updated_at, metadata
                FROM notebook_memories 
                WHERE memory_id = :memory_id
            """), {'memory_id': memory_id})
            
            row = result.fetchone()
            db.commit()
            
            self.logger.info(f"Successfully created memory {memory_id}")
            
            return MemoryResponse(
                id=str(row.id),
                notebook_id=str(row.notebook_id),
                memory_id=row.memory_id,
                name=row.name,
                description=row.description,
                content=row.content,
                milvus_collection=row.milvus_collection,
                chunk_count=row.chunk_count,
                created_at=row.created_at,
                updated_at=row.updated_at,
                metadata=json.loads(row.metadata) if row.metadata else None
            )
            
        except ValueError:
            raise
        except IntegrityError as e:
            db.rollback()
            self.logger.error(f"Integrity error creating memory: {str(e)}")
            raise
        except DatabaseError as e:
            db.rollback()
            self.logger.error(f"Database error creating memory: {str(e)}")
            raise
        except Exception as e:
            db.rollback()
            self.logger.error(f"Unexpected error creating memory: {str(e)}")
            raise
    
    async def get_memories(
        self,
        db: Session,
        notebook_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> 'MemoryListResponse':
        """
        Get memories for a notebook with pagination.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            page: Page number (1-based)
            page_size: Number of memories per page
            
        Returns:
            List of memories with pagination info
        """
        try:
            self.logger.info(f"Getting memories for notebook {notebook_id}")
            
            # Import here to avoid circular imports
            from app.models.notebook_models import MemoryResponse, MemoryListResponse
            
            # Verify notebook exists
            if not await self.notebook_exists(db, notebook_id):
                raise ValueError(f"Notebook {notebook_id} not found")
            
            # Calculate offset
            offset = (page - 1) * page_size
            
            # Get total count
            count_query = text("""
                SELECT COUNT(*) as total
                FROM notebook_memories
                WHERE notebook_id = :notebook_id
            """)
            
            count_result = db.execute(count_query, {'notebook_id': notebook_id})
            total_count = count_result.fetchone().total
            
            # Get paginated memories
            memories_query = text("""
                SELECT id, notebook_id, memory_id, name, description, content,
                       milvus_collection, chunk_count, created_at, updated_at, metadata
                FROM notebook_memories
                WHERE notebook_id = :notebook_id
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """)
            
            result = db.execute(memories_query, {
                'notebook_id': notebook_id,
                'limit': page_size,
                'offset': offset
            })
            
            memories = []
            for row in result.fetchall():
                memories.append(MemoryResponse(
                    id=str(row.id),
                    notebook_id=str(row.notebook_id),
                    memory_id=row.memory_id,
                    name=row.name,
                    description=row.description,
                    content=row.content,
                    milvus_collection=row.milvus_collection,
                    chunk_count=row.chunk_count,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    metadata=json.loads(row.metadata) if row.metadata else None
                ))
            
            return MemoryListResponse(
                memories=memories,
                total_count=total_count,
                page=page,
                page_size=page_size
            )
            
        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"Error getting memories: {str(e)}")
            raise
    
    async def get_memory(
        self,
        db: Session,
        notebook_id: str,
        memory_id: str
    ) -> Optional['MemoryResponse']:
        """
        Get a specific memory by ID.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            memory_id: Memory ID
            
        Returns:
            Memory response or None if not found
        """
        try:
            self.logger.info(f"Getting memory {memory_id} for notebook {notebook_id}")
            
            # Import here to avoid circular imports
            from app.models.notebook_models import MemoryResponse
            
            query = text("""
                SELECT id, notebook_id, memory_id, name, description, content,
                       milvus_collection, chunk_count, created_at, updated_at, metadata
                FROM notebook_memories
                WHERE notebook_id = :notebook_id AND memory_id = :memory_id
            """)
            
            result = db.execute(query, {
                'notebook_id': notebook_id,
                'memory_id': memory_id
            })
            
            row = result.fetchone()
            
            if not row:
                return None
            
            return MemoryResponse(
                id=str(row.id),
                notebook_id=str(row.notebook_id),
                memory_id=row.memory_id,
                name=row.name,
                description=row.description,
                content=row.content,
                milvus_collection=row.milvus_collection,
                chunk_count=row.chunk_count,
                created_at=row.created_at,
                updated_at=row.updated_at,
                metadata=json.loads(row.metadata) if row.metadata else None
            )
            
        except Exception as e:
            self.logger.error(f"Error getting memory: {str(e)}")
            raise
    
    async def update_memory(
        self,
        db: Session,
        notebook_id: str,
        memory_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional['MemoryResponse']:
        """
        Update a memory.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            memory_id: Memory ID
            name: Updated name
            description: Updated description
            content: Updated content (will trigger re-processing)
            metadata: Updated metadata
            
        Returns:
            Updated memory response or None if not found
        """
        try:
            self.logger.info(f"Updating memory {memory_id} for notebook {notebook_id}")
            
            # Import here to avoid circular imports
            from app.models.notebook_models import MemoryResponse
            
            # Get current memory
            current_memory = await self.get_memory(db, notebook_id, memory_id)
            if not current_memory:
                return None
            
            # Prepare update fields
            update_fields = []
            update_params = {'notebook_id': notebook_id, 'memory_id': memory_id}
            
            if name is not None and name.strip():
                update_fields.append("name = :name")
                update_params['name'] = name.strip()
            
            if description is not None:
                update_fields.append("description = :description")
                update_params['description'] = description
            
            if metadata is not None:
                update_fields.append("metadata = :metadata")
                update_params['metadata'] = json.dumps(metadata)
            
            # Handle content update (requires re-processing)
            if content is not None and content.strip():
                update_fields.append("content = :content")
                update_fields.append("chunk_count = :chunk_count")
                update_params['content'] = content.strip()
                
                # Re-process content
                chunk_count, target_collection = await self._process_memory_content(
                    db=db,
                    memory_id=memory_id,
                    content=content.strip(),
                    metadata=metadata or current_memory.metadata or {},
                    is_update=True
                )
                update_params['chunk_count'] = chunk_count
            
            if not update_fields:
                # No changes to make
                return current_memory
            
            update_fields.append("updated_at = NOW()")
            
            update_query = text(f"""
                UPDATE notebook_memories 
                SET {', '.join(update_fields)}
                WHERE notebook_id = :notebook_id AND memory_id = :memory_id
                RETURNING id, notebook_id, memory_id, name, description, content,
                         milvus_collection, chunk_count, created_at, updated_at, metadata
            """)
            
            result = db.execute(update_query, update_params)
            row = result.fetchone()
            
            if not row:
                return None
            
            db.commit()
            
            self.logger.info(f"Successfully updated memory {memory_id}")
            
            return MemoryResponse(
                id=str(row.id),
                notebook_id=str(row.notebook_id),
                memory_id=row.memory_id,
                name=row.name,
                description=row.description,
                content=row.content,
                milvus_collection=row.milvus_collection,
                chunk_count=row.chunk_count,
                created_at=row.created_at,
                updated_at=row.updated_at,
                metadata=json.loads(row.metadata) if row.metadata else None
            )
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error updating memory: {str(e)}")
            raise
    
    async def delete_memory(
        self,
        db: Session,
        notebook_id: str,
        memory_id: str
    ) -> bool:
        """
        Delete a memory and all its associated data.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            memory_id: Memory ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            self.logger.info(f"Deleting memory {memory_id} for notebook {notebook_id}")
            
            # Get memory details before deletion
            memory = await self.get_memory(db, notebook_id, memory_id)
            if not memory:
                return False
            
            # Delete from Milvus
            await self._delete_memory_from_milvus(memory_id, memory.milvus_collection)
            
            # Delete from knowledge_graph_documents
            kg_delete_query = text("""
                DELETE FROM knowledge_graph_documents
                WHERE document_id = :memory_id AND content_type = 'memory'
            """)
            db.execute(kg_delete_query, {'memory_id': memory_id})
            
            # Delete from notebook_memories
            memory_delete_query = text("""
                DELETE FROM notebook_memories
                WHERE notebook_id = :notebook_id AND memory_id = :memory_id
            """)
            
            result = db.execute(memory_delete_query, {
                'notebook_id': notebook_id,
                'memory_id': memory_id
            })
            
            deleted = result.rowcount > 0
            
            if deleted:
                db.commit()
                self.logger.info(f"Successfully deleted memory {memory_id}")
            else:
                db.rollback()
            
            return deleted
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error deleting memory: {str(e)}")
            raise
    
    async def _process_memory_content(
        self,
        db: Session,
        memory_id: str,
        content: str,
        metadata: Dict[str, Any],
        is_update: bool = False
    ) -> int:
        """
        Process memory content: chunk, embed, and store in Milvus.
        
        Args:
            db: Database session
            memory_id: Memory ID
            content: Content to process
            metadata: Memory metadata
            is_update: Whether this is an update operation
            
        Returns:
            Number of chunks created
        """
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from app.core.embedding_settings_cache import get_embedding_settings
            from app.core.vector_db_settings_cache import get_vector_db_settings
            from app.core.collection_registry_cache import get_collection_config
            from pymilvus import connections, Collection, utility
            import requests
            import hashlib
            
            # Get settings
            embedding_settings = get_embedding_settings()
            vector_settings_raw = get_vector_db_settings()
            
            if not embedding_settings or not vector_settings_raw:
                raise ValueError("Missing embedding or vector database settings")
            
            # Migrate and get active Milvus configuration
            from app.utils.vector_db_migration import migrate_vector_db_settings
            vector_db_cfg = migrate_vector_db_settings(vector_settings_raw)
            
            # Find active Milvus database
            milvus_db = None
            for db_config in vector_db_cfg.get("databases", []):
                if db_config.get("id") == "milvus" and db_config.get("enabled"):
                    milvus_db = db_config
                    break
            
            if not milvus_db:
                raise ValueError("No active Milvus configuration found")
            
            # Use notebooks collection - matches document upload pattern 
            target_collection = 'notebooks'
            collection_config = get_collection_config(target_collection)
            
            if not collection_config:
                raise ValueError(f"Collection '{target_collection}' not found. Please initialize it first.")
            
            # Split content into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            chunks = text_splitter.split_text(content)
            
            # Generate embeddings
            endpoint = embedding_settings.get('embedding_endpoint')
            if not endpoint:
                raise ValueError("Embedding endpoint not configured")
            
            embeddings = []
            for chunk in chunks:
                payload = {"texts": [chunk.lower().strip()]}
                response = requests.post(endpoint, json=payload, timeout=30)
                response.raise_for_status()
                embedding_data = response.json()["embeddings"][0]
                embeddings.append(embedding_data)
            
            # Connect to Milvus
            milvus_cfg = milvus_db.get("config", {})
            milvus_uri = milvus_cfg.get("MILVUS_URI")
            milvus_token = milvus_cfg.get("MILVUS_TOKEN")
            connections.connect(uri=milvus_uri, token=milvus_token)
            
            # Ensure collection exists
            from app.api.v1.endpoints.notebooks import ensure_milvus_collection
            collection = ensure_milvus_collection(
                target_collection,
                len(embeddings[0]) if embeddings else 384,
                milvus_uri,
                milvus_token
            )
            
            # If updating, delete existing chunks
            if is_update:
                delete_expr = f'doc_id == "{memory_id}"'
                collection.delete(expr=delete_expr)
                collection.flush()
            
            # Prepare data for insertion
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]
            now_iso = datetime.now().isoformat()
            
            data = [
                chunk_ids,
                embeddings,
                chunks,
                [f"memory:{memory_id}" for _ in chunks],  # source
                [0 for _ in chunks],  # page
                ['memory' for _ in chunks],  # doc_type
                [now_iso for _ in chunks],  # uploaded_at
                [f'memory_{i}' for i in range(len(chunks))],  # section
                ['' for _ in chunks],  # author
                [hashlib.sha256(chunk.encode()).hexdigest() for chunk in chunks],  # hash
                [memory_id for _ in chunks],  # doc_id
                ['' for _ in chunks],  # bm25_tokens
                [0 for _ in chunks],  # bm25_term_count
                [0 for _ in chunks],  # bm25_unique_terms
                ['' for _ in chunks],  # bm25_top_terms
                [now_iso for _ in chunks],  # creation_date
                [now_iso for _ in chunks],  # last_modified_date
            ]
            
            # Insert into Milvus
            collection.insert(data)
            collection.flush()
            
            # Create or update knowledge_graph_documents record with content-based deduplication
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # First check if content already exists
            existing_check_query = text("""
                SELECT document_id, chunks_processed 
                FROM knowledge_graph_documents 
                WHERE file_hash = :file_hash AND content_type = 'memory'
                LIMIT 1
            """)
            
            existing_result = db.execute(existing_check_query, {'file_hash': content_hash}).fetchone()
            
            if existing_result and not is_update:
                # Content already exists, don't create duplicate
                self.logger.info(f"Memory content already exists with document_id {existing_result.document_id}, skipping duplicate creation")
                return existing_result.chunks_processed, target_collection
            elif existing_result and is_update:
                # For updates, use the existing document_id to maintain consistency
                existing_doc_id = existing_result.document_id
                # Update the existing record
                kg_update_query = text("""
                    UPDATE knowledge_graph_documents 
                    SET chunks_processed = :chunks,
                        total_chunks = :chunks,
                        processing_completed_at = NOW(),
                        updated_at = NOW()
                    WHERE document_id = :doc_id
                """)
                
                db.execute(kg_update_query, {
                    'doc_id': existing_doc_id,
                    'chunks': len(chunks)
                })
            else:
                # Content doesn't exist, create new record
                kg_insert_query = text("""
                    INSERT INTO knowledge_graph_documents 
                    (document_id, filename, file_hash, file_type, content_type, 
                     milvus_collection, processing_status, chunks_processed, 
                     total_chunks, processing_completed_at)
                    VALUES (:doc_id, :filename, :file_hash, 'memory', 'memory',
                            :collection, 'completed', :chunks, :chunks, NOW())
                """)
                
                db.execute(kg_insert_query, {
                    'doc_id': memory_id,
                    'filename': f'memory_{memory_id}',
                    'file_hash': content_hash,
                    'collection': target_collection,
                    'chunks': len(chunks)
                })
            
            return len(chunks), target_collection
            
        except Exception as e:
            self.logger.error(f"Error processing memory content: {str(e)}")
            raise
    
    async def _delete_memory_from_milvus(
        self,
        memory_id: str,
        collection_name: Optional[str]
    ) -> None:
        """
        Delete memory chunks from Milvus.
        
        Args:
            memory_id: Memory ID
            collection_name: Milvus collection name
        """
        try:
            if not collection_name:
                self.logger.warning(f"No collection specified for memory {memory_id}")
                return
            
            from app.core.vector_db_settings_cache import get_vector_db_settings
            from pymilvus import connections, Collection, utility
            
            vector_settings_raw = get_vector_db_settings()
            if not vector_settings_raw:
                return
            
            # Migrate and get active Milvus configuration
            from app.utils.vector_db_migration import migrate_vector_db_settings
            vector_db_cfg = migrate_vector_db_settings(vector_settings_raw)
            
            # Find active Milvus database
            milvus_db = None
            for db_config in vector_db_cfg.get("databases", []):
                if db_config.get("id") == "milvus" and db_config.get("enabled"):
                    milvus_db = db_config
                    break
            
            if not milvus_db:
                self.logger.warning("No active Milvus configuration found")
                return
            
            milvus_cfg = milvus_db.get("config", {})
            milvus_uri = milvus_cfg.get("MILVUS_URI")
            milvus_token = milvus_cfg.get("MILVUS_TOKEN")
            connections.connect(uri=milvus_uri, token=milvus_token)
            
            if not utility.has_collection(collection_name):
                self.logger.warning(f"Collection {collection_name} not found")
                return
            
            collection = Collection(collection_name)
            collection.load()
            
            delete_expr = f'doc_id == "{memory_id}"'
            collection.delete(expr=delete_expr)
            collection.flush()
            
            self.logger.info(f"Deleted memory {memory_id} chunks from Milvus")
            
        except Exception as e:
            self.logger.error(f"Error deleting memory from Milvus: {str(e)}")
            # Don't raise here - we still want to delete from database