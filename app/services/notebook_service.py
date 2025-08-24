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