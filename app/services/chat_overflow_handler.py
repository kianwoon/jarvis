"""
Chat Overflow Handler Service
Handles large input overflow in chat conversations using intelligent chunking and Redis storage layers.
"""

import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import tiktoken
import numpy as np
from sqlalchemy.orm import Session

from app.core.redis_client import get_redis_client
from app.core.config import get_settings
from app.core.overflow_settings_cache import get_overflow_config, reload_overflow_settings
from app.schemas.overflow import OverflowConfig

logger = logging.getLogger(__name__)

class ChatOverflowHandler:
    """
    Handles overflow content when users paste large amounts of text in chat.
    Uses Redis L1/L2 storage layers with intelligent chunking and retrieval.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = get_redis_client()
        
        # Load configuration from database
        self.config = self._load_config()
        
        # Token encoder for accurate counting
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def _load_config(self) -> OverflowConfig:
        """
        Load overflow configuration from cached settings.
        Uses cache for performance.
        """
        return get_overflow_config()
    
    def reload_config(self) -> OverflowConfig:
        """
        Reload configuration from database.
        Useful for dynamic config updates without restart.
        """
        reload_overflow_settings()  # Force cache reload
        self.config = get_overflow_config()
        logger.info("Reloaded overflow configuration from database")
        return self.config
        
    def detect_overflow(self, content: str) -> Tuple[bool, int]:
        """
        Detect if content exceeds overflow threshold.
        
        Args:
            content: Input text content
            
        Returns:
            Tuple of (is_overflow, token_count)
        """
        try:
            token_count = len(self.encoder.encode(content))
            is_overflow = token_count > self.config.overflow_threshold_tokens
            
            if is_overflow:
                logger.info(f"Overflow detected: {token_count} tokens (threshold: {self.config.overflow_threshold_tokens})")
            
            return is_overflow, token_count
        except Exception as e:
            logger.error(f"Error detecting overflow: {str(e)}")
            # Conservative estimate if encoding fails
            estimated_tokens = len(content) // 4
            return estimated_tokens > self.config.overflow_threshold_tokens, estimated_tokens
    
    async def chunk_intelligently(self, content: str, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Chunk content intelligently using semantic boundaries.
        
        Args:
            content: Content to chunk
            conversation_id: Conversation identifier
            
        Returns:
            List of chunk dictionaries with metadata
        """
        try:
            # Perform intelligent text chunking with overlap
            chunks = self._chunk_text_with_overlap(
                text=content,
                chunk_size=self.config.chunk_size_tokens,
                overlap_size=self.config.chunk_overlap_tokens
            )
            
            # Enhance chunks with metadata
            enhanced_chunks = []
            for i, chunk_text in enumerate(chunks):
                chunk_id = hashlib.md5(f"{conversation_id}_{i}_{chunk_text[:100]}".encode()).hexdigest()
                
                # Generate embedding for semantic search (disabled for now)
                # TODO: Re-enable when vector service is available
                # embedding = await self.vector_service.create_embedding(chunk_text)
                embedding = []  # Empty for now
                
                # Extract summary and keywords if enabled
                summary = self._extract_summary(chunk_text) if self.config.enable_keyword_extraction else ""
                keywords = self._extract_keywords(chunk_text) if self.config.enable_keyword_extraction else []
                
                enhanced_chunks.append({
                    "chunk_id": chunk_id,
                    "conversation_id": conversation_id,
                    "position": i,
                    "content": chunk_text,
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    "summary": summary,
                    "keywords": keywords,
                    "token_count": len(self.encoder.encode(chunk_text)),
                    "created_at": datetime.utcnow().isoformat(),
                    "access_count": 0
                })
                
            logger.info(f"Created {len(enhanced_chunks)} chunks for conversation {conversation_id}")
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Error chunking content: {str(e)}")
            # Fallback to simple chunking
            return self._simple_chunk(content, conversation_id)
    
    async def store_overflow(self, chunks: List[Dict[str, Any]], conversation_id: str, 
                           storage_layer: str = "L2") -> bool:
        """
        Store overflow chunks in appropriate Redis layer.
        
        Args:
            chunks: List of chunk dictionaries
            conversation_id: Conversation identifier
            storage_layer: "L1" for hot storage, "L2" for warm storage
            
        Returns:
            Success status
        """
        try:
            pipeline = self.redis_client.pipeline()
            
            # Determine TTL based on storage layer
            if storage_layer == "L1":
                ttl_seconds = self.config.l1_ttl_hours * 3600
                key_prefix = f"conv:{conversation_id}:overflow:recent"
            else:  # L2
                ttl_seconds = self.config.l2_ttl_days * 86400
                key_prefix = f"conv:{conversation_id}:overflow:chunks"
            
            # Store each chunk
            for chunk in chunks:
                chunk_key = f"{key_prefix}:{chunk['chunk_id']}"
                pipeline.setex(
                    chunk_key,
                    ttl_seconds,
                    json.dumps(chunk)
                )
                
                # Add to conversation's chunk index
                index_key = f"conv:{conversation_id}:overflow:index"
                pipeline.sadd(index_key, chunk['chunk_id'])
                pipeline.expire(index_key, ttl_seconds)
            
            # Store metadata about overflow
            metadata_key = f"conv:{conversation_id}:overflow:metadata"
            metadata = {
                "total_chunks": len(chunks),
                "total_tokens": sum(c["token_count"] for c in chunks),
                "storage_layer": storage_layer,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(seconds=ttl_seconds)).isoformat()
            }
            pipeline.setex(metadata_key, ttl_seconds, json.dumps(metadata))
            
            # Execute pipeline
            pipeline.execute()
            
            logger.info(f"Stored {len(chunks)} overflow chunks in {storage_layer} for conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing overflow: {str(e)}")
            return False
    
    async def retrieve_relevant_chunks(self, query: str, conversation_id: str, 
                                      top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant overflow chunks based on semantic similarity to query.
        
        Args:
            query: User query to match against
            conversation_id: Conversation identifier
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of relevant chunk dictionaries
        """
        try:
            top_k = top_k or self.config.retrieval_top_k
            
            # Get chunk index for conversation
            index_key = f"conv:{conversation_id}:overflow:index"
            chunk_ids = self.redis_client.smembers(index_key)
            
            if not chunk_ids:
                logger.info(f"No overflow chunks found for conversation {conversation_id}")
                return []
            
            # For now, use simple keyword matching instead of embeddings
            # TODO: Re-enable embedding-based search when vector service is available
            
            # Retrieve all chunks and score by keyword matching
            scored_chunks = []
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for chunk_id in chunk_ids:
                # Try L1 first, then L2
                chunk_data = None
                for prefix in [f"conv:{conversation_id}:overflow:recent", 
                             f"conv:{conversation_id}:overflow:chunks"]:
                    chunk_key = f"{prefix}:{chunk_id}"
                    chunk_json = self.redis_client.get(chunk_key)
                    if chunk_json:
                        chunk_data = json.loads(chunk_json)
                        break
                
                if chunk_data:
                    # Calculate simple keyword similarity score
                    chunk_lower = chunk_data["content"].lower()
                    chunk_words = set(chunk_lower.split())
                    
                    # Score based on word overlap
                    common_words = query_words.intersection(chunk_words)
                    similarity = len(common_words) / max(len(query_words), 1)
                    
                    # Boost score if query appears as substring
                    if query_lower in chunk_lower:
                        similarity = min(1.0, similarity + 0.5)
                    
                    scored_chunks.append({
                        "chunk": chunk_data,
                        "similarity": similarity
                    })
            
            # Sort by similarity and return top-k
            scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
            top_chunks = [item["chunk"] for item in scored_chunks[:top_k]]
            
            # Update access counts and check for auto-promotion
            if self.config.auto_promote_to_l1:
                for chunk in top_chunks:
                    chunk["access_count"] = chunk.get("access_count", 0) + 1
                    chunk["last_accessed"] = datetime.utcnow().isoformat()
                    
                    # Check if should promote to L1
                    if chunk["access_count"] >= self.config.promotion_threshold_accesses:
                        # Check if not already in L1
                        l1_key = f"conv:{conversation_id}:overflow:recent:{chunk['chunk_id']}"
                        if not self.redis_client.exists(l1_key):
                            await self.promote_to_l1([chunk["chunk_id"]], conversation_id)
                            logger.info(f"Auto-promoted chunk {chunk['chunk_id']} to L1 after {chunk['access_count']} accesses")
            
            logger.info(f"Retrieved {len(top_chunks)} relevant chunks for conversation {conversation_id}")
            return top_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    async def get_overflow_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information about overflow content for a conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Summary dictionary or None if no overflow
        """
        try:
            metadata_key = f"conv:{conversation_id}:overflow:metadata"
            metadata_json = self.redis_client.get(metadata_key)
            
            if not metadata_json:
                return None
            
            metadata = json.loads(metadata_json)
            
            # Add remaining TTL information
            ttl = self.redis_client.ttl(metadata_key)
            if ttl > 0:
                metadata["ttl_remaining_hours"] = round(ttl / 3600, 1)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting overflow summary: {str(e)}")
            return None
    
    async def promote_to_l1(self, chunk_ids: List[str], conversation_id: str) -> bool:
        """
        Promote specific chunks from L2 to L1 for faster access.
        
        Args:
            chunk_ids: List of chunk IDs to promote
            conversation_id: Conversation identifier
            
        Returns:
            Success status
        """
        try:
            promoted_chunks = []
            
            for chunk_id in chunk_ids:
                # Find chunk in L2
                l2_key = f"conv:{conversation_id}:overflow:chunks:{chunk_id}"
                chunk_json = self.redis_client.get(l2_key)
                
                if chunk_json:
                    chunk_data = json.loads(chunk_json)
                    promoted_chunks.append(chunk_data)
            
            if promoted_chunks:
                # Store in L1 with shorter TTL
                await self.store_overflow(promoted_chunks, conversation_id, storage_layer="L1")
                logger.info(f"Promoted {len(promoted_chunks)} chunks to L1 for conversation {conversation_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error promoting chunks: {str(e)}")
            return False
    
    def _extract_summary(self, text: str, max_length: int = 100) -> str:
        """
        Extract a brief summary from chunk text.
        
        Args:
            text: Chunk text
            max_length: Maximum summary length
            
        Returns:
            Summary string
        """
        # Simple extraction: first sentence or first N characters
        sentences = text.split('. ')
        if sentences:
            summary = sentences[0]
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return summary
        return text[:max_length] + "..." if len(text) > max_length else text
    
    def _chunk_text_with_overlap(self, text: str, chunk_size: int, overlap_size: int) -> List[str]:
        """
        Chunk text with overlap for better context preservation.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in tokens
            overlap_size: Size of overlap between chunks in tokens
            
        Returns:
            List of text chunks
        """
        # Encode text to tokens
        tokens = self.encoder.encode(text)
        
        if len(tokens) <= chunk_size:
            return [text]
        
        chunks = []
        step_size = chunk_size - overlap_size
        
        for i in range(0, len(tokens), step_size):
            chunk_tokens = tokens[i:i + chunk_size]
            
            # Decode tokens back to text
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Stop if we've covered all tokens
            if i + chunk_size >= len(tokens):
                break
        
        return chunks
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract keywords from chunk text.
        
        Args:
            text: Chunk text
            max_keywords: Maximum number of keywords
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction: most common words > 5 chars
        words = text.lower().split()
        word_freq = {}
        
        for word in words:
            # Clean and filter words
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 5 and word not in ['the', 'this', 'that', 'these', 'those']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def _simple_chunk(self, content: str, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Fallback simple chunking method.
        
        Args:
            content: Content to chunk
            conversation_id: Conversation identifier
            
        Returns:
            List of simple chunks
        """
        chunks = []
        tokens = self.encoder.encode(content)
        
        for i in range(0, len(tokens), self.config.chunk_size_tokens - self.config.chunk_overlap_tokens):
            chunk_tokens = tokens[i:i + self.config.chunk_size_tokens]
            chunk_text = self.encoder.decode(chunk_tokens)
            
            chunk_id = hashlib.md5(f"{conversation_id}_{i}_{chunk_text[:50]}".encode()).hexdigest()
            
            chunks.append({
                "chunk_id": chunk_id,
                "conversation_id": conversation_id,
                "position": i // (self.config.chunk_size_tokens - self.config.chunk_overlap_tokens),
                "content": chunk_text,
                "embedding": [],  # Will need to generate separately
                "summary": self._extract_summary(chunk_text),
                "keywords": self._extract_keywords(chunk_text),
                "token_count": len(chunk_tokens),
                "created_at": datetime.utcnow().isoformat()
            })
        
        return chunks
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score between -1 and 1
        """
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0