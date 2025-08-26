"""
Document Summarization Service

Creates concise summaries of documents for hierarchical RAG retrieval.
These summaries enable document-level filtering before chunk-level retrieval.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.notebook_llm_settings_cache import get_notebook_llm_settings
from app.core.db import SessionLocal
from app.llm.ollama import OllamaLLM
from app.models.llm_models import LLMConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentSummarizationService:
    """
    Service for creating and managing document summaries for hierarchical RAG
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,  # Larger chunks for summarization
            chunk_overlap=400,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
    async def create_document_summary(
        self,
        document_id: str,
        document_content: str,
        document_metadata: Dict[str, Any],
        max_summary_length: int = 500
    ) -> Dict[str, Any]:
        """
        Create a concise summary of a document for hierarchical retrieval
        
        Args:
            document_id: Unique document identifier
            document_content: Full document text content
            document_metadata: Document metadata (filename, type, etc.)
            max_summary_length: Target summary length in tokens
            
        Returns:
            Dictionary with summary and metadata
        """
        try:
            # Skip summarization for very short documents
            if len(document_content) < 1000:
                return {
                    "document_id": document_id,
                    "summary": document_content[:max_summary_length * 4],  # Rough token-to-char conversion
                    "summary_type": "excerpt",
                    "metadata": document_metadata,
                    "created_at": datetime.utcnow().isoformat(),
                    "content_hash": hashlib.md5(document_content.encode()).hexdigest()
                }
            
            # Split document into chunks for processing
            chunks = self.text_splitter.split_text(document_content)
            
            if len(chunks) == 1:
                # Single chunk - create extractive summary
                summary = await self._create_extractive_summary(chunks[0], max_summary_length)
            else:
                # Multiple chunks - create hierarchical summary
                summary = await self._create_hierarchical_summary(chunks, max_summary_length, document_metadata)
            
            return {
                "document_id": document_id,
                "summary": summary,
                "summary_type": "generated",
                "metadata": {
                    **document_metadata,
                    "chunk_count": len(chunks),
                    "original_length": len(document_content)
                },
                "created_at": datetime.utcnow().isoformat(),
                "content_hash": hashlib.md5(document_content.encode()).hexdigest()
            }
            
        except Exception as e:
            logger.error(f"Failed to create summary for document {document_id}: {str(e)}")
            # Return fallback summary
            return {
                "document_id": document_id,
                "summary": document_content[:max_summary_length * 4],
                "summary_type": "fallback",
                "metadata": document_metadata,
                "created_at": datetime.utcnow().isoformat(),
                "content_hash": hashlib.md5(document_content.encode()).hexdigest(),
                "error": str(e)
            }
    
    async def _create_extractive_summary(self, content: str, max_length: int) -> str:
        """
        Create extractive summary by selecting key sentences
        """
        try:
            # Use LLM for better extractive summarization
            notebook_settings = get_notebook_llm_settings()
            llm_config = notebook_settings.get('notebook_llm', {})
            
            if not llm_config:
                # Simple fallback: take first paragraphs
                sentences = content.split('. ')
                return '. '.join(sentences[:5]) + '.'
            
            # Initialize LLM
            llm_config_obj = LLMConfig(
                model_name=llm_config['model'],
                temperature=0.1,  # Low temperature for consistent summaries
                top_p=0.9,
                max_tokens=max_length
            )
            llm = OllamaLLM(llm_config_obj)
            
            prompt = f"""Create a concise summary of the following text. Focus on the main topics, key points, and important information. Keep it under {max_length} words.

Text to summarize:
{content[:4000]}  # Limit input to prevent context overflow

Summary:"""
            
            response = await llm.agenerate([prompt])
            if response and hasattr(response, 'generations') and response.generations:
                summary = response.generations[0][0].text.strip()
                return summary if summary else content[:max_length * 4]
            
            return content[:max_length * 4]
            
        except Exception as e:
            logger.warning(f"Extractive summarization failed, using fallback: {str(e)}")
            sentences = content.split('. ')
            return '. '.join(sentences[:3]) + '.'
    
    async def _create_hierarchical_summary(
        self, 
        chunks: List[str], 
        max_length: int, 
        metadata: Dict[str, Any]
    ) -> str:
        """
        Create hierarchical summary from multiple chunks
        """
        try:
            # Create summaries for individual chunks first
            chunk_summaries = []
            notebook_settings = get_notebook_llm_settings()
            llm_config = notebook_settings.get('notebook_llm', {})
            
            if not llm_config:
                # Fallback: use first sentences from each chunk
                for chunk in chunks[:3]:  # Limit to first 3 chunks
                    sentences = chunk.split('. ')
                    chunk_summaries.append('. '.join(sentences[:2]) + '.')
            else:
                # Use LLM for chunk summarization
                llm_config_obj = LLMConfig(
                    model_name=llm_config['model'],
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=100  # Short summaries for each chunk
                )
                llm = OllamaLLM(llm_config_obj)
                
                # Summarize up to 5 chunks to avoid overwhelming context
                for i, chunk in enumerate(chunks[:5]):
                    prompt = f"""Summarize this text section in 2-3 sentences, focusing on the main points:

{chunk[:2000]}

Summary:"""
                    
                    try:
                        response = await llm.agenerate([prompt])
                        if response and hasattr(response, 'generations') and response.generations:
                            summary = response.generations[0][0].text.strip()
                            if summary:
                                chunk_summaries.append(summary)
                        else:
                            # Fallback for this chunk
                            sentences = chunk.split('. ')
                            chunk_summaries.append('. '.join(sentences[:2]) + '.')
                    except Exception as e:
                        logger.warning(f"Failed to summarize chunk {i}: {str(e)}")
                        sentences = chunk.split('. ')
                        chunk_summaries.append('. '.join(sentences[:2]) + '.')
                
                # Create final summary from chunk summaries
                if chunk_summaries:
                    combined_summary = " ".join(chunk_summaries)
                    
                    # If combined summary is too long, create a meta-summary
                    if len(combined_summary) > max_length * 6:  # Rough token limit
                        final_prompt = f"""Create a comprehensive summary from these section summaries. Keep it under {max_length} words:

{combined_summary[:3000]}

Final Summary:"""
                        
                        response = await llm.agenerate([final_prompt])
                        if response and hasattr(response, 'generations') and response.generations:
                            final_summary = response.generations[0][0].text.strip()
                            if final_summary:
                                return final_summary
                    
                    return combined_summary[:max_length * 4]
            
            return " ".join(chunk_summaries)[:max_length * 4]
            
        except Exception as e:
            logger.error(f"Hierarchical summarization failed: {str(e)}")
            # Fallback: use first chunk's beginning
            if chunks:
                return chunks[0][:max_length * 4]
            return "Summary unavailable"
    
    async def batch_create_summaries(
        self,
        documents: List[Tuple[str, str, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Create summaries for multiple documents in batch
        
        Args:
            documents: List of (document_id, content, metadata) tuples
            
        Returns:
            List of summary dictionaries
        """
        summaries = []
        
        for document_id, content, metadata in documents:
            try:
                summary = await self.create_document_summary(
                    document_id, content, metadata
                )
                summaries.append(summary)
                
                # Add small delay to prevent overwhelming the LLM service
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to create summary for document {document_id}: {str(e)}")
                # Add error placeholder
                summaries.append({
                    "document_id": document_id,
                    "summary": "Summary generation failed",
                    "summary_type": "error",
                    "metadata": metadata,
                    "created_at": datetime.utcnow().isoformat(),
                    "error": str(e)
                })
        
        logger.info(f"Created summaries for {len(summaries)} documents")
        return summaries
    
    async def update_summary_if_changed(
        self,
        document_id: str,
        new_content: str,
        new_metadata: Dict[str, Any],
        existing_summary: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update summary only if document content has changed
        
        Args:
            document_id: Document identifier
            new_content: Current document content
            new_metadata: Current document metadata
            existing_summary: Previously created summary
            
        Returns:
            New summary if content changed, None otherwise
        """
        # Check if content has changed using hash
        new_hash = hashlib.md5(new_content.encode()).hexdigest()
        existing_hash = existing_summary.get("content_hash", "")
        
        if new_hash == existing_hash:
            logger.debug(f"Document {document_id} content unchanged, keeping existing summary")
            return None
        
        logger.info(f"Document {document_id} content changed, creating new summary")
        return await self.create_document_summary(document_id, new_content, new_metadata)


# Factory function
def get_document_summarization_service() -> DocumentSummarizationService:
    """Get singleton instance of document summarization service"""
    if not hasattr(get_document_summarization_service, '_instance'):
        get_document_summarization_service._instance = DocumentSummarizationService()
    return get_document_summarization_service._instance