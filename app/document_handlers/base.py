"""
Base document handler interface for different file types
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import os

@dataclass
class ExtractedChunk:
    """Represents an extracted chunk from a document"""
    content: str
    metadata: Dict[str, Any]
    quality_score: float = 1.0  # 0-1 score for chunk quality
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "quality_score": self.quality_score
        }

@dataclass
class ExtractionPreview:
    """Preview of document extraction for user validation"""
    total_chunks: int
    sample_chunks: List[ExtractedChunk]  # First few chunks
    file_metadata: Dict[str, Any]
    warnings: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chunks": self.total_chunks,
            "sample_chunks": [chunk.to_dict() for chunk in self.sample_chunks],
            "file_metadata": self.file_metadata,
            "warnings": self.warnings or []
        }

class DocumentHandler(ABC):
    """Abstract base class for document handlers"""
    
    # File size limits (in MB)
    MAX_FILE_SIZE_MB = 50
    
    # Quality thresholds
    MIN_CONTENT_LENGTH = 50  # Minimum characters for a valid chunk
    MIN_QUALITY_SCORE = 0.3  # Minimum quality score to include chunk
    
    def __init__(self):
        self.supported_extensions = []
        
    @abstractmethod
    def validate(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if file can be processed
        Returns: (is_valid, error_message)
        """
        # Common validation
        if not os.path.exists(file_path):
            return False, "File not found"
            
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            return False, f"File too large: {file_size_mb:.1f}MB (max: {self.MAX_FILE_SIZE_MB}MB)"
            
        return True, None
        
    @abstractmethod
    def extract(self, file_path: str, options: Dict[str, Any] = None) -> List[ExtractedChunk]:
        """
        Extract content from document
        Options can include sheet selection, page range, etc.
        """
        pass
        
    @abstractmethod
    def get_preview(self, file_path: str, max_chunks: int = 5) -> ExtractionPreview:
        """Get a preview of extraction without full processing"""
        pass
        
    def calculate_chunk_quality(self, chunk: ExtractedChunk) -> float:
        """
        Calculate quality score for a chunk (0-1)
        Factors: content length, structure, information density
        """
        content = chunk.content.strip()
        
        # Length score
        length_score = min(len(content) / 500, 1.0)  # Optimal at 500+ chars
        
        # Structure score (has proper sentences/data)
        has_punctuation = any(p in content for p in ['.', '!', '?', ';'])
        has_newlines = '\n' in content
        structure_score = 0.5
        if has_punctuation:
            structure_score += 0.25
        if has_newlines:
            structure_score += 0.25
            
        # Information density (not just whitespace/numbers)
        word_count = len(content.split())
        density_score = min(word_count / 50, 1.0)  # Optimal at 50+ words
        
        # Combine scores
        quality_score = (length_score * 0.4 + structure_score * 0.3 + density_score * 0.3)
        
        return round(quality_score, 2)
        
    def generate_chunk_id(self, file_id: str, chunk_index: int, **kwargs) -> str:
        """Generate unique chunk ID"""
        parts = [file_id, f"chunk_{chunk_index}"]
        for key, value in kwargs.items():
            parts.append(f"{key}_{value}")
        return "_".join(parts)
        
    def filter_quality_chunks(self, chunks: List[ExtractedChunk]) -> List[ExtractedChunk]:
        """Filter out low quality chunks"""
        filtered = []
        for chunk in chunks:
            # Skip empty or too short
            if len(chunk.content.strip()) < self.MIN_CONTENT_LENGTH:
                continue
                
            # Calculate and check quality
            chunk.quality_score = self.calculate_chunk_quality(chunk)
            if chunk.quality_score < self.MIN_QUALITY_SCORE:
                continue
                
            filtered.append(chunk)
            
        return filtered
    
    def calculate_quality_score(self, content: str) -> float:
        """
        Calculate quality score for text content (0-1)
        This is a convenience method that creates a temporary chunk
        """
        temp_chunk = ExtractedChunk(
            content=content,
            metadata={}
        )
        return self.calculate_chunk_quality(temp_chunk)