"""
BM25 Text Processor for Enhanced RAG Retrieval

This module provides BM25 preprocessing and scoring capabilities to enhance
the existing hybrid search system with proper statistical text ranking.
"""

import math
import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class BM25Stats:
    """Statistics for BM25 calculation"""
    total_documents: int
    average_doc_length: float
    term_doc_frequencies: Dict[str, int]  # How many docs contain each term
    

class BM25Processor:
    """
    Enhanced BM25 processor that integrates with existing Milvus infrastructure
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 processor with standard parameters
        
        Args:
            k1: Controls term frequency saturation (typically 1.2-2.0)
            b: Controls document length normalization (typically 0.75)
        """
        self.k1 = k1
        self.b = b
        
        # Enhanced stop words for better keyword extraction
        self.stop_words = {
            'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'a', 'an', 'what', 'how', 'when', 'where', 'why', 'like', 'it', 'this', 'that', 'these',
            'those', 'can', 'could', 'would', 'should', 'may', 'might', 'must', 'shall', 'will',
            'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been', 'being', 'was', 'were',
            'find', 'get', 'show', 'tell', 'about', 'info', 'information', 'news', 'give', 'provide',
            'need', 'want', 'please', 'help', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'our', 'their', 'mine', 'yours',
            'also', 'just', 'now', 'then', 'here', 'there', 'up', 'down', 'out', 'off', 'over',
            'under', 'again', 'further', 'once', 'before', 'after', 'above', 'below', 'between'
        }
    
    def tokenize_and_clean(self, text: str) -> List[str]:
        """
        Tokenize text and remove stop words, maintaining compatibility with existing system
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of cleaned tokens
        """
        if not text:
            return []
        
        # Convert to lowercase and extract words (alphanumeric + common punctuation)
        text = text.lower().strip()
        
        # Remove special characters but keep alphanumeric, spaces, and hyphens
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Split on whitespace and filter
        tokens = [
            word.strip('-') for word in text.split()
            if len(word) > 2 and word not in self.stop_words and word.strip('-')
        ]
        
        return tokens
    
    def calculate_term_frequencies(self, text: str) -> Dict[str, int]:
        """
        Calculate term frequencies for a document
        
        Args:
            text: Document text
            
        Returns:
            Dictionary mapping terms to their frequencies
        """
        tokens = self.tokenize_and_clean(text)
        return dict(Counter(tokens))
    
    def prepare_document_for_bm25(self, text: str, existing_metadata: dict = None) -> dict:
        """
        Prepare document metadata for BM25 indexing
        
        Args:
            text: Document content
            existing_metadata: Existing document metadata to extend
            
        Returns:
            Enhanced metadata with BM25 fields
        """
        if existing_metadata is None:
            existing_metadata = {}
        
        # Calculate BM25-specific fields
        tokens = self.tokenize_and_clean(text)
        term_frequencies = self.calculate_term_frequencies(text)
        
        # Add BM25 metadata
        bm25_metadata = existing_metadata.copy()
        bm25_metadata.update({
            'bm25_tokens': json.dumps(tokens[:100]),  # Store first 100 tokens for debugging
            'bm25_term_count': len(tokens),
            'bm25_unique_terms': len(term_frequencies),
            'bm25_top_terms': json.dumps(
                dict(sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)[:20])
            )  # Top 20 terms with frequencies
        })
        
        return bm25_metadata
    
    def calculate_bm25_score(
        self, 
        query_terms: List[str], 
        doc_term_frequencies: Dict[str, int],
        doc_length: int,
        corpus_stats: BM25Stats
    ) -> float:
        """
        Calculate BM25 score for a document given query terms
        
        Args:
            query_terms: List of query terms
            doc_term_frequencies: Term frequencies in the document
            doc_length: Length of document in tokens
            corpus_stats: Corpus statistics for IDF calculation
            
        Returns:
            BM25 score
        """
        if not query_terms or not doc_term_frequencies or corpus_stats.total_documents == 0:
            return 0.0
        
        score = 0.0
        
        for term in query_terms:
            # Term frequency in document
            tf = doc_term_frequencies.get(term, 0)
            if tf == 0:
                continue
            
            # Inverse Document Frequency
            df = corpus_stats.term_doc_frequencies.get(term, 0)
            if df == 0:
                continue
            
            idf = math.log((corpus_stats.total_documents - df + 0.5) / (df + 0.5))
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / corpus_stats.average_doc_length)
            )
            
            score += idf * (numerator / denominator)
        
        return max(0.0, score)
    
    def enhance_existing_relevance_score(
        self,
        query: str,
        content: str,
        corpus_stats: Optional[BM25Stats] = None,
        existing_score: float = 0.0
    ) -> float:
        """
        Enhance existing relevance score with BM25 calculation
        
        Args:
            query: Search query
            content: Document content
            corpus_stats: Corpus statistics (if available)
            existing_score: Existing relevance score from current system
            
        Returns:
            Enhanced score combining existing logic with BM25
        """
        if not query or not content:
            return existing_score
        
        # Get query terms
        query_terms = self.tokenize_and_clean(query)
        if not query_terms:
            return existing_score
        
        # Calculate document metrics
        doc_term_frequencies = self.calculate_term_frequencies(content)
        doc_length = len(self.tokenize_and_clean(content))
        
        if not doc_term_frequencies or doc_length == 0:
            return existing_score
        
        # If we have corpus stats, calculate proper BM25
        bm25_score = 0.0
        if corpus_stats:
            bm25_score = self.calculate_bm25_score(
                query_terms, doc_term_frequencies, doc_length, corpus_stats
            )
            # Normalize BM25 score to 0-1 range (approximate)
            bm25_score = min(1.0, bm25_score / 10.0)  # Typical BM25 scores are 0-10
        
        # Calculate simple term matching as fallback
        matching_terms = sum(1 for term in query_terms if term in doc_term_frequencies)
        term_coverage = matching_terms / len(query_terms) if query_terms else 0.0
        
        # Combine scores
        if corpus_stats:
            # Use proper BM25 with higher weight
            final_score = (
                existing_score * 0.4 +      # Existing TF-IDF-like score
                bm25_score * 0.5 +          # Proper BM25 score
                term_coverage * 0.1         # Basic term coverage
            )
        else:
            # Fallback without corpus stats
            final_score = (
                existing_score * 0.7 +      # Keep existing logic as primary
                term_coverage * 0.3         # Add simple term matching boost
            )
        
        return min(1.0, max(0.0, final_score))
    
    def extract_searchable_terms(self, text: str, max_terms: int = 50) -> List[str]:
        """
        Extract key terms for search indexing
        
        Args:
            text: Input text
            max_terms: Maximum number of terms to return
            
        Returns:
            List of important terms for indexing
        """
        tokens = self.tokenize_and_clean(text)
        if not tokens:
            return []
        
        # Calculate term frequencies
        term_freq = Counter(tokens)
        
        # Score terms by frequency and length (longer terms often more important)
        scored_terms = []
        for term, freq in term_freq.items():
            # Score based on frequency and length
            length_bonus = min(2.0, len(term) / 5.0)  # Bonus for longer terms
            score = freq * (1.0 + length_bonus)
            scored_terms.append((term, score))
        
        # Sort by score and return top terms
        scored_terms.sort(key=lambda x: x[1], reverse=True)
        return [term for term, _ in scored_terms[:max_terms]]


class BM25CorpusManager:
    """
    Manages corpus statistics for BM25 calculation
    """
    
    def __init__(self):
        self.processor = BM25Processor()
    
    def calculate_corpus_stats_from_milvus(self, collection) -> BM25Stats:
        """
        Calculate corpus statistics from Milvus collection
        
        Args:
            collection: Milvus collection object
            
        Returns:
            BM25Stats object with corpus information
        """
        try:
            # Load collection to ensure it's ready
            collection.load()
            
            # Get total document count
            total_docs = collection.num_entities
            
            if total_docs == 0:
                return BM25Stats(
                    total_documents=0,
                    average_doc_length=0.0,
                    term_doc_frequencies={}
                )
            
            # Query all documents to calculate statistics
            # Note: This is expensive for large collections, consider sampling
            iterator = collection.query_iterator(
                expr="",  # Empty expression = all documents
                output_fields=["content", "bm25_term_count"],
                batch_size=1000
            )
            
            total_length = 0
            term_doc_freq = defaultdict(int)
            processed_docs = 0
            
            for batch in iterator:
                for entity in batch:
                    content = entity.get("content", "")
                    stored_length = entity.get("bm25_term_count", 0)
                    
                    # Use stored length if available, otherwise calculate
                    if stored_length > 0:
                        doc_length = stored_length
                    else:
                        doc_length = len(self.processor.tokenize_and_clean(content))
                    
                    total_length += doc_length
                    processed_docs += 1
                    
                    # Calculate which terms appear in this document
                    doc_terms = set(self.processor.tokenize_and_clean(content))
                    for term in doc_terms:
                        term_doc_freq[term] += 1
            
            # Calculate average document length
            avg_length = total_length / processed_docs if processed_docs > 0 else 0.0
            
            return BM25Stats(
                total_documents=processed_docs,
                average_doc_length=avg_length,
                term_doc_frequencies=dict(term_doc_freq)
            )
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate corpus stats: {e}")
            # Return empty stats as fallback
            return BM25Stats(
                total_documents=0,
                average_doc_length=0.0,
                term_doc_frequencies={}
            )
    
    def get_cached_corpus_stats(self, collection_name: str) -> Optional[BM25Stats]:
        """
        Get cached corpus statistics (implement Redis caching later)
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Cached BM25Stats or None
        """
        # TODO: Implement Redis caching for corpus statistics
        # For now, return None to force recalculation
        return None
    
    def cache_corpus_stats(self, collection_name: str, stats: BM25Stats):
        """
        Cache corpus statistics (implement Redis caching later)
        
        Args:
            collection_name: Name of the collection
            stats: Statistics to cache
        """
        # TODO: Implement Redis caching for corpus statistics
        pass