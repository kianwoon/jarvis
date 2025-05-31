"""
Focused RAG search implementation that prioritizes entity-centric results
"""
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
import re

class FocusedRAGSearch:
    """Implements entity-focused search strategy for better relevance"""
    
    def __init__(self, collection, embeddings):
        self.collection = collection
        self.embeddings = embeddings
        
    def search(self, query_analysis: Dict[str, Any], top_k: int = 50) -> List[Document]:
        """
        Execute focused search based on query analysis
        """
        strategy = query_analysis['search_strategy']
        
        if strategy == 'entity_focused':
            return self._entity_focused_search(query_analysis, top_k)
        elif strategy == 'entity_broad':
            return self._entity_broad_search(query_analysis, top_k)
        else:
            return self._general_search(query_analysis, top_k)
    
    def _entity_focused_search(self, analysis: Dict[str, Any], top_k: int) -> List[Document]:
        """
        Search specifically for entity + aspects
        """
        core_entity = analysis['core_entity']
        aspects = analysis['aspects']
        
        print(f"[DEBUG] Entity-focused search: entity='{core_entity}', aspects={aspects}")
        
        # Phase 1: Find all documents containing the core entity
        entity_docs = self._search_by_entity(core_entity, limit=200)
        print(f"[DEBUG] Found {len(entity_docs)} documents containing '{core_entity}'")
        
        if not entity_docs:
            # Fallback to vector search if no keyword matches
            return self._vector_search_with_boost(analysis['original_query'], core_entity, top_k)
        
        # Phase 2: Score documents by aspect relevance
        scored_docs = []
        for doc in entity_docs:
            score = self._calculate_aspect_relevance(doc, aspects, core_entity)
            scored_docs.append((score, doc))
        
        # Sort by score and return top results
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Log top results for debugging
        print(f"[DEBUG] Top 3 scored results:")
        for i, (score, doc) in enumerate(scored_docs[:3]):
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  {i+1}. Score={score:.3f}, content='{content_preview}...'")
        
        return [doc for score, doc in scored_docs[:top_k]]
    
    def _entity_broad_search(self, analysis: Dict[str, Any], top_k: int) -> List[Document]:
        """
        Search for entity without specific aspects
        """
        core_entity = analysis['core_entity']
        
        print(f"[DEBUG] Entity-broad search: entity='{core_entity}'")
        
        # Get all documents about the entity
        entity_docs = self._search_by_entity(core_entity, limit=top_k * 2)
        
        if not entity_docs:
            return self._vector_search_with_boost(analysis['original_query'], core_entity, top_k)
        
        # Score by entity prominence
        scored_docs = []
        for doc in entity_docs:
            score = self._calculate_entity_prominence(doc, core_entity)
            scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]
    
    def _general_search(self, analysis: Dict[str, Any], top_k: int) -> List[Document]:
        """
        Fallback to general search when no clear entity is found
        """
        print(f"[DEBUG] General search fallback")
        
        # Use standard vector search
        query_text = analysis['original_query']
        query_embedding = self.embeddings.embed_query(query_text)
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["content", "source", "page", "hash", "doc_id", "section", "doc_type"]
        )
        
        docs = []
        for hits in results:
            for hit in hits:
                doc = Document(
                    page_content=hit.entity.get("content", ""),
                    metadata={
                        "source": hit.entity.get("source", ""),
                        "page": hit.entity.get("page", 0),
                        "hash": hit.entity.get("hash", ""),
                        "doc_id": hit.entity.get("doc_id", ""),
                        "section": hit.entity.get("section", ""),
                        "doc_type": hit.entity.get("doc_type", ""),
                        "distance": hit.distance
                    }
                )
                docs.append(doc)
        
        return docs
    
    def _search_by_entity(self, entity: str, limit: int = 100) -> List[Document]:
        """
        Search for documents containing the entity in content, source, or section fields
        Case-insensitive search by checking multiple variations
        """
        # Use dynamic case variation generation
        try:
            from app.langchain.dynamic_case_helper import generate_case_variations
            variations = generate_case_variations(entity)
        except ImportError:
            # Fallback to basic variations if helper not available
            variations = [entity.lower(), entity.upper(), entity.title()]
        
        print(f"[DEBUG] Dynamic case variations for '{entity}': {variations}")
        
        # Build expressions for each field and variation
        content_conditions = [f'content like "%{var}%"' for var in variations]
        source_conditions = [f'source like "%{var}%"' for var in variations]
        section_conditions = [f'section like "%{var}%"' for var in variations]
        
        # Combine all conditions
        all_conditions = content_conditions + source_conditions + section_conditions
        expr = " or ".join(all_conditions)
        
        print(f"[DEBUG] Case-insensitive search with {len(variations)} variations")
        
        try:
            results = self.collection.query(
                expr=expr,
                output_fields=["content", "source", "page", "hash", "doc_id", "section", "doc_type"],
                limit=limit
            )
            
            docs = []
            for r in results:
                doc = Document(
                    page_content=r.get("content", ""),
                    metadata={
                        "source": r.get("source", ""),
                        "page": r.get("page", 0),
                        "hash": r.get("hash", ""),
                        "doc_id": r.get("doc_id", ""),
                        "section": r.get("section", ""),
                        "doc_type": r.get("doc_type", "")
                    }
                )
                docs.append(doc)
            
            return docs
            
        except Exception as e:
            print(f"[ERROR] Entity search failed: {e}")
            return []
    
    def _calculate_aspect_relevance(self, doc: Document, aspects: List[str], entity: str) -> float:
        """
        Calculate how relevant a document is to the requested aspects
        """
        content_lower = doc.page_content.lower()
        source_lower = doc.metadata.get('source', '').lower()
        section_lower = doc.metadata.get('section', '').lower()
        
        score = 0.0
        
        # Count entity occurrences (case-insensitive)
        entity_lower = entity.lower()
        entity_count = content_lower.count(entity_lower)
        entity_in_source = entity_lower in source_lower
        entity_in_section = entity_lower in section_lower
        
        # Strong signal if entity is in the source/filename
        if entity_in_source:
            score += 2.0
            
        # Also boost if entity is in section header
        if entity_in_section:
            score += 1.5
        
        # Score based on entity frequency
        score += min(entity_count * 0.1, 1.0)
        
        # Score based on aspects
        for aspect in aspects:
            aspect_lower = aspect.lower()
            aspect_words = aspect_lower.split()
            
            # Check for exact aspect match
            if aspect_lower in content_lower:
                score += 1.0
            
            # Check for individual aspect words
            word_matches = sum(1 for word in aspect_words if word in content_lower)
            score += word_matches * 0.3
            
            # Bonus if aspect appears near entity
            if self._terms_near_each_other(content_lower, entity_lower, aspect_lower, window=50):
                score += 0.5
        
        return score
    
    def _calculate_entity_prominence(self, doc: Document, entity: str) -> float:
        """
        Calculate how prominent the entity is in the document
        """
        content_lower = doc.page_content.lower()
        source_lower = doc.metadata.get('source', '').lower() 
        section_lower = doc.metadata.get('section', '').lower()
        entity_lower = entity.lower()
        
        score = 0.0
        
        # Source/filename bonus
        if entity_lower in source_lower:
            score += 3.0
            
        # Section header bonus  
        if entity_lower in section_lower:
            score += 2.0
        
        # Frequency score (case-insensitive)
        entity_count = content_lower.count(entity_lower)
        score += min(entity_count * 0.2, 2.0)
        
        # Position score (earlier is better)
        first_occurrence = content_lower.find(entity_lower)
        if first_occurrence != -1:
            position_score = 1.0 - (first_occurrence / max(len(content_lower), 1))
            score += position_score * 0.5
        
        return score
    
    def _terms_near_each_other(self, text: str, term1: str, term2: str, window: int = 50) -> bool:
        """
        Check if two terms appear near each other in the text
        """
        # Find all occurrences of both terms
        indices1 = [m.start() for m in re.finditer(re.escape(term1), text)]
        indices2 = [m.start() for m in re.finditer(re.escape(term2), text)]
        
        # Check if any occurrences are within the window
        for i1 in indices1:
            for i2 in indices2:
                if abs(i1 - i2) <= window:
                    return True
        
        return False
    
    def _vector_search_with_boost(self, query: str, boost_term: str, top_k: int) -> List[Document]:
        """
        Vector search with boosting for documents containing specific term
        """
        # Modify query to emphasize the boost term
        boosted_query = f"{boost_term} {query}"
        
        query_embedding = self.embeddings.embed_query(boosted_query)
        
        # Search with larger limit to allow for filtering
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k * 2,
            output_fields=["content", "source", "page", "hash", "doc_id", "section", "doc_type"]
        )
        
        docs = []
        boost_docs = []
        
        for hits in results:
            for hit in hits:
                doc = Document(
                    page_content=hit.entity.get("content", ""),
                    metadata={
                        "source": hit.entity.get("source", ""),
                        "page": hit.entity.get("page", 0),
                        "hash": hit.entity.get("hash", ""),
                        "doc_id": hit.entity.get("doc_id", ""),
                        "section": hit.entity.get("section", ""),
                        "doc_type": hit.entity.get("doc_type", ""),
                        "distance": hit.distance
                    }
                )
                
                # Separate docs containing boost term
                if boost_term.lower() in doc.page_content.lower() or \
                   boost_term.lower() in doc.metadata.get('source', '').lower():
                    boost_docs.append(doc)
                else:
                    docs.append(doc)
        
        # Return boosted docs first, then others
        combined = boost_docs + docs
        return combined[:top_k]