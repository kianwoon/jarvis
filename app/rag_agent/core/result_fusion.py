"""
Result fusion engine for combining and ranking multi-collection search results

This module implements sophisticated fusion strategies to combine results from
multiple collections into coherent, well-sourced responses.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass

from app.rag_agent.utils.types import (
    StepResult, CollectionResult, Source, FusionMethod,
    SearchContext, RAGOptions
)

logger = logging.getLogger(__name__)


@dataclass
class FusedResponse:
    """Final fused response with sources and metadata"""
    content: str
    sources: List[Source]
    confidence_score: float
    fusion_method: FusionMethod
    collection_contributions: Dict[str, float]  # collection -> contribution score


class ResultFusion:
    """Intelligent fusion of multi-collection search results"""
    
    def __init__(self):
        # Collection authority weights for different query types
        self.authority_weights = {
            "regulatory_compliance": {
                "compliance": 1.0,
                "policy": 0.8,
                "legal": 0.9,
                "general": 0.3
            },
            "product_documentation": {
                "product": 1.0,
                "pricing": 1.0,
                "features": 0.9,
                "technical": 0.7,
                "general": 0.4
            },
            "risk_management": {
                "risk": 1.0,
                "compliance": 0.8,
                "audit": 0.9,
                "general": 0.3
            },
            "technical_docs": {
                "technical": 1.0,
                "api": 1.0,
                "implementation": 0.9,
                "general": 0.4
            },
            "customer_support": {
                "support": 1.0,
                "troubleshooting": 1.0,
                "customer": 0.9,
                "general": 0.5
            }
        }
    
    async def fuse_results(
        self,
        step_results: List[StepResult],
        original_query: str,
        routing_decision,
        context: Optional[SearchContext] = None,
        options: Optional[RAGOptions] = None
    ) -> FusedResponse:
        """
        Fuse results from multiple execution steps into a coherent response
        
        Args:
            step_results: Results from all execution steps
            original_query: Original user query
            routing_decision: LLM routing decision
            context: Search context
            options: RAG options
            
        Returns:
            FusedResponse with synthesized content and sources
        """
        
        if not step_results or not any(sr.collection_results for sr in step_results):
            return self._create_empty_response()
        
        # Collect all sources from all steps
        all_sources = []
        collection_results = []
        
        for step_result in step_results:
            for collection_result in step_result.collection_results:
                collection_results.append(collection_result)
                all_sources.extend(collection_result.sources)
        
        if not all_sources:
            return self._create_empty_response()
        
        logger.info(f"Fusing {len(all_sources)} sources from {len(collection_results)} collections")
        
        # Determine optimal fusion strategy
        fusion_method = self._select_fusion_strategy(
            collection_results, original_query, context
        )
        
        # Apply fusion strategy
        if fusion_method == FusionMethod.COLLECTION_AUTHORITY:
            fused_response = await self._fuse_by_collection_authority(
                all_sources, original_query, context
            )
        elif fusion_method == FusionMethod.TEMPORAL_PRIORITY:
            fused_response = await self._fuse_by_temporal_priority(
                all_sources, original_query, context
            )
        elif fusion_method == FusionMethod.CROSS_VALIDATION:
            fused_response = await self._fuse_by_cross_validation(
                all_sources, original_query, context
            )
        else:  # RELEVANCE_WEIGHTED
            fused_response = await self._fuse_by_relevance_weight(
                all_sources, original_query, context
            )
        
        # Enhance with collection contributions
        fused_response.collection_contributions = self._calculate_collection_contributions(
            collection_results
        )
        
        fused_response.fusion_method = fusion_method
        
        logger.info(f"Fusion completed with {fusion_method.value} method, "
                   f"confidence: {fused_response.confidence_score:.2f}")
        
        return fused_response
    
    def _select_fusion_strategy(
        self,
        collection_results: List[CollectionResult],
        query: str,
        context: Optional[SearchContext] = None
    ) -> FusionMethod:
        """Select optimal fusion strategy based on results and query characteristics"""
        
        # Analyze query type
        query_lower = query.lower()
        
        # For compliance/regulatory queries, use authority-based fusion
        if any(word in query_lower for word in ["policy", "regulation", "compliance", "requirement", "rule"]):
            return FusionMethod.COLLECTION_AUTHORITY
        
        # For comparative queries, use cross-validation
        if any(word in query_lower for word in ["compare", "difference", "versus", "vs", "better"]):
            return FusionMethod.CROSS_VALIDATION
        
        # For time-sensitive queries, use temporal priority
        if any(word in query_lower for word in ["latest", "recent", "current", "new", "updated"]):
            return FusionMethod.TEMPORAL_PRIORITY
        
        # Analyze collection diversity
        collection_types = set(cr.collection_name.split('_')[0] for cr in collection_results)
        
        if len(collection_types) > 2:
            # Multiple collection types - use authority weighting
            return FusionMethod.COLLECTION_AUTHORITY
        
        # Default to relevance-weighted fusion
        return FusionMethod.RELEVANCE_WEIGHTED
    
    async def _fuse_by_relevance_weight(
        self,
        sources: List[Source],
        query: str,
        context: Optional[SearchContext] = None
    ) -> FusedResponse:
        """Fuse results based purely on relevance scores"""
        
        # Sort sources by score
        sorted_sources = sorted(sources, key=lambda s: s.score, reverse=True)
        
        # Take top sources and deduplicate
        top_sources = self._deduplicate_sources(sorted_sources[:15])
        
        # Generate response content
        content = await self._generate_response_content(
            top_sources, query, "relevance_weighted"
        )
        
        # Calculate confidence
        confidence = self._calculate_fusion_confidence(top_sources)
        
        return FusedResponse(
            content=content,
            sources=top_sources[:10],  # Limit final sources
            confidence_score=confidence,
            fusion_method=FusionMethod.RELEVANCE_WEIGHTED,
            collection_contributions={}
        )
    
    async def _fuse_by_collection_authority(
        self,
        sources: List[Source],
        query: str,
        context: Optional[SearchContext] = None
    ) -> FusedResponse:
        """Fuse results based on collection authority for query type"""
        
        # Classify query type
        query_type = self._classify_query_type(query)
        
        # Weight sources by collection authority
        weighted_sources = []
        for source in sources:
            collection_type = self._get_collection_type(source.collection_name)
            
            # Get authority weight
            authority_weight = self.authority_weights.get(collection_type, {}).get(
                query_type, 0.5
            )
            
            # Apply authority weighting to score
            weighted_score = source.score * authority_weight
            
            # Create weighted source
            weighted_source = Source(
                collection_name=source.collection_name,
                document_id=source.document_id,
                content=source.content,
                score=weighted_score,
                metadata=source.metadata,
                page=source.page,
                section=source.section
            )
            weighted_sources.append(weighted_source)
        
        # Sort by weighted score
        sorted_sources = sorted(weighted_sources, key=lambda s: s.score, reverse=True)
        top_sources = self._deduplicate_sources(sorted_sources[:15])
        
        # Generate authority-aware response
        content = await self._generate_response_content(
            top_sources, query, "collection_authority"
        )
        
        confidence = self._calculate_fusion_confidence(top_sources)
        
        return FusedResponse(
            content=content,
            sources=top_sources[:10],
            confidence_score=confidence,
            fusion_method=FusionMethod.COLLECTION_AUTHORITY,
            collection_contributions={}
        )
    
    async def _fuse_by_cross_validation(
        self,
        sources: List[Source],
        query: str,
        context: Optional[SearchContext] = None
    ) -> FusedResponse:
        """Fuse results by cross-validating information across collections"""
        
        # Group sources by collection
        collection_groups = defaultdict(list)
        for source in sources:
            collection_groups[source.collection_name].append(source)
        
        # Find information that appears in multiple collections
        validated_sources = []
        
        for collection_name, collection_sources in collection_groups.items():
            for source in collection_sources:
                # Check if similar information exists in other collections
                validation_score = self._calculate_cross_validation_score(
                    source, collection_groups, collection_name
                )
                
                # Boost score for cross-validated information
                if validation_score > 0.3:
                    source.score = source.score * (1 + validation_score)
                
                validated_sources.append(source)
        
        # Sort by validated scores
        sorted_sources = sorted(validated_sources, key=lambda s: s.score, reverse=True)
        top_sources = self._deduplicate_sources(sorted_sources[:15])
        
        # Generate cross-validated response
        content = await self._generate_response_content(
            top_sources, query, "cross_validation"
        )
        
        confidence = self._calculate_fusion_confidence(top_sources)
        
        return FusedResponse(
            content=content,
            sources=top_sources[:10],
            confidence_score=confidence,
            fusion_method=FusionMethod.CROSS_VALIDATION,
            collection_contributions={}
        )
    
    async def _fuse_by_temporal_priority(
        self,
        sources: List[Source],
        query: str,
        context: Optional[SearchContext] = None
    ) -> FusedResponse:
        """Fuse results prioritizing more recent information"""
        
        # Extract temporal information and boost recent sources
        temporal_sources = []
        
        for source in sources:
            # Extract date from metadata or content
            recency_score = self._calculate_recency_score(source)
            
            # Boost score based on recency
            temporal_score = source.score * (1 + recency_score * 0.3)
            
            temporal_source = Source(
                collection_name=source.collection_name,
                document_id=source.document_id,
                content=source.content,
                score=temporal_score,
                metadata=source.metadata,
                page=source.page,
                section=source.section
            )
            temporal_sources.append(temporal_source)
        
        # Sort by temporal-weighted score
        sorted_sources = sorted(temporal_sources, key=lambda s: s.score, reverse=True)
        top_sources = self._deduplicate_sources(sorted_sources[:15])
        
        # Generate temporal-aware response
        content = await self._generate_response_content(
            top_sources, query, "temporal_priority"
        )
        
        confidence = self._calculate_fusion_confidence(top_sources)
        
        return FusedResponse(
            content=content,
            sources=top_sources[:10],
            confidence_score=confidence,
            fusion_method=FusionMethod.TEMPORAL_PRIORITY,
            collection_contributions={}
        )
    
    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """Remove duplicate or very similar sources"""
        
        deduplicated = []
        seen_content_hashes = set()
        
        for source in sources:
            # Create content hash for deduplication
            content_preview = source.content[:200].lower().strip()
            content_hash = hash(content_preview)
            
            if content_hash not in seen_content_hashes:
                deduplicated.append(source)
                seen_content_hashes.add(content_hash)
            
            # Limit to prevent excessive deduplication processing
            if len(deduplicated) >= 20:
                break
        
        return deduplicated
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for authority weighting"""
        
        query_lower = query.lower()
        
        # Policy/compliance queries
        if any(word in query_lower for word in ["policy", "regulation", "compliance", "rule"]):
            return "compliance"
        
        # Product queries
        if any(word in query_lower for word in ["product", "pricing", "feature", "offering"]):
            return "product"
        
        # Technical queries
        if any(word in query_lower for word in ["api", "technical", "code", "implementation"]):
            return "technical"
        
        # Risk queries
        if any(word in query_lower for word in ["risk", "assessment", "mitigation"]):
            return "risk"
        
        # Support queries
        if any(word in query_lower for word in ["help", "support", "troubleshoot", "problem"]):
            return "support"
        
        # Legal queries
        if any(word in query_lower for word in ["legal", "contract", "agreement", "terms"]):
            return "legal"
        
        return "general"
    
    def _get_collection_type(self, collection_name: str) -> str:
        """Extract collection type from collection name"""
        
        # Handle naming patterns
        if "regulatory" in collection_name or "compliance" in collection_name:
            return "regulatory_compliance"
        elif "product" in collection_name:
            return "product_documentation"
        elif "risk" in collection_name:
            return "risk_management"
        elif "technical" in collection_name or "api" in collection_name:
            return "technical_docs"
        elif "support" in collection_name or "customer" in collection_name:
            return "customer_support"
        elif "audit" in collection_name:
            return "audit_reports"
        elif "training" in collection_name:
            return "training_materials"
        
        return "general"
    
    def _calculate_cross_validation_score(
        self,
        source: Source,
        collection_groups: Dict[str, List[Source]],
        exclude_collection: str
    ) -> float:
        """Calculate how well this source is validated by other collections"""
        
        validation_score = 0.0
        source_keywords = set(re.findall(r'\b\w{4,}\b', source.content.lower()))
        
        for collection_name, other_sources in collection_groups.items():
            if collection_name == exclude_collection:
                continue
            
            for other_source in other_sources:
                other_keywords = set(re.findall(r'\b\w{4,}\b', other_source.content.lower()))
                
                # Calculate keyword overlap
                overlap = len(source_keywords.intersection(other_keywords))
                total_keywords = len(source_keywords.union(other_keywords))
                
                if total_keywords > 0:
                    similarity = overlap / total_keywords
                    if similarity > 0.2:  # Threshold for meaningful overlap
                        validation_score += similarity * 0.1  # Scale down the boost
        
        return min(validation_score, 1.0)
    
    def _calculate_recency_score(self, source: Source) -> float:
        """Calculate recency score based on metadata or content analysis"""
        
        recency_score = 0.0
        
        # Check metadata for dates
        uploaded_at = source.metadata.get('uploaded_at', '')
        if uploaded_at:
            # Simple heuristic: more recent uploads get higher scores
            # This would need proper date parsing in production
            if '2024' in uploaded_at:
                recency_score += 0.5
            elif '2023' in uploaded_at:
                recency_score += 0.3
        
        # Check content for temporal indicators
        content_lower = source.content.lower()
        if any(word in content_lower for word in ['current', 'latest', 'new', 'updated', 'recent']):
            recency_score += 0.2
        
        return min(recency_score, 1.0)
    
    def _calculate_fusion_confidence(self, sources: List[Source]) -> float:
        """Calculate confidence score for fused results"""
        
        if not sources:
            return 0.0
        
        # Base confidence on top source scores
        top_scores = [s.score for s in sources[:5]]
        avg_top_score = sum(top_scores) / len(top_scores)
        
        # Boost for number of sources
        source_count_boost = min(len(sources) / 10, 0.2)
        
        # Boost for collection diversity
        unique_collections = len(set(s.collection_name for s in sources))
        diversity_boost = min(unique_collections / 5, 0.1)
        
        final_confidence = avg_top_score + source_count_boost + diversity_boost
        return min(final_confidence, 1.0)
    
    def _calculate_collection_contributions(
        self,
        collection_results: List[CollectionResult]
    ) -> Dict[str, float]:
        """Calculate each collection's contribution to the final result"""
        
        contributions = {}
        total_score = sum(cr.relevance_score for cr in collection_results)
        
        if total_score > 0:
            for cr in collection_results:
                contributions[cr.collection_name] = cr.relevance_score / total_score
        
        return contributions
    
    async def _generate_response_content(
        self,
        sources: List[Source],
        query: str,
        fusion_type: str
    ) -> str:
        """Generate response content from fused sources"""
        
        if not sources:
            return "No relevant information found."
        
        # Group sources by collection for organized response
        collection_groups = defaultdict(list)
        for source in sources:
            collection_groups[source.collection_name].append(source)
        
        # Build response sections
        response_parts = []
        
        # Add main answer from top sources
        top_sources = sources[:3]
        main_content = []
        
        for source in top_sources:
            # Extract key sentences from source content
            sentences = source.content.split('.')
            relevant_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:2]
            
            if relevant_sentences:
                main_content.extend(relevant_sentences)
        
        if main_content:
            response_parts.append("Based on the available information:")
            response_parts.append(" ".join(main_content[:3]) + ".")
            response_parts.append("")
        
        # Add collection-specific insights if multiple collections
        if len(collection_groups) > 1:
            response_parts.append("Additional details from different sources:")
            
            for collection_name, collection_sources in list(collection_groups.items())[:3]:
                best_source = max(collection_sources, key=lambda s: s.score)
                
                # Extract brief insight
                sentences = best_source.content.split('.')
                insight = next((s.strip() for s in sentences if len(s.strip()) > 15), "")
                
                if insight:
                    collection_display = collection_name.replace('_', ' ').title()
                    response_parts.append(f"• From {collection_display}: {insight}.")
        
        # Add source references
        response_parts.append("")
        response_parts.append("Sources:")
        
        for i, source in enumerate(sources[:5], 1):
            collection_display = source.collection_name.replace('_', ' ').title()
            doc_ref = source.document_id
            
            if source.page:
                doc_ref += f" (page {source.page})"
            
            response_parts.append(f"{i}. {collection_display}: {doc_ref}")
        
        return "\n".join(response_parts)
    
    def _create_empty_response(self) -> FusedResponse:
        """Create empty response for no results"""
        
        return FusedResponse(
            content="No relevant information found in the knowledge base.",
            sources=[],
            confidence_score=0.0,
            fusion_method=FusionMethod.RELEVANCE_WEIGHTED,
            collection_contributions={}
        )