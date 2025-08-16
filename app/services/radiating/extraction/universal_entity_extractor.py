"""
Universal Entity Extractor for Radiating Coverage System

Extracts entities from any domain using LLM intelligence.
No hardcoded entity types - discovers entities dynamically based on content.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib

from app.core.radiating_settings_cache import get_extraction_config, get_radiating_settings, get_prompt
from .web_search_integration import WebSearchIntegration

logger = logging.getLogger(__name__)

@dataclass
class ExtractedEntity:
    """Represents an extracted entity"""
    text: str
    entity_type: str
    confidence: float
    context: str
    metadata: Dict[str, Any]
    entity_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.entity_id:
            self.entity_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for entity"""
        hash_input = f"{self.text}_{self.entity_type}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

class UniversalEntityExtractor:
    """
    Extracts entities from text using LLM without predefined types.
    Adapts to any domain and discovers entity types dynamically.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the universal entity extractor.
        
        Args:
            llm_client: Optional LLM client for extraction
        """
        self.config = get_extraction_config()
        self.llm_client = llm_client
        self._init_llm_if_needed()
        
        # Cache for discovered entity types
        self.discovered_types: Dict[str, Dict[str, Any]] = {}
        
        # Initialize web search integration
        self.web_search = WebSearchIntegration()
        
    def _init_llm_if_needed(self):
        """Initialize LLM client if not provided"""
        if not self.llm_client:
            try:
                from app.llm.ollama import JarvisLLM
                settings = get_radiating_settings()
                model_config = settings.get('model_config', {})
                
                # Initialize JarvisLLM with max_tokens from config
                max_tokens = model_config.get('max_tokens', 4096)
                self.llm_client = JarvisLLM(mode='non-thinking', max_tokens=max_tokens)
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                self.llm_client = None
    
    async def extract_entities(
        self,
        text: str,
        domain_hints: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> List[ExtractedEntity]:
        """
        Extract entities from text without predefined types.
        
        Args:
            text: Text to extract entities from
            domain_hints: Optional domain hints to guide extraction
            context: Optional context about the text
            
        Returns:
            List of extracted entities with confidence scores
        """
        
        if not self.llm_client:
            logger.warning("LLM client not available for entity extraction")
            return []
        
        try:
            # Detect if this is a technology/tools/frameworks query
            is_comprehensive_query = self._is_comprehensive_technology_query(text)
            
            # First pass: Discover entity types if needed
            if self.config.get('enable_universal_discovery', True):
                discovered_types = await self._discover_entity_types(
                    text, domain_hints, is_comprehensive=is_comprehensive_query
                )
                self._update_discovered_types(discovered_types)
            
            # Second pass: Extract entities (pass the comprehensive flag)
            entities = await self._extract_entities_with_llm(
                text, domain_hints, context, is_comprehensive=is_comprehensive_query
            )
            
            # Score confidence for each entity
            scored_entities = self._score_entity_confidence(entities, text)
            
            # Adjust confidence threshold based on query type
            # Lower threshold for comprehensive technology queries to capture more entities
            if is_comprehensive_query:
                min_confidence = 0.3  # Much lower threshold for comprehensive queries
            else:
                min_confidence = self.config.get('entity_confidence_threshold', 0.6)
            
            filtered = [e for e in scored_entities if e.confidence >= min_confidence]
            
            # Adjust max entities based on query type
            # Allow many more entities for comprehensive queries
            if is_comprehensive_query:
                max_entities = 100  # Allow up to 100 entities for comprehensive queries
            else:
                max_entities = self.config.get('max_entities_per_query', 20)
            
            return filtered[:max_entities]
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _is_comprehensive_technology_query(self, text: str) -> bool:
        """
        Detect if the query is asking for comprehensive technology/tool listings.
        
        Args:
            text: The query text to analyze
            
        Returns:
            True if this is a comprehensive technology query
        """
        text_lower = text.lower()
        
        # Keywords that indicate comprehensive listing requests
        comprehensive_keywords = [
            'technologies', 'technology', 'tools', 'tool', 'frameworks', 'framework',
            'libraries', 'library', 'platforms', 'platform', 'systems', 'system',
            'solutions', 'solution', 'software', 'packages', 'package',
            'implementations', 'implementation', 'stacks', 'stack',
            'essential', 'list', 'what are', 'which', 'available', 'options',
            'alternatives', 'examples', 'popular', 'common', 'best', 'top',
            'recommended', 'suggest', 'recommendation', 'modern', 'latest', 'current'
        ]
        
        # Check for technology domain indicators - emphasize LLM-era technologies
        tech_domains = [
            'llm', 'large language model', 'gpt', 'chatgpt', 'claude', 'gemini',
            'rag', 'retrieval augmented generation', 'vector database', 'vector db',
            'embeddings', 'embedding', 'langchain', 'llamaindex', 'ollama',
            'inference', 'prompt engineering', 'prompt', 'agent', 'agents',
            'ai', 'artificial intelligence', 'generative ai', 'genai',
            'open source llm', 'local llm', 'fine-tuning', 'quantization',
            'machine learning', 'ml', 'deep learning', 'neural', 'nlp',
            'computer vision', 'cv', 'data science', 'analytics', 'big data',
            'cloud', 'devops', 'backend', 'frontend', 'full stack', 'web',
            'mobile', 'database', 'blockchain', 'iot', 'embedded', 'quantum',
            'robotics', 'automation', 'open source'
        ]
        
        # Count matching keywords
        keyword_matches = sum(1 for kw in comprehensive_keywords if kw in text_lower)
        domain_matches = sum(1 for domain in tech_domains if domain in text_lower)
        
        # Consider it comprehensive if we have strong signals
        is_comprehensive = (
            (keyword_matches >= 2) or  # Multiple comprehensive keywords
            (keyword_matches >= 1 and domain_matches >= 1) or  # Combo of keyword + domain
            ('what are' in text_lower and any(kw in text_lower for kw in ['technologies', 'tools', 'frameworks'])) or
            ('essential' in text_lower and any(kw in text_lower for kw in ['technologies', 'tools'])) or
            ('list' in text_lower and any(kw in text_lower for kw in ['technologies', 'tools', 'frameworks']))
        )
        
        if is_comprehensive:
            logger.info(f"Detected comprehensive technology query: {text[:100]}...")
        
        return is_comprehensive
    
    async def _discover_entity_types(
        self,
        text: str,
        domain_hints: Optional[List[str]] = None,
        is_comprehensive: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Discover entity types present in the text"""
        
        domain_context = f"in the context of {', '.join(domain_hints)}" if domain_hints else ""
        
        if is_comprehensive:
            # Comprehensive type discovery for technology queries
            prompt_template = get_prompt(
                'entity_extraction', 
                'discovery_comprehensive',
                # Fallback prompt if not in database
                "You are analyzing a query about MODERN LLM-ERA technologies...\n{domain_context}\nQuery/Text: {text}"
            )
            prompt = prompt_template.format(
                domain_context=domain_context,
                text=text[:2000]
            )
        else:
            # Regular type discovery
            prompt_template = get_prompt(
                'entity_extraction',
                'discovery_regular',
                # Fallback prompt if not in database
                "Analyze this text and identify the types of entities present {domain_context}.\nText: {text}"
            )
            prompt = prompt_template.format(
                domain_context=domain_context,
                text=text[:2000]
            )
        
        try:
            response = await self.llm_client.invoke(prompt)
            result = json.loads(response)
            
            entity_types = {}
            for type_info in result.get('entity_types', []):
                type_name = type_info['type']
                entity_types[type_name] = {
                    'description': type_info.get('description', ''),
                    'examples': type_info.get('examples', []),
                    'confidence': type_info.get('confidence', 0.7),
                    'domain_specific': True
                }
            
            if is_comprehensive:
                logger.info(f"Discovered {len(entity_types)} entity types for comprehensive extraction")
            
            return entity_types
            
        except Exception as e:
            logger.debug(f"Entity type discovery failed: {e}")
            return {}
    
    async def _extract_entities_with_llm(
        self,
        text: str,
        domain_hints: Optional[List[str]] = None,
        context: Optional[str] = None,
        is_comprehensive: bool = False
    ) -> List[ExtractedEntity]:
        """Extract entities using LLM"""
        
        domain_context = f"Domain context: {', '.join(domain_hints)}" if domain_hints else ""
        additional_context = f"Additional context: {context}" if context else ""
        
        # Include discovered types in prompt if available
        type_guidance = ""
        if self.discovered_types:
            type_examples = []
            for type_name, type_info in list(self.discovered_types.items())[:10]:
                type_examples.append(f"- {type_name}: {type_info['description']}")
            type_guidance = "Consider these entity types:\n" + "\n".join(type_examples)
        
        # Create different prompts for comprehensive vs regular extraction
        if is_comprehensive:
            # Comprehensive extraction prompt - explicitly request many entities
            prompt_template = get_prompt(
                'entity_extraction',
                'extraction_comprehensive',
                # Fallback prompt if not in database
                "You are an expert on MODERN LLM-ERA technologies...\n{domain_context}\n{additional_context}\nQuery/Text: {text}"
            )
            prompt = prompt_template.format(
                domain_context=domain_context,
                additional_context=additional_context,
                text=text
            )
        else:
            # Regular extraction prompt
            prompt_template = get_prompt(
                'entity_extraction',
                'extraction_regular',
                # Fallback prompt if not in database
                "Extract all important entities from this text.\n{domain_context}\n{additional_context}\n{type_guidance}\nText: {text}"
            )
            prompt = prompt_template.format(
                domain_context=domain_context,
                additional_context=additional_context,
                type_guidance=type_guidance,
                text=text
            )
        
        try:
            response = await self.llm_client.invoke(prompt)
            entities_data = json.loads(response)
            
            # Log the number of entities extracted
            if is_comprehensive:
                logger.info(f"Comprehensive extraction yielded {len(entities_data)} entities")
            
            entities = []
            for entity_data in entities_data:
                entity = ExtractedEntity(
                    text=entity_data['text'],
                    entity_type=entity_data['type'],
                    confidence=entity_data.get('confidence', 0.7),
                    context=entity_data.get('context', ''),
                    metadata={
                        'reason': entity_data.get('reason', ''),
                        'extraction_method': 'llm_comprehensive' if is_comprehensive else 'llm_universal',
                        'is_comprehensive': is_comprehensive
                    }
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}")
            return []
    
    def _score_entity_confidence(
        self,
        entities: List[ExtractedEntity],
        original_text: str
    ) -> List[ExtractedEntity]:
        """
        Score confidence for extracted entities based on various factors.
        """
        
        scored_entities = []
        text_lower = original_text.lower()
        
        for entity in entities:
            # Base confidence from extraction
            confidence = entity.confidence
            
            # Boost confidence if entity appears multiple times
            occurrences = text_lower.count(entity.text.lower())
            if occurrences > 1:
                confidence += min(0.1 * (occurrences - 1), 0.2)
            
            # Boost confidence if entity type is well-established
            if entity.entity_type in self.discovered_types:
                type_confidence = self.discovered_types[entity.entity_type].get('confidence', 0.7)
                confidence = (confidence + type_confidence) / 2
            
            # Apply pattern detection if enabled
            if self.config.get('enable_pattern_detection', True):
                pattern_score = self._check_entity_patterns(entity, original_text)
                confidence = (confidence + pattern_score) / 2
            
            # Cap confidence at 1.0
            entity.confidence = min(confidence, 1.0)
            scored_entities.append(entity)
        
        return scored_entities
    
    def _check_entity_patterns(self, entity: ExtractedEntity, text: str) -> float:
        """
        Check for patterns that indicate entity importance.
        Returns a confidence score based on patterns.
        """
        
        score = 0.5  # Base score
        
        # Check capitalization (often indicates proper nouns)
        if entity.text[0].isupper():
            score += 0.1
        
        # Check if in quotes (often indicates special terms)
        if f'"{entity.text}"' in text or f"'{entity.text}'" in text:
            score += 0.15
        
        # Check if followed by definition or explanation
        patterns = [
            f"{entity.text} is ",
            f"{entity.text}, which",
            f"{entity.text} (", 
            f"{entity.text} means",
            f"{entity.text} refers to"
        ]
        
        for pattern in patterns:
            if pattern in text:
                score += 0.2
                break
        
        return min(score, 1.0)
    
    def _update_discovered_types(self, new_types: Dict[str, Dict[str, Any]]):
        """Update the cache of discovered entity types"""
        
        for type_name, type_info in new_types.items():
            if type_name in self.discovered_types:
                # Merge with existing type info
                existing = self.discovered_types[type_name]
                
                # Update confidence with weighted average
                old_confidence = existing.get('confidence', 0.7)
                new_confidence = type_info.get('confidence', 0.7)
                existing['confidence'] = (old_confidence + new_confidence) / 2
                
                # Merge examples
                existing_examples = set(existing.get('examples', []))
                new_examples = set(type_info.get('examples', []))
                existing['examples'] = list(existing_examples.union(new_examples))[:5]
                
            else:
                # Add new type
                self.discovered_types[type_name] = type_info
    
    async def extract_with_context_preservation(
        self,
        text: str,
        previous_entities: List[ExtractedEntity],
        domain_hints: Optional[List[str]] = None
    ) -> List[ExtractedEntity]:
        """
        Extract entities while preserving context from previous extractions.
        Useful for maintaining consistency across multiple text chunks.
        
        Args:
            text: Text to extract from
            previous_entities: Entities from previous extractions
            domain_hints: Optional domain hints
            
        Returns:
            List of extracted entities
        """
        
        if not self.config.get('enable_context_preservation', True):
            return await self.extract_entities(text, domain_hints)
        
        # Build context from previous entities
        context_parts = []
        
        # Get unique entity types from previous entities
        previous_types = set(e.entity_type for e in previous_entities[:10])
        if previous_types:
            context_parts.append(f"Previously found entity types: {', '.join(previous_types)}")
        
        # Get key previous entities
        key_entities = [e.text for e in previous_entities[:5]]
        if key_entities:
            context_parts.append(f"Related to: {', '.join(key_entities)}")
        
        context = " ".join(context_parts) if context_parts else None
        
        # Extract with context
        entities = await self.extract_entities(text, domain_hints, context)
        
        # Boost confidence for entities related to previous ones
        for entity in entities:
            for prev_entity in previous_entities:
                if self._are_related(entity, prev_entity):
                    entity.confidence = min(entity.confidence * 1.1, 1.0)
                    entity.metadata['related_to'] = prev_entity.text
                    break
        
        return entities
    
    def _are_related(self, entity1: ExtractedEntity, entity2: ExtractedEntity) -> bool:
        """Check if two entities are related"""
        
        # Same type indicates potential relationship
        if entity1.entity_type == entity2.entity_type:
            return True
        
        # Check for textual similarity
        text1_lower = entity1.text.lower()
        text2_lower = entity2.text.lower()
        
        if text1_lower in text2_lower or text2_lower in text1_lower:
            return True
        
        # Check for shared words (for compound entities)
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        
        if words1.intersection(words2):
            return True
        
        return False
    
    def get_discovered_types_summary(self) -> Dict[str, Any]:
        """Get a summary of discovered entity types"""
        
        return {
            'total_types': len(self.discovered_types),
            'types': [
                {
                    'name': type_name,
                    'description': info.get('description', ''),
                    'confidence': info.get('confidence', 0.0),
                    'examples': info.get('examples', [])[:3]
                }
                for type_name, info in list(self.discovered_types.items())[:20]
            ]
        }
    
    def reset_discovered_types(self):
        """Reset the cache of discovered entity types"""
        self.discovered_types = {}
        logger.info("Reset discovered entity types cache")
    
    async def extract_entities_with_web_search(
        self,
        text: str,
        domain_hints: Optional[List[str]] = None,
        context: Optional[str] = None,
        force_web_search: bool = False
    ) -> List[ExtractedEntity]:
        """
        Extract entities from text using both LLM and web search for comprehensive coverage.
        This method is designed to discover the latest AI/LLM technologies beyond the LLM's
        knowledge cutoff.
        
        Args:
            text: Text to extract entities from
            domain_hints: Optional domain hints to guide extraction
            context: Optional context about the text
            force_web_search: Force web search even if not automatically triggered
            
        Returns:
            List of extracted entities combining LLM and web search results
        """
        logger.info("Starting entity extraction with web search augmentation")
        
        # First, perform standard LLM extraction
        llm_entities = await self.extract_entities(text, domain_hints, context)
        logger.info(f"LLM extraction found {len(llm_entities)} entities")
        
        # Check if web search should be used
        should_search = force_web_search or self.web_search.should_use_web_search(text)
        
        if not should_search:
            logger.info("Web search not triggered for this query")
            return llm_entities
        
        logger.info("Web search triggered - discovering latest technologies")
        
        try:
            # Extract focus areas from the query
            focus_areas = self._extract_focus_areas(text, domain_hints)
            
            # Perform web search for latest technologies
            web_entities = await self.web_search.search_for_technologies(text, focus_areas)
            logger.info(f"Web search discovered {len(web_entities)} additional entities")
            
            # Merge entities, prioritizing newer information
            merged_entities = await self.web_search.merge_entities(llm_entities, web_entities)
            
            # Apply final filtering and scoring
            final_entities = self._finalize_entities(merged_entities, text)
            
            logger.info(f"Final entity count after merging: {len(final_entities)} "
                       f"(LLM: {len(llm_entities)}, Web: {len(web_entities)})")
            
            return final_entities
            
        except Exception as e:
            logger.error(f"Web search augmentation failed: {e}")
            # Fall back to LLM entities if web search fails
            return llm_entities
    
    def _extract_focus_areas(self, text: str, domain_hints: Optional[List[str]]) -> List[str]:
        """
        Extract focus areas from the query for targeted web searches.
        
        Args:
            text: Query text
            domain_hints: Optional domain hints
            
        Returns:
            List of focus areas for web search
        """
        focus_areas = []
        text_lower = text.lower()
        
        # Technology categories to look for
        tech_categories = {
            'llm': ['LLM', 'large language model', 'language model'],
            'rag': ['RAG', 'retrieval augmented generation', 'retrieval'],
            'vector_db': ['vector database', 'vector db', 'embeddings database'],
            'agent': ['agent', 'multi-agent', 'autonomous agent'],
            'inference': ['inference', 'serving', 'deployment'],
            'prompt': ['prompt', 'prompt engineering', 'prompting'],
            'embedding': ['embedding', 'embeddings', 'text embedding'],
            'framework': ['framework', 'library', 'toolkit'],
            'open_source': ['open source', 'opensource', 'oss']
        }
        
        for category, keywords in tech_categories.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                focus_areas.append(category)
        
        # Add domain hints if provided
        if domain_hints:
            focus_areas.extend(domain_hints[:2])  # Limit to avoid too many searches
        
        # Default focus areas if none detected
        if not focus_areas:
            focus_areas = ['AI tools', 'LLM frameworks']
        
        return focus_areas
    
    def _finalize_entities(self, entities: List[ExtractedEntity], original_text: str) -> List[ExtractedEntity]:
        """
        Apply final filtering, scoring, and deduplication to merged entities.
        
        Args:
            entities: Merged list of entities
            original_text: Original query text
            
        Returns:
            Finalized list of entities
        """
        # Remove duplicates based on text similarity
        unique_entities = []
        seen_texts = set()
        seen_similar = set()
        
        for entity in entities:
            entity_lower = entity.text.lower()
            
            # Skip if exact duplicate
            if entity_lower in seen_texts:
                continue
            
            # Check for similar entities (e.g., "LangChain" vs "Langchain")
            is_similar = False
            for seen in seen_similar:
                if self._are_similar_texts(entity_lower, seen):
                    is_similar = True
                    break
            
            if not is_similar:
                unique_entities.append(entity)
                seen_texts.add(entity_lower)
                seen_similar.add(entity_lower)
        
        # Re-score entities based on relevance to original query
        for entity in unique_entities:
            # Boost entities that appear in the original text
            if entity.text.lower() in original_text.lower():
                entity.confidence = min(1.0, entity.confidence + 0.2)
            
            # Boost entities from web search for technology queries
            if hasattr(entity, 'metadata') and entity.metadata.get('source') == 'web_search':
                # Web-sourced entities get a slight boost for freshness
                entity.confidence = min(1.0, entity.confidence + 0.05)
        
        # Sort by confidence
        unique_entities.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply limits based on query type
        is_comprehensive = self._is_comprehensive_technology_query(original_text)
        
        if is_comprehensive:
            # Allow more entities for comprehensive queries
            max_entities = min(100, len(unique_entities))
        else:
            # Standard limit for regular queries
            max_entities = self.config.get('max_entities_per_query', 30)
        
        return unique_entities[:max_entities]
    
    def _are_similar_texts(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are similar enough to be considered duplicates.
        
        Args:
            text1: First text (lowercase)
            text2: Second text (lowercase)
            
        Returns:
            True if texts are similar
        """
        # Remove common separators and spaces
        clean1 = text1.replace('-', '').replace('_', '').replace(' ', '').replace('.', '')
        clean2 = text2.replace('-', '').replace('_', '').replace(' ', '').replace('.', '')
        
        # Check exact match after cleaning
        if clean1 == clean2:
            return True
        
        # Check if one is contained in the other
        if clean1 in clean2 or clean2 in clean1:
            return True
        
        # Check Levenshtein distance for very similar strings
        if len(clean1) > 3 and len(clean2) > 3:
            # Simple character difference check
            if abs(len(clean1) - len(clean2)) <= 2:
                differences = sum(1 for a, b in zip(clean1, clean2) if a != b)
                if differences <= 2:
                    return True
        
        return False