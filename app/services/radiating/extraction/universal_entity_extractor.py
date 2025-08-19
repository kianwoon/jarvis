"""
Universal Entity Extractor for Radiating Coverage System

Extracts entities from any domain using LLM intelligence.
No hardcoded entity types - discovers entities dynamically based on content.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import asyncio
from datetime import datetime

from app.core.radiating_settings_cache import get_extraction_config, get_radiating_settings, get_prompt, get_model_config
from app.core.timeout_settings_cache import get_entity_extraction_timeout
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
                
                # Get model configuration from settings
                model_config = get_model_config()
                
                # Initialize JarvisLLM with configuration from settings
                # Override max_tokens to 4096 for entity extraction to prevent timeouts
                # Entity extraction doesn't need huge token limits
                self.llm_client = JarvisLLM(
                    model=model_config.get('model', 'llama3.1:8b'),
                    mode=model_config.get('llm_mode', 'non-thinking'),  # Use mode from config
                    max_tokens=4096,  # Fixed at 4096 for entity extraction regardless of config
                    temperature=model_config.get('temperature', 0.7),
                    model_server=model_config['model_server']  # Required, no fallback
                )
                
                # Set a JSON-only system prompt for radiating
                self.llm_client.system_prompt = "You are a JSON-only API that extracts information from text. You must ALWAYS respond with valid JSON and nothing else. Never include explanations, markdown, or any text outside the JSON structure."
                    
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                self.llm_client = None
    
    async def extract_entities(
        self,
        text: str,
        domain_hints: Optional[List[str]] = None,
        context: Optional[str] = None,
        prefer_web_search: bool = True
    ) -> List[ExtractedEntity]:
        """
        Extract entities from text without predefined types.
        
        Args:
            text: Text to extract entities from
            domain_hints: Optional domain hints to guide extraction
            context: Optional context about the text
            prefer_web_search: When True, use web search as primary source (default: True)
            
        Returns:
            List of extracted entities with confidence scores
        """
        
        # If prefer_web_search is True and it's a technology query, use web-first approach
        if prefer_web_search and self._is_comprehensive_technology_query(text):
            logger.info("Using web-first extraction for technology query")
            return await self.extract_entities_web_first(text, domain_hints, context)
        
        # Otherwise, use the traditional LLM-first approach
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
                min_confidence = self.config.get('entity_confidence_threshold', 0.4)
            
            filtered = [e for e in scored_entities if e.confidence >= min_confidence]
            
            # Adjust max entities based on query type
            # Allow many more entities for comprehensive queries
            if is_comprehensive_query:
                max_entities = 100  # Allow up to 100 entities for comprehensive queries
            else:
                max_entities = self.config.get('max_entities_per_query', 50)
            
            return filtered[:max_entities]
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def extract_entities_web_first(
        self,
        text: str,
        domain_hints: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> List[ExtractedEntity]:
        """
        Extract entities with web search as the primary source.
        Only falls back to LLM if web search fails or returns insufficient results.
        
        Args:
            text: Text to extract entities from
            domain_hints: Optional domain hints to guide extraction
            context: Optional context about the text
            
        Returns:
            List of extracted entities with web sources prioritized
        """
        logger.info("Starting web-first entity extraction")
        
        # Always try web search first for technology queries
        web_entities = []
        llm_entities = []
        
        try:
            # Extract focus areas for targeted searches
            focus_areas = self._extract_focus_areas(text, domain_hints)
            logger.info(f"Focus areas for web search: {focus_areas}")
            
            # Generate multiple query variations for comprehensive coverage
            query_variations = self.web_search.generate_query_variations(text, max_variations=8)
            logger.info(f"Generated {len(query_variations)} query variations for comprehensive search")
            
            # Perform web searches with variations
            all_web_results = []
            
            # Search with the original query first
            original_results = await self.web_search.search_for_technologies(text, focus_areas)
            all_web_results.extend(original_results)
            
            # Then search with variations for more coverage
            for variation in query_variations[:5]:  # Limit to top 5 variations
                variation_results = await self.web_search._execute_web_search(variation)
                extracted = await self.web_search._extract_entities_from_results(variation_results)
                all_web_results.extend(extracted)
            
            # Convert web results to ExtractedEntity objects
            for web_entity in all_web_results:
                entity = ExtractedEntity(
                    text=web_entity['text'],
                    entity_type=web_entity['type'],
                    confidence=web_entity.get('confidence', 0.7) + 0.1,  # Boost web confidence
                    context=web_entity.get('context', ''),
                    metadata={
                        'source': 'web_search',
                        'url': web_entity.get('url', ''),
                        'extraction_method': 'web_first',
                        'search_query': text
                    }
                )
                web_entities.append(entity)
            
            logger.info(f"Web search found {len(web_entities)} entities")
            
            # Store entities in Neo4j with web-sourced label
            if web_entities:
                search_metadata = {
                    'query': text,
                    'focus_areas': focus_areas,
                    'variations_used': len(query_variations),
                    'timestamp': datetime.now().isoformat()
                }
                await self.web_search.store_entities_in_neo4j(
                    [{'text': e.text, 'type': e.entity_type, 'confidence': e.confidence, 
                      'url': e.metadata.get('url', ''), 'context': e.context} 
                     for e in web_entities],
                    search_metadata
                )
            
            # Extract relationships from search snippets
            if all_web_results:
                relationships = self.web_search.extract_relationships_from_snippets(
                    [{'snippet': e.get('context', ''), 'url': e.get('url', '')} 
                     for e in all_web_results]
                )
                logger.info(f"Extracted {len(relationships)} relationships from web search")
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            web_entities = []
        
        # Only use LLM as fallback if web search didn't find enough entities
        if len(web_entities) < 5:
            logger.info(f"Web search found only {len(web_entities)} entities, falling back to LLM")
            
            if self.llm_client:
                try:
                    # Use standard LLM extraction
                    llm_entities = await self._extract_entities_with_llm(
                        text, domain_hints, context, 
                        is_comprehensive=self._is_comprehensive_technology_query(text)
                    )
                    llm_entities = self._score_entity_confidence(llm_entities, text)
                    logger.info(f"LLM extraction found {len(llm_entities)} additional entities")
                except Exception as e:
                    logger.error(f"LLM extraction failed: {e}")
                    llm_entities = []
        
        # Combine results with web entities having higher priority
        combined_entities = []
        seen_texts = set()
        
        # Add all web entities first (higher priority)
        for entity in web_entities:
            entity_lower = entity.text.lower()
            if entity_lower not in seen_texts:
                combined_entities.append(entity)
                seen_texts.add(entity_lower)
        
        # Add LLM entities that aren't duplicates
        for entity in llm_entities:
            entity_lower = entity.text.lower()
            if entity_lower not in seen_texts:
                # Slightly reduce confidence for LLM entities when web search was primary
                entity.confidence = entity.confidence * 0.9
                entity.metadata['extraction_method'] = 'llm_fallback'
                combined_entities.append(entity)
                seen_texts.add(entity_lower)
        
        # Sort by confidence
        combined_entities.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply appropriate limits
        is_comprehensive = self._is_comprehensive_technology_query(text)
        # Higher limits for web-first extraction
        if is_comprehensive:
            max_entities = 150
        else:
            max_entities = self.config.get('max_entities_per_query', 50)
            # If we have many web entities, increase limit
            web_count = sum(1 for e in combined_entities if hasattr(e, 'metadata') and e.metadata.get('source') == 'web_search')
            if web_count > 20:
                max_entities = min(max_entities * 2, 100)
                logger.info(f"Web-first: Found {web_count} web entities, increased limit to {max_entities}")
        
        final_entities = combined_entities[:max_entities]
        
        logger.info(f"Web-first extraction complete: {len(final_entities)} total entities "
                   f"(Web: {len(web_entities)}, LLM: {len(llm_entities)})")
        
        return final_entities
    
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
                """TEXT: {text}

List entity types found in text.

JSON format:
{{
  "entity_types": [
    {{"type": "Technology", "description": "tools/frameworks", "examples": ["Python"], "confidence": 0.9}}
  ]
}}"""
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
                """TEXT: {text}

List entity types found.

JSON format:
{{
  "entity_types": [
    {{"type": "Person", "description": "people", "examples": ["John"], "confidence": 0.8}}
  ]
}}"""
            )
            prompt = prompt_template.format(
                domain_context=domain_context,
                text=text[:2000]
            )
        
        try:
            # Add JSON instruction to prompt
            json_prompt = prompt + "\n\nREMEMBER: Respond with ONLY the JSON, no other text."
            response = await self.llm_client.invoke(json_prompt)
            
            # Handle empty or invalid response
            if not response or response.strip() == "" or response.strip() == "[]":
                logger.warning("[UniversalEntityExtractor] Empty response from LLM for type discovery")
                return {}
            
            # Try multiple extraction methods
            result = self._extract_json_from_response(response, 'dict')
            
            if result is None:
                # Try regex extraction as fallback
                result = self._extract_json_with_regex(response, 'dict')
            
            if result is None:
                logger.error(f"[UniversalEntityExtractor] Could not extract JSON from response: {response[:200]}")
                return {}
            
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
                """TEXT: {text}

Extract all entities (technologies, languages, frameworks, products, concepts).

JSON format:
[
  {{"text": "Python", "type": "Language", "confidence": 0.9, "context": "main language", "reason": "core tech"}}
]"""
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
                """TEXT: {text}

Extract entities.

JSON format:
[
  {{"text": "Entity", "type": "Type", "confidence": 0.8, "context": "context", "reason": "why"}}
]"""
            )
            prompt = prompt_template.format(
                domain_context=domain_context,
                additional_context=additional_context,
                type_guidance=type_guidance,
                text=text
            )
        
        try:
            # Add JSON instruction to prompt
            json_prompt = prompt + "\n\nREMEMBER: Respond with ONLY the JSON, no other text."
            
            # Add timeout protection
            timeout = get_entity_extraction_timeout()
            response = await asyncio.wait_for(
                self.llm_client.invoke(json_prompt),
                timeout=timeout
            )
            
            # Handle empty or invalid response
            if not response or response.strip() == "" or response.strip() == "[]":
                logger.warning("[UniversalEntityExtractor] Empty response from LLM for entity extraction")
                return []
            
            # Try multiple extraction methods
            entities_data = self._extract_json_from_response(response, 'list')
            
            if entities_data is None:
                # Try regex extraction as fallback
                entities_data = self._extract_json_with_regex(response, 'list')
            
            if entities_data is None:
                logger.error(f"[UniversalEntityExtractor] Could not extract JSON from response: {response[:200]}")
                return []
            
            # Handle different response formats
            if isinstance(entities_data, dict):
                # If it's a dict with 'entities' key, use that
                if 'entities' in entities_data:
                    entities_data = entities_data['entities']
                else:
                    logger.error(f"[UniversalEntityExtractor] Dict response without 'entities' key: {entities_data}")
                    return []
            
            # Ensure entities_data is a list
            if not isinstance(entities_data, list):
                logger.error(f"[UniversalEntityExtractor] Expected list but got {type(entities_data)}: {entities_data}")
                return []
            
            # Log the number of entities extracted
            if is_comprehensive:
                logger.info(f"Comprehensive extraction yielded {len(entities_data)} entities")
            
            entities = []
            for entity_data in entities_data:
                # Ensure entity_data is a dict
                if not isinstance(entity_data, dict):
                    logger.warning(f"[UniversalEntityExtractor] Skipping non-dict entity: {entity_data}")
                    continue
                    
                # Handle different field names with comprehensive mapping
                # LLM might return: text/name/entity for the entity text
                entity_text = (entity_data.get('text') or 
                              entity_data.get('name') or 
                              entity_data.get('entity'))
                
                # LLM might return: type/category/entity_type for the type
                entity_type = (entity_data.get('type') or 
                              entity_data.get('category') or 
                              entity_data.get('entity_type'))
                
                # Check required fields
                if not entity_text or not entity_type:
                    logger.warning(f"[UniversalEntityExtractor] Missing required fields in entity: {entity_data}")
                    logger.debug(f"[UniversalEntityExtractor] Entity data: {entity_data}")
                    continue
                
                # Get confidence with multiple fallbacks
                confidence = (entity_data.get('confidence') or 
                             entity_data.get('score') or 
                             entity_data.get('relevance') or 
                             0.7)
                
                # Ensure confidence is a float between 0 and 1
                try:
                    confidence = float(confidence)
                    confidence = max(0.0, min(1.0, confidence))
                except (TypeError, ValueError):
                    confidence = 0.7
                    
                entity = ExtractedEntity(
                    text=entity_text,
                    entity_type=entity_type,
                    confidence=confidence,
                    context=entity_data.get('context', ''),
                    metadata={
                        'reason': entity_data.get('reason', ''),
                        'extraction_method': 'llm_comprehensive' if is_comprehensive else 'llm_universal',
                        'is_comprehensive': is_comprehensive,
                        'original_fields': list(entity_data.keys())  # Track what fields the LLM actually returned
                    }
                )
                entities.append(entity)
                
            # Log extraction success
            if entities:
                logger.info(f"[UniversalEntityExtractor] Successfully extracted {len(entities)} entities")
                if is_comprehensive:
                    logger.debug(f"[UniversalEntityExtractor] Comprehensive extraction - first entity: {entities[0].__dict__ if entities else 'None'}")
            
            return entities
            
        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}", exc_info=True)
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
        knowledge cutoff. Enhanced to use advanced WebSearchIntegration features.
        
        Args:
            text: Text to extract entities from
            domain_hints: Optional domain hints to guide extraction
            context: Optional context about the text
            force_web_search: Force web search even if not automatically triggered
            
        Returns:
            List of extracted entities combining LLM and web search results
        """
        logger.info("Starting enhanced entity extraction with web search augmentation")
        
        # Check if web search should be used (now prioritized)
        should_search = force_web_search or self.web_search.should_use_web_search(text)
        
        if should_search:
            logger.info("Web search triggered - using enhanced web-first approach")
            
            try:
                # Extract focus areas from the query
                focus_areas = self._extract_focus_areas(text, domain_hints)
                logger.info(f"Focus areas identified: {focus_areas}")
                
                # Generate multiple query variations for comprehensive coverage
                query_variations = self.web_search.generate_query_variations(text, max_variations=10)
                logger.info(f"Generated {len(query_variations)} query variations for comprehensive search")
                
                # Perform web searches with all variations
                all_web_results = []
                all_web_entities = []
                
                # Search with the original query first
                original_results = await self.web_search.search_for_technologies(text, focus_areas)
                all_web_entities.extend(original_results)
                
                # Then search with variations for maximum coverage
                for i, variation in enumerate(query_variations[:7], 1):  # Use top 7 variations
                    logger.info(f"Searching variation {i}/{min(7, len(query_variations))}: {variation[:50]}...")
                    variation_results = await self.web_search._execute_web_search(variation)
                    extracted = await self.web_search._extract_entities_from_results(variation_results)
                    all_web_entities.extend(extracted)
                    all_web_results.extend(variation_results)
                
                logger.info(f"Web search discovered {len(all_web_entities)} total entities")
                
                # Convert web entities to ExtractedEntity objects
                web_extracted_entities = []
                for web_entity in all_web_entities:
                    entity = ExtractedEntity(
                        text=web_entity['text'],
                        entity_type=web_entity['type'],
                        confidence=web_entity.get('confidence', 0.7) + 0.15,  # Higher boost for web entities
                        context=web_entity.get('context', ''),
                        metadata={
                            'source': 'web_search',
                            'url': web_entity.get('url', ''),
                            'extraction_method': 'web_search_enhanced',
                            'search_query': text,
                            'variation_source': True
                        }
                    )
                    web_extracted_entities.append(entity)
                
                # Store entities in Neo4j with enhanced metadata
                if web_extracted_entities:
                    search_metadata = {
                        'query': text,
                        'focus_areas': focus_areas,
                        'variations_used': len(query_variations),
                        'total_results': len(all_web_results),
                        'timestamp': datetime.now().isoformat(),
                        'enhanced_search': True
                    }
                    await self.web_search.store_entities_in_neo4j(
                        [{'text': e.text, 'type': e.entity_type, 'confidence': e.confidence, 
                          'url': e.metadata.get('url', ''), 'context': e.context} 
                         for e in web_extracted_entities],
                        search_metadata
                    )
                    logger.info(f"Stored {len(web_extracted_entities)} entities in Neo4j")
                
                # Extract relationships from all search snippets
                if all_web_results:
                    relationships = self.web_search.extract_relationships_from_snippets(all_web_results)
                    logger.info(f"Extracted {len(relationships)} relationships from web search results")
                    
                    # Store relationships in Neo4j if available
                    if relationships:
                        await self._store_relationships_in_neo4j(relationships)
                
                # Only perform LLM extraction if web search found < 10 entities
                llm_entities = []
                if len(web_extracted_entities) < 10:
                    logger.info(f"Web search found only {len(web_extracted_entities)} entities, adding LLM extraction")
                    llm_entities = await self.extract_entities(text, domain_hints, context, prefer_web_search=False)
                    logger.info(f"LLM extraction found {len(llm_entities)} additional entities")
                
                # Merge entities with web entities having priority
                merged_entities = await self.web_search.merge_entities(llm_entities, 
                    [{'text': e.text, 'type': e.entity_type, 'confidence': e.confidence, 
                      'context': e.context, 'url': e.metadata.get('url', '')} 
                     for e in web_extracted_entities])
                
                # Apply final filtering and scoring
                final_entities = self._finalize_entities(merged_entities, text)
                
                logger.info(f"Enhanced extraction complete: {len(final_entities)} total entities "
                           f"(Web: {len(web_extracted_entities)}, LLM: {len(llm_entities)})")
                
                return final_entities
                
            except Exception as e:
                logger.error(f"Enhanced web search failed: {e}", exc_info=True)
                # Fall back to LLM-only extraction
                logger.info("Falling back to LLM-only extraction")
                return await self.extract_entities(text, domain_hints, context, prefer_web_search=False)
        
        else:
            # No web search needed, use standard LLM extraction
            logger.info("Web search not triggered, using LLM extraction only")
            return await self.extract_entities(text, domain_hints, context, prefer_web_search=False)
    
    async def _store_relationships_in_neo4j(self, relationships: List[Dict]):
        """
        Store extracted relationships in Neo4j.
        
        Args:
            relationships: List of relationship dictionaries to store
        """
        try:
            from app.services.radiating.storage.radiating_neo4j_service import RadiatingNeo4jService
            
            neo4j_service = RadiatingNeo4jService()
            if not neo4j_service.is_enabled():
                logger.warning("Neo4j is not enabled, skipping relationship storage")
                return
            
            timestamp = datetime.now().isoformat()
            
            with neo4j_service.driver.session() as session:
                for rel in relationships:
                    # Create or merge the relationship in Neo4j
                    session.run("""
                        MERGE (source:Entity {name: $source})
                        MERGE (target:Entity {name: $target})
                        MERGE (source)-[r:RELATES_TO {type: $rel_type}]->(target)
                        SET r.confidence = $confidence,
                            r.context = $context,
                            r.source_url = $source_url,
                            r.extraction_method = $extraction_method,
                            r.discovery_timestamp = $timestamp,
                            r.last_updated = timestamp()
                        RETURN r
                    """, {
                        'source': rel['source'],
                        'target': rel['target'],
                        'rel_type': rel['type'],
                        'confidence': rel.get('confidence', 0.5),
                        'context': rel.get('context', ''),
                        'source_url': rel.get('source_url', ''),
                        'extraction_method': rel.get('extraction_method', 'web_search'),
                        'timestamp': timestamp
                    })
            
            logger.info(f"Stored {len(relationships)} relationships in Neo4j")
            
        except Exception as e:
            logger.error(f"Failed to store relationships in Neo4j: {e}")
    
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
        logger.info(f"Finalizing {len(entities)} entities before deduplication")
        
        # Blacklist of error-related entity names that shouldn't be processed
        error_entity_blacklist = {
            'api', 'error', 'google search', 'google', 'search',
            'api error', 'search error', 'failed', 'failure',
            'invalid', 'unable', 'could not', 'bad request',
            'not found', 'unauthorized', 'forbidden', 'rate limit',
            'quota', 'authentication', 'service', 'unavailable',
            'exception', 'timeout', 'connection', 'refused'
        }
        
        # Remove duplicates and filter error entities
        unique_entities = []
        seen_texts = set()
        seen_similar = set()
        filtered_count = 0
        
        for entity in entities:
            entity_lower = entity.text.lower()
            
            # Filter out error-related entities
            if entity_lower in error_entity_blacklist:
                # Only filter if confidence is low and it's likely an error entity
                if entity.confidence < 0.7 or entity.entity_type in ['Entity', 'Unknown']:
                    logger.debug(f"Filtering out error-related entity: '{entity.text}' (type: {entity.entity_type}, confidence: {entity.confidence})")
                    filtered_count += 1
                    continue
            
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
        
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} error-related entities")
        logger.info(f"After deduplication: {len(unique_entities)} unique entities")
        
        # Count web-sourced entities
        web_entity_count = sum(1 for e in unique_entities 
                               if hasattr(e, 'metadata') and e.metadata.get('source') == 'web_search')
        logger.info(f"Web-sourced entities: {web_entity_count}")
        
        # Re-score entities based on relevance to original query
        for entity in unique_entities:
            # Boost entities that appear in the original text
            if entity.text.lower() in original_text.lower():
                entity.confidence = min(1.0, entity.confidence + 0.2)
            
            # Boost entities from web search for technology queries (higher boost)
            if hasattr(entity, 'metadata') and entity.metadata.get('source') == 'web_search':
                # Web-sourced entities get a significant boost for freshness and relevance
                entity.confidence = min(1.0, entity.confidence + 0.15)
        
        # Sort by confidence (web entities should be at the top due to boost)
        unique_entities.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply limits based on query type
        is_comprehensive = self._is_comprehensive_technology_query(original_text)
        
        if is_comprehensive:
            # Allow many more entities for comprehensive queries
            max_entities = min(150, len(unique_entities))
            logger.info(f"Comprehensive query detected - allowing up to {max_entities} entities")
        else:
            # Increased standard limit for regular queries
            max_entities = self.config.get('max_entities_per_query', 50)
            # If we have many web entities, increase the limit further
            if web_entity_count > 20:
                max_entities = min(max_entities * 2, 100)
                logger.info(f"Many web entities found - increased limit to {max_entities}")
        
        final_entities = unique_entities[:max_entities]
        logger.info(f"Returning {len(final_entities)} entities (limit was {max_entities})")
        
        return final_entities
    
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
    
    def _extract_json_from_response(self, response: str, expected_type: str = 'any') -> Optional[Any]:
        """Extract JSON from response with robust handling"""
        if not response:
            return None
        
        response = response.strip()
        
        # Try direct parsing first
        try:
            result = json.loads(response)
            if self._validate_json_type(result, expected_type):
                return result
        except json.JSONDecodeError:
            pass
        
        # Remove markdown code blocks
        cleaned = self._clean_markdown(response)
        if cleaned != response:
            try:
                result = json.loads(cleaned)
                if self._validate_json_type(result, expected_type):
                    return result
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON in the text
        json_candidates = self._find_json_candidates(response)
        for candidate in json_candidates:
            try:
                result = json.loads(candidate)
                if self._validate_json_type(result, expected_type):
                    return result
            except json.JSONDecodeError:
                continue
        
        # Handle wrapped responses
        try:
            result = json.loads(response)
            if isinstance(result, dict):
                # Look for common wrapper keys
                for key in ['results', 'items', 'data', 'entities', 'entity_types', 'concepts']:
                    if key in result:
                        if self._validate_json_type(result[key], expected_type):
                            return result[key]
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _clean_markdown(self, text: str) -> str:
        """Remove markdown code blocks from text"""
        # Remove ```json blocks
        if '```json' in text:
            pattern = r'```json\s*([\s\S]*?)```'
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Remove generic ``` blocks
        if '```' in text:
            pattern = r'```\s*([\s\S]*?)```'
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return text
    
    def _find_json_candidates(self, text: str) -> List[str]:
        """Find potential JSON structures in text"""
        candidates = []
        
        # Find array structures
        array_pattern = r'\[[\s\S]*?\]'
        arrays = re.findall(array_pattern, text)
        candidates.extend(arrays)
        
        # Find object structures
        object_pattern = r'\{[\s\S]*?\}'
        objects = re.findall(object_pattern, text)
        candidates.extend(objects)
        
        # Find nested structures (more complex)
        # This handles cases where we have nested arrays/objects
        stack_pattern = r'[\[\{][\s\S]*?[\]\}]'
        nested = re.findall(stack_pattern, text)
        for item in nested:
            if item not in candidates:
                candidates.append(item)
        
        return candidates
    
    def _validate_json_type(self, data: Any, expected_type: str) -> bool:
        """Validate if data matches expected type"""
        if expected_type == 'any':
            return True
        elif expected_type == 'list':
            return isinstance(data, list)
        elif expected_type == 'dict':
            return isinstance(data, dict)
        return False
    
    def _extract_json_with_regex(self, response: str, json_type: str = 'any') -> Optional[Any]:
        """Extract JSON using regex as final fallback"""
        
        # Clean the response first
        response = self._clean_markdown(response)
        
        if json_type == 'list' or json_type == 'any':
            # Look for array pattern with better handling
            patterns = [
                r'\[[^\[\]]*\]',  # Simple array
                r'\[[\s\S]*?\](?![\]\}])',  # Array with possible nesting
                r'\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*\]'  # Array of objects
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response)
                for match in matches:
                    try:
                        result = json.loads(match)
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        continue
        
        if json_type == 'dict' or json_type == 'any':
            # Look for object pattern with better handling
            patterns = [
                r'\{[^\{\}]*\}',  # Simple object
                r'\{[\s\S]*?\}(?![\]\}])',  # Object with possible nesting
                r'\{\s*"[^"]+"\s*:\s*[\s\S]*?\}'  # Object with quoted keys
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response)
                for match in matches:
                    try:
                        result = json.loads(match)
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError:
                        continue
        
        return None