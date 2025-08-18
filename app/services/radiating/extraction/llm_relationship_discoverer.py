"""
LLM-Powered Relationship Discoverer for Radiating System

This module discovers relationships between entities using LLM knowledge
when Neo4j returns empty results, focusing on technology domain relationships.
"""

import asyncio
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import aiohttp

from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.models.radiating_relationship import RadiatingRelationship
from app.core.llm_settings_cache import get_llm_settings
from app.core.config import get_settings
from app.core.radiating_settings_cache import get_prompt, get_model_config
from app.services.radiating.circuit_breaker import protected_llm_call

logger = logging.getLogger(__name__)


class LLMRelationshipDiscoverer:
    """
    Discovers relationships between entities using LLM's knowledge base,
    particularly for technology, AI, and business domains.
    """
    
    # Technology-focused relationship types (specific, not generic)
    TECHNOLOGY_RELATIONSHIPS = {
        # Integration & Dependencies
        'INTEGRATES_WITH': 'Can be integrated or connected with',
        'DEPENDS_ON': 'Requires or is built on top of',
        'USES': 'Utilizes or employs in operations',
        'REQUIRES': 'Needs as a prerequisite',
        'BUILT_ON': 'Is constructed using or based on',
        'WRAPS': 'Provides wrapper or interface for',
        'EMBEDS': 'Embeds or includes internally',
        
        # Competition & Alternatives
        'COMPETES_WITH': 'Directly competes or is an alternative to',
        'REPLACES': 'Can replace or supersede',
        'ALTERNATIVE_TO': 'Serves as an alternative option',
        'MIGRATES_FROM': 'Common migration source',
        'MIGRATES_TO': 'Common migration target',
        
        # Extension & Enhancement
        'EXTENDS': 'Extends functionality or capabilities of',
        'ENHANCES': 'Improves or augments',
        'COMPLEMENTS': 'Works well together with',
        'SUPPORTS': 'Provides support or enables',
        'OPTIMIZES': 'Optimizes performance or efficiency of',
        'ACCELERATES': 'Speeds up or accelerates',
        
        # Implementation & Development
        'IMPLEMENTS': 'Implements or realizes',
        'PROVIDES': 'Offers or supplies functionality',
        'ENABLES': 'Makes possible or facilitates',
        'POWERS': 'Provides the underlying power or engine for',
        'ORCHESTRATES': 'Orchestrates or manages',
        'MANAGES': 'Manages or controls',
        'MONITORS': 'Monitors or observes',
        
        # Data & Communication
        'CONNECTS_TO': 'Establishes connection or communication with',
        'QUERIES': 'Queries or retrieves data from',
        'STORES_IN': 'Stores data or information in',
        'PROCESSES_WITH': 'Processes data using',
        'STREAMS_TO': 'Streams data or events to',
        'CACHES_IN': 'Caches data in',
        'INDEXES_WITH': 'Uses for indexing or search',
        'TRANSFORMS_WITH': 'Transforms data using',
        
        # Organizational & Business
        'DEVELOPED_BY': 'Was created or developed by',
        'OWNED_BY': 'Is owned or maintained by',
        'USED_BY': 'Is utilized by',
        'DEPLOYED_ON': 'Is deployed or runs on',
        'ACQUIRED_BY': 'Was acquired by',
        'PARTNERS_WITH': 'Has partnership with',
        'FUNDED_BY': 'Received funding from',
        
        # Compatibility & Standards
        'COMPATIBLE_WITH': 'Is compatible or works with',
        'FOLLOWS': 'Follows standard or protocol',
        'BASED_ON': 'Is based on or derived from',
        'INSPIRED_BY': 'Takes inspiration or concepts from',
        'CONFORMS_TO': 'Conforms to standard or specification',
        'EXPOSES': 'Exposes API or interface',
        
        # Version & Evolution
        'SUCCESSOR_OF': 'Is the successor or next version of',
        'PREDECESSOR_OF': 'Is the predecessor or previous version of',
        'FORKS_FROM': 'Is a fork of',
        'MERGES_WITH': 'Merges or combines with'
    }
    
    def __init__(self):
        """Initialize the LLM Relationship Discoverer"""
        self.settings = get_settings()
        self.llm_settings = get_llm_settings()
        # Use radiating model config instead of knowledge graph config
        self.model_config = get_model_config()
        self.discovered_cache = {}  # Cache to avoid redundant LLM calls
    
    async def discover_relationships(self, 
                                    entities: List[RadiatingEntity],
                                    max_relationships_per_pair: int = 3,
                                    confidence_threshold: float = 0.5) -> List[RadiatingRelationship]:
        """
        Discover relationships between entities using LLM knowledge.
        
        Args:
            entities: List of RadiatingEntity objects to analyze
            max_relationships_per_pair: Maximum relationships between any entity pair
            confidence_threshold: Minimum confidence for discovered relationships
            
        Returns:
            List of discovered RadiatingRelationship objects
        """
        if not entities or len(entities) < 2:
            logger.warning("Need at least 2 entities to discover relationships")
            return []
        
        # Extract entity information for LLM processing
        entity_info = self._prepare_entity_info(entities)
        
        # Check cache first
        cache_key = self._generate_cache_key(entity_info)
        if cache_key in self.discovered_cache:
            logger.info(f"Using cached relationships for {len(entities)} entities")
            return self.discovered_cache[cache_key]
        
        logger.info(f"Discovering relationships for {len(entities)} entities using LLM")
        
        try:
            # Call LLM to discover relationships
            llm_response = await self._call_llm_for_discovery(entity_info)
            
            # Parse LLM response into relationships
            relationships = self._parse_llm_relationships(
                llm_response, 
                entities,
                max_relationships_per_pair,
                confidence_threshold
            )
            
            # Cache the results
            self.discovered_cache[cache_key] = relationships
            
            logger.info(f"Discovered {len(relationships)} relationships from LLM")
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to discover relationships via LLM: {e}")
            return []
    
    def _prepare_entity_info(self, entities: List[RadiatingEntity]) -> List[Dict[str, Any]]:
        """Prepare entity information for LLM processing"""
        entity_info = []
        for entity in entities:
            info = {
                'name': entity.canonical_form,
                'type': entity.label,
                'id': entity.get_entity_id()
            }
            
            # Add domain metadata if available
            if entity.domain_metadata:
                info['domain'] = entity.domain_metadata.get('domain', 'technology')
                info['description'] = entity.domain_metadata.get('description', '')
            
            # Add any properties that might help LLM understand the entity
            if entity.properties:
                info['properties'] = entity.properties
                
            entity_info.append(info)
            
        return entity_info
    
    def _generate_cache_key(self, entity_info: List[Dict[str, Any]]) -> str:
        """Generate cache key for entity set"""
        # Sort entities by name for consistent caching
        sorted_names = sorted([e['name'] for e in entity_info])
        return f"rel_discovery_{'_'.join(sorted_names)}"
    
    async def _call_llm_for_discovery(self, entity_info: List[Dict[str, Any]]) -> str:
        """Call LLM to discover relationships between entities"""
        
        # Build the prompt for relationship discovery
        prompt = self._build_discovery_prompt(entity_info)
        
        # Call LLM with retry logic using improved timeout and retry settings
        from app.core.timeout_settings_cache import get_radiating_timeout, get_radiating_max_retries, get_radiating_retry_delay
        
        max_retries = get_radiating_max_retries()
        retry_delay = get_radiating_retry_delay()
        timeout_seconds = get_radiating_timeout('llm_call_timeout', 30)
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"LLM discovery attempt {attempt + 1}/{max_retries}")
                
                payload = {
                    "model": self.model_config['model'],
                    "prompt": prompt,
                    "temperature": self.model_config['temperature'],
                    "max_tokens": self.model_config['max_tokens'],
                    "stream": False
                }
                
                # Use circuit breaker protection for LLM calls
                async def _protected_llm_call():
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.model_config['model_server']}/api/generate",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                response_text = result.get('response', '')
                                
                                if not response_text.strip():
                                    raise Exception("Empty response from LLM")
                                
                                return response_text
                            else:
                                raise Exception(f"LLM API error {response.status}")
                
                # Use circuit breaker protection
                response_text = await protected_llm_call(_protected_llm_call)
                
                if response_text is None:
                    # Circuit breaker is open, use fallback
                    logger.warning("LLM circuit breaker open for relationship discovery")
                    return "{'relationships': []}"  # Return empty JSON as fallback
                
                logger.debug(f"LLM discovery successful on attempt {attempt + 1}")
                logger.debug(f"LLM response (first 500 chars): {response_text[:500]}")
                return response_text
                            
            except asyncio.TimeoutError:
                logger.warning(f"LLM discovery timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                raise
                
            except Exception as e:
                logger.warning(f"LLM discovery error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                raise
        
        raise Exception(f"LLM discovery failed after {max_retries} attempts")
    
    def _build_discovery_prompt(self, entity_info: List[Dict[str, Any]]) -> str:
        """Build prompt for relationship discovery"""
        
        # Format entity list with additional context
        entity_list = "\n".join([
            f"- {e['name']} ({e['type']}){' - ' + e.get('description', '') if e.get('description') else ''}"
            for e in entity_info
        ])
        
        # Format relationship types with descriptions (limit to most relevant)
        # Prioritize common relationships to keep prompt concise
        priority_relationships = [
            'INTEGRATES_WITH', 'DEPENDS_ON', 'USES', 'REQUIRES', 'BUILT_ON',
            'COMPETES_WITH', 'ALTERNATIVE_TO', 'EXTENDS', 'SUPPORTS', 'IMPLEMENTS',
            'PROVIDES', 'ENABLES', 'CONNECTS_TO', 'DEVELOPED_BY', 'COMPATIBLE_WITH'
        ]
        
        relationship_types = "\n".join([
            f"- {rel_type}: {self.TECHNOLOGY_RELATIONSHIPS[rel_type]}"
            for rel_type in priority_relationships
        ])
        
        # Create a very explicit prompt that demands JSON-only response
        prompt = f"""You are an expert in technology, AI, ML, software systems, cloud computing, databases, and business relationships.

Analyze these entities and discover meaningful relationships between them:

ENTITIES:
{entity_list}

RELATIONSHIP TYPES (use ONLY these):
{relationship_types}

IMPORTANT: Return ONLY a valid JSON object with NO additional text, explanations, or markdown formatting.
The JSON MUST follow this exact structure:

{{
  "relationships": [
    {{
      "source": "entity_name_1",
      "target": "entity_name_2",
      "type": "RELATIONSHIP_TYPE",
      "confidence": 0.8,
      "context": "Brief explanation",
      "bidirectional": false
    }}
  ]
}}

Discover up to 10 most important relationships. Start your response directly with {{ and end with }}.
"""
        
        return prompt
    
    def _parse_llm_relationships(self,
                                response: str,
                                entities: List[RadiatingEntity],
                                max_per_pair: int,
                                confidence_threshold: float) -> List[RadiatingRelationship]:
        """Parse LLM response into RadiatingRelationship objects"""
        relationships = []
        
        try:
            # Clean the response - remove any markdown formatting
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith('```'):
                # Find the actual JSON content within code blocks
                lines = response.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith('```'):
                        if not in_json:
                            in_json = True
                            continue
                        else:
                            break
                    elif in_json:
                        json_lines.append(line)
                response = '\n'.join(json_lines)
            
            # Try to extract JSON - look for the outermost curly braces
            # First try to parse the entire response as JSON
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON using regex
                # Look for JSON object pattern
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if not json_match:
                    # Try a more permissive pattern
                    start_idx = response.find('{')
                    end_idx = response.rfind('}')
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        json_str = response[start_idx:end_idx+1]
                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse extracted JSON: {e}")
                            logger.debug(f"Attempted to parse: {json_str[:200]}...")
                            return relationships
                    else:
                        logger.error("No JSON structure found in LLM response")
                        logger.debug(f"Response: {response[:500]}...")
                        return relationships
                else:
                    try:
                        data = json.loads(json_match.group())
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse matched JSON: {e}")
                        return relationships
            
            # Create entity lookup maps
            entity_map = {
                entity.canonical_form.lower(): entity
                for entity in entities
            }
            
            # Also map by partial names for flexibility
            for entity in entities:
                name_parts = entity.canonical_form.lower().split()
                for part in name_parts:
                    if len(part) > 3 and part not in entity_map:
                        entity_map[part] = entity
            
            # Track relationships per entity pair
            pair_counts = {}
            
            # Process each discovered relationship
            for rel_data in data.get('relationships', []):
                try:
                    source_name = rel_data.get('source', '').lower()
                    target_name = rel_data.get('target', '').lower()
                    rel_type = rel_data.get('type', '').upper()
                    confidence = float(rel_data.get('confidence', 0.7))
                    context = rel_data.get('context', '')
                    bidirectional = rel_data.get('bidirectional', False)
                    
                    # Skip if confidence too low
                    if confidence < confidence_threshold:
                        continue
                    
                    # Find matching entities
                    source_entity = self._find_entity_match(source_name, entity_map)
                    target_entity = self._find_entity_match(target_name, entity_map)
                    
                    if not source_entity or not target_entity:
                        logger.debug(f"Could not match entities: {source_name} -> {target_name}")
                        continue
                    
                    # Skip self-relationships
                    if source_entity.get_entity_id() == target_entity.get_entity_id():
                        continue
                    
                    # Check pair count limit
                    pair_key = tuple(sorted([source_entity.get_entity_id(), target_entity.get_entity_id()]))
                    if pair_key not in pair_counts:
                        pair_counts[pair_key] = 0
                    
                    if pair_counts[pair_key] >= max_per_pair:
                        continue
                    
                    # Validate relationship type
                    if rel_type not in self.TECHNOLOGY_RELATIONSHIPS:
                        # Try to map to closest known type
                        rel_type = self._map_to_known_type(rel_type)
                        if not rel_type:
                            continue
                    
                    # Create RadiatingRelationship
                    relationship = RadiatingRelationship(
                        source_entity=source_entity.canonical_form,
                        target_entity=target_entity.canonical_form,
                        relationship_type=rel_type,
                        confidence=confidence,
                        context=context,
                        properties={
                            'discovered_by': 'LLM',
                            'discovery_method': 'knowledge_base',
                            'bidirectional': bidirectional
                        },
                        traversal_weight=confidence * 0.8,  # Slightly lower weight for LLM-discovered
                        bidirectional=bidirectional,
                        strength_score=confidence,
                        discovery_context={
                            'source': 'LLM_discovery',
                            'timestamp': datetime.now().isoformat(),
                            'model': self.model_config['model']
                        }
                    )
                    
                    relationships.append(relationship)
                    pair_counts[pair_key] += 1
                    
                    # Add inverse relationship if bidirectional
                    if bidirectional and pair_counts[pair_key] < max_per_pair:
                        inverse_rel = relationship.get_inverse_relationship()
                        inverse_rel.relationship_type = rel_type  # Keep same type, not "inverse_"
                        relationships.append(inverse_rel)
                        pair_counts[pair_key] += 1
                    
                except Exception as e:
                    logger.debug(f"Error parsing relationship: {e}")
                    continue
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Try to extract relationships using regex as fallback
            relationships = self._fallback_parse_relationships(response, entities)
        
        return relationships
    
    def _find_entity_match(self, name: str, entity_map: Dict[str, RadiatingEntity]) -> Optional[RadiatingEntity]:
        """Find matching entity from map with fuzzy matching"""
        name_lower = name.lower().strip()
        
        # Exact match
        if name_lower in entity_map:
            return entity_map[name_lower]
        
        # Partial match - check if name is substring
        for key, entity in entity_map.items():
            if name_lower in key or key in name_lower:
                return entity
        
        # Word-based match
        name_words = set(name_lower.split())
        for key, entity in entity_map.items():
            key_words = set(key.split())
            if len(name_words.intersection(key_words)) > 0:
                return entity
        
        return None
    
    def _map_to_known_type(self, rel_type: str) -> Optional[str]:
        """Map unknown relationship type to known type"""
        rel_type_upper = rel_type.upper()
        
        # Direct match
        if rel_type_upper in self.TECHNOLOGY_RELATIONSHIPS:
            return rel_type_upper
        
        # Common variations
        mappings = {
            'USES_AS': 'USES',
            'BUILT_WITH': 'BUILT_ON',
            'RUNS_ON': 'DEPLOYED_ON',
            'ALTERNATIVE': 'ALTERNATIVE_TO',
            'COMPETITOR': 'COMPETES_WITH',
            'INTEGRATED_WITH': 'INTEGRATES_WITH',
            'CONNECTED_TO': 'CONNECTS_TO',
            'UTILIZED_BY': 'USED_BY',
            'CREATED_BY': 'DEVELOPED_BY',
            'MAINTAINED_BY': 'OWNED_BY'
        }
        
        if rel_type_upper in mappings:
            return mappings[rel_type_upper]
        
        # Partial match
        for known_type in self.TECHNOLOGY_RELATIONSHIPS:
            if known_type in rel_type_upper or rel_type_upper in known_type:
                return known_type
        
        return None
    
    def _fallback_parse_relationships(self, 
                                     response: str,
                                     entities: List[RadiatingEntity]) -> List[RadiatingRelationship]:
        """Fallback parsing using regex when JSON parsing fails"""
        relationships = []
        
        # Create entity name list for matching
        entity_names = [e.canonical_form.lower() for e in entities]
        entity_map = {e.canonical_form.lower(): e for e in entities}
        
        # Look for patterns like "X RELATIONSHIP_TYPE Y"
        for rel_type in self.TECHNOLOGY_RELATIONSHIPS:
            pattern = rf"(\w+[\w\s]*)\s+{rel_type}\s+(\w+[\w\s]*)"
            matches = re.finditer(pattern, response, re.IGNORECASE)
            
            for match in matches:
                source_text = match.group(1).strip().lower()
                target_text = match.group(2).strip().lower()
                
                # Try to match to known entities
                source_entity = self._find_entity_match(source_text, entity_map)
                target_entity = self._find_entity_match(target_text, entity_map)
                
                if source_entity and target_entity and source_entity != target_entity:
                    relationship = RadiatingRelationship(
                        source_entity=source_entity.canonical_form,
                        target_entity=target_entity.canonical_form,
                        relationship_type=rel_type,
                        confidence=0.6,  # Lower confidence for regex extraction
                        context="Extracted via pattern matching",
                        properties={'discovered_by': 'LLM_fallback'},
                        traversal_weight=0.5,
                        strength_score=0.6
                    )
                    relationships.append(relationship)
        
        return relationships
    
    async def discover_entity_neighbors(self,
                                       entity: RadiatingEntity,
                                       potential_entities: List[RadiatingEntity],
                                       max_neighbors: int = 10) -> List[Tuple[RadiatingEntity, RadiatingRelationship]]:
        """
        Discover neighbors for a specific entity from a pool of potential entities.
        
        Args:
            entity: The entity to find neighbors for
            potential_entities: Pool of entities to check for relationships
            max_neighbors: Maximum number of neighbors to return
            
        Returns:
            List of tuples (neighbor_entity, relationship)
        """
        if not potential_entities:
            return []
        
        # Include the source entity in the discovery
        all_entities = [entity] + [e for e in potential_entities if e.get_entity_id() != entity.get_entity_id()]
        
        # Discover all relationships
        relationships = await self.discover_relationships(all_entities)
        
        # Filter for relationships involving the source entity
        neighbors = []
        source_id = entity.get_entity_id()
        source_canonical = entity.canonical_form.lower()
        
        for rel in relationships:
            neighbor_entity = None
            
            # Check if source entity is involved
            if rel.source_entity.lower() == source_canonical:
                # Find target entity
                for e in potential_entities:
                    if e.canonical_form.lower() == rel.target_entity.lower():
                        neighbor_entity = e
                        break
                        
            elif rel.target_entity.lower() == source_canonical:
                # Find source entity (inverse relationship)
                for e in potential_entities:
                    if e.canonical_form.lower() == rel.source_entity.lower():
                        neighbor_entity = e
                        # Create inverse relationship
                        rel = rel.get_inverse_relationship()
                        break
            
            if neighbor_entity:
                neighbors.append((neighbor_entity, rel))
                
                if len(neighbors) >= max_neighbors:
                    break
        
        return neighbors