"""
Pure LLM-Driven Knowledge Graph Extraction Service

Handles entity and relationship extraction from document chunks using ONLY LLM intelligence.
No spaCy, no regex patterns, no hardcoded types - pure AI-driven knowledge discovery.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings
from app.services.neo4j_service import get_neo4j_service
from app.services.knowledge_graph_types import (
    ExtractedEntity, 
    ExtractedRelationship, 
    GraphExtractionResult
)
from app.document_handlers.base import ExtractedChunk

logger = logging.getLogger(__name__)
# Enable debug logging for relationship storage tracking
logger.setLevel(logging.DEBUG)

class KnowledgeGraphExtractionService:
    """Pure LLM-driven service for extracting knowledge graphs from documents"""
    
    def __init__(self):
        self.config = get_knowledge_graph_settings()
        # Import here to avoid circular import
        from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
        self.llm_extractor = LLMKnowledgeExtractor()
        logger.info("🚀 Initialized Pure LLM Knowledge Graph Extraction Service")
    
    async def extract_from_chunk(self, chunk: ExtractedChunk, document_id: str = None) -> GraphExtractionResult:
        """Extract knowledge graph using pure LLM intelligence"""
        start_time = datetime.now()
        
        try:
            logger.info(f"🧠 Starting LLM knowledge extraction for chunk {chunk.chunk_id}")
            
            # Use LLM to extract entities and relationships with no predefined constraints
            extraction_result = await self.llm_extractor.extract_knowledge(
                text=chunk.text,
                context={
                    'document_id': document_id,
                    'chunk_id': chunk.chunk_id,
                    'metadata': chunk.metadata
                }
            )
            
            # Convert LLM results to our format
            entities = []
            relationships = []
            warnings = []
            
            # Process LLM-discovered entities (extraction_result is LLMExtractionResult object)
            for entity in extraction_result.entities:
                try:
                    entities.append(entity)
                except Exception as e:
                    warnings.append(f"Failed to process LLM entity: {e}")
            
            # Process LLM-discovered relationships  
            for relationship in extraction_result.relationships:
                try:
                    relationships.append(relationship)
                except Exception as e:
                    warnings.append(f"Failed to process LLM relationship: {e}")
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"🎯 LLM extraction complete: {len(entities)} entities, {len(relationships)} relationships")
            
            return GraphExtractionResult(
                chunk_id=chunk.chunk_id,
                entities=entities,
                relationships=relationships,
                processing_time_ms=processing_time,
                source_metadata=chunk.metadata,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"❌ LLM knowledge extraction failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return GraphExtractionResult(
                chunk_id=chunk.chunk_id,
                entities=[],
                relationships=[],
                processing_time_ms=processing_time,
                source_metadata=chunk.metadata,
                warnings=[f"LLM extraction failed: {str(e)}"]
            )
    
    async def store_in_neo4j(self, result: GraphExtractionResult, document_id: str = None) -> Dict[str, Any]:
        """Store LLM-extracted knowledge graph in Neo4j"""
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            logger.warning("Neo4j not enabled - cannot store knowledge graph")
            return {'success': False, 'error': 'Neo4j not enabled'}
        
        try:
            entities_stored = 0
            relationships_stored = 0
            relationship_failures = 0
            
            # Create mapping of entity names to Neo4j IDs
            entity_name_to_id = {}
            
            # Store all LLM-discovered entities and build mapping
            for entity in result.entities:
                try:
                    entity_id = neo4j_service.create_entity(
                        entity_type=entity.label,  # Use LLM-determined type directly
                        properties={
                            'name': entity.canonical_form,
                            'type': entity.label,  # Explicitly set the type property for frontend visualization
                            'original_text': entity.text,
                            'confidence': entity.confidence,
                            'document_id': document_id,
                            'chunk_id': result.chunk_id,
                            'discovered_by': 'llm',
                            'created_at': datetime.now().isoformat()
                        }
                    )
                    if entity_id:
                        entities_stored += 1
                        # Build mapping for relationships: both canonical form and original text
                        entity_name_to_id[entity.canonical_form.lower()] = entity_id
                        entity_name_to_id[entity.text.lower()] = entity_id
                        logger.debug(f"📝 Entity mapping: '{entity.canonical_form}' -> {entity_id}")
                except Exception as e:
                    logger.warning(f"Failed to store LLM entity {entity.text}: {e}")
            
            # Store all LLM-discovered relationships using ID mapping
            for relationship in result.relationships:
                try:
                    # Map entity names to Neo4j IDs
                    source_name = relationship.source_entity.strip()
                    target_name = relationship.target_entity.strip()
                    
                    source_id = entity_name_to_id.get(source_name.lower())
                    target_id = entity_name_to_id.get(target_name.lower())
                    
                    if not source_id:
                        logger.warning(f"❌ Source entity not found: '{source_name}' (available: {list(entity_name_to_id.keys())[:5]}...)")
                        relationship_failures += 1
                        continue
                    
                    if not target_id:
                        logger.warning(f"❌ Target entity not found: '{target_name}' (available: {list(entity_name_to_id.keys())[:5]}...)")
                        relationship_failures += 1
                        continue
                    
                    logger.debug(f"🔗 Creating relationship: {source_id} --[{relationship.relationship_type}]--> {target_id}")
                    
                    # Build properties dictionary and sanitize for Neo4j
                    raw_properties = {
                        'confidence': relationship.confidence,
                        'context': relationship.context,
                        'document_id': document_id,
                        'chunk_id': result.chunk_id,
                        'discovered_by': 'llm',
                        'created_at': datetime.now().isoformat(),
                        'source_name': source_name,  # Keep original names for reference
                        'target_name': target_name,
                        **relationship.properties
                    }
                    
                    # Sanitize properties to avoid Map{} errors in Neo4j
                    sanitized_properties = self._sanitize_neo4j_properties(raw_properties)
                    
                    success = neo4j_service.create_relationship(
                        from_id=source_id,
                        to_id=target_id,
                        relationship_type=relationship.relationship_type,  # Use LLM-determined type
                        properties=sanitized_properties
                    )
                    if success:
                        relationships_stored += 1
                        logger.debug(f"✅ Relationship stored successfully")
                    else:
                        logger.warning(f"❌ Neo4j relationship creation returned False")
                        relationship_failures += 1
                        
                except Exception as e:
                    logger.warning(f"❌ Failed to store LLM relationship '{source_name}' -> '{target_name}': {e}")
                    relationship_failures += 1
            
            logger.info(f"✅ Stored LLM knowledge: {entities_stored} entities, {relationships_stored} relationships ({relationship_failures} relationship failures)")
            
            # Debug logging for entity storage tracking
            if entities_stored != len(result.entities):
                logger.warning(f"🔍 Entity storage mismatch: Expected {len(result.entities)}, actually stored {entities_stored}")
                for i, entity in enumerate(result.entities):
                    logger.debug(f"   Entity {i+1}: {entity.canonical_form} ({entity.label}) - Chunk: {result.chunk_id}")
            else:
                logger.debug(f"✅ All {entities_stored} entities stored successfully from chunk {result.chunk_id}")
            
            # Automatically run anti-silo analysis after storage to reduce isolated nodes
            if entities_stored > 0:
                try:
                    logger.info("🔗 Running automatic anti-silo analysis after entity storage...")
                    logger.info(f"📊 Pre-anti-silo status: {entities_stored} entities stored, {relationships_stored} relationships created")
                    
                    anti_silo_result = await self.run_global_anti_silo_analysis()
                    
                    if anti_silo_result.get('success'):
                        logger.info(f"🎯 Anti-silo analysis COMPLETED:")
                        logger.info(f"   ✅ Initial silos: {anti_silo_result.get('initial_silo_count', 0)}")
                        logger.info(f"   ✅ Final silos: {anti_silo_result.get('final_silo_count', 0)}")
                        logger.info(f"   ✅ Connections made: {anti_silo_result.get('connections_made', 0)}")
                        logger.info(f"   ✅ Hubs connected: {anti_silo_result.get('hubs_connected', 0)}")
                        logger.info(f"   ✅ Nodes removed: {anti_silo_result.get('nodes_removed', 0)}")
                        logger.info(f"   ✅ Total reduction: {anti_silo_result.get('reduction', 0)} silos eliminated")
                        
                        remaining_silos = anti_silo_result.get('remaining_silos', [])
                        if remaining_silos:
                            logger.info(f"   ⚠️  Remaining {len(remaining_silos)} isolated nodes:")
                            for silo in remaining_silos[:5]:  # Show first 5
                                logger.info(f"      - {silo.get('name', 'Unknown')} ({silo.get('type', 'Unknown type')})")
                        else:
                            logger.info("   🎉 NO isolated nodes remaining!")
                    else:
                        logger.error(f"❌ Anti-silo analysis FAILED: {anti_silo_result.get('error', 'Unknown error')}")
                        logger.error("   This means isolated nodes may remain in the knowledge graph")
                        
                except Exception as e:
                    logger.error(f"❌ Automatic anti-silo analysis CRASHED: {e}")
                    logger.error("   This is a critical error - isolated nodes will remain")
                    logger.exception("Full exception details:")
            
            return {
                'success': True,
                'entities_stored': entities_stored,
                'relationships_stored': relationships_stored,
                'relationship_failures': relationship_failures,
                'processing_time_ms': result.processing_time_ms
            }
            
        except Exception as e:
            logger.error(f"Failed to store LLM knowledge graph: {e}")
            return {'success': False, 'error': str(e)}
    
    def _sanitize_neo4j_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize properties for Neo4j by converting complex types to primitives"""
        sanitized = {}
        
        for key, value in properties.items():
            if value is None:
                # Skip None values
                continue
            elif isinstance(value, (str, int, float, bool)):
                # Primitive types - Neo4j compatible
                sanitized[key] = value
            elif isinstance(value, dict):
                if value:  # Non-empty dict - serialize to JSON string
                    sanitized[key] = json.dumps(value)
                # Skip empty dicts (the source of Map{} error)
            elif isinstance(value, list):
                if value:  # Non-empty list - serialize to JSON string if complex, keep if primitive
                    if all(isinstance(item, (str, int, float, bool)) for item in value):
                        sanitized[key] = value  # Keep primitive arrays
                    else:
                        sanitized[key] = json.dumps(value)  # Serialize complex arrays
                # Skip empty lists
            else:
                # Other types - convert to string
                sanitized[key] = str(value)
        
        return sanitized
    
    async def run_global_anti_silo_analysis(self) -> Dict[str, Any]:
        """Run comprehensive global anti-silo analysis to connect isolated nodes"""
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            logger.warning("Neo4j not enabled - cannot run anti-silo analysis")
            return {'success': False, 'error': 'Neo4j not enabled'}
        
        try:
            logger.info("🌍 Starting global anti-silo analysis...")
            
            # Get isolated nodes
            isolated_nodes = neo4j_service.get_truly_isolated_nodes()
            initial_silo_count = len(isolated_nodes)
            
            logger.info(f"🔍 Found {initial_silo_count} isolated nodes")
            
            # Separate Hub nodes from regular entities
            hub_nodes = [node for node in isolated_nodes if 'HUB' in node.get('labels', [])]
            regular_nodes = [node for node in isolated_nodes if 'HUB' not in node.get('labels', [])]
            
            logger.info(f"📊 Isolated breakdown: {len(hub_nodes)} Hub nodes, {len(regular_nodes)} regular entities")
            
            connections_made = 0
            hubs_connected = 0
            nodes_removed = 0
            
            # Strategy 1: Connect Hub nodes to their corresponding entity types
            for hub_node in hub_nodes:
                # Handle None values properly - node.get() can return None even with default
                hub_type_raw = hub_node.get('type', '')
                hub_type = (hub_type_raw or '').upper()
                if hub_type:
                    connections = await self._connect_hub_to_entities(neo4j_service, hub_node, hub_type)
                    if connections > 0:
                        hubs_connected += 1
                        connections_made += connections
                        logger.info(f"🔗 Connected Hub_{hub_type} to {connections} entities")
            
            # Strategy 2: Remove Hub nodes that have no entities to connect to
            empty_hubs_removed = await self._remove_empty_hubs(neo4j_service)
            nodes_removed += empty_hubs_removed
            
            # Strategy 3: Connect regular entities based on semantic similarity
            semantic_connections = await self._connect_entities_by_similarity(neo4j_service, regular_nodes)
            connections_made += semantic_connections
            
            logger.info(f"🧠 Semantic connections made: {semantic_connections}")
            
            # Get final isolated count
            final_isolated_nodes = neo4j_service.get_truly_isolated_nodes()
            final_silo_count = len(final_isolated_nodes)
            
            result = {
                'success': True,
                'initial_silo_count': initial_silo_count,
                'final_silo_count': final_silo_count,
                'connections_made': connections_made,
                'hubs_connected': hubs_connected,
                'nodes_removed': nodes_removed,
                'reduction': initial_silo_count - final_silo_count,
                'remaining_silos': [{'name': node.get('name'), 'type': node.get('type'), 'labels': node.get('labels')} 
                                   for node in final_isolated_nodes],
                'message': f'Anti-silo analysis complete: {connections_made} connections made, {nodes_removed} nodes removed'
            }
            
            logger.info(f"✅ Anti-silo analysis complete: {initial_silo_count} → {final_silo_count} isolated nodes")
            return result
            
        except Exception as e:
            logger.error(f"❌ Global anti-silo analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _connect_hub_to_entities(self, neo4j_service, hub_node: Dict[str, Any], hub_type: str) -> int:
        """Connect a Hub node to all entities of its corresponding type"""
        try:
            # Map hub types to entity labels
            type_mapping = {
                'CONCEPT': ['CONCEPT'],
                'PERSON': ['PERSON'],
                'ORG': ['ORGANIZATION', 'ORG'],
                'ORGANIZATION': ['ORGANIZATION', 'ORG'],
                'LOCATION': ['LOCATION'],
                'TECHNOLOGY': ['TECHNOLOGY'],
                'TEMPORAL': ['TEMPORAL', 'DATE', 'TIME'],
                'NUMERIC': ['NUMERIC', 'NUMBER', 'QUANTITY']
            }
            
            entity_labels = type_mapping.get(hub_type, [hub_type])
            hub_id = hub_node.get('id')
            
            if not hub_id:
                return 0
            
            connections_made = 0
            
            # Find entities of matching types and connect them to the hub
            for label in entity_labels:
                query = """
                MATCH (hub) WHERE hub.id = $hub_id
                MATCH (entity) WHERE $label IN labels(entity) AND NOT (hub)-[:BELONGS_TO_TYPE]-(entity)
                CREATE (entity)-[:BELONGS_TO_TYPE]->(hub)
                RETURN count(*) as connections
                """
                
                result = neo4j_service.execute_cypher(query, {'hub_id': hub_id, 'label': label})
                if result:
                    connections_made += result[0].get('connections', 0)
            
            return connections_made
            
        except Exception as e:
            logger.warning(f"Failed to connect hub {hub_node.get('name', 'unknown')}: {e}")
            return 0
    
    async def _remove_empty_hubs(self, neo4j_service) -> int:
        """Remove Hub nodes that have no entities to organize"""
        try:
            # Find Hub nodes that still have no relationships after connection attempts
            query = """
            MATCH (hub:HUB)
            WHERE NOT EXISTS((hub)--())
            DETACH DELETE hub
            RETURN count(*) as removed
            """
            
            result = neo4j_service.execute_cypher(query)
            removed_count = result[0].get('removed', 0) if result else 0
            
            if removed_count > 0:
                logger.info(f"🗑️ Removed {removed_count} empty Hub nodes")
            
            return removed_count
            
        except Exception as e:
            logger.warning(f"Failed to remove empty hubs: {e}")
            return 0
    
    async def _connect_entities_by_similarity(self, neo4j_service, isolated_nodes: List[Dict]) -> int:
        """Connect isolated entities based on semantic similarity and domain knowledge"""
        if not isolated_nodes:
            logger.info("🔍 No isolated nodes provided for similarity connection")
            return 0
            
        logger.info(f"🔍 Starting similarity-based connection for {len(isolated_nodes)} isolated nodes")
        connections_made = 0
        
        try:
            # Get knowledge graph settings for similarity threshold
            kg_settings = get_knowledge_graph_settings()
            similarity_threshold = kg_settings.get('anti_silo', {}).get('similarity_threshold', 0.5)
            logger.info(f"📊 Using similarity threshold: {similarity_threshold}")
            
            # Define domain-based connection rules for technology entities
            technology_entities = []
            business_entities = []
            geographic_entities = []
            
            # Categorize isolated nodes
            logger.info("🏷️ Categorizing isolated nodes by type and content...")
            for node in isolated_nodes:
                name = node.get('name', '').lower()
                # Handle None values properly - node.get() can return None even with default
                node_type_raw = node.get('type', '')
                node_type = (node_type_raw or '').upper()
                
                logger.debug(f"   Analyzing node: '{node.get('name')}' (type: {node_type})")
                
                # Categorize by type first, then by content analysis
                if node_type in ['TECHNOLOGY']:
                    technology_entities.append(node)
                    logger.debug(f"      -> Categorized as TECHNOLOGY (by type)")
                elif node_type in ['LOCATION', 'GEOGRAPHIC']:
                    geographic_entities.append(node)
                    logger.debug(f"      -> Categorized as GEOGRAPHIC (by type)")
                elif any(tech_term in name for tech_term in ['blockchain', 'mainframe', 'ai', 'ml', 'database', 'cloud', 'api', 'kafka', 'redis']):
                    technology_entities.append(node)
                    logger.debug(f"      -> Categorized as TECHNOLOGY (by content)")
                elif any(geo_term in name for geo_term in ['singapore', 'hong kong', 'china', 'india', 'indonesia']):
                    geographic_entities.append(node)
                    logger.debug(f"      -> Categorized as GEOGRAPHIC (by content)")
                else:
                    business_entities.append(node)
                    logger.debug(f"      -> Categorized as BUSINESS (fallback)")
            
            logger.info(f"🏷️ Categorization complete:")
            logger.info(f"   - Technology entities: {len(technology_entities)}")
            logger.info(f"   - Business entities: {len(business_entities)}")
            logger.info(f"   - Geographic entities: {len(geographic_entities)}")
            
            # Log entity names for debugging
            if technology_entities:
                tech_names = [e.get('name', 'Unknown') for e in technology_entities]
                logger.info(f"   📱 Technology entities: {tech_names}")
            if business_entities:
                business_names = [e.get('name', 'Unknown') for e in business_entities[:5]]  # Show first 5
                logger.info(f"   🏢 Business entities (first 5): {business_names}")
            if geographic_entities:
                geo_names = [e.get('name', 'Unknown') for e in geographic_entities]
                logger.info(f"   🌍 Geographic entities: {geo_names}")
            
            # Connect technology entities to DBS Bank (if it exists)
            if technology_entities:
                logger.info(f"🔗 Attempting to connect {len(technology_entities)} technology entities to DBS Bank...")
                tech_connections = await self._connect_technology_entities_to_dbs(neo4j_service, technology_entities)
                connections_made += tech_connections
                logger.info(f"   ✅ Made {tech_connections} DBS-technology connections")
            
            # Connect business entities using industry patterns
            if business_entities:
                logger.info(f"🏢 Attempting to connect {len(business_entities)} business entities...")
                business_connections = await self._connect_business_entities(neo4j_service, business_entities)
                connections_made += business_connections
                logger.info(f"   ✅ Made {business_connections} business entity connections")
            
            # AGGRESSIVE: Connect ALL entity types to each other using domain knowledge
            all_entities = technology_entities + business_entities + geographic_entities
            if len(all_entities) > 1:
                logger.info(f"🚀 AGGRESSIVE MODE: Cross-connecting {len(all_entities)} entities of all types...")
                cross_connections = await self._connect_all_entity_types_aggressively(neo4j_service, all_entities)
                connections_made += cross_connections
                logger.info(f"   ✅ Made {cross_connections} aggressive cross-type connections")
            
            # NUCLEAR OPTION: Eliminate ALL remaining silo nodes by force
            settings = get_knowledge_graph_settings()
            nuclear_enabled = settings.get('extraction', {}).get('enable_nuclear_option', True)  # Default True for backward compatibility
            
            if nuclear_enabled:
                logger.info("☢️  NUCLEAR ANTI-SILO: Checking for any remaining isolated nodes...")
                nuclear_connections = await self._nuclear_anti_silo_elimination(neo4j_service)
                connections_made += nuclear_connections
                logger.info(f"   ☢️  Made {nuclear_connections} nuclear connections to eliminate all silos")
            else:
                logger.info("☢️  Nuclear option disabled in settings - skipping aggressive elimination")
            
            # Connect entities within same categories
            logger.info("🔗 Connecting entities within same categories...")
            
            if len(technology_entities) > 1:
                tech_internal_connections = await self._connect_similar_entities(neo4j_service, technology_entities, "RELATED_TECHNOLOGY")
                connections_made += tech_internal_connections
                logger.info(f"   ✅ Made {tech_internal_connections} technology-technology connections")
            
            if len(geographic_entities) > 1:
                geo_internal_connections = await self._connect_similar_entities(neo4j_service, geographic_entities, "RELATED_LOCATION")
                connections_made += geo_internal_connections
                logger.info(f"   ✅ Made {geo_internal_connections} geographic-geographic connections")
            
            if len(business_entities) > 1:
                business_internal_connections = await self._connect_similar_entities(neo4j_service, business_entities, "COMPETES_WITH")
                connections_made += business_internal_connections
                logger.info(f"   ✅ Made {business_internal_connections} business-business connections")
            
            logger.info(f"🎯 Similarity connection summary: {connections_made} total connections made")
            return connections_made
            
        except Exception as e:
            logger.error(f"❌ Failed to connect entities by similarity: {e}")
            logger.exception("Full exception details:")
            return 0
    
    async def _connect_technology_entities_to_dbs(self, neo4j_service, tech_entities: List[Dict]) -> int:
        """Connect technology entities to DBS Bank if it exists"""
        connections_made = 0
        
        try:
            # Find DBS Bank node
            dbs_query = """
            MATCH (dbs:CONCEPT)
            WHERE dbs.name =~ '(?i).*dbs.*bank.*'
            RETURN dbs.id as id, dbs.name as name
            LIMIT 1
            """
            
            result = neo4j_service.execute_cypher(dbs_query)
            if result and len(result) > 0:
                dbs_id = result[0]['id']
                logger.info(f"🏦 Found DBS Bank node: {result[0]['name']}")
                
                for tech_entity in tech_entities:
                    tech_id = tech_entity.get('id')
                    tech_name = tech_entity.get('name')
                    
                    if tech_id and tech_id != dbs_id:
                        # Create relationship: DBS Bank -> EVALUATES -> Technology
                        relationship_created = neo4j_service.create_relationship(
                            from_id=dbs_id,
                            to_id=tech_id,
                            relationship_type="EVALUATES",
                            properties={
                                'created_by': 'anti_silo_analysis',
                                'confidence': 0.8,
                                'reasoning': f'DBS Bank likely evaluates {tech_name} as part of technology assessment'
                            }
                        )
                        
                        if relationship_created:
                            connections_made += 1
                            logger.info(f"🔗 Connected DBS Bank -> EVALUATES -> {tech_name}")
            
            return connections_made
            
        except Exception as e:
            logger.error(f"❌ Failed to connect tech entities to DBS: {e}")
            return 0
    
    async def _connect_business_entities(self, neo4j_service, business_entities: List[Dict]) -> int:
        """Connect business entities using industry and domain knowledge patterns"""
        connections_made = 0
        
        try:
            logger.info(f"🏢 Analyzing {len(business_entities)} business entities for connections...")
            
            # Find all existing organizations in the graph to connect to
            org_query = """
            MATCH (org)
            WHERE org.name IS NOT NULL 
            AND (org.type = 'ORGANIZATION' OR org.name =~ '(?i).*(bank|corp|company|group|inc|ltd).*')
            RETURN org.id as id, org.name as name, org.type as type
            """
            
            existing_orgs = neo4j_service.execute_cypher(org_query)
            logger.info(f"🔍 Found {len(existing_orgs)} existing organizations in graph")
            
            # Create a map of existing organizations for quick lookup
            org_map = {org['id']: org for org in existing_orgs}
            
            # Connect each business entity to relevant organizations
            for business_entity in business_entities:
                business_id = business_entity.get('id')
                business_name = business_entity.get('name', '').lower()
                
                if not business_id:
                    continue
                
                logger.info(f"   🏢 Connecting '{business_entity.get('name')}' to ecosystem...")
                entity_connections = 0
                
                for org in existing_orgs:
                    org_id = org['id']
                    org_name = org['name'].lower()
                    
                    # Skip self-connections
                    if business_id == org_id:
                        continue
                    
                    # Determine relationship type based on business context
                    relationship_type, confidence, reasoning = self._determine_business_relationship(
                        business_name, org_name, business_entity.get('name'), org['name']
                    )
                    
                    if relationship_type and confidence >= 0.3:
                        # Create the relationship
                        relationship_created = neo4j_service.create_relationship(
                            from_id=business_id,
                            to_id=org_id,
                            relationship_type=relationship_type,
                            properties={
                                'created_by': 'anti_silo_business_analysis',
                                'confidence': confidence,
                                'reasoning': reasoning,
                                'domain': 'business'
                            }
                        )
                        
                        if relationship_created:
                            connections_made += 1
                            entity_connections += 1
                            logger.info(f"      🔗 {business_entity.get('name')} -> {relationship_type} -> {org['name']}")
                
                logger.info(f"   ✅ Made {entity_connections} connections for '{business_entity.get('name')}'")
            
            return connections_made
            
        except Exception as e:
            logger.error(f"❌ Failed to connect business entities: {e}")
            logger.exception("Full exception details:")
            return 0
    
    def _determine_business_relationship(self, entity1_lower: str, entity2_lower: str, 
                                       entity1_name: str, entity2_name: str) -> tuple:
        """Determine the most appropriate business relationship between two entities (AGGRESSIVE MODE)"""
        
        # Financial services relationships
        if 'bank' in entity1_lower and 'bank' in entity2_lower:
            return 'COMPETES_WITH', 0.8, f'{entity1_name} and {entity2_name} are both banks in the financial sector'
        
        if 'bank' in entity1_lower and any(term in entity2_lower for term in ['group', 'corp', 'company']):
            return 'PARTNERS_WITH', 0.7, f'{entity1_name} may partner with {entity2_name} in financial services'
        
        # Technology/fintech relationships
        if any(term in entity1_lower for term in ['ant', 'alibaba', 'tencent']) and 'bank' in entity2_lower:
            return 'PROVIDES_SERVICES_TO', 0.8, f'{entity1_name} provides fintech services to banking sector including {entity2_name}'
        
        if 'ant group' in entity1_lower and any(term in entity2_lower for term in ['alibaba', 'alipay']):
            return 'PART_OF', 0.9, f'{entity1_name} is part of the Alibaba ecosystem including {entity2_name}'
        
        # Technology-Organization connections (AGGRESSIVE)
        tech_terms = ['database', 'sql', 'stack', 'base', 'cloud', 'platform', 'system', 'api', 'kafka', 'redis', 'postgresql', 'mariadb']
        org_terms = ['bank', 'corp', 'company', 'group', 'inc', 'ltd', 'organization', 'enterprise']
        
        entity1_is_tech = any(term in entity1_lower for term in tech_terms)
        entity2_is_tech = any(term in entity2_lower for term in tech_terms)
        entity1_is_org = any(term in entity1_lower for term in org_terms)
        entity2_is_org = any(term in entity2_lower for term in org_terms)
        
        # Tech-Org connections
        if entity1_is_tech and entity2_is_org:
            return 'USED_BY', 0.6, f'{entity1_name} technology is likely used by {entity2_name}'
        if entity1_is_org and entity2_is_tech:
            return 'USES', 0.6, f'{entity1_name} likely uses {entity2_name} technology'
        
        # Tech-Tech connections
        if entity1_is_tech and entity2_is_tech:
            return 'INTEGRATES_WITH', 0.5, f'{entity1_name} and {entity2_name} are related technologies'
        
        # Geographic/market relationships (EXPANDED)
        geo_terms = ['singapore', 'china', 'asia', 'hong kong', 'indonesia', 'thailand', 'malaysia', 'vietnam']
        if any(geo in entity1_lower for geo in geo_terms) and entity2_is_org:
            return 'OPERATES_IN_REGION', 0.5, f'{entity2_name} operates in {entity1_name} region'
        if entity1_is_org and any(geo in entity2_lower for geo in geo_terms):
            return 'OPERATES_IN_REGION', 0.5, f'{entity1_name} operates in {entity2_name} region'
        
        # Concept-Organization connections (AGGRESSIVE)
        concept_terms = ['digital', 'transformation', 'migration', 'performance', 'scalability', 'security', 'strategy', 'innovation']
        entity1_is_concept = any(term in entity1_lower for term in concept_terms)
        entity2_is_concept = any(term in entity2_lower for term in concept_terms)
        
        if entity1_is_concept and entity2_is_org:
            return 'IMPLEMENTED_BY', 0.4, f'{entity1_name} concept is implemented by {entity2_name}'
        if entity1_is_org and entity2_is_concept:
            return 'IMPLEMENTS', 0.4, f'{entity1_name} implements {entity2_name} initiatives'
        
        # General business relationships for organizations (LOWERED THRESHOLD)
        if entity1_is_org and entity2_is_org:
            return 'INDUSTRY_PEER', 0.4, f'{entity1_name} and {entity2_name} are industry peers'
        
        # Financial/fintech ecosystem (EXPANDED)
        fintech_terms = ['payment', 'fintech', 'financial', 'banking', 'wallet', 'alipay', 'wechat', 'pay']
        entity1_is_fintech = any(term in entity1_lower for term in fintech_terms)
        entity2_is_fintech = any(term in entity2_lower for term in fintech_terms)
        
        if entity1_is_fintech and entity2_is_org:
            return 'SERVES', 0.4, f'{entity1_name} serves {entity2_name} in financial sector'
        if entity1_is_org and entity2_is_fintech:
            return 'SERVED_BY', 0.4, f'{entity1_name} is served by {entity2_name} in financial sector'
        
        # Business ecosystem connections (VERY PERMISSIVE)
        business_terms = ['business', 'enterprise', 'solution', 'service', 'product']
        entity1_is_business = any(term in entity1_lower for term in business_terms)
        entity2_is_business = any(term in entity2_lower for term in business_terms)
        
        if (entity1_is_business or entity1_is_org) and (entity2_is_business or entity2_is_org):
            return 'RELATED_IN_ECOSYSTEM', 0.3, f'{entity1_name} and {entity2_name} are part of the same business ecosystem'
        
        # Default: still try to connect if entities seem related (LAST RESORT)
        if len(entity1_name) > 3 and len(entity2_name) > 3:  # Avoid tiny words
            return 'CONTEXTUALLY_RELATED', 0.3, f'{entity1_name} and {entity2_name} appear in similar business context'
        
        # Truly no relationship found
        return None, 0.0, 'No relationship identified'
    
    async def _connect_all_entity_types_aggressively(self, neo4j_service, all_entities: List[Dict]) -> int:
        """AGGRESSIVE MODE: Connect entities across all types using broad domain knowledge"""
        connections_made = 0
        
        try:
            logger.info(f"🚀 AGGRESSIVE CROSS-TYPE CONNECTION: Processing {len(all_entities)} entities")
            
            # Get all existing entities in the graph for maximum connectivity
            all_nodes_query = """
            MATCH (n)
            WHERE n.name IS NOT NULL
            RETURN n.id as id, n.name as name, n.type as type
            ORDER BY n.name
            """
            
            all_existing_nodes = neo4j_service.execute_cypher(all_nodes_query)
            logger.info(f"🔍 Found {len(all_existing_nodes)} total nodes in graph for aggressive connection")
            
            # Connect each isolated entity to ALL relevant existing entities
            for isolated_entity in all_entities:
                isolated_id = isolated_entity.get('id')
                isolated_name = isolated_entity.get('name', '').lower()
                isolated_type = isolated_entity.get('type', '').upper()
                
                if not isolated_id:
                    continue
                
                logger.info(f"   🔗 Aggressively connecting '{isolated_entity.get('name')}' ({isolated_type})")
                entity_connections = 0
                
                for existing_node in all_existing_nodes:
                    existing_id = existing_node['id']
                    existing_name = existing_node['name'].lower()
                    existing_type = existing_node.get('type', '').upper()
                    
                    # Skip self-connections
                    if isolated_id == existing_id:
                        continue
                    
                    # Determine aggressive relationship
                    relationship_type, confidence = self._determine_aggressive_relationship(
                        isolated_name, existing_name, isolated_type, existing_type,
                        isolated_entity.get('name'), existing_node['name']
                    )
                    
                    if relationship_type and confidence >= 0.2:  # Very permissive threshold
                        # Create the relationship
                        relationship_created = neo4j_service.create_relationship(
                            from_id=isolated_id,
                            to_id=existing_id,
                            relationship_type=relationship_type,
                            properties={
                                'created_by': 'aggressive_anti_silo',
                                'confidence': confidence,
                                'reasoning': f'Aggressive domain knowledge connection between {isolated_type} and {existing_type}',
                                'aggressive_mode': True
                            }
                        )
                        
                        if relationship_created:
                            connections_made += 1
                            entity_connections += 1
                            logger.debug(f"      🔗 {isolated_entity.get('name')} -> {relationship_type} -> {existing_node['name']}")
                
                logger.info(f"   ✅ Made {entity_connections} connections for '{isolated_entity.get('name')}'")
                
                # Early termination if we've made enough connections for this entity
                if entity_connections >= 5:  # Limit to prevent over-connection
                    logger.info(f"   🛑 Stopping at {entity_connections} connections (sufficient connectivity)")
            
            return connections_made
            
        except Exception as e:
            logger.error(f"❌ Failed aggressive cross-type connection: {e}")
            logger.exception("Full exception details:")
            return 0
    
    def _determine_aggressive_relationship(self, entity1_lower: str, entity2_lower: str, 
                                         entity1_type: str, entity2_type: str,
                                         entity1_name: str, entity2_name: str) -> tuple:
        """AGGRESSIVE MODE: Determine relationships with very permissive logic"""
        
        # Skip if names are too similar (likely duplicates)
        if entity1_lower == entity2_lower:
            return None, 0.0
        
        # Type-based aggressive connections
        type_matrix = {
            # Organization connections
            ('ORGANIZATION', 'TECHNOLOGY'): ('USES', 0.5),
            ('TECHNOLOGY', 'ORGANIZATION'): ('USED_BY', 0.5),
            ('ORGANIZATION', 'LOCATION'): ('OPERATES_IN', 0.4),
            ('LOCATION', 'ORGANIZATION'): ('HOSTS', 0.4),
            ('ORGANIZATION', 'CONCEPT'): ('IMPLEMENTS', 0.3),
            ('CONCEPT', 'ORGANIZATION'): ('IMPLEMENTED_BY', 0.3),
            ('ORGANIZATION', 'PROJECT'): ('SPONSORS', 0.4),
            ('PROJECT', 'ORGANIZATION'): ('SPONSORED_BY', 0.4),
            ('ORGANIZATION', 'PRODUCT'): ('OFFERS', 0.4),
            ('PRODUCT', 'ORGANIZATION'): ('OFFERED_BY', 0.4),
            
            # Technology connections
            ('TECHNOLOGY', 'TECHNOLOGY'): ('INTEGRATES_WITH', 0.3),
            ('TECHNOLOGY', 'CONCEPT'): ('ENABLES', 0.3),
            ('CONCEPT', 'TECHNOLOGY'): ('ENABLED_BY', 0.3),
            ('TECHNOLOGY', 'PROJECT'): ('SUPPORTS', 0.3),
            ('PROJECT', 'TECHNOLOGY'): ('SUPPORTED_BY', 0.3),
            
            # Geographic connections
            ('LOCATION', 'LOCATION'): ('RELATED_REGION', 0.2),
            ('LOCATION', 'CONCEPT'): ('INFLUENCES', 0.2),
            ('CONCEPT', 'LOCATION'): ('INFLUENCED_BY', 0.2),
            
            # Concept connections
            ('CONCEPT', 'CONCEPT'): ('RELATED_CONCEPT', 0.2),
            ('PROJECT', 'CONCEPT'): ('IMPLEMENTS', 0.3),
            ('CONCEPT', 'PROJECT'): ('IMPLEMENTED_BY', 0.3),
            
            # Product connections
            ('PRODUCT', 'TECHNOLOGY'): ('BUILT_ON', 0.3),
            ('TECHNOLOGY', 'PRODUCT'): ('POWERS', 0.3),
            ('PRODUCT', 'CONCEPT'): ('EMBODIES', 0.2),
            ('CONCEPT', 'PRODUCT'): ('EMBODIED_IN', 0.2),
        }
        
        # Check type matrix first
        key = (entity1_type, entity2_type)
        if key in type_matrix:
            rel_type, confidence = type_matrix[key]
            return rel_type, confidence
        
        # Keyword-based aggressive connections
        financial_terms = ['bank', 'financial', 'payment', 'fintech', 'wallet']
        tech_terms = ['database', 'sql', 'stack', 'cloud', 'platform', 'system', 'api']
        business_terms = ['business', 'enterprise', 'company', 'corporation', 'group']
        
        entity1_financial = any(term in entity1_lower for term in financial_terms)
        entity2_financial = any(term in entity2_lower for term in financial_terms)
        entity1_tech = any(term in entity1_lower for term in tech_terms)
        entity2_tech = any(term in entity2_lower for term in tech_terms)
        entity1_business = any(term in entity1_lower for term in business_terms)
        entity2_business = any(term in entity2_lower for term in business_terms)
        
        # Financial ecosystem connections
        if entity1_financial and entity2_financial:
            return 'FINANCIAL_ECOSYSTEM', 0.3
        if entity1_financial and entity2_tech:
            return 'USES_TECH', 0.3
        if entity1_tech and entity2_financial:
            return 'SERVES_FINANCE', 0.3
        
        # Business ecosystem connections
        if entity1_business and entity2_business:
            return 'BUSINESS_NETWORK', 0.2
        if entity1_business and entity2_tech:
            return 'ADOPTS_TECH', 0.2
        if entity1_tech and entity2_business:
            return 'ENABLES_BUSINESS', 0.2
        
        # Geographic connections (very permissive)
        asian_locations = ['singapore', 'china', 'hong kong', 'asia', 'indonesia', 'malaysia']
        entity1_asian = any(loc in entity1_lower for loc in asian_locations)
        entity2_asian = any(loc in entity2_lower for loc in asian_locations)
        
        if entity1_asian and entity2_asian:
            return 'SAME_REGION', 0.2
        if entity1_asian and (entity2_financial or entity2_business):
            return 'REGIONAL_PRESENCE', 0.2
        if (entity1_financial or entity1_business) and entity2_asian:
            return 'OPERATES_IN_REGION', 0.2
        
        # Last resort: connect anything that seems business-related
        business_context_terms = ['digital', 'transformation', 'strategy', 'solution', 'service', 'platform', 'management']
        entity1_business_context = any(term in entity1_lower for term in business_context_terms)
        entity2_business_context = any(term in entity2_lower for term in business_context_terms)
        
        if entity1_business_context and entity2_business_context:
            return 'BUSINESS_CONTEXT', 0.2
        
        # Truly no connection
        return None, 0.0
    
    async def _nuclear_anti_silo_elimination(self, neo4j_service) -> int:
        """NUCLEAR OPTION: Brute force eliminate ALL silo nodes by connecting them to something"""
        connections_made = 0
        
        try:
            logger.info("☢️  NUCLEAR ANTI-SILO: Finding all isolated nodes for elimination...")
            
            # Get ALL truly isolated nodes (0 relationships)
            isolated_query = """
            MATCH (n)
            WHERE n.name IS NOT NULL
            WITH n, [(n)-[r]-(other) | r] as relationships
            WHERE size(relationships) = 0
            RETURN n.id as id, n.name as name, n.type as type
            ORDER BY n.name
            """
            
            isolated_nodes = neo4j_service.execute_cypher(isolated_query)
            logger.info(f"☢️  Found {len(isolated_nodes)} isolated nodes for nuclear elimination")
            
            if not isolated_nodes:
                logger.info("☢️  No isolated nodes found - nuclear option not needed")
                return 0
            
            # Log the isolated nodes for debugging
            isolated_names = [node['name'] for node in isolated_nodes]
            logger.info(f"☢️  Isolated nodes: {isolated_names}")
            
            # Find the most connected node in the graph to use as a hub
            hub_query = """
            MATCH (hub)
            WHERE hub.name IS NOT NULL
            WITH hub, size([(hub)-[]-(other) | other]) as connection_count
            WHERE connection_count > 0
            RETURN hub.id as id, hub.name as name, hub.type as type, connection_count
            ORDER BY connection_count DESC
            LIMIT 1
            """
            
            hub_result = neo4j_service.execute_cypher(hub_query)
            
            if not hub_result:
                # If no hub exists, connect isolated nodes to each other
                logger.info("☢️  No hub found - connecting isolated nodes to each other")
                return await self._connect_isolated_to_each_other(neo4j_service, isolated_nodes)
            
            hub_node = hub_result[0]
            hub_id = hub_node['id']
            hub_name = hub_node['name']
            hub_connections = hub_node['connection_count']
            
            logger.info(f"☢️  Using hub node: '{hub_name}' (type: {hub_node.get('type', 'Unknown')}, connections: {hub_connections})")
            
            # Connect EVERY isolated node to the hub with multiple strategies
            for isolated_node in isolated_nodes:
                isolated_id = isolated_node['id']
                isolated_name = isolated_node['name']
                isolated_type = isolated_node.get('type', 'UNKNOWN')
                
                if isolated_id == hub_id:
                    continue  # Skip if the isolated node is somehow the hub
                
                logger.info(f"☢️  Nuclear connecting: '{isolated_name}' -> '{hub_name}'")
                
                # Strategy 1: Document co-occurrence relationship
                cooccurrence_created = neo4j_service.create_relationship(
                    from_id=isolated_id,
                    to_id=hub_id,
                    relationship_type="MENTIONED_TOGETHER",
                    properties={
                        'created_by': 'nuclear_anti_silo',
                        'confidence': 0.3,
                        'reasoning': f'Nuclear anti-silo: {isolated_name} and {hub_name} mentioned in same document context',
                        'nuclear_connection': True,
                        'connection_strategy': 'document_cooccurrence'
                    }
                )
                
                if cooccurrence_created:
                    connections_made += 1
                    logger.info(f"      ✅ Nuclear connection 1: {isolated_name} -> MENTIONED_TOGETHER -> {hub_name}")
                
                # Strategy 2: Ecosystem relationship (bidirectional for better connectivity)
                ecosystem_created = neo4j_service.create_relationship(
                    from_id=isolated_id,
                    to_id=hub_id,
                    relationship_type="PART_OF_ECOSYSTEM",
                    properties={
                        'created_by': 'nuclear_anti_silo',
                        'confidence': 0.25,
                        'reasoning': f'Nuclear anti-silo: {isolated_name} is part of the same business ecosystem as {hub_name}',
                        'nuclear_connection': True,
                        'connection_strategy': 'ecosystem_membership'
                    }
                )
                
                if ecosystem_created:
                    connections_made += 1
                    logger.info(f"      ✅ Nuclear connection 2: {isolated_name} -> PART_OF_ECOSYSTEM -> {hub_name}")
                
                # Strategy 3: Type-specific connection if possible
                type_relationship = self._get_nuclear_type_relationship(isolated_type, hub_node.get('type', 'UNKNOWN'))
                if type_relationship:
                    rel_type, rel_confidence = type_relationship
                    type_created = neo4j_service.create_relationship(
                        from_id=isolated_id,
                        to_id=hub_id,
                        relationship_type=rel_type,
                        properties={
                            'created_by': 'nuclear_anti_silo',
                            'confidence': rel_confidence,
                            'reasoning': f'Nuclear anti-silo: Type-based connection between {isolated_type} and hub',
                            'nuclear_connection': True,
                            'connection_strategy': 'type_based'
                        }
                    )
                    
                    if type_created:
                        connections_made += 1
                        logger.info(f"      ✅ Nuclear connection 3: {isolated_name} -> {rel_type} -> {hub_name}")
            
            # ADDITIONAL STRATEGY: Document-based connections
            doc_connections = await self._connect_by_document_cooccurrence(neo4j_service, isolated_nodes)
            connections_made += doc_connections
            
            logger.info(f"☢️  NUCLEAR ELIMINATION COMPLETE: {connections_made} total connections made")
            
            # Verify no nodes remain isolated
            remaining_isolated = neo4j_service.execute_cypher(isolated_query)
            if remaining_isolated:
                logger.warning(f"☢️  WARNING: {len(remaining_isolated)} nodes still isolated after nuclear option!")
                remaining_names = [node['name'] for node in remaining_isolated]
                logger.warning(f"☢️  Remaining isolated: {remaining_names}")
            else:
                logger.info("☢️  SUCCESS: All nodes now connected!")
            
            return connections_made
            
        except Exception as e:
            logger.error(f"☢️  Nuclear anti-silo elimination failed: {e}")
            logger.exception("Nuclear elimination exception details:")
            return 0
    
    def _get_nuclear_type_relationship(self, isolated_type: str, hub_type: str) -> tuple:
        """Get appropriate relationship type for nuclear connections based on entity types"""
        
        # Nuclear type matrix - very permissive
        nuclear_matrix = {
            ('ORGANIZATION', 'ORGANIZATION'): ('INDUSTRY_PEER', 0.2),
            ('ORGANIZATION', 'TECHNOLOGY'): ('USES', 0.3),
            ('TECHNOLOGY', 'ORGANIZATION'): ('USED_BY', 0.3),
            ('ORGANIZATION', 'LOCATION'): ('OPERATES_IN', 0.3),
            ('LOCATION', 'ORGANIZATION'): ('HOSTS', 0.3),
            ('TECHNOLOGY', 'TECHNOLOGY'): ('COMPATIBLE_WITH', 0.2),
            ('CONCEPT', 'ORGANIZATION'): ('IMPLEMENTED_BY', 0.2),
            ('ORGANIZATION', 'CONCEPT'): ('IMPLEMENTS', 0.2),
            ('CONCEPT', 'TECHNOLOGY'): ('ENABLED_BY', 0.2),
            ('TECHNOLOGY', 'CONCEPT'): ('ENABLES', 0.2),
            ('PRODUCT', 'ORGANIZATION'): ('OFFERED_BY', 0.2),
            ('ORGANIZATION', 'PRODUCT'): ('OFFERS', 0.2),
            ('PROJECT', 'ORGANIZATION'): ('SPONSORED_BY', 0.2),
            ('ORGANIZATION', 'PROJECT'): ('SPONSORS', 0.2),
        }
        
        key = (isolated_type, hub_type)
        if key in nuclear_matrix:
            return nuclear_matrix[key]
        
        # Generic fallback
        return ('CONTEXTUALLY_CONNECTED', 0.15)
    
    async def _connect_by_document_cooccurrence(self, neo4j_service, isolated_nodes: List[Dict]) -> int:
        """Connect isolated nodes based on document co-occurrence"""
        connections_made = 0
        
        try:
            logger.info("📄 Connecting isolated nodes by document co-occurrence...")
            
            for isolated_node in isolated_nodes:
                isolated_id = isolated_node['id']
                isolated_name = isolated_node['name']
                
                # Find other entities with the same document_id
                cooccurrence_query = """
                MATCH (isolated), (other)
                WHERE isolated.id = $isolated_id 
                AND other.document_id = isolated.document_id
                AND other.id <> isolated.id
                AND other.name IS NOT NULL
                RETURN other.id as id, other.name as name, other.type as type
                LIMIT 5
                """
                
                cooccurring_entities = neo4j_service.execute_cypher(cooccurrence_query, {'isolated_id': isolated_id})
                
                for other_entity in cooccurring_entities:
                    other_id = other_entity['id']
                    other_name = other_entity['name']
                    
                    # Create document co-occurrence relationship
                    cooc_created = neo4j_service.create_relationship(
                        from_id=isolated_id,
                        to_id=other_id,
                        relationship_type="DOCUMENT_COOCCURRENCE",
                        properties={
                            'created_by': 'nuclear_document_cooccurrence',
                            'confidence': 0.4,
                            'reasoning': f'{isolated_name} and {other_name} appear in the same document',
                            'nuclear_connection': True,
                            'connection_strategy': 'document_cooccurrence'
                        }
                    )
                    
                    if cooc_created:
                        connections_made += 1
                        logger.info(f"📄 Document connection: {isolated_name} -> DOCUMENT_COOCCURRENCE -> {other_name}")
            
            return connections_made
            
        except Exception as e:
            logger.error(f"📄 Document co-occurrence connection failed: {e}")
            return 0
    
    async def _connect_isolated_to_each_other(self, neo4j_service, isolated_nodes: List[Dict]) -> int:
        """Last resort: connect isolated nodes to each other"""
        connections_made = 0
        
        try:
            logger.info("🔗 Last resort: connecting isolated nodes to each other...")
            
            # Connect each isolated node to the first other isolated node
            if len(isolated_nodes) >= 2:
                hub_isolated = isolated_nodes[0]  # Use first isolated node as mini-hub
                hub_id = hub_isolated['id']
                hub_name = hub_isolated['name']
                
                for other_isolated in isolated_nodes[1:]:
                    other_id = other_isolated['id']
                    other_name = other_isolated['name']
                    
                    # Create mutual connection
                    connected = neo4j_service.create_relationship(
                        from_id=other_id,
                        to_id=hub_id,
                        relationship_type="ISOLATED_CLUSTER",
                        properties={
                            'created_by': 'nuclear_isolated_cluster',
                            'confidence': 0.2,
                            'reasoning': f'Last resort: connecting isolated entities {other_name} and {hub_name}',
                            'nuclear_connection': True,
                            'connection_strategy': 'isolated_cluster'
                        }
                    )
                    
                    if connected:
                        connections_made += 1
                        logger.info(f"🔗 Isolated cluster: {other_name} -> ISOLATED_CLUSTER -> {hub_name}")
            
            return connections_made
            
        except Exception as e:
            logger.error(f"🔗 Isolated cluster connection failed: {e}")
            return 0
    
    async def _connect_similar_entities(self, neo4j_service, entities: List[Dict], relationship_type: str) -> int:
        """Connect entities within the same category"""
        connections_made = 0
        
        try:
            # Connect each entity to every other entity in the same category
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    entity1_id = entity1.get('id')
                    entity2_id = entity2.get('id')
                    entity1_name = entity1.get('name')
                    entity2_name = entity2.get('name')
                    
                    if entity1_id and entity2_id:
                        # Create bidirectional relationships
                        relationship_created = neo4j_service.create_relationship(
                            from_id=entity1_id,
                            to_id=entity2_id,
                            relationship_type=relationship_type,
                            properties={
                                'created_by': 'anti_silo_analysis',
                                'confidence': 0.7,
                                'reasoning': f'{entity1_name} and {entity2_name} are related concepts'
                            }
                        )
                        
                        if relationship_created:
                            connections_made += 1
                            logger.info(f"🔗 Connected {entity1_name} -> {relationship_type} -> {entity2_name}")
            
            return connections_made
            
        except Exception as e:
            logger.error(f"❌ Failed to connect similar entities: {e}")
            return 0

# Singleton service instance
_knowledge_graph_service: Optional[KnowledgeGraphExtractionService] = None

def get_knowledge_graph_service() -> KnowledgeGraphExtractionService:
    """Get or create the pure LLM knowledge graph service"""
    global _knowledge_graph_service
    if _knowledge_graph_service is None:
        _knowledge_graph_service = KnowledgeGraphExtractionService()
    return _knowledge_graph_service

async def extract_knowledge_graph(chunk: ExtractedChunk, document_id: str = None) -> GraphExtractionResult:
    """Main entry point for pure LLM knowledge graph extraction"""
    service = get_knowledge_graph_service()
    return await service.extract_from_chunk(chunk, document_id)