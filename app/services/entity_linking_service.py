"""
Cross-Document Entity Linking Service

Provides intelligent entity resolution and linking across multiple documents
to create a unified knowledge graph with reduced entity duplication and
improved connectivity between related entities.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import hashlib

from app.services.neo4j_service import get_neo4j_service
from app.services.knowledge_graph_types import ExtractedEntity
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings

logger = logging.getLogger(__name__)

@dataclass
class EntityCandidate:
    """Candidate entity for linking with similarity scores"""
    entity_id: str
    entity_name: str
    entity_type: str
    similarity_score: float
    matching_reasons: List[str]
    confidence: float
    properties: Dict[str, Any]

@dataclass
class EntityLinkingResult:
    """Result of entity linking process"""
    original_entity: ExtractedEntity
    linked_entity_id: Optional[str]
    is_new_entity: bool
    similarity_score: float
    linking_confidence: float
    alternative_candidates: List[EntityCandidate]
    reasoning: str

class EntityLinkingService:
    """Service for cross-document entity linking and resolution"""
    
    def __init__(self):
        self.neo4j_service = get_neo4j_service()
        self.config = get_knowledge_graph_settings()
        self.similarity_threshold = 0.5  # Reduced from 0.7 to allow more fuzzy matching
        
        # Enhanced entity type equivalences for comprehensive cross-type matching
        self.type_equivalences = {
            'PERSON': ['PERSON', 'EXECUTIVE', 'RESEARCHER', 'ENTREPRENEUR'],
            'ORGANIZATION': ['ORGANIZATION', 'COMPANY', 'UNIVERSITY', 'STARTUP', 'ORG'],
            'ORG': ['ORG', 'ORGANIZATION', 'COMPANY', 'TECHNOLOGY', 'STARTUP'],  # Allow ORG to match TECHNOLOGY
            'CONCEPT': ['CONCEPT', 'TECHNOLOGY', 'METHODOLOGY', 'PRODUCT'],
            'TECHNOLOGY': ['TECHNOLOGY', 'ORG', 'CONCEPT', 'PRODUCT', 'ORGANIZATION'],  # Technology can match multiple types
            'LOCATION': ['LOCATION', 'CITY', 'COUNTRY', 'FACILITY'],
            'TEMPORAL': ['TEMPORAL', 'DATE', 'TIME', 'YEAR'],
            'COMPANY': ['COMPANY', 'ORG', 'ORGANIZATION', 'TECHNOLOGY'],  # For legacy compatibility
        }
        
        # Enhanced variations and aliases with comprehensive company hierarchies and geographic relationships
        self.name_variations = {
            'company_suffixes': ['inc', 'corp', 'llc', 'ltd', 'company', 'corporation', 'bank', 'technology', 'group', 'holdings'],
            'title_prefixes': ['dr', 'prof', 'mr', 'ms', 'mrs', 'ceo', 'cto', 'cio', 'cfo', 'founder', 'president', 'director'],
            
            # Company hierarchies and subsidiaries 
            'company_hierarchies': {
                'dbs bank': ['dbs', 'development bank of singapore'],
                'ping an technology': ['ping an', 'ping an group', 'ping an insurance'],
                'ant group': ['ant', 'ant financial', 'alipay', 'ant fintech'],
                'alibaba group': ['alibaba', 'alibaba cloud'],
                'tencent holdings': ['tencent', 'tencent technology'],
                'microsoft corporation': ['microsoft', 'msft'],
                'google inc': ['google', 'alphabet'],
                'amazon.com': ['amazon', 'aws', 'amazon web services'],
                'meta platforms': ['meta', 'facebook'],
                'oracle corporation': ['oracle'],
                'ibm': ['international business machines'],
                'jpmorgan chase': ['jpmorgan', 'jp morgan'],
                'goldman sachs group': ['goldman sachs'],
                'bank of america': ['boa', 'bofa'],
                'wells fargo bank': ['wells fargo'],
                'hsbc holdings': ['hsbc'],
                'standard chartered': ['stanchart'],
                'ocbc': ['oversea-chinese banking corporation'],
                'uob': ['united overseas bank'],
            },
            
            # Geographic relationships and aliases
            'geographic_relationships': {
                'hong kong': ['hk', 'hong kong sar'],
                'singapore': ['sg', 'republic of singapore'],
                'indonesia': ['id', 'republic of indonesia'],
                'malaysia': ['my', 'malaysia federation'],
                'thailand': ['th', 'kingdom of thailand'],
                'vietnam': ['vn', 'socialist republic of vietnam'],
                'philippines': ['ph', 'republic of the philippines'],
                'china': ['prc', 'peoples republic of china', 'mainland china'],
                'india': ['bharat', 'republic of india'],
                'japan': ['jp', 'nippon'],
                'south korea': ['korea', 'republic of korea', 'rok'],
                'taiwan': ['tw', 'republic of china', 'chinese taipei'],
                'australia': ['au', 'commonwealth of australia'],
                'new zealand': ['nz', 'aotearoa'],
                'united states': ['usa', 'us', 'america', 'united states of america'],
                'united kingdom': ['uk', 'britain', 'great britain', 'england'],
                'east asia': ['east', 'asia pacific', 'eastern asia', 'far east'],
                'southeast asia': ['sea', 'asean', 'south east asia'],
                'south asia': ['indian subcontinent'],
                'middle east': ['mena', 'middle east and north africa'],
            },
            
            # Technology and database aliases
            'technology_aliases': {
                'apache kafka': ['kafka', 'apache kafka cluster'],
                'postgresql': ['postgres', 'pg', 'postgressql'],
                'mysql': ['my sql', 'mysql database'],
                'mongodb': ['mongo', 'mongo db'],
                'elasticsearch': ['elastic search', 'es', 'elastic'],
                'redis': ['redis cache', 'redis database'],
                'apache hadoop': ['hadoop', 'big data platform'],
                'apache spark': ['spark'],
                'kubernetes': ['k8s', 'container orchestration'],
                'docker': ['docker container', 'containerization'],
                'microsoft azure': ['azure', 'azure cloud'],
                'google cloud platform': ['gcp', 'google cloud'],
                'amazon web services': ['aws', 'amazon aws'],
                'temenos': ['core banking', 'banking platform'],
                'mainframe': ['legacy system', 'z/os'],
                'microservices': ['micro services'],
                'api gateway': ['gateway'],
            },
            
            # Standard abbreviations (expanded)
            'abbreviations': {
                'artificial intelligence': ['ai', 'a.i.'],
                'machine learning': ['ml', 'm.l.'],
                'natural language processing': ['nlp'],
                'computer vision': ['cv'],
                'deep learning': ['dl'],
                'application programming interface': ['api'],
                'representational state transfer': ['rest'],
                'structured query language': ['sql'],
                'chief executive officer': ['ceo'],
                'chief technology officer': ['cto'],
                'chief information officer': ['cio'],
                'chief financial officer': ['cfo'],
                'research and development': ['r&d', 'rnd'],
                'user interface': ['ui'],
                'user experience': ['ux'],
                'business intelligence': ['bi'],
                'extract transform load': ['etl'],
                'software as a service': ['saas'],
                'platform as a service': ['paas'],
                'infrastructure as a service': ['iaas'],
            }
        }
    
    async def link_entities_in_document(self, entities: List[ExtractedEntity], 
                                       document_id: str) -> List[EntityLinkingResult]:
        """Link entities from a document to existing entities in the knowledge graph"""
        if not self.neo4j_service.is_enabled():
            logger.warning("Neo4j not enabled, skipping entity linking")
            return []
        
        linking_results = []
        
        for entity in entities:
            try:
                # Find potential matches in existing graph
                candidates = await self._find_entity_candidates(entity)
                
                # Determine best match
                linking_result = await self._determine_best_match(
                    entity, candidates, document_id
                )
                
                linking_results.append(linking_result)
                
                logger.debug(f"🔗 Entity linking: {entity.canonical_form} -> {linking_result.linked_entity_id} (confidence: {linking_result.linking_confidence:.3f})")
                
            except Exception as e:
                logger.error(f"Entity linking failed for {entity.canonical_form}: {e}")
                # Create fallback result
                linking_results.append(EntityLinkingResult(
                    original_entity=entity,
                    linked_entity_id=None,
                    is_new_entity=True,
                    similarity_score=0.0,
                    linking_confidence=0.0,
                    alternative_candidates=[],
                    reasoning=f"Linking failed: {str(e)}"
                ))
        
        # Log linking statistics
        linked_count = sum(1 for r in linking_results if not r.is_new_entity)
        logger.info(f"🔗 ENTITY LINKING: {linked_count}/{len(entities)} entities linked to existing entities")
        
        return linking_results
    
    async def _find_entity_candidates(self, entity: ExtractedEntity) -> List[EntityCandidate]:
        """Find potential matching entities in the existing knowledge graph"""
        candidates = []
        
        try:
            # Get compatible entity types
            compatible_types = self._get_compatible_types(entity.label)
            
            # Query for potential matches using multiple strategies
            
            # Strategy 1: Exact name match
            exact_matches = await self._find_exact_name_matches(entity, compatible_types)
            candidates.extend(exact_matches)
            
            # Strategy 2: Fuzzy name matching
            fuzzy_matches = await self._find_fuzzy_name_matches(entity, compatible_types)
            candidates.extend(fuzzy_matches)
            
            # Strategy 3: Alias and variation matching
            alias_matches = await self._find_alias_matches(entity, compatible_types)
            candidates.extend(alias_matches)
            
            # Strategy 4: Partial name matching
            partial_matches = await self._find_partial_name_matches(entity, compatible_types)
            candidates.extend(partial_matches)
            
            # Remove duplicates and sort by similarity
            candidates = self._deduplicate_candidates(candidates)
            candidates.sort(key=lambda c: c.similarity_score, reverse=True)
            
            # Limit to top candidates
            return candidates[:10]
            
        except Exception as e:
            logger.error(f"Failed to find entity candidates for {entity.canonical_form}: {e}")
            return []
    
    async def _find_exact_name_matches(self, entity: ExtractedEntity, 
                                     compatible_types: List[str]) -> List[EntityCandidate]:
        """Find entities with exact name matches"""
        candidates = []
        
        try:
            # Enhanced query for exact matches with cross-type support
            query = """
            MATCH (n)
            WHERE n.name = $name AND (labels(n)[0] IN $types OR n.type IN $types)
            RETURN n.id as id, n.name as name, 
                   COALESCE(n.type, labels(n)[0]) as type, 
                   COALESCE(n.confidence, 0.0) as confidence,
                   labels(n)[0] as label, properties(n) as properties
            """
            
            results = self.neo4j_service.execute_cypher(query, {
                'name': entity.canonical_form,
                'types': compatible_types
            })
            
            for result in results:
                candidate = EntityCandidate(
                    entity_id=result['id'],
                    entity_name=result['name'],
                    entity_type=result['label'],
                    similarity_score=1.0,  # Exact match
                    matching_reasons=['exact_name_match'],
                    confidence=result.get('confidence', 0.0),
                    properties=result.get('properties', {})
                )
                candidates.append(candidate)
                
        except Exception as e:
            logger.error(f"Exact name matching failed: {e}")
        
        return candidates
    
    async def _find_fuzzy_name_matches(self, entity: ExtractedEntity,
                                     compatible_types: List[str]) -> List[EntityCandidate]:
        """Find entities with similar names using fuzzy matching"""
        candidates = []
        
        try:
            # Enhanced query for fuzzy matches with cross-type support  
            query = """
            MATCH (n)
            WHERE (labels(n)[0] IN $types OR n.type IN $types)
            RETURN n.id as id, n.name as name, 
                   COALESCE(n.type, labels(n)[0]) as type, 
                   COALESCE(n.confidence, 0.0) as confidence,
                   labels(n)[0] as label, properties(n) as properties
            LIMIT 1000
            """
            
            results = self.neo4j_service.execute_cypher(query, {
                'types': compatible_types
            })
            
            entity_name_clean = self._clean_entity_name(entity.canonical_form)
            
            for result in results:
                existing_name_clean = self._clean_entity_name(result['name'])
                
                # Calculate similarity
                similarity = SequenceMatcher(None, entity_name_clean, existing_name_clean).ratio()
                
                if similarity >= self.similarity_threshold:
                    reasons = ['fuzzy_name_match']
                    if similarity >= 0.9:
                        reasons.append('high_similarity')
                    
                    candidate = EntityCandidate(
                        entity_id=result['id'],
                        entity_name=result['name'],
                        entity_type=result['label'],
                        similarity_score=similarity,
                        matching_reasons=reasons,
                        confidence=result.get('confidence', 0.0),
                        properties=result.get('properties', {})
                    )
                    candidates.append(candidate)
                    
        except Exception as e:
            logger.error(f"Fuzzy name matching failed: {e}")
        
        return candidates
    
    async def _find_alias_matches(self, entity: ExtractedEntity,
                                compatible_types: List[str]) -> List[EntityCandidate]:
        """Find entities using alias and variation matching"""
        candidates = []
        
        try:
            # Generate possible aliases for the entity
            aliases = self._generate_entity_aliases(entity.canonical_form, entity.label)
            
            if not aliases:
                return candidates
            
            # Enhanced query for alias matches with cross-type support
            query = """
            MATCH (n)
            WHERE n.name IN $aliases AND (labels(n)[0] IN $types OR n.type IN $types)
            RETURN n.id as id, n.name as name, 
                   COALESCE(n.type, labels(n)[0]) as type, 
                   COALESCE(n.confidence, 0.0) as confidence,
                   labels(n)[0] as label, properties(n) as properties
            """
            
            results = self.neo4j_service.execute_cypher(query, {
                'aliases': aliases,
                'types': compatible_types
            })
            
            for result in results:
                # Calculate similarity based on alias match
                similarity = 0.8  # High but not perfect for alias matches
                
                candidate = EntityCandidate(
                    entity_id=result['id'],
                    entity_name=result['name'],
                    entity_type=result['label'],
                    similarity_score=similarity,
                    matching_reasons=['alias_match'],
                    confidence=result.get('confidence', 0.0),
                    properties=result.get('properties', {})
                )
                candidates.append(candidate)
                
        except Exception as e:
            logger.error(f"Alias matching failed: {e}")
        
        return candidates
    
    async def _find_partial_name_matches(self, entity: ExtractedEntity,
                                       compatible_types: List[str]) -> List[EntityCandidate]:
        """Find entities with partial name matches (for long entity names)"""
        candidates = []
        
        try:
            entity_words = entity.canonical_form.lower().split()
            
            # Only do partial matching for entities with multiple words
            if len(entity_words) < 2:
                return candidates
            
            # Generate partial name queries
            partial_names = []
            
            # Try combinations of words
            if len(entity_words) >= 2:
                partial_names.append(' '.join(entity_words[:2]))  # First two words
                partial_names.append(' '.join(entity_words[-2:]))  # Last two words
            
            if len(entity_words) >= 3:
                partial_names.append(' '.join(entity_words[1:-1]))  # Middle words
            
            # Enhanced query for partial matches with cross-type support
            for partial_name in partial_names:
                query = """
                MATCH (n)
                WHERE toLower(n.name) CONTAINS $partial_name AND (labels(n)[0] IN $types OR n.type IN $types)
                RETURN n.id as id, n.name as name, 
                       COALESCE(n.type, labels(n)[0]) as type, 
                       COALESCE(n.confidence, 0.0) as confidence,
                       labels(n)[0] as label, properties(n) as properties
                """
                
                results = self.neo4j_service.execute_cypher(query, {
                    'partial_name': partial_name.lower(),
                    'types': compatible_types
                })
                
                for result in results:
                    # Calculate similarity for partial match
                    similarity = self._calculate_partial_similarity(
                        entity.canonical_form, result['name']
                    )
                    
                    if similarity >= 0.6:  # Lower threshold for partial matches
                        candidate = EntityCandidate(
                            entity_id=result['id'],
                            entity_name=result['name'],
                            entity_type=result['label'],
                            similarity_score=similarity,
                            matching_reasons=['partial_name_match'],
                            confidence=result.get('confidence', 0.0),
                            properties=result.get('properties', {})
                        )
                        candidates.append(candidate)
                        
        except Exception as e:
            logger.error(f"Partial name matching failed: {e}")
        
        return candidates
    
    def _get_compatible_types(self, entity_type: str) -> List[str]:
        """Get list of entity types compatible for linking"""
        return self.type_equivalences.get(entity_type, [entity_type])
    
    def _clean_entity_name(self, name: str) -> str:
        """Clean entity name for better matching"""
        name = name.lower().strip()
        
        # Remove common suffixes and prefixes
        for suffix in self.name_variations['company_suffixes']:
            if name.endswith(f' {suffix}'):
                name = name[:-len(suffix)-1].strip()
        
        for prefix in self.name_variations['title_prefixes']:
            if name.startswith(f'{prefix} '):
                name = name[len(prefix)+1:].strip()
        
        # Remove punctuation
        name = re.sub(r'[^\w\s]', '', name)
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        return name
    
    def _generate_entity_aliases(self, name: str, entity_type: str) -> List[str]:
        """Generate comprehensive aliases for an entity including hierarchies and geographic relationships"""
        aliases = []
        name_lower = name.lower()
        
        # Check company hierarchies and subsidiaries
        for parent_company, subsidiaries in self.name_variations['company_hierarchies'].items():
            if parent_company in name_lower:
                aliases.extend(subsidiaries)
            for subsidiary in subsidiaries:
                if subsidiary in name_lower:
                    aliases.append(parent_company)
                    aliases.extend([s for s in subsidiaries if s != subsidiary])
        
        # Check geographic relationships
        for location, aliases_list in self.name_variations['geographic_relationships'].items():
            if location in name_lower:
                aliases.extend(aliases_list)
            for alias in aliases_list:
                if alias in name_lower:
                    aliases.append(location)
                    aliases.extend([a for a in aliases_list if a != alias])
        
        # Check technology aliases
        for tech_name, tech_aliases in self.name_variations['technology_aliases'].items():
            if tech_name in name_lower:
                aliases.extend(tech_aliases)
            for alias in tech_aliases:
                if alias in name_lower:
                    aliases.append(tech_name)
                    aliases.extend([a for a in tech_aliases if a != alias])
        
        # Check standard abbreviation mappings
        for full_name, abbrevs in self.name_variations['abbreviations'].items():
            if full_name in name_lower:
                for abbrev in abbrevs:
                    aliases.append(name.replace(full_name, abbrev))
                    aliases.append(name.replace(full_name, abbrev.upper()))
            
            for abbrev in abbrevs:
                if abbrev in name_lower:
                    aliases.append(name.replace(abbrev, full_name))
        
        # For organizations, try without suffixes
        if entity_type in ['ORGANIZATION', 'COMPANY', 'ORG']:
            for suffix in self.name_variations['company_suffixes']:
                if name_lower.endswith(f' {suffix}'):
                    base_name = name[:-len(suffix)-1].strip()
                    aliases.append(base_name)
                    # Also check if base name has hierarchies
                    base_lower = base_name.lower()
                    for parent, subs in self.name_variations['company_hierarchies'].items():
                        if parent == base_lower:
                            aliases.extend(subs)
                        elif base_lower in subs:
                            aliases.append(parent)
        
        # For people, try without titles
        if entity_type == 'PERSON':
            for prefix in self.name_variations['title_prefixes']:
                if name_lower.startswith(f'{prefix} '):
                    aliases.append(name[len(prefix)+1:].strip())
        
        # For locations, add variations with common geographic suffixes
        if entity_type in ['LOCATION', 'CITY', 'COUNTRY']:
            geographic_suffixes = ['region', 'area', 'province', 'state', 'territory']
            for suffix in geographic_suffixes:
                if name_lower.endswith(f' {suffix}'):
                    aliases.append(name[:-len(suffix)-1].strip())
        
        # Clean and deduplicate aliases
        aliases = [alias.strip() for alias in aliases if alias.strip() and len(alias.strip()) > 1]
        return list(set(aliases))  # Remove duplicates
    
    def _calculate_partial_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity score for partial matches"""
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        
        # Jaccard similarity for word sets
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        
        # Also consider sequence similarity
        sequence_similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
        
        # Weighted combination
        return 0.6 * jaccard_similarity + 0.4 * sequence_similarity
    
    def _deduplicate_candidates(self, candidates: List[EntityCandidate]) -> List[EntityCandidate]:
        """Remove duplicate candidates keeping the highest scoring ones"""
        seen_ids = set()
        deduplicated = []
        
        for candidate in sorted(candidates, key=lambda c: c.similarity_score, reverse=True):
            if candidate.entity_id not in seen_ids:
                seen_ids.add(candidate.entity_id)
                deduplicated.append(candidate)
        
        return deduplicated
    
    async def _determine_best_match(self, entity: ExtractedEntity, 
                                  candidates: List[EntityCandidate],
                                  document_id: str) -> EntityLinkingResult:
        """Determine the best matching entity from candidates"""
        
        if not candidates:
            return EntityLinkingResult(
                original_entity=entity,
                linked_entity_id=None,
                is_new_entity=True,
                similarity_score=0.0,
                linking_confidence=0.0,
                alternative_candidates=[],
                reasoning="No matching candidates found"
            )
        
        best_candidate = candidates[0]  # Highest scoring candidate
        
        # Additional validation for the best candidate
        linking_confidence = self._calculate_linking_confidence(
            entity, best_candidate, candidates
        )
        
        # Decide whether to link or create new entity with smart thresholds
        confidence_threshold = self._get_confidence_threshold_for_entity_type(entity.label)
        should_link = (
            best_candidate.similarity_score >= self.similarity_threshold and
            linking_confidence >= confidence_threshold
        )
        
        if should_link:
            return EntityLinkingResult(
                original_entity=entity,
                linked_entity_id=best_candidate.entity_id,
                is_new_entity=False,
                similarity_score=best_candidate.similarity_score,
                linking_confidence=linking_confidence,
                alternative_candidates=candidates[1:5],  # Top alternatives
                reasoning=f"Linked via {', '.join(best_candidate.matching_reasons)}"
            )
        else:
            return EntityLinkingResult(
                original_entity=entity,
                linked_entity_id=None,
                is_new_entity=True,
                similarity_score=best_candidate.similarity_score,
                linking_confidence=linking_confidence,
                alternative_candidates=candidates[:5],
                reasoning=f"Similarity too low ({best_candidate.similarity_score:.3f}) or confidence too low ({linking_confidence:.3f})"
            )
    
    def _get_confidence_threshold_for_entity_type(self, entity_type: str) -> float:
        """Get confidence threshold based on entity type - different types need different confidence levels"""
        thresholds = {
            'TECHNOLOGY': 0.4,    # Lower threshold for technology entities (acronyms, variations)
            'LOCATION': 0.5,      # Medium threshold for geographic entities
            'ORG': 0.5,          # Medium threshold for organizations
            'ORGANIZATION': 0.5,  # Medium threshold for organizations
            'COMPANY': 0.5,       # Medium threshold for companies
            'PERSON': 0.6,        # Higher threshold for people (avoid false positives)
            'TEMPORAL': 0.3,      # Very low threshold for dates/times (usually clear)
            'CONCEPT': 0.5,       # Medium threshold for concepts
        }
        return thresholds.get(entity_type, 0.5)  # Default to 0.5
    
    def _calculate_linking_confidence(self, entity: ExtractedEntity,
                                    best_candidate: EntityCandidate,
                                    all_candidates: List[EntityCandidate]) -> float:
        """Calculate confidence score for entity linking decision"""
        confidence_factors = []
        
        # Factor 1: Similarity score
        confidence_factors.append(best_candidate.similarity_score)
        
        # Factor 2: Gap between best and second-best candidate
        if len(all_candidates) > 1:
            gap = best_candidate.similarity_score - all_candidates[1].similarity_score
            confidence_factors.append(min(gap * 2, 1.0))  # Boost confidence if clear winner
        else:
            confidence_factors.append(0.8)  # High confidence if only one candidate
        
        # Factor 3: Entity confidence scores
        avg_confidence = (entity.confidence + best_candidate.confidence) / 2
        confidence_factors.append(avg_confidence)
        
        # Factor 4: Matching method quality
        method_scores = {
            'exact_name_match': 1.0,
            'high_similarity': 0.9,
            'fuzzy_name_match': 0.7,
            'alias_match': 0.8,
            'partial_name_match': 0.6
        }
        
        method_score = max(
            method_scores.get(reason, 0.5) 
            for reason in best_candidate.matching_reasons
        )
        confidence_factors.append(method_score)
        
        # Calculate weighted average
        weights = [0.3, 0.2, 0.2, 0.3]  # Adjust based on importance
        return sum(w * f for w, f in zip(weights, confidence_factors))

# Singleton instance
_entity_linking_service: Optional[EntityLinkingService] = None

def get_entity_linking_service() -> EntityLinkingService:
    """Get or create entity linking service singleton"""
    global _entity_linking_service
    if _entity_linking_service is None:
        _entity_linking_service = EntityLinkingService()
    return _entity_linking_service