"""
LLM-Enhanced Knowledge Graph Extractor

Provides LLM-powered entity and relationship extraction with context understanding,
domain awareness, and confidence scoring for improved knowledge graph quality.
"""

import asyncio
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings
from app.core.llm_settings_cache import get_llm_settings
from app.services.knowledge_graph_types import ExtractedEntity, ExtractedRelationship
from app.services.dynamic_schema_manager import dynamic_schema_manager

logger = logging.getLogger(__name__)

@dataclass
class LLMExtractionResult:
    """Result of LLM-based entity and relationship extraction"""
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    confidence_score: float
    reasoning: str
    processing_time_ms: float
    llm_model_used: str
    extraction_metadata: Dict[str, Any]

class LLMKnowledgeExtractor:
    """Enhanced LLM-powered knowledge extraction service with multi-pass extraction and business intelligence"""
    
    def __init__(self):
        self.kg_settings = get_knowledge_graph_settings()
        self.llm_settings = get_llm_settings()
        self.model_config = self._get_extraction_model_config()
        
        # Enhanced hierarchical entity types with business intelligence focus
        self.hierarchical_entity_types = {
            # People and Roles
            'PERSON', 'EXECUTIVE', 'RESEARCHER', 'ENTREPRENEUR', 'ANALYST', 'MANAGER', 'DIRECTOR',
            'CEO', 'CTO', 'CIO', 'CFO', 'FOUNDER', 'PRESIDENT', 'EMPLOYEE', 'CONSULTANT',
            'STAKEHOLDER', 'CUSTOMER', 'PARTNER', 'INVESTOR', 'SHAREHOLDER',
            
            # Organizations and Entities
            'ORGANIZATION', 'ORG', 'COMPANY', 'CORPORATION', 'BANK', 'FINTECH', 'STARTUP',
            'UNIVERSITY', 'RESEARCH_INSTITUTE', 'GOVERNMENT', 'AGENCY', 'DEPARTMENT',
            'SUBSIDIARY', 'DIVISION', 'UNIT', 'TEAM', 'COMMITTEE', 'BOARD',
            'ASSOCIATION', 'FOUNDATION', 'NGO', 'CONSORTIUM', 'ALLIANCE',
            
            # Business and Financial
            'PRODUCT', 'SERVICE', 'PLATFORM', 'SOLUTION', 'OFFERING', 'PORTFOLIO',
            'REVENUE', 'COST', 'PROFIT', 'LOSS', 'INVESTMENT', 'FUNDING', 'BUDGET',
            'MARKET', 'SEGMENT', 'INDUSTRY', 'SECTOR', 'VERTICAL', 'ECONOMY',
            'BUSINESS_MODEL', 'STRATEGY', 'INITIATIVE', 'PROGRAM', 'PROJECT',
            'KPI', 'METRIC', 'TARGET', 'GOAL', 'OBJECTIVE', 'MILESTONE',
            
            # Technology and Systems
            'TECHNOLOGY', 'SYSTEM', 'SOFTWARE', 'HARDWARE', 'INFRASTRUCTURE',
            'DATABASE', 'APPLICATION', 'API', 'FRAMEWORK', 'LIBRARY', 'TOOL',
            'METHODOLOGY', 'PROTOCOL', 'STANDARD', 'SPECIFICATION', 'ALGORITHM',
            'MODEL', 'ARCHITECTURE', 'DESIGN', 'IMPLEMENTATION',
            
            # Geographic and Temporal
            'LOCATION', 'CITY', 'COUNTRY', 'REGION', 'CONTINENT', 'FACILITY',
            'OFFICE', 'HEADQUARTERS', 'BRANCH', 'MARKET_REGION',
            'TEMPORAL', 'DATE', 'TIME', 'YEAR', 'QUARTER', 'MONTH', 'PERIOD',
            'EVENT', 'MEETING', 'CONFERENCE', 'ANNOUNCEMENT', 'LAUNCH',
            
            # Regulatory and Compliance
            'REGULATION', 'POLICY', 'LAW', 'COMPLIANCE', 'STANDARD', 'REQUIREMENT',
            'RISK', 'THREAT', 'OPPORTUNITY', 'CHALLENGE', 'ISSUE', 'CONCERN',
            
            # General Concepts
            'CONCEPT', 'PRINCIPLE', 'APPROACH', 'TREND', 'PATTERN', 'PHENOMENON',
            'FACTOR', 'ELEMENT', 'COMPONENT', 'ASPECT', 'DIMENSION', 'ATTRIBUTE'
        }
        
        # Enhanced relationship taxonomy with comprehensive business relationships
        self.relationship_taxonomy = {
            'organizational': {
                'RELATED_TO', 'PART_OF', 'CONTAINS', 'MEMBER_OF', 'OWNS', 'OWNED_BY',
                'MANAGES', 'MANAGED_BY', 'REPORTS_TO', 'SUPERVISES', 'WORKS_FOR', 
                'EMPLOYS', 'SUBSIDIARY_OF', 'PARENT_OF', 'DIVISION_OF', 'UNIT_OF',
                'DEPARTMENT_OF', 'TEAM_MEMBER', 'BOARD_MEMBER', 'EXECUTIVE_OF',
                'FOUNDED_BY', 'LED_BY', 'HEADED_BY', 'CHAIRED_BY'
            },
            'business_operational': {
                'PARTNERS_WITH', 'COMPETES_WITH', 'COLLABORATES_WITH', 'SUPPLIES',
                'SUPPLIED_BY', 'SERVES', 'SERVED_BY', 'CLIENTS_OF', 'VENDOR_TO',
                'ACQUIRES', 'ACQUIRED_BY', 'MERGES_WITH', 'JOINT_VENTURE',
                'STRATEGIC_ALLIANCE', 'DISTRIBUTES', 'DISTRIBUTED_BY',
                'LICENSES', 'LICENSED_BY', 'FRANCHISES', 'FRANCHISED_BY'
            },
            'financial': {
                'INVESTS_IN', 'INVESTED_BY', 'FUNDS', 'FUNDED_BY', 'SPONSORS',
                'SPONSORED_BY', 'FINANCES', 'FINANCED_BY', 'LOANS_TO', 'BORROWS_FROM',
                'INSURES', 'INSURED_BY', 'UNDERWRITES', 'UNDERWRITTEN_BY',
                'REVENUE_FROM', 'COST_TO', 'PROFITS_FROM'
            },
            'product_service': {
                'OFFERS', 'OFFERED_BY', 'PROVIDES', 'PROVIDED_BY', 'DEVELOPS',
                'DEVELOPED_BY', 'MANUFACTURES', 'MANUFACTURED_BY', 'SELLS',
                'SOLD_BY', 'MARKETS', 'MARKETED_BY', 'SUPPORTS', 'SUPPORTED_BY',
                'MAINTAINS', 'MAINTAINED_BY', 'UPGRADES', 'UPGRADED_BY'
            },
            'geographic_operational': {
                'LOCATED_IN', 'HOSTS', 'OPERATES_IN', 'BASED_IN', 'HEADQUARTERED_IN',
                'OFFICE_IN', 'BRANCH_IN', 'FACILITY_IN', 'PRESENCE_IN',
                'SERVES_REGION', 'COVERS_MARKET', 'EXPANDS_TO', 'EXITS_FROM'
            },
            'temporal_business': {
                'FOUNDED_IN', 'ESTABLISHED_IN', 'LAUNCHED_IN', 'ANNOUNCED_IN',
                'OCCURRED_IN', 'DURING', 'BEFORE', 'AFTER', 'CONCURRENT_WITH',
                'PLANNED_FOR', 'SCHEDULED_FOR', 'DEADLINE_BY', 'EXPIRES_IN'
            },
            'strategic': {
                'TARGETS', 'TARGETED_BY', 'FOCUSES_ON', 'PRIORITIZES', 'EMPHASIZES',
                'STRATEGY_FOR', 'INITIATIVE_FOR', 'GOAL_OF', 'OBJECTIVE_OF',
                'ADDRESSES', 'ADDRESSES_BY', 'SOLVES', 'SOLVED_BY', 'IMPROVES',
                'IMPROVED_BY', 'OPTIMIZES', 'OPTIMIZED_BY', 'TRANSFORMS',
                'TRANSFORMED_BY', 'DISRUPTS', 'DISRUPTED_BY'
            },
            'technological': {
                'USES', 'USED_BY', 'IMPLEMENTS', 'IMPLEMENTED_BY', 'INTEGRATES_WITH',
                'INTEGRATED_BY', 'ENABLES', 'ENABLED_BY', 'POWERS', 'POWERED_BY',
                'BUILT_ON', 'BUILDS', 'COMPATIBLE_WITH', 'REPLACES', 'REPLACED_BY',
                'UPGRADES', 'UPGRADED_BY', 'MIGRATES_TO', 'MIGRATES_FROM',
                'CONNECTS_TO', 'CONNECTED_BY', 'INTERFACES_WITH'
            },
            'regulatory_compliance': {
                'COMPLIES_WITH', 'REGULATED_BY', 'GOVERNED_BY', 'SUBJECT_TO',
                'REQUIRES', 'REQUIRED_BY', 'MANDATES', 'MANDATED_BY',
                'CERTIFIES', 'CERTIFIED_BY', 'AUDITS', 'AUDITED_BY',
                'MONITORS', 'MONITORED_BY', 'OVERSEES', 'OVERSEEN_BY'
            },
            'market_competitive': {
                'COMPETES_IN', 'MARKET_LEADER_IN', 'MARKET_SHARE_IN', 'DOMINATES',
                'CHALLENGED_BY', 'THREATENS', 'THREATENED_BY', 'DISRUPTS',
                'DISRUPTED_BY', 'INNOVATES_IN', 'FIRST_MOVER_IN', 'FOLLOWER_IN'
            },
            'knowledge_information': {
                'INFLUENCES', 'INFLUENCED_BY', 'INFORMS', 'INFORMED_BY',
                'BASED_ON', 'BASIS_FOR', 'DERIVED_FROM', 'DERIVES', 'REFERENCES',
                'REFERENCED_BY', 'CITES', 'CITED_BY', 'MENTIONS', 'MENTIONED_BY',
                'DISCUSSES', 'DISCUSSED_IN', 'ANALYZES', 'ANALYZED_BY'
            }
        }
        
        # Multi-pass extraction configuration
        self.extraction_passes = {
            'core_entities': {
                'focus': ['ORGANIZATION', 'PERSON', 'TECHNOLOGY', 'LOCATION', 'PRODUCT'],
                'min_confidence': 0.6,
                'aggressive_matching': True
            },
            'business_concepts': {
                'focus': ['STRATEGY', 'INITIATIVE', 'MARKET', 'REVENUE', 'INVESTMENT', 'RISK'],
                'min_confidence': 0.5,
                'contextual_enhancement': True
            },
            'relationships_deep': {
                'focus': 'relationships',
                'inference_enabled': True,
                'cross_reference': True,
                'min_confidence': 0.4
            },
            'temporal_causal': {
                'focus': ['temporal_business', 'strategic'],
                'causal_inference': True,
                'timeline_analysis': True
            }
        }
    
    def _get_extraction_model_config(self) -> Dict[str, Any]:
        """Get LLM configuration optimized for knowledge extraction"""
        # Get model config from the nested model_config section
        model_config = self.kg_settings.get('model_config', {})
        
        config = {
            'model': model_config.get('model', 'qwen3:30b-a3b-q4_K_M'),
            'temperature': model_config.get('temperature', 0.1),
            'max_tokens': model_config.get('max_tokens', 4096),
            'model_server': model_config.get('model_server', 'http://host.docker.internal:11434')
        }
        return config
    
    async def extract_knowledge(self, text: str, context: Optional[Dict[str, Any]] = None) -> LLMExtractionResult:
        """Wrapper method to maintain compatibility with existing code"""
        return await self.extract_with_llm(text, context)
    
    async def extract_with_llm(self, text: str, context: Optional[Dict[str, Any]] = None,
                              domain_hints: Optional[List[str]] = None) -> LLMExtractionResult:
        """Enhanced multi-pass extraction for 10x better business document analysis"""
        start_time = datetime.now()
        
        try:
            # Determine if this is a business document requiring enhanced extraction
            is_business_document = self._is_business_document(text, domain_hints)
            
            if is_business_document and len(text) > 1000:  # Use multi-pass for substantial business documents
                logger.info("🎯 Business document detected - using enhanced multi-pass extraction")
                return await self._multi_pass_business_extraction(text, context, domain_hints, start_time)
            else:
                # Standard single-pass extraction for smaller or non-business documents
                return await self._single_pass_extraction(text, context, domain_hints, start_time)
                
        except Exception as e:
            logger.error(f"Enhanced LLM knowledge extraction failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return LLMExtractionResult(
                entities=[],
                relationships=[],
                confidence_score=0.0,
                reasoning=f"Enhanced extraction failed: {str(e)}",
                processing_time_ms=processing_time,
                llm_model_used=self.model_config['model'],
                extraction_metadata={'error': str(e), 'extraction_type': 'failed'}
            )
    
    def _is_business_document(self, text: str, domain_hints: Optional[List[str]]) -> bool:
        """Detect if document contains business/strategic content requiring enhanced extraction"""
        business_indicators = [
            'strategy', 'business', 'revenue', 'investment', 'market', 'customer', 'product',
            'technology', 'digital transformation', 'innovation', 'partnership', 'acquisition',
            'growth', 'expansion', 'competitive', 'industry', 'sector', 'financial', 'banking',
            'fintech', 'platform', 'solution', 'service', 'enterprise', 'corporate', 'executive',
            'management', 'governance', 'compliance', 'risk', 'opportunity', 'initiative',
            'program', 'project', 'roadmap', 'vision', 'mission', 'objective', 'goal', 'target',
            'kpi', 'metric', 'performance', 'efficiency', 'optimization', 'transformation'
        ]
        
        if domain_hints and any('business' in hint.lower() or 'strategy' in hint.lower() for hint in domain_hints):
            return True
            
        text_lower = text.lower()
        business_score = sum(1 for indicator in business_indicators if indicator in text_lower)
        
        return business_score >= 5  # Threshold for business document detection
    
    async def _multi_pass_business_extraction(self, text: str, context: Optional[Dict[str, Any]], 
                                            domain_hints: Optional[List[str]], start_time: datetime) -> LLMExtractionResult:
        """Multi-pass extraction optimized for business documents with 10x better yield"""
        all_entities = []
        all_relationships = []
        pass_metadata = {}
        
        logger.info(f"🔄 Starting 4-pass business extraction on {len(text):,} character text")
        
        # Pass 1: Core Business Entities (Organizations, People, Technologies, Products)
        logger.info("🎯 Pass 1: Core business entities extraction")
        pass1_result = await self._extraction_pass_core_entities(text, context, domain_hints)
        all_entities.extend(pass1_result['entities'])
        all_relationships.extend(pass1_result['relationships'])
        pass_metadata['pass1_core_entities'] = {
            'entities_found': len(pass1_result['entities']),
            'relationships_found': len(pass1_result['relationships']),
            'processing_time_ms': pass1_result['processing_time_ms']
        }
        
        # Pass 2: Business Concepts and Strategic Elements
        logger.info("📈 Pass 2: Business concepts and strategic elements")
        pass2_result = await self._extraction_pass_business_concepts(text, all_entities, context, domain_hints)
        all_entities.extend(pass2_result['entities'])
        all_relationships.extend(pass2_result['relationships'])
        pass_metadata['pass2_business_concepts'] = {
            'entities_found': len(pass2_result['entities']),
            'relationships_found': len(pass2_result['relationships']),
            'processing_time_ms': pass2_result['processing_time_ms']
        }
        
        # Pass 3: Deep Relationship Analysis and Inference
        logger.info("🔗 Pass 3: Deep relationship analysis and inference")
        pass3_result = await self._extraction_pass_deep_relationships(text, all_entities, context)
        all_relationships.extend(pass3_result['relationships'])
        pass_metadata['pass3_deep_relationships'] = {
            'relationships_found': len(pass3_result['relationships']),
            'processing_time_ms': pass3_result['processing_time_ms']
        }
        
        # Pass 4: Temporal and Causal Analysis
        logger.info("⏰ Pass 4: Temporal and causal relationship analysis")
        pass4_result = await self._extraction_pass_temporal_causal(text, all_entities, all_relationships, context)
        all_entities.extend(pass4_result['entities'])
        all_relationships.extend(pass4_result['relationships'])
        pass_metadata['pass4_temporal_causal'] = {
            'entities_found': len(pass4_result['entities']),
            'relationships_found': len(pass4_result['relationships']),
            'processing_time_ms': pass4_result['processing_time_ms']
        }
        
        # Consolidation: Remove duplicates and enhance confidence
        logger.info("🔧 Consolidating and deduplicating multi-pass results")
        consolidated_entities = self._consolidate_entities(all_entities)
        consolidated_relationships = self._consolidate_relationships(all_relationships, consolidated_entities)
        
        # Final enhancement: Co-occurrence relationships for high-frequency entities
        logger.info("💡 Generating co-occurrence relationships for frequently mentioned entities")
        cooccurrence_relationships = await self._generate_cooccurrence_relationships(
            text, consolidated_entities, consolidated_relationships
        )
        consolidated_relationships.extend(cooccurrence_relationships)
        
        # Calculate processing time and metrics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        confidence_score = self._calculate_multi_pass_confidence(consolidated_entities, consolidated_relationships, pass_metadata)
        
        total_entities = len(consolidated_entities)
        total_relationships = len(consolidated_relationships)
        
        logger.info(f"✅ Multi-pass extraction complete: {total_entities} entities, {total_relationships} relationships")
        logger.info(f"   Entity density: {total_entities / (len(text) / 1000):.1f} entities per 1K chars")
        logger.info(f"   Relationship density: {total_relationships / max(total_entities, 1):.1f} relationships per entity")
        
        return LLMExtractionResult(
            entities=consolidated_entities,
            relationships=consolidated_relationships,
            confidence_score=confidence_score,
            reasoning=f"Multi-pass business extraction completed: 4 passes with {total_entities} entities and {total_relationships} relationships",
            processing_time_ms=processing_time,
            llm_model_used=self.model_config['model'],
            extraction_metadata={
                'extraction_type': 'multi_pass_business',
                'passes_completed': 4,
                'text_length': len(text),
                'business_document': True,
                'entity_density_per_1k': total_entities / (len(text) / 1000),
                'relationship_density_per_entity': total_relationships / max(total_entities, 1),
                'pass_breakdown': pass_metadata,
                'domain_hints': domain_hints or [],
                'context_provided': context is not None
            }
        )
    
    async def _single_pass_extraction(self, text: str, context: Optional[Dict[str, Any]], 
                                    domain_hints: Optional[List[str]], start_time: datetime) -> LLMExtractionResult:
        """Standard single-pass extraction for non-business or smaller documents"""
        # Build sophisticated extraction prompt
        extraction_prompt = await self._build_extraction_prompt(text, context, domain_hints)
        
        # Call LLM for extraction
        llm_response = await self._call_llm_for_extraction(extraction_prompt)
        
        # Parse and validate LLM response
        parsed_result = self._parse_llm_response(llm_response)
        
        # Extract discoveries from parsed_result
        discoveries = parsed_result.get('discoveries', {})
        
        # Process discoveries asynchronously
        if discoveries:
            asyncio.create_task(self._process_discoveries_async(discoveries))
        
        # Enhance entities with hierarchical classification
        enhanced_entities = self._enhance_entities_with_hierarchy(parsed_result.get('entities', []))
        
        # Validate and score relationships
        validated_relationships = self._validate_and_score_relationships(
            parsed_result.get('relationships', []), enhanced_entities
        )
        
        # Fallback strategy: If no relationships found but entities exist, try to infer relationships
        if len(validated_relationships) == 0 and len(enhanced_entities) > 1:
            logger.info("🔄 No relationships found, attempting fallback relationship inference...")
            fallback_relationships = await self._infer_fallback_relationships(enhanced_entities, text)
            validated_relationships.extend(fallback_relationships)
            logger.info(f"Fallback inference created {len(fallback_relationships)} relationships")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Generate overall confidence score
        confidence_score = self._calculate_overall_confidence(
            enhanced_entities, validated_relationships
        )
        
        return LLMExtractionResult(
            entities=enhanced_entities,
            relationships=validated_relationships,
            confidence_score=confidence_score,
            reasoning=parsed_result.get('reasoning', 'Single-pass extraction completed'),
            processing_time_ms=processing_time,
            llm_model_used=self.model_config['model'],
            extraction_metadata={
                'extraction_type': 'single_pass',
                'domain_hints': domain_hints or [],
                'context_provided': context is not None,
                'text_length': len(text),
                'total_extractions': len(enhanced_entities) + len(validated_relationships),
                'discoveries_made': bool(discoveries),
                'new_entity_types': len(discoveries.get('new_entity_types', [])),
                'new_relationship_types': len(discoveries.get('new_relationship_types', [])),
                'discovery_details': {
                    'entity_types_discovered': discoveries.get('new_entity_types', []),
                    'relationship_types_discovered': discoveries.get('new_relationship_types', []),
                    'discovery_processing_initiated': bool(discoveries)
                }
            }
        )
            
        except Exception as e:
            logger.error(f"LLM knowledge extraction failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return LLMExtractionResult(
                entities=[],
                relationships=[],
                confidence_score=0.0,
                reasoning=f"Extraction failed: {str(e)}",
                processing_time_ms=processing_time,
                llm_model_used=self.model_config['model'],
                extraction_metadata={'error': str(e)}
            )
    
    async def _build_extraction_prompt(self, text: str, context: Optional[Dict[str, Any]] = None,
                                      domain_hints: Optional[List[str]] = None) -> str:
        """Build sophisticated extraction prompt with context and domain awareness"""
        from app.services.settings_prompt_service import get_prompt_service as get_settings_prompt_service
        
        prompt_service = get_settings_prompt_service()
        
        # Context information
        context_info = ""
        if context:
            context_info = f"""
CONTEXT INFORMATION:
- Document type: {context.get('document_type', 'unknown')}
- Source: {context.get('source', 'unknown')}
- Date: {context.get('date', 'unknown')}
- Domain: {context.get('domain', 'general')}
"""
        
        # Domain-specific guidance
        domain_guidance = ""
        if domain_hints:
            domain_guidance = f"""
DOMAIN FOCUS: Pay special attention to {', '.join(domain_hints)} related entities and relationships.
"""
        
        # Get dynamic schema from DynamicSchemaManager
        dynamic_schema = await dynamic_schema_manager.get_combined_schema()
        entity_types = dynamic_schema['entity_types']
        relationship_types = dynamic_schema['relationship_types']
        
        return prompt_service.get_prompt(
            "knowledge_extraction",
            variables={
                "text": text,
                "context_info": context_info,
                "domain_guidance": domain_guidance,
                "entity_types": entity_types,
                "relationship_types": relationship_types
            }
        )
    
    async def _call_llm_for_extraction(self, prompt: str) -> str:
        """Call LLM API for knowledge extraction"""
        try:
            # This would integrate with your existing LLM service
            # For now, implementing a basic HTTP client call
            import aiohttp
            
            payload = {
                "model": self.model_config['model'],
                "prompt": prompt,
                "temperature": self.model_config['temperature'],
                "max_tokens": self.model_config['max_tokens'],
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.model_config['model_server']}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '')
                    else:
                        raise Exception(f"LLM API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM JSON response with enhanced robustness and debugging"""
        original_response = response
        
        try:
            # Clean response - remove any non-JSON text
            response = response.strip()
            
            logger.debug(f"Parsing LLM response, original length: {len(original_response)}")
            
            # Remove <think> tags if present (common with reasoning models)
            if '<think>' in response and '</think>' in response:
                # Extract everything after </think>
                response = response.split('</think>')[-1].strip()
                logger.debug("Removed <think> tags from response")
            
            # Handle code blocks
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
                logger.debug("Extracted JSON from code block")
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
                logger.debug("Extracted content from code block")
            
            # Clean any remaining whitespace and newlines
            response = response.strip()
            
            # Try multiple JSON parsing strategies
            parsed = None
            
            # Strategy 1: Direct JSON parsing
            try:
                parsed = json.loads(response)
                logger.debug("✅ Direct JSON parsing successful")
            except json.JSONDecodeError as e:
                logger.debug(f"Direct JSON parsing failed: {e}")
                
                # Strategy 2: Try to find JSON object in the response
                json_start = response.find('{')
                json_end = response.rfind('}')
                
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_content = response[json_start:json_end+1]
                    try:
                        parsed = json.loads(json_content)
                        logger.debug("✅ Extracted JSON object parsing successful")
                    except json.JSONDecodeError:
                        logger.debug("Extracted JSON object parsing failed")
                
                # Strategy 3: Try to clean and parse
                if not parsed:
                    # Remove trailing commas and other common issues
                    cleaned_response = self._clean_json_response(response)
                    try:
                        parsed = json.loads(cleaned_response)
                        logger.debug("✅ Cleaned JSON parsing successful")
                    except json.JSONDecodeError:
                        logger.debug("Cleaned JSON parsing failed")
            
            if not parsed:
                raise ValueError("All JSON parsing strategies failed")
            
            # Validate and fix structure
            if not isinstance(parsed.get('entities'), list):
                logger.warning("Entities field missing or invalid, initializing empty list")
                parsed['entities'] = []
                
            if not isinstance(parsed.get('relationships'), list):
                logger.warning("Relationships field missing or invalid, initializing empty list")  
                parsed['relationships'] = []
                
            if not isinstance(parsed.get('discoveries', {}), dict):
                logger.warning("Discoveries field invalid, initializing empty dict")
                parsed['discoveries'] = {}
            
            # Log parsing results
            entities_count = len(parsed.get('entities', []))
            relationships_count = len(parsed.get('relationships', []))
            logger.info(f"JSON parsing successful: {entities_count} entities, {relationships_count} relationships")
            
            if relationships_count == 0 and entities_count > 1:
                logger.warning("⚠️ No relationships found despite multiple entities - potential extraction issue!")
                logger.debug(f"Sample entities: {[e.get('canonical_form', e.get('text', 'unknown')) for e in parsed.get('entities', [])][:5]}")
                
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response length: {len(original_response)}")
            logger.error(f"First 500 chars: {original_response[:500]}")
            logger.error(f"Last 500 chars: {original_response[-500:]}")
            
            # Try to extract entities and relationships manually as fallback
            fallback_result = self._emergency_parse_response(original_response)
            return fallback_result
    
    def _clean_json_response(self, response: str) -> str:
        """Clean common JSON formatting issues"""
        # Remove trailing commas
        response = re.sub(r',\s*}', '}', response)
        response = re.sub(r',\s*]', ']', response)
        
        # Fix common quote issues
        response = response.replace("'", '"')
        
        # Remove any text before first { and after last }
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            response = response[start:end]
            
        return response
    
    def _emergency_parse_response(self, response: str) -> Dict[str, Any]:
        """Emergency parsing when JSON parsing fails completely"""
        logger.warning("🚨 Emergency parsing mode activated - JSON parsing completely failed")
        
        result = {
            'entities': [],
            'relationships': [],
            'reasoning': 'Emergency parsing - JSON parse failed',
            'discoveries': {}
        }
        
        # Try to extract entities using regex patterns
        entity_patterns = [
            r'"text":\s*"([^"]+)"',
            r'"canonical_form":\s*"([^"]+)"',
            r'"name":\s*"([^"]+)"'
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                if len(match) > 2 and match not in [e.get('text', '') for e in result['entities']]:
                    result['entities'].append({
                        'text': match,
                        'canonical_form': match,
                        'type': 'CONCEPT',
                        'confidence': 0.5
                    })
        
        # Try to extract relationships
        rel_patterns = [
            r'"source_entity":\s*"([^"]+)".*?"target_entity":\s*"([^"]+)".*?"relationship_type":\s*"([^"]+)"',
        ]
        
        for pattern in rel_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                if len(match) == 3:
                    result['relationships'].append({
                        'source_entity': match[0],
                        'target_entity': match[1], 
                        'relationship_type': match[2],
                        'confidence': 0.3,
                        'context': 'Emergency extracted'
                    })
        
        logger.info(f"Emergency parsing recovered: {len(result['entities'])} entities, {len(result['relationships'])} relationships")
        return result
    
    def _enhance_entities_with_hierarchy(self, raw_entities: List[Dict[str, Any]]) -> List[ExtractedEntity]:
        """Convert raw entities to ExtractedEntity objects with hierarchical enhancement"""
        enhanced_entities = []
        
        for raw_entity in raw_entities:
            try:
                # Extract basic information
                text = raw_entity.get('text', '')
                canonical_form = raw_entity.get('canonical_form', text).strip().title()
                entity_type = raw_entity.get('type', 'CONCEPT').upper()
                subtype = raw_entity.get('subtype', '').upper()
                confidence = float(raw_entity.get('confidence', 0.7))
                
                # Validate entity type
                if entity_type not in self.hierarchical_entity_types:
                    entity_type = 'CONCEPT'
                
                # Create enhanced entity
                entity = ExtractedEntity(
                    text=text,
                    label=entity_type,
                    start_char=raw_entity.get('start_char', 0),
                    end_char=raw_entity.get('end_char', len(text)),
                    confidence=confidence,
                    canonical_form=canonical_form
                )
                
                enhanced_entities.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to process entity {raw_entity}: {e}")
                continue
        
        return enhanced_entities
    
    def _validate_and_score_relationships(self, raw_relationships: List[Dict[str, Any]], 
                                        entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Validate relationships with fuzzy matching and enhanced debugging"""
        validated_relationships = []
        
        # Create multiple entity name mappings for flexible matching
        entity_names_exact = {e.canonical_form.lower(): e.canonical_form for e in entities}
        entity_names_text = {e.text.lower(): e.canonical_form for e in entities}
        entity_words = {}  # Map individual words to entity names
        
        # Build word-based mapping for partial matching
        for entity in entities:
            words = entity.canonical_form.lower().split()
            for word in words:
                if len(word) > 2:  # Only significant words
                    if word not in entity_words:
                        entity_words[word] = []
                    entity_words[word].append(entity.canonical_form)
        
        logger.info(f"Processing {len(raw_relationships)} raw relationships against {len(entities)} entities")
        logger.debug(f"Available entities: {list(entity_names_exact.keys())[:10]}...")  # Log first 10
        
        relationships_found = 0
        relationships_skipped = 0
        
        for raw_rel in raw_relationships:
            try:
                source = raw_rel.get('source_entity', '').strip()
                target = raw_rel.get('target_entity', '').strip()
                rel_type = raw_rel.get('relationship_type', '').upper()
                confidence = float(raw_rel.get('confidence', 0.5))
                
                logger.debug(f"Processing relationship: '{source}' -> '{target}' ({rel_type})")
                
                # Try multiple matching strategies
                source_canonical = self._find_matching_entity(source, entity_names_exact, entity_names_text, entity_words)
                target_canonical = self._find_matching_entity(target, entity_names_exact, entity_names_text, entity_words)
                
                if not source_canonical:
                    logger.debug(f"  ❌ Source entity not found: '{source}'")
                    relationships_skipped += 1
                    continue
                    
                if not target_canonical:
                    logger.debug(f"  ❌ Target entity not found: '{target}'")
                    relationships_skipped += 1
                    continue
                
                logger.debug(f"  ✅ Matched: '{source}' -> '{source_canonical}', '{target}' -> '{target_canonical}'")
                
                # Validate relationship type
                if not self._is_valid_relationship_type(rel_type):
                    logger.debug(f"  ⚠️ Invalid relationship type '{rel_type}', using 'RELATED_TO'")
                    rel_type = 'RELATED_TO'
                
                # Create relationship with enhanced properties
                relationship = ExtractedRelationship(
                    source_entity=source_canonical,
                    target_entity=target_canonical,
                    relationship_type=rel_type,
                    confidence=confidence,
                    context=raw_rel.get('context', raw_rel.get('evidence', '')),
                    properties={
                        'llm_extracted': True,
                        'evidence': raw_rel.get('evidence', ''),
                        'temporal_info': raw_rel.get('temporal_info', ''),
                        'attributes': raw_rel.get('attributes', {}),
                        'reasoning': raw_rel.get('reasoning', ''),
                        'original_source': source,
                        'original_target': target,
                        'fuzzy_matched': source_canonical != source or target_canonical != target
                    }
                )
                
                validated_relationships.append(relationship)
                relationships_found += 1
                logger.debug(f"  ✅ Relationship created: {source_canonical} -[{rel_type}]-> {target_canonical}")
                
            except Exception as e:
                logger.warning(f"Failed to process relationship {raw_rel}: {e}")
                relationships_skipped += 1
                continue
        
        logger.info(f"Relationship validation complete: {relationships_found} found, {relationships_skipped} skipped")
        
        if relationships_found == 0 and len(raw_relationships) > 0:
            logger.warning("⚠️ NO RELATIONSHIPS EXTRACTED despite raw relationships being present!")
            logger.warning(f"Raw relationship sample: {raw_relationships[0] if raw_relationships else 'None'}")
            logger.warning(f"Available entities sample: {list(entity_names_exact.keys())[:5]}")
        
        return validated_relationships
    
    def _find_matching_entity(self, entity_name: str, entity_names_exact: Dict[str, str], 
                             entity_names_text: Dict[str, str], entity_words: Dict[str, List[str]]) -> Optional[str]:
        """Find matching entity using multiple strategies with fuzzy matching"""
        if not entity_name:
            return None
            
        entity_lower = entity_name.lower().strip()
        
        # Strategy 1: Exact canonical form match
        if entity_lower in entity_names_exact:
            return entity_names_exact[entity_lower]
        
        # Strategy 2: Exact text match
        if entity_lower in entity_names_text:
            return entity_names_text[entity_lower]
        
        # Strategy 3: Partial word matching
        entity_name_words = entity_lower.split()
        for word in entity_name_words:
            if len(word) > 2 and word in entity_words:
                # Return the first match (could be improved with scoring)
                return entity_words[word][0]
        
        # Strategy 4: Fuzzy substring matching  
        for canonical_name in entity_names_exact.values():
            canonical_lower = canonical_name.lower()
            
            # Check if entity name is a substring of canonical name
            if entity_lower in canonical_lower or canonical_lower in entity_lower:
                return canonical_name
            
            # Check if significant words overlap
            canonical_words = set(canonical_lower.split())
            entity_words_set = set(entity_name_words)
            
            # If more than half the words match, consider it a match
            overlap = len(canonical_words.intersection(entity_words_set))
            min_words = min(len(canonical_words), len(entity_words_set))
            
            if min_words > 0 and overlap / min_words >= 0.5:
                return canonical_name
        
        # Strategy 5: Similarity matching for common variations
        for canonical_name in entity_names_exact.values():
            if self._entities_similar(entity_lower, canonical_name.lower()):
                return canonical_name
                
        return None
    
    def _entities_similar(self, entity1: str, entity2: str) -> bool:
        """Check if two entities are similar using simple heuristics"""
        # Remove common suffixes/prefixes
        entity1_clean = entity1.replace(' inc', '').replace(' ltd', '').replace(' corp', '').replace(' company', '')
        entity2_clean = entity2.replace(' inc', '').replace(' ltd', '').replace(' corp', '').replace(' company', '')
        
        # Check cleaned versions
        if entity1_clean == entity2_clean:
            return True
            
        # Check for abbreviations (e.g., "DBS" vs "DBS Bank")
        if entity1_clean in entity2_clean or entity2_clean in entity1_clean:
            return True
            
        return False
    
    def _is_valid_relationship_type(self, rel_type: str) -> bool:
        """Check if relationship type is valid in our taxonomy"""
        for category, types in self.relationship_taxonomy.items():
            if rel_type in types:
                return True
        return False
    
    async def _infer_fallback_relationships(self, entities: List[ExtractedEntity], text: str) -> List[ExtractedRelationship]:
        """Infer relationships when LLM extraction failed to find any"""
        fallback_relationships = []
        
        # Strategy 1: Co-occurrence based relationships
        text_lower = text.lower()
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Check if entities appear close to each other in text
                entity1_pos = text_lower.find(entity1.text.lower())
                entity2_pos = text_lower.find(entity2.text.lower())
                
                if entity1_pos != -1 and entity2_pos != -1:
                    distance = abs(entity1_pos - entity2_pos)
                    
                    # If entities are within 200 characters, create a relationship
                    if distance < 200:
                        # Determine relationship type based on entity types
                        rel_type = self._infer_relationship_type(entity1, entity2, text)
                        
                        # Extract context around the entities
                        context_start = max(0, min(entity1_pos, entity2_pos) - 50)
                        context_end = min(len(text), max(entity1_pos + len(entity1.text), entity2_pos + len(entity2.text)) + 50)
                        context = text[context_start:context_end]
                        
                        relationship = ExtractedRelationship(
                            source_entity=entity1.canonical_form,
                            target_entity=entity2.canonical_form,
                            relationship_type=rel_type,
                            confidence=0.4,  # Lower confidence for inferred relationships
                            context=context,
                            properties={
                                'fallback_inferred': True,
                                'inference_method': 'co_occurrence',
                                'distance': distance,
                                'evidence': f"Entities found within {distance} characters"
                            }
                        )
                        
                        fallback_relationships.append(relationship)
                        logger.debug(f"Inferred relationship: {entity1.canonical_form} -[{rel_type}]-> {entity2.canonical_form}")
        
        # Strategy 2: Pattern-based inference for common relationships
        pattern_relationships = self._infer_pattern_relationships(entities, text)
        fallback_relationships.extend(pattern_relationships)
        
        return fallback_relationships
    
    def _infer_relationship_type(self, entity1: ExtractedEntity, entity2: ExtractedEntity, text: str) -> str:
        """Infer the most likely relationship type between two entities"""
        
        # Rule-based inference based on entity types
        type1, type2 = entity1.label, entity2.label
        
        # Organization-Person relationships
        if (type1 == 'ORGANIZATION' and type2 == 'PERSON') or (type1 == 'PERSON' and type2 == 'ORGANIZATION'):
            return 'WORKS_FOR' if type1 == 'PERSON' else 'EMPLOYS'
        
        # Organization-Organization relationships
        if type1 == 'ORGANIZATION' and type2 == 'ORGANIZATION':
            return 'PARTNERS_WITH'
        
        # Technology-Organization relationships
        if (type1 == 'TECHNOLOGY' and type2 == 'ORGANIZATION') or (type1 == 'ORGANIZATION' and type2 == 'TECHNOLOGY'):
            return 'USES' if type2 == 'TECHNOLOGY' else 'USED_BY'
        
        # Location-Organization relationships
        if (type1 == 'LOCATION' and type2 == 'ORGANIZATION') or (type1 == 'ORGANIZATION' and type2 == 'LOCATION'):
            return 'LOCATED_IN' if type1 == 'ORGANIZATION' else 'HOSTS'
        
        # Concept relationships
        if type1 == 'CONCEPT' or type2 == 'CONCEPT':
            return 'IMPLEMENTS'
        
        # Default fallback
        return 'RELATED_TO'
    
    def _infer_pattern_relationships(self, entities: List[ExtractedEntity], text: str) -> List[ExtractedRelationship]:
        """Infer relationships based on textual patterns"""
        pattern_relationships = []
        text_lower = text.lower()
        
        # Common relationship patterns
        patterns = [
            (r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:part\s+of|belongs\s+to|member\s+of)\s+(\w+(?:\s+\w+)*)', 'PART_OF'),
            (r'(\w+(?:\s+\w+)*)\s+(?:owns|operates|manages)\s+(\w+(?:\s+\w+)*)', 'OWNS'),
            (r'(\w+(?:\s+\w+)*)\s+(?:founded|established|created)\s+(\w+(?:\s+\w+)*)', 'FOUNDED'),
            (r'(\w+(?:\s+\w+)*)\s+(?:uses|utilizes|employs)\s+(\w+(?:\s+\w+)*)', 'USES'),
            (r'(\w+(?:\s+\w+)*)\s+(?:partners\s+with|collaborates\s+with)\s+(\w+(?:\s+\w+)*)', 'PARTNERS_WITH'),
            (r'(\w+(?:\s+\w+)*)\s+(?:located\s+in|based\s+in)\s+(\w+(?:\s+\w+)*)', 'LOCATED_IN'),
        ]
        
        entity_map = {entity.canonical_form.lower(): entity for entity in entities}
        
        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                source_text = match.group(1).strip()
                target_text = match.group(2).strip()
                
                # Find matching entities
                source_entity = None
                target_entity = None
                
                for entity_name, entity in entity_map.items():
                    if source_text in entity_name or entity_name in source_text:
                        source_entity = entity
                    if target_text in entity_name or entity_name in target_text:
                        target_entity = entity
                
                if source_entity and target_entity and source_entity != target_entity:
                    relationship = ExtractedRelationship(
                        source_entity=source_entity.canonical_form,
                        target_entity=target_entity.canonical_form,
                        relationship_type=rel_type,
                        confidence=0.6,  # Medium confidence for pattern-based
                        context=match.group(0),
                        properties={
                            'fallback_inferred': True,
                            'inference_method': 'pattern_based',
                            'pattern': pattern,
                            'evidence': f"Pattern match: {match.group(0)}"
                        }
                    )
                    pattern_relationships.append(relationship)
                    logger.debug(f"Pattern inferred: {source_entity.canonical_form} -[{rel_type}]-> {target_entity.canonical_form}")
        
        return pattern_relationships
    
    def _calculate_overall_confidence(self, entities: List[ExtractedEntity], 
                                    relationships: List[ExtractedRelationship]) -> float:
        """Calculate overall confidence score for the extraction"""
        if not entities and not relationships:
            return 0.0
        
        total_confidence = 0.0
        total_items = 0
        
        # Entity confidence
        for entity in entities:
            total_confidence += entity.confidence
            total_items += 1
        
        # Relationship confidence
        for rel in relationships:
            total_confidence += rel.confidence
            total_items += 1
        
        return total_confidence / total_items if total_items > 0 else 0.0
    
    async def _process_discoveries_async(self, discoveries: Dict[str, Any]) -> None:
        """Process new entity and relationship type discoveries asynchronously"""
        try:
            from app.services.dynamic_schema_manager import DiscoveredEntityType, DiscoveredRelationshipType
            from datetime import datetime
            
            # Process new entity types
            new_entity_types = discoveries.get('new_entity_types', [])
            for entity_type_data in new_entity_types:
                if entity_type_data.get('confidence', 0) >= 0.7:  # High confidence threshold
                    entity_type = DiscoveredEntityType(
                        type=entity_type_data['type'],
                        description=entity_type_data['description'],
                        examples=entity_type_data.get('examples', []),
                        confidence=entity_type_data['confidence'],
                        frequency=1,
                        first_seen=datetime.utcnow(),
                        last_seen=datetime.utcnow(),
                        status='pending'
                    )
                    await dynamic_schema_manager._update_entity_cache([entity_type])
            
            # Process new relationship types
            new_relationship_types = discoveries.get('new_relationship_types', [])
            for rel_type_data in new_relationship_types:
                if rel_type_data.get('confidence', 0) >= 0.7:  # High confidence threshold
                    relationship_type = DiscoveredRelationshipType(
                        type=rel_type_data['type'],
                        description=rel_type_data['description'],
                        inverse=rel_type_data.get('inverse'),
                        examples=rel_type_data.get('examples', []),
                        confidence=rel_type_data.get('confidence', 0.5),
                        frequency=1,
                        first_seen=datetime.utcnow(),
                        last_seen=datetime.utcnow(),
                        status='pending'
                    )
                    await dynamic_schema_manager._update_relationship_cache([relationship_type])
                    
        except Exception as e:
            logger.error(f"Error processing discoveries: {e}")


# Singleton instance
_llm_extractor: Optional[LLMKnowledgeExtractor] = None

def get_llm_knowledge_extractor() -> LLMKnowledgeExtractor:
    """Get or create LLM knowledge extractor singleton"""
    global _llm_extractor
    if _llm_extractor is None:
        _llm_extractor = LLMKnowledgeExtractor()
    return _llm_extractor