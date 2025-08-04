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
from app.core.timeout_settings_cache import get_knowledge_graph_timeout, get_kg_base_timeout, get_kg_max_timeout, get_kg_fallback_timeout
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
        
        # COMPREHENSIVE BUSINESS INTELLIGENCE ENTITY TYPES (4x more coverage)
        self.hierarchical_entity_types = {
            # People and Roles - Enhanced Coverage
            'PERSON', 'EXECUTIVE', 'RESEARCHER', 'ENTREPRENEUR', 'ANALYST', 'MANAGER', 'DIRECTOR',
            'CEO', 'CTO', 'CIO', 'CFO', 'COO', 'CMO', 'CHRO', 'CCO', 'FOUNDER', 'PRESIDENT', 'CHAIRMAN',
            'EMPLOYEE', 'CONSULTANT', 'ADVISOR', 'BOARD_MEMBER', 'EXECUTIVE_COMMITTEE',
            'STAKEHOLDER', 'CUSTOMER', 'CLIENT', 'PARTNER', 'INVESTOR', 'SHAREHOLDER', 'VENDOR',
            'SUPPLIER', 'CONTRACTOR', 'REPRESENTATIVE', 'SPOKESPERSON', 'AMBASSADOR',
            
            # Organizations and Entities - Comprehensive Structure
            'ORGANIZATION', 'ORG', 'COMPANY', 'CORPORATION', 'BANK', 'FINTECH', 'STARTUP',
            'UNIVERSITY', 'RESEARCH_INSTITUTE', 'GOVERNMENT', 'AGENCY', 'DEPARTMENT', 'MINISTRY',
            'SUBSIDIARY', 'DIVISION', 'UNIT', 'TEAM', 'COMMITTEE', 'BOARD', 'GROUP',
            'ASSOCIATION', 'FOUNDATION', 'NGO', 'CONSORTIUM', 'ALLIANCE', 'PARTNERSHIP',
            'BUSINESS_UNIT', 'BUSINESS_DIVISION', 'REGIONAL_OFFICE', 'LOCAL_OFFICE', 'HEADQUARTERS',
            'SUBSIDIARY_COMPANY', 'JOINT_VENTURE', 'HOLDING_COMPANY', 'PARENT_COMPANY',
            
            # Business and Financial - Detailed Coverage
            'PRODUCT', 'SERVICE', 'PLATFORM', 'SOLUTION', 'OFFERING', 'PORTFOLIO', 'BRAND',
            'REVENUE', 'COST', 'PROFIT', 'LOSS', 'INVESTMENT', 'FUNDING', 'BUDGET', 'CAPITAL',
            'MARKET', 'SEGMENT', 'INDUSTRY', 'SECTOR', 'VERTICAL', 'ECONOMY', 'MARKET_SHARE',
            'BUSINESS_MODEL', 'STRATEGY', 'INITIATIVE', 'PROGRAM', 'PROJECT', 'CAMPAIGN',
            'KPI', 'METRIC', 'TARGET', 'GOAL', 'OBJECTIVE', 'MILESTONE', 'BENCHMARK',
            'PERFORMANCE', 'EFFICIENCY', 'PRODUCTIVITY', 'GROWTH', 'EXPANSION', 'ACQUISITION',
            'MERGER', 'DIVESTITURE', 'SPINOFF', 'IPO', 'FUNDING_ROUND', 'VALUATION',
            
            # Products and Services - Granular Types
            'MOBILE_APP', 'WEB_PLATFORM', 'DIGITAL_SERVICE', 'BANKING_PRODUCT', 'PAYMENT_SYSTEM',
            'CREDIT_CARD', 'LOAN_PRODUCT', 'INVESTMENT_PRODUCT', 'INSURANCE_PRODUCT',
            'CORPORATE_BANKING', 'RETAIL_BANKING', 'WEALTH_MANAGEMENT', 'ASSET_MANAGEMENT',
            'TRADING_PLATFORM', 'E_COMMERCE', 'MARKETPLACE', 'SUBSCRIPTION_SERVICE',
            
            # Technology and Systems - Comprehensive Tech Stack
            'TECHNOLOGY', 'SYSTEM', 'SOFTWARE', 'HARDWARE', 'INFRASTRUCTURE', 'CLOUD',
            'DATABASE', 'APPLICATION', 'API', 'FRAMEWORK', 'LIBRARY', 'TOOL', 'PLATFORM_TECH',
            'METHODOLOGY', 'PROTOCOL', 'STANDARD', 'SPECIFICATION', 'ALGORITHM',
            'MODEL', 'ARCHITECTURE', 'DESIGN', 'IMPLEMENTATION', 'INTEGRATION',
            'ARTIFICIAL_INTELLIGENCE', 'MACHINE_LEARNING', 'BLOCKCHAIN', 'FINTECH_SOLUTION',
            'CORE_BANKING', 'PAYMENT_GATEWAY', 'MOBILE_BANKING', 'INTERNET_BANKING',
            'DATA_ANALYTICS', 'BIG_DATA', 'CYBERSECURITY', 'CLOUD_COMPUTING',
            
            # Geographic and Temporal - Enhanced Coverage
            'LOCATION', 'CITY', 'COUNTRY', 'REGION', 'CONTINENT', 'FACILITY', 'MARKET_LOCATION',
            'OFFICE', 'HEADQUARTERS', 'BRANCH', 'MARKET_REGION', 'GEOGRAPHIC_MARKET',
            'ASIA_PACIFIC', 'SOUTHEAST_ASIA', 'NORTH_AMERICA', 'EUROPE', 'MIDDLE_EAST',
            'TEMPORAL', 'DATE', 'TIME', 'YEAR', 'QUARTER', 'MONTH', 'PERIOD', 'FISCAL_YEAR',
            'EVENT', 'MEETING', 'CONFERENCE', 'ANNOUNCEMENT', 'LAUNCH', 'MILESTONE_DATE',
            'TIMELINE', 'PHASE', 'ROLLOUT', 'IMPLEMENTATION_DATE', 'DEADLINE',
            
            # Regulatory and Compliance - Business Context
            'REGULATION', 'POLICY', 'LAW', 'COMPLIANCE', 'STANDARD', 'REQUIREMENT', 'GUIDELINE',
            'RISK', 'THREAT', 'OPPORTUNITY', 'CHALLENGE', 'ISSUE', 'CONCERN', 'REGULATORY_BODY',
            'CENTRAL_BANK', 'FINANCIAL_AUTHORITY', 'REGULATOR', 'COMPLIANCE_FRAMEWORK',
            'RISK_MANAGEMENT', 'GOVERNANCE', 'AUDIT', 'INTERNAL_CONTROL',
            
            # Business Processes and Operations
            'PROCESS', 'WORKFLOW', 'OPERATION', 'FUNCTION', 'CAPABILITY', 'COMPETENCY',
            'DIGITAL_TRANSFORMATION', 'AUTOMATION', 'OPTIMIZATION', 'MODERNIZATION',
            'CUSTOMER_JOURNEY', 'USER_EXPERIENCE', 'CUSTOMER_SERVICE', 'SUPPORT',
            'SALES', 'MARKETING', 'PROCUREMENT', 'SUPPLY_CHAIN', 'LOGISTICS',
            
            # Financial Metrics and Indicators - Detailed
            'FINANCIAL_METRIC', 'ROI', 'ROE', 'ROA', 'NPV', 'IRR', 'EBITDA', 'NET_INCOME',
            'GROSS_MARGIN', 'OPERATING_MARGIN', 'COST_SAVINGS', 'COST_REDUCTION',
            'EFFICIENCY_GAIN', 'PRODUCTIVITY_IMPROVEMENT', 'MARKET_CAPITALIZATION',
            'BOOK_VALUE', 'EARNINGS_PER_SHARE', 'DIVIDEND_YIELD',
            
            # Competitive and Market Intelligence
            'COMPETITOR', 'COMPETITION', 'COMPETITIVE_ADVANTAGE', 'MARKET_POSITION',
            'MARKET_LEADER', 'MARKET_CHALLENGER', 'MARKET_FOLLOWER', 'NICHE_PLAYER',
            'DIFFERENTIATION', 'VALUE_PROPOSITION', 'UNIQUE_SELLING_POINT',
            
            # Innovation and Development
            'INNOVATION', 'R_AND_D', 'RESEARCH_AND_DEVELOPMENT', 'PROTOTYPE', 'PILOT',
            'PROOF_OF_CONCEPT', 'INNOVATION_LAB', 'INCUBATOR', 'ACCELERATOR',
            'PATENT', 'INTELLECTUAL_PROPERTY', 'TRADE_SECRET', 'COPYRIGHT',
            
            # General Concepts - Enhanced
            'CONCEPT', 'PRINCIPLE', 'APPROACH', 'TREND', 'PATTERN', 'PHENOMENON',
            'FACTOR', 'ELEMENT', 'COMPONENT', 'ASPECT', 'DIMENSION', 'ATTRIBUTE',
            'THEME', 'PRIORITY', 'FOCUS_AREA', 'STRATEGIC_PILLAR', 'VALUE_DRIVER'
        }
        
        # Enhanced relationship taxonomy with comprehensive business relationships
        self.relationship_taxonomy = {
            'organizational': {
                'PART_OF', 'CONTAINS', 'MEMBER_OF', 'OWNS', 'OWNED_BY',  # REMOVED: RELATED_TO (generic)
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
        
        # AGGRESSIVE MULTI-PASS EXTRACTION CONFIGURATION (4x better yield)
        self.extraction_passes = {
            'core_entities': {
                'focus': ['ORGANIZATION', 'PERSON', 'TECHNOLOGY', 'LOCATION', 'PRODUCT', 'SERVICE', 
                         'BANK', 'FINTECH', 'SUBSIDIARY', 'DIVISION', 'BUSINESS_UNIT'],
                'min_confidence': 0.6,  # BALANCED: Set to 0.6 for quality entities with reasonable volume
                'aggressive_matching': True,
                'business_context_boost': True
            },
            'business_concepts': {
                'focus': ['STRATEGY', 'INITIATIVE', 'MARKET', 'REVENUE', 'INVESTMENT', 'RISK',
                         'DIGITAL_TRANSFORMATION', 'COMPETITIVE_ADVANTAGE', 'MARKET_SHARE',
                         'CUSTOMER_JOURNEY', 'INNOVATION', 'GROWTH', 'EXPANSION'],
                'min_confidence': 0.6,  # BALANCED: Set to 0.6 for quality concepts with reasonable volume
                'contextual_enhancement': True,
                'pattern_recognition': True
            },
            'financial_metrics': {
                'focus': ['KPI', 'METRIC', 'TARGET', 'GOAL', 'ROI', 'COST_SAVINGS', 'REVENUE',
                         'PROFIT', 'INVESTMENT', 'FUNDING', 'BUDGET', 'FINANCIAL_METRIC'],
                'min_confidence': 0.6,  # BALANCED: Set to 0.6 for quality metrics with reasonable volume
                'number_pattern_extraction': True,
                'percentage_extraction': True
            },
            'operational_entities': {
                'focus': ['PROCESS', 'WORKFLOW', 'OPERATION', 'CAPABILITY', 'PLATFORM_TECH',
                         'CORE_BANKING', 'PAYMENT_SYSTEM', 'MOBILE_BANKING', 'AUTOMATION'],
                'min_confidence': 0.6,  # BALANCED: Set to 0.6 for quality operations with reasonable volume
                'system_integration_focus': True
            },
            'market_competitive': {
                'focus': ['COMPETITOR', 'MARKET_POSITION', 'COMPETITIVE_ADVANTAGE', 'MARKET_LEADER',
                         'DIFFERENTIATION', 'VALUE_PROPOSITION', 'MARKET_SEGMENT'],
                'min_confidence': 0.6,  # BALANCED: Set to 0.6 for quality competitive entities with reasonable volume
                'competitive_analysis': True
            },
            'relationships_deep': {
                'focus': 'relationships',
                'inference_enabled': True,
                'cross_reference': True,
                'min_confidence': 0.6,  # BALANCED: Set to 0.6 for quality relationships with reasonable volume
                'business_relationship_patterns': True
            },
            'temporal_causal': {
                'focus': ['temporal_business', 'strategic', 'TIMELINE', 'MILESTONE', 'PHASE'],
                'causal_inference': True,
                'timeline_analysis': True,
                'min_confidence': 0.6  # BALANCED: Set to 0.6 for quality temporal relationships with reasonable volume
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
    
    def _calculate_dynamic_timeout(self, text_length: int, extraction_mode: str = "standard", 
                                  pass_number: int = 1) -> int:
        """Calculate dynamic timeout based on content size, complexity, and extraction mode"""
        try:
            # Get base timeout from configuration
            base_timeout = get_kg_base_timeout()
            max_timeout = get_kg_max_timeout()
            
            # Get configuration parameters
            large_threshold = get_knowledge_graph_timeout("large_document_threshold", 20000)
            ultra_threshold = get_knowledge_graph_timeout("ultra_large_document_threshold", 50000)
            content_multiplier = get_knowledge_graph_timeout("content_size_multiplier", 0.01)
            complexity_multiplier = get_knowledge_graph_timeout("complexity_multiplier", 1.2)
            pass_multiplier = get_knowledge_graph_timeout("pass_timeout_multiplier", 1.5)
            
            # Calculate base timeout with content size scaling
            size_factor = max(1.0, text_length * content_multiplier / 1000)  # Scale per 1000 chars
            calculated_timeout = int(base_timeout * size_factor)
            
            # Apply extraction mode multipliers
            if extraction_mode in ["ULTRA-AGGRESSIVE", "ultra_aggressive"]:
                calculated_timeout = int(calculated_timeout * complexity_multiplier * 1.5)
            elif extraction_mode in ["AGGRESSIVE", "aggressive"]:
                calculated_timeout = int(calculated_timeout * complexity_multiplier)
            
            # Apply pass-specific scaling (later passes may need more time for relationship analysis)
            if pass_number > 1:
                calculated_timeout = int(calculated_timeout * (pass_multiplier ** (pass_number - 1)))
            
            # Apply document size thresholds
            if text_length > ultra_threshold:
                calculated_timeout = int(calculated_timeout * 2.0)  # Double for ultra-large documents
                logger.info(f"📊 Ultra-large document detected ({text_length:,} chars), applying 2x timeout multiplier")
            elif text_length > large_threshold:
                calculated_timeout = int(calculated_timeout * 1.5)  # 1.5x for large documents
                logger.info(f"📊 Large document detected ({text_length:,} chars), applying 1.5x timeout multiplier")
            
            # Ensure we don't exceed maximum timeout
            final_timeout = min(calculated_timeout, max_timeout)
            
            # Log timeout calculation for debugging
            logger.info(f"⏱️  Dynamic timeout calculation:")
            logger.info(f"   📄 Text length: {text_length:,} characters")
            logger.info(f"   🎯 Extraction mode: {extraction_mode}")
            logger.info(f"   🔢 Pass number: {pass_number}")
            logger.info(f"   ⚡ Base timeout: {base_timeout}s")
            logger.info(f"   📈 Calculated timeout: {calculated_timeout}s")
            logger.info(f"   ✅ Final timeout: {final_timeout}s")
            
            return final_timeout
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating dynamic timeout: {e}, using fallback")
            return get_kg_fallback_timeout()
    
    async def _fallback_extraction_strategy(self, text: str, context: Optional[Dict[str, Any]], 
                                          domain_hints: Optional[List[str]], pass_number: int) -> Optional[Dict[str, Any]]:
        """Progressive fallback strategy when extraction passes time out"""
        try:
            # Get fallback timeout (shorter than original)
            fallback_timeout = get_kg_fallback_timeout()
            logger.info(f"🔄 Attempting fallback extraction for Pass {pass_number} with {fallback_timeout}s timeout")
            
            # Use simplified extraction with reduced content if text is very large
            simplified_text = text
            if len(text) > 50000:  # Ultra-large document
                # Take first 70% and last 15% to preserve context and conclusion
                split_point1 = int(len(text) * 0.7)
                split_point2 = int(len(text) * 0.85)
                simplified_text = text[:split_point1] + "\n\n[CONTENT TRUNCATED FOR PROCESSING]\n\n" + text[split_point2:]
                logger.info(f"📊 Text simplified from {len(text):,} to {len(simplified_text):,} characters for fallback")
            
            # Try simplified extraction based on pass number
            if pass_number == 1:
                result = await asyncio.wait_for(
                    self._simplified_core_entities_extraction(simplified_text, context, domain_hints),
                    timeout=fallback_timeout
                )
            elif pass_number == 2:
                # For Pass 2, we need existing entities, use empty list if none available
                result = await asyncio.wait_for(
                    self._simplified_business_concepts_extraction(simplified_text, [], context, domain_hints),
                    timeout=fallback_timeout
                )
            else:
                logger.warning(f"⚠️ No fallback strategy defined for Pass {pass_number}")
                return None
            
            logger.info(f"✅ Fallback extraction succeeded for Pass {pass_number}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"❌ Fallback extraction also timed out for Pass {pass_number} after {fallback_timeout}s")
            return None
        except Exception as e:
            logger.error(f"❌ Fallback extraction failed for Pass {pass_number}: {e}")
            return None
    
    async def _simplified_core_entities_extraction(self, text: str, context: Optional[Dict[str, Any]], 
                                                 domain_hints: Optional[List[str]]) -> Dict[str, Any]:
        """Simplified core entity extraction for fallback scenarios"""
        start_time = datetime.now()
        
        # Use a much simpler prompt focused on just the most critical entities
        simplified_prompt = f"""
Extract the most important business entities from this text. Focus only on:
- Companies/Organizations
- Key People (executives, leaders)
- Main Products/Services
- Technologies mentioned

TEXT ({len(text):,} characters):
{text}

Return ONLY a JSON object with:
{{"entities": [list of entities], "relationships": [list of relationships], "reasoning": "brief explanation"}}

Be concise but accurate. Extract only the most critical entities.
"""
        
        try:
            # Use the fallback timeout for API calls too
            fallback_api_timeout = get_kg_fallback_timeout()
            response = await self._call_llm_for_extraction(simplified_prompt, fallback_api_timeout)
            parsed_result = self.extract_json_from_response(response)
            
            # Convert to proper format
            entities = []
            relationships = []
            
            for entity_data in parsed_result.get('entities', []):
                if isinstance(entity_data, dict):
                    entities.append(ExtractedEntity(
                        canonical_form=entity_data.get('name', 'Unknown'),
                        entity_type=entity_data.get('type', 'ENTITY'),
                        confidence=0.7,  # Lower confidence for simplified extraction
                        aliases=entity_data.get('aliases', []),
                        properties=entity_data.get('properties', {})
                    ))
            
            for rel_data in parsed_result.get('relationships', []):
                if isinstance(rel_data, dict):
                    relationships.append(ExtractedRelationship(
                        source_entity=rel_data.get('source', 'Unknown'),
                        target_entity=rel_data.get('target', 'Unknown'),
                        relationship_type=rel_data.get('type', 'USES'),  # CHANGED: Default to specific type
                        confidence=0.6,  # Lower confidence for simplified extraction
                        properties=rel_data.get('properties', {})
                    ))
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'entities': entities,
                'relationships': relationships,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Simplified core entity extraction failed: {e}")
            return {'entities': [], 'relationships': [], 'processing_time_ms': 0}
    
    async def _simplified_business_concepts_extraction(self, text: str, existing_entities: List[ExtractedEntity],
                                                     context: Optional[Dict[str, Any]], domain_hints: Optional[List[str]]) -> Dict[str, Any]:
        """Simplified business concepts extraction for fallback scenarios"""
        start_time = datetime.now()
        
        # Use a much simpler prompt focused on just key business concepts
        simplified_prompt = f"""
Extract key business concepts from this text. Focus only on:
- Business strategies/initiatives
- Financial metrics/KPIs
- Market segments
- Key processes

TEXT ({len(text):,} characters):
{text}

Return ONLY a JSON object with:
{{"entities": [list of business concept entities], "relationships": [list of relationships], "reasoning": "brief explanation"}}

Be concise but accurate. Extract only the most important business concepts.
"""
        
        try:
            # Use the fallback timeout for API calls too
            fallback_api_timeout = get_kg_fallback_timeout()
            response = await self._call_llm_for_extraction(simplified_prompt, fallback_api_timeout)
            parsed_result = self.extract_json_from_response(response)
            
            # Convert to proper format (similar to simplified_core_entities_extraction)
            entities = []
            relationships = []
            
            for entity_data in parsed_result.get('entities', []):
                if isinstance(entity_data, dict):
                    entities.append(ExtractedEntity(
                        canonical_form=entity_data.get('name', 'Unknown'),
                        entity_type=entity_data.get('type', 'BUSINESS_CONCEPT'),
                        confidence=0.6,  # Lower confidence for simplified extraction
                        aliases=entity_data.get('aliases', []),
                        properties=entity_data.get('properties', {})
                    ))
            
            for rel_data in parsed_result.get('relationships', []):
                if isinstance(rel_data, dict):
                    relationships.append(ExtractedRelationship(
                        source_entity=rel_data.get('source', 'Unknown'),
                        target_entity=rel_data.get('target', 'Unknown'),
                        relationship_type=rel_data.get('type', 'USES'),  # CHANGED: Default to specific type
                        confidence=0.5,  # Lower confidence for simplified extraction
                        properties=rel_data.get('properties', {})
                    ))
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'entities': entities,
                'relationships': relationships,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Simplified business concepts extraction failed: {e}")
            return {'entities': [], 'relationships': [], 'processing_time_ms': 0}
    
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
            
            if is_business_document and len(text) > 500:  # Lowered threshold for aggressive extraction
                logger.info("🎯 Business document detected - using ULTRA-AGGRESSIVE multi-pass extraction")
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
        """Enhanced business document detection with comprehensive patterns"""
        # COMPREHENSIVE BUSINESS INDICATORS (4x more patterns)
        business_indicators = [
            # Core Business Terms
            'strategy', 'business', 'revenue', 'investment', 'market', 'customer', 'product',
            'technology', 'digital transformation', 'innovation', 'partnership', 'acquisition',
            'growth', 'expansion', 'competitive', 'industry', 'sector', 'financial', 'banking',
            'fintech', 'platform', 'solution', 'service', 'enterprise', 'corporate', 'executive',
            'management', 'governance', 'compliance', 'risk', 'opportunity', 'initiative',
            'program', 'project', 'roadmap', 'vision', 'mission', 'objective', 'goal', 'target',
            'kpi', 'metric', 'performance', 'efficiency', 'optimization', 'transformation',
            
            # Financial & Banking Specific
            'dbs', 'bank', 'banking', 'fintech', 'financial services', 'payment', 'credit',
            'loan', 'deposit', 'wealth management', 'asset management', 'trading', 'investment banking',
            'retail banking', 'corporate banking', 'treasury', 'risk management', 'capital',
            'liquidity', 'regulatory capital', 'basel', 'monetary authority', 'central bank',
            
            # Technology & Digital
            'digitalization', 'automation', 'artificial intelligence', 'machine learning',
            'blockchain', 'cloud computing', 'data analytics', 'cybersecurity', 'api',
            'mobile banking', 'internet banking', 'core banking', 'payment gateway',
            'digital wallet', 'cryptocurrency', 'regtech', 'suptech',
            
            # Organizational
            'subsidiary', 'division', 'business unit', 'regional office', 'headquarters',
            'board of directors', 'executive committee', 'management team', 'stakeholder',
            'shareholder', 'investor', 'analyst', 'rating agency',
            
            # Market & Competition
            'market share', 'competitive advantage', 'market position', 'market leader',
            'competitor', 'competition', 'differentiation', 'value proposition',
            'customer segment', 'target market', 'market penetration',
            
            # Operations & Processes
            'operational excellence', 'process improvement', 'cost optimization',
            'productivity', 'efficiency gains', 'automation', 'workflow',
            'customer experience', 'customer journey', 'user experience',
            
            # Strategic & Planning
            'strategic planning', 'business planning', 'budget', 'forecast',
            'milestone', 'timeline', 'roadmap', 'implementation', 'rollout',
            'pilot program', 'proof of concept', 'business case'
        ]
        
        # Enhanced domain hint detection
        if domain_hints:
            business_hint_patterns = ['business', 'strategy', 'financial', 'banking', 'corporate', 'technology']
            if any(pattern in hint.lower() for hint in domain_hints for pattern in business_hint_patterns):
                return True
            
        text_lower = text.lower()
        business_score = sum(1 for indicator in business_indicators if indicator in text_lower)
        
        # Also check for specific business document patterns
        document_patterns = [
            'confidential', 'strategy document', 'business plan', 'annual report',
            'quarterly report', 'investor presentation', 'board presentation',
            'executive summary', 'strategic review', 'business review'
        ]
        pattern_score = sum(1 for pattern in document_patterns if pattern in text_lower)
        
        # Lower threshold for more aggressive business document detection
        return business_score >= 3 or pattern_score >= 1
    
    def _determine_optimal_pass_count(self, text_length: int) -> int:
        """Determine optimal number of extraction passes based on document size for performance"""
        if text_length < 50000:  # < 50k chars: Use 2-pass extraction
            return 2
        elif text_length < 100000:  # < 100k chars: Use 3-pass extraction 
            return 3
        else:  # >= 100k chars: Use full 4-pass extraction
            return 4
    
    async def _multi_pass_business_extraction(self, text: str, context: Optional[Dict[str, Any]], 
                                            domain_hints: Optional[List[str]], start_time: datetime) -> LLMExtractionResult:
        """Multi-pass extraction optimized for business documents with adaptive pass count"""
        all_entities = []
        all_relationships = []
        pass_metadata = {}
        
        # Determine optimal pass count based on document size
        text_length = len(text)
        pass_count = self._determine_optimal_pass_count(text_length)
        
        logger.info(f"🔄 Starting {pass_count}-pass business extraction on {text_length:,} character text")
        logger.info(f"   📊 Performance optimization: {text_length:,} chars → {pass_count} passes")
        
        # Calculate dynamic timeout for Pass 1 with ULTRA-AGGRESSIVE mode
        pass1_timeout = self._calculate_dynamic_timeout(text_length, "ULTRA-AGGRESSIVE", 1)
        
        # Pass 1: Core Business Entities (Organizations, People, Technologies, Products)
        logger.info("🎯 Pass 1: Core business entities extraction")
        try:
            pass1_result = await asyncio.wait_for(
                self._extraction_pass_core_entities(text, context, domain_hints, pass1_timeout),
                timeout=pass1_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"⏰ Pass 1 timed out after {pass1_timeout} seconds")
            # Implement progressive fallback strategy
            logger.warning("🔄 Activating fallback strategy for Pass 1 timeout")
            pass1_result = await self._fallback_extraction_strategy(text, context, domain_hints, pass_number=1)
            if pass1_result is None:
                logger.error("❌ Fallback strategy also failed, using empty result")
                pass1_result = {'entities': [], 'relationships': [], 'processing_time_ms': pass1_timeout * 1000}
        all_entities.extend(pass1_result['entities'])
        all_relationships.extend(pass1_result['relationships'])
        pass_metadata['pass1_core_entities'] = {
            'entities_found': len(pass1_result['entities']),
            'relationships_found': len(pass1_result['relationships']),
            'processing_time_ms': pass1_result['processing_time_ms']
        }
        
        # Calculate dynamic timeout for Pass 2 with ULTRA-AGGRESSIVE mode
        pass2_timeout = self._calculate_dynamic_timeout(text_length, "ULTRA-AGGRESSIVE", 2)
        
        # Pass 2: Business Concepts and Strategic Elements
        logger.info("📈 Pass 2: Business concepts and strategic elements")
        try:
            pass2_result = await asyncio.wait_for(
                self._extraction_pass_business_concepts(text, all_entities, context, domain_hints, pass2_timeout),
                timeout=pass2_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"⏰ Pass 2 timed out after {pass2_timeout} seconds")
            # Implement progressive fallback strategy
            logger.warning("🔄 Activating fallback strategy for Pass 2 timeout")
            pass2_result = await self._fallback_extraction_strategy(text, context, domain_hints, pass_number=2)
            if pass2_result is None:
                logger.error("❌ Fallback strategy also failed, using empty result")
                pass2_result = {'entities': [], 'relationships': [], 'processing_time_ms': pass2_timeout * 1000}
        all_entities.extend(pass2_result['entities'])
        all_relationships.extend(pass2_result['relationships'])
        pass_metadata['pass2_business_concepts'] = {
            'entities_found': len(pass2_result['entities']),
            'relationships_found': len(pass2_result['relationships']),
            'processing_time_ms': pass2_result['processing_time_ms']
        }
        
        # Pass 3: Deep Relationship Analysis and Inference (only for 3+ pass documents)
        if pass_count >= 3:
            logger.info("🔗 Pass 3: Deep relationship analysis and inference")
            pass3_result = await self._extraction_pass_deep_relationships(text, all_entities, context)
            all_relationships.extend(pass3_result['relationships'])
            pass_metadata['pass3_deep_relationships'] = {
                'relationships_found': len(pass3_result['relationships']),
                'processing_time_ms': pass3_result['processing_time_ms']
            }
        
        # Pass 4: Temporal and Causal Analysis (only for 4-pass documents)
        if pass_count >= 4:
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
        
        # EMERGENCY FIX: DISABLE duplicate co-occurrence generation to prevent exponential explosion
        # This was causing 244 → 1238 → 3764 relationship explosion
        # Co-occurrence relationships are already generated in Pass 3 via _infer_cooccurrence_relationships()
        logger.info("🚑 SKIPPING duplicate co-occurrence generation to prevent relationship explosion")
        # cooccurrence_relationships = await self._generate_cooccurrence_relationships(
        #     text, consolidated_entities, consolidated_relationships
        # )
        # consolidated_relationships.extend(cooccurrence_relationships)
        
        # NUCLEAR HARD CAP: Absolute maximum relationships to prevent system overload
        NUCLEAR_RELATIONSHIP_LIMIT = 150  # Absolute maximum that cannot be bypassed
        if len(consolidated_relationships) > NUCLEAR_RELATIONSHIP_LIMIT:
            original_count = len(consolidated_relationships)
            # Keep highest confidence relationships
            consolidated_relationships.sort(key=lambda r: r.confidence, reverse=True)
            consolidated_relationships = consolidated_relationships[:NUCLEAR_RELATIONSHIP_LIMIT]
            logger.warning(f"🚑 NUCLEAR CAP APPLIED: Reduced relationships from {original_count} to {NUCLEAR_RELATIONSHIP_LIMIT}")
            logger.warning(f"🚑 This prevents system overload from exponential relationship explosion")
        
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
        try:
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
    
    async def _call_llm_for_extraction(self, prompt: str, dynamic_timeout: Optional[int] = None) -> str:
        """Call LLM API for knowledge extraction with robust timeout and retry logic"""
        import aiohttp
        import asyncio
        
        max_retries = 3
        # Use dynamic timeout if provided, otherwise get from configuration
        if dynamic_timeout is not None:
            timeout_seconds = dynamic_timeout
        else:
            timeout_seconds = get_knowledge_graph_timeout("api_call_timeout", 180)
        
        retry_delays = [1, 5, 10]  # Progressive backoff
        
        logger.debug(f"🔄 LLM API call with {timeout_seconds}s timeout")
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"🔄 LLM API call attempt {attempt + 1}/{max_retries}")
                
                payload = {
                    "model": self.model_config['model'],
                    "prompt": prompt,
                    "temperature": self.model_config['temperature'],
                    "max_tokens": self.model_config['max_tokens'],
                    "stream": False
                }
                
                # Use connection pooling for better reliability
                connector = aiohttp.TCPConnector(
                    limit=1,  # Single connection for this request
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                )
                
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.post(
                        f"{self.model_config['model_server']}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_text = result.get('response', '')
                            
                            if not response_text.strip():
                                raise Exception("Empty response from LLM API")
                                
                            logger.debug(f"✅ LLM API call successful on attempt {attempt + 1}")
                            logger.debug(f"Response length: {len(response_text):,} characters")
                            return response_text
                        else:
                            response_text = await response.text()
                            raise Exception(f"LLM API HTTP error {response.status}: {response_text}")
                            
            except asyncio.TimeoutError:
                error_msg = f"LLM API timeout after {timeout_seconds}s on attempt {attempt + 1}"
                logger.warning(f"⏱️ {error_msg}")
                
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    logger.info(f"🔄 Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                raise Exception(error_msg)
                
            except aiohttp.ClientError as e:
                error_msg = f"LLM API client error on attempt {attempt + 1}: {e}"
                logger.warning(f"🌐 {error_msg}")
                
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    logger.info(f"🔄 Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                raise Exception(error_msg)
                
            except Exception as e:
                error_msg = f"LLM API error on attempt {attempt + 1}: {e}"
                logger.warning(f"❌ {error_msg}")
                
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    logger.info(f"🔄 Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                raise Exception(error_msg)
        
        # Should never reach here due to raises above, but safety net
        raise Exception(f"LLM API call failed after {max_retries} attempts")
    
    def extract_json_from_response(self, response: str) -> dict:
        """Extract JSON from LLM response, handling markdown and extra text"""
        try:
            # Remove markdown code blocks
            clean_response = re.sub(r'```json\s*|\s*```', '', response.strip())
            
            # Find JSON object boundaries
            start = clean_response.find('{')
            end = clean_response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = clean_response[start:end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON object found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parsing failed: {e}")
            # Fall back to existing emergency parser
            return self._emergency_parse_response(response)

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
            
            # BULLETPROOF JSON parsing - simple and reliable
            parsed = None
            
            # Strategy 1: Direct JSON parsing
            try:
                parsed = json.loads(response)
                logger.debug("✅ Direct JSON parsing successful")
            except json.JSONDecodeError as e:
                logger.debug(f"Direct JSON parsing failed: {e}")
                
                # Strategy 2: BULLETPROOF extraction - find first { and last }
                json_start = response.find('{')
                json_end = response.rfind('}')
                
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_content = response[json_start:json_end+1]
                    try:
                        parsed = json.loads(json_content)
                        logger.debug("✅ Bulletproof JSON extraction successful")
                    except json.JSONDecodeError as e2:
                        logger.debug(f"Bulletproof extraction failed: {e2}")
                        
                        # Strategy 3: Handle JSON with explanatory text - clean invalid JSON keys
                        try:
                            # Remove keys that contain explanatory text (parentheses, special chars)
                            cleaned_content = re.sub(r'"[^"]*\([^)]*\)[^"]*":\s*"[^"]*"[,\s]*', '', json_content)
                            # Remove trailing commas before closing braces/brackets  
                            cleaned_content = re.sub(r',(\s*[}\]])', r'\1', cleaned_content)
                            parsed = json.loads(cleaned_content)
                            logger.debug("✅ Cleaned JSON parsing successful")
                        except json.JSONDecodeError as e3:
                            logger.debug(f"Cleaned JSON parsing failed: {e3}")
            
            if not parsed:
                logger.error(f"All JSON parsing failed. Response preview: {response[:200]}...")
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
            
            # Log parsing results with enhanced debugging
            entities_count = len(parsed.get('entities', []))
            relationships_count = len(parsed.get('relationships', []))
            logger.info(f"JSON parsing successful: {entities_count} entities, {relationships_count} relationships")
            
            # Enhanced debugging for entity names - handle both string and dict entities
            if entities_count > 0:
                sample_entities = parsed.get('entities', [])[:5]
                entity_names = []
                for e in sample_entities:
                    if isinstance(e, str):
                        # Handle string entities from emergency parsing
                        entity_names.append(e.strip() if e else 'EMPTY_NAME')
                    elif isinstance(e, dict):
                        # Handle dict entities from normal parsing
                        name = e.get('text', '') or e.get('name', '') or e.get('canonical_form', '') or e.get('entity', '')
                        entity_names.append(name.strip() if name else 'EMPTY_NAME')
                    else:
                        entity_names.append('INVALID_ENTITY_TYPE')
                logger.debug(f"Sample entity names: {entity_names}")
                
                # Check for empty entity names - handle both string and dict entities
                empty_names = []
                for e in parsed.get('entities', []):
                    if isinstance(e, str):
                        if not e.strip():
                            empty_names.append(e)
                    elif isinstance(e, dict):
                        if not (e.get('text', '') or e.get('name', '') or e.get('canonical_form', '') or e.get('entity', '')).strip():
                            empty_names.append(e)
                    else:
                        empty_names.append(e)  # Invalid type
                        
                if empty_names:
                    logger.error(f"🚨 FOUND {len(empty_names)} ENTITIES WITH EMPTY NAMES!")
                    logger.error(f"Empty entities sample: {empty_names[:3]}")
            
            if relationships_count == 0 and entities_count > 1:
                logger.warning("⚠️ No relationships found despite multiple entities - potential extraction issue!")
                
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
        """Emergency parsing when JSON parsing fails completely - enhanced to handle string arrays"""
        logger.warning("🚨 Emergency parsing mode activated - JSON parsing completely failed")
        
        result = {
            'entities': [],
            'relationships': [],
            'reasoning': 'Emergency parsing - JSON parse failed',
            'discoveries': {}
        }
        
        # CRITICAL FIX: Look for string arrays in entities field
        # Pattern: "entities": ["Entity Name 1", "Entity Name 2", ...]
        string_array_pattern = r'"entities":\s*\[\s*("(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")*)\s*\]'
        array_matches = re.search(string_array_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if array_matches:
            entity_list_str = array_matches.group(1)
            # Extract individual quoted strings
            entity_names = re.findall(r'"([^"]+)"', entity_list_str)
            
            logger.info(f"🚑 Found string array with {len(entity_names)} entities")
            
            for entity_name in entity_names:
                if len(entity_name.strip()) > 1:  # Skip very short entities
                    result['entities'].append(entity_name.strip())  # Store as string (will be processed by enhanced method)
                    logger.debug(f"🚑 Emergency entity: '{entity_name.strip()}'")
        
        # FALLBACK: Try traditional regex patterns for structured entities
        if not result['entities']:
            entity_patterns = [
                r'"text":\s*"([^"]+)"',
                r'"canonical_form":\s*"([^"]+)"',
                r'"name":\s*"([^"]+)"',
                r'"entity":\s*"([^"]+)"'
            ]
            
            for pattern in entity_patterns:
                matches = re.findall(pattern, response)
                for match in matches:
                    if len(match) > 2 and match not in [e if isinstance(e, str) else e.get('text', '') for e in result['entities']]:
                        result['entities'].append({
                            'text': match,
                            'canonical_form': match,
                            'type': 'CONCEPT',
                            'confidence': 0.5
                        })
        
        # Try to extract relationships with more patterns
        rel_patterns = [
            r'"source_entity":\s*"([^"]+)".*?"target_entity":\s*"([^"]+)".*?"relationship_type":\s*"([^"]+)"',
            r'"source":\s*"([^"]+)".*?"target":\s*"([^"]+)".*?"relationship":\s*"([^"]+)"',
            r'"from":\s*"([^"]+)".*?"to":\s*"([^"]+)".*?"type":\s*"([^"]+)"'
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
        
        # LAST RESORT: Extract any quoted strings that look like business entities
        if not result['entities']:
            logger.warning("🚑 Last resort: extracting quoted strings as potential entities")
            all_quotes = re.findall(r'"([^"]{3,50})"', response)  # 3-50 char quoted strings
            
            business_keywords = [
                'strategy', 'bank', 'digital', 'transformation', 'technology', 'system', 
                'platform', 'service', 'product', 'market', 'customer', 'business', 
                'innovation', 'investment', 'revenue', 'growth', 'competitive', 'advantage',
                'dbs', 'singapore', 'asia', 'financial', 'modernization', 'initiative',
                'program', 'project', 'development', 'management', 'operations', 'excellence'
            ]
            
            for quoted_text in all_quotes:
                if any(keyword in quoted_text.lower() for keyword in business_keywords):
                    if quoted_text not in [e if isinstance(e, str) else e.get('text', '') for e in result['entities']]:
                        result['entities'].append(quoted_text)
                        logger.debug(f"🚑 Last resort entity: '{quoted_text}'")
        
        entity_count = len(result['entities'])
        relationship_count = len(result['relationships'])
        
        logger.info(f"🚑 Emergency parsing recovered: {entity_count} entities, {relationship_count} relationships")
        
        if entity_count == 0:
            logger.error("🚨 EMERGENCY PARSING COMPLETE FAILURE - NO ENTITIES FOUND!")
            logger.error(f"Response length: {len(response):,} characters")
            logger.error(f"Response preview: {response[:1000]}...")
            
        return result
    
    def _enhance_entities_with_hierarchy(self, raw_entities: List[Any]) -> List[ExtractedEntity]:
        """Convert raw entities to ExtractedEntity objects with hierarchical enhancement - handles both string and dict entities"""
        enhanced_entities = []
        
        logger.debug(f"🔧 Processing {len(raw_entities)} raw entities")
        
        for i, raw_entity in enumerate(raw_entities):
            try:
                entity_name = ""
                entity_type = "CONCEPT"
                confidence = 0.7
                canonical_form = ""
                
                # CRITICAL FIX: Handle both string arrays and dictionary arrays
                if isinstance(raw_entity, str):
                    # Entity is just a string name (common in LLM responses)
                    entity_name = raw_entity.strip()
                    canonical_form = entity_name.title()
                    entity_type = self._infer_entity_type_from_name(entity_name)
                    logger.debug(f"  📝 String entity [{i}]: '{entity_name}' -> {entity_type}")
                    
                elif isinstance(raw_entity, dict):
                    # Entity is a dictionary with structured data
                    # Extract basic information with multiple fallback strategies
                    text = raw_entity.get('text', '').strip()
                    name = raw_entity.get('name', '').strip()
                    canonical_form = raw_entity.get('canonical_form', '').strip()
                    
                    # Try multiple ways to get entity name
                    entity_name = text or name or canonical_form
                    
                    # Additional fallback strategies for different JSON formats
                    if not entity_name:
                        entity_name = (raw_entity.get('entity', '') or 
                                     raw_entity.get('entity_name', '') or
                                     raw_entity.get('label', '') or
                                     raw_entity.get('value', '') or
                                     raw_entity.get('canonical_form', '')).strip()
                    
                    # Extract type and confidence if available
                    entity_type = raw_entity.get('type', 'CONCEPT').upper()
                    confidence = float(raw_entity.get('confidence', 0.7))
                    
                    # Set canonical form if not already set
                    if not canonical_form:
                        canonical_form = entity_name.title()
                        
                    logger.debug(f"  📚 Dict entity [{i}]: '{entity_name}' (type: {entity_type}, conf: {confidence})")
                    
                else:
                    logger.warning(f"  ❓ Unknown entity format [{i}]: {type(raw_entity)} - {raw_entity}")
                    continue
                
                # More lenient name validation for business entities
                if not entity_name or len(entity_name.strip()) < 1:
                    logger.warning(f"  ❌ Skipping entity [{i}] with empty name: {raw_entity}")
                    continue
                
                # Filter out very short non-meaningful entities
                if len(entity_name.strip()) == 1 and not entity_name.isalpha():
                    logger.warning(f"  ❌ Skipping single non-alphabetic character [{i}]: '{entity_name}'")
                    continue
                
                # ENHANCED BUSINESS VALUE FILTERING with dynamic thresholds
                business_value_score = self._calculate_business_value_score(entity_name, entity_type)
                
                # REBALANCED: Lower thresholds for better entity coverage, especially organizations
                if entity_type in ['ORGANIZATION', 'ORG', 'COMPANY', 'BANK', 'CORPORATION']:
                    threshold = 0.4  # Lower threshold for organization entities
                elif entity_type == 'TECHNOLOGY':
                    threshold = 0.45  # Keep current threshold for technology
                else:
                    threshold = 0.5  # Reduced from 0.6 to 0.5 for general entities
                
                if business_value_score < threshold:
                    logger.warning(f"  ❌ Skipping low business value entity [{i}]: '{entity_name}' (score: {business_value_score:.2f}, threshold: {threshold})")
                    continue
                
                # Enhanced stop words including business-generic terms
                if self._is_generic_business_term(entity_name):
                    logger.warning(f"  ❌ Skipping generic business term [{i}]: '{entity_name}'")
                    continue
                
                # Normalize canonical form
                if not canonical_form:
                    canonical_form = entity_name.strip().title()
                
                # Validate and normalize entity type
                if entity_type not in self.hierarchical_entity_types:
                    # Try to infer a better type if possible
                    inferred_type = self._infer_entity_type_from_name(entity_name)
                    entity_type = inferred_type if inferred_type in self.hierarchical_entity_types else 'CONCEPT'
                
                # Create enhanced entity with business value score
                entity = ExtractedEntity(
                    text=entity_name.strip(),
                    label=entity_type,
                    start_char=raw_entity.get('start_char', 0) if isinstance(raw_entity, dict) else 0,
                    end_char=raw_entity.get('end_char', len(entity_name)) if isinstance(raw_entity, dict) else len(entity_name),
                    confidence=confidence,
                    canonical_form=canonical_form,
                    properties={
                        'business_value_score': business_value_score,
                        'strategic_relevance': raw_entity.get('strategic_relevance', '') if isinstance(raw_entity, dict) else '',
                        'evidence': raw_entity.get('evidence', '') if isinstance(raw_entity, dict) else ''
                    }
                )
                
                enhanced_entities.append(entity)
                logger.debug(f"  ✅ Created entity [{i}]: '{entity_name}' (type: {entity_type}, business_value: {business_value_score:.2f}, canonical: '{canonical_form}')")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to process entity [{i}] {raw_entity}: {e}")
                continue
        
        logger.info(f"🔧 Enhanced entities: {len(raw_entities)} raw → {len(enhanced_entities)} valid entities")
        
        # Enhanced debugging for empty entities issue
        if len(enhanced_entities) == 0 and len(raw_entities) > 0:
            logger.error(f"🚨 CRITICAL: All {len(raw_entities)} entities were filtered out!")
            logger.error(f"Raw entities types: {[type(e).__name__ for e in raw_entities[:5]]}")
            logger.error(f"Raw entities sample: {raw_entities[:3]}")
            
            # Try emergency entity creation from strings
            emergency_entities = []
            for raw in raw_entities[:10]:  # Try first 10 entities
                if isinstance(raw, str) and len(raw.strip()) > 1:
                    try:
                        emergency_entity = ExtractedEntity(
                            text=raw.strip(),
                            label='CONCEPT',
                            start_char=0,
                            end_char=len(raw.strip()),
                            confidence=0.5,
                            canonical_form=raw.strip().title()
                        )
                        emergency_entities.append(emergency_entity)
                        logger.info(f"🚑 Emergency entity created: '{raw.strip()}'")
                    except Exception as emergency_e:
                        logger.error(f"🚑 Emergency entity creation failed for '{raw}': {emergency_e}")
            
            if emergency_entities:
                logger.warning(f"🚑 Using {len(emergency_entities)} emergency entities")
                return emergency_entities
            
        return enhanced_entities
    
    def _infer_entity_type_from_name(self, entity_name: str) -> str:
        """Enhanced entity type inference with high-value business pattern matching"""
        name_lower = entity_name.lower()
        
        # HIGH-VALUE PATTERN MATCHING
        
        # Executive patterns with names (highest value)
        if re.search(r'\b(ceo|chief executive officer|president|chairman|founder)\b', name_lower) and \
           len(name_lower.split()) >= 3:  # Must include actual name
            return 'EXECUTIVE'
            
        # Financial metrics with specific amounts (highest value)
        if re.search(r'\$[\d,.]+(billion|million|k|thousand)', name_lower) or \
           re.search(r'\d+(\.\d+)?%', name_lower) or \
           re.search(r'(revenue|profit|earnings|income|cost|investment|funding|capital|budget)', name_lower) and \
           (re.search(r'\$', name_lower) or re.search(r'\d', name_lower)):
            return 'FINANCIAL_METRIC'
        
        # TECHNOLOGY PATTERNS (Enhanced to properly classify tech entities)
        
        # Specific database and data technologies
        database_patterns = [
            r'\b(oceanbase|sofastack|tdsql|redis|mongodb|postgresql|mysql|oracle|cassandra)\b',
            r'\b(elasticsearch|clickhouse|snowflake|databricks|neo4j|dynamodb|bigquery)\b',
            r'\b(hadoop|spark|flink|kafka|apache kafka|rabbitmq|activemq)\b'
        ]
        if any(re.search(pattern, name_lower) for pattern in database_patterns):
            return 'TECHNOLOGY'
            
        # Cloud and infrastructure technologies
        cloud_tech_patterns = [
            r'\b(aws|amazon web services|azure|microsoft azure|google cloud|gcp)\b',
            r'\b(kubernetes|docker|terraform|ansible|jenkins|gitlab|github)\b',
            r'\b(openstack|vmware|citrix|alibaba cloud|tencent cloud)\b',
            r'\b(microservices|serverless|containerization|service mesh)\b'
        ]
        if any(re.search(pattern, name_lower) for pattern in cloud_tech_patterns):
            return 'TECHNOLOGY'
            
        # AI/ML and strategic technologies
        ai_ml_patterns = [
            r'\b(artificial intelligence|machine learning|deep learning|neural network)\b',
            r'\b(natural language processing|computer vision|blockchain|smart contract)\b',
            r'\b(quantum computing|edge computing|iot|internet of things)\b',
            r'\b(augmented reality|virtual reality|ar|vr|digital identity|biometric)\b'
        ]
        if any(re.search(pattern, name_lower) for pattern in ai_ml_patterns):
            return 'TECHNOLOGY'
            
        # Business technology strategies and methodologies
        tech_strategy_patterns = [
            r'\b(cloud migration|digital transformation|devops|cicd|api economy)\b',
            r'\b(cloud native|hybrid cloud|multi-cloud|data lake|data warehouse)\b',
            r'\b(real-time analytics|stream processing|event-driven architecture)\b',
            r'\b(mainframe decommissioning|legacy modernization|innovation strategy)\b'
        ]
        if any(re.search(pattern, name_lower) for pattern in tech_strategy_patterns):
            return 'TECHNOLOGY'
            
        # Fintech and banking technologies
        fintech_patterns = [
            r'\b(core banking|payment gateway|digital wallet|mobile banking|open banking)\b',
            r'\b(api banking|regtech|compliance automation|fraud detection|kyc|aml)\b',
            r'\b(risk management system|pci dss|cryptocurrency|digital payment)\b'
        ]
        if any(re.search(pattern, name_lower) for pattern in fintech_patterns):
            return 'TECHNOLOGY'
            
        # Programming languages and frameworks (in business context)
        programming_patterns = [
            r'\b(java|python|javascript|typescript|react|angular|vue)\s+\w+\b',  # Must be compound
            r'\b(spring boot|node\.js|express|\.net|microservice)\b',
            r'\b\w+\s+(framework|library|runtime|sdk|api)\b'
        ]
        if any(re.search(pattern, name_lower) for pattern in programming_patterns):
            return 'TECHNOLOGY'
        
        # Specific organization names (must be complete)
        org_patterns = [
            r'\b\w+\s+(bank|corp|corporation|company|ltd|inc|group|holdings|division|subsidiary)\b',
            r'\b(dbs|ant financial|grab|gojek|shopee|lazada|sea limited)\b'
        ]
        if any(re.search(pattern, name_lower) for pattern in org_patterns):
            return 'ORGANIZATION'
        
        # Branded products/services (must be specific)
        product_patterns = [
            r'\b\w+\s+(app|platform|service|system|solution)\b',  # Like "PayLah! app"
            r'\b(payLah|grabpay|shopeepay|gojek|grab|uber)\b'
        ]
        if any(re.search(pattern, name_lower) for pattern in product_patterns):
            return 'PRODUCT'
        
        # Strategic initiatives (must be named)
        strategy_patterns = [
            r'\b\w+\s+(strategy|initiative|program|project|transformation)\b',  # Like "Digital Transformation 2025"
            r'\b(business transformation|organizational change|process improvement)\s+\w+\b'
        ]
        if any(re.search(pattern, name_lower) for pattern in strategy_patterns):
            return 'STRATEGY'
        
        # Specific geographic markets
        location_patterns = [
            r'\b(singapore|hong kong|indonesia|thailand|malaysia|philippines|vietnam)\b',
            r'\b(southeast asia|asia pacific|asean|greater china)\b',
            r'\b\w+\s+(market|region|office|headquarters|branch|operations)\b'
        ]
        if any(re.search(pattern, name_lower) for pattern in location_patterns):
            return 'LOCATION'
        
        # Technology stacks (must be specific)
        tech_patterns = [
            r'\b(aws|azure|google cloud|kubernetes|docker|microservices)\b',
            r'\b\w+\s+(api|database|system|platform|infrastructure)\b'
        ]
        if any(re.search(pattern, name_lower) for pattern in tech_patterns):
            return 'TECHNOLOGY'
        
        # Temporal events (must be specific)
        temporal_patterns = [
            r'\b(q[1-4]\s+20\d{2}|quarter\s+[1-4]|h[12]\s+20\d{2})\b',  # Q3 2024, Quarter 1, H1 2024
            r'\b(20\d{2}\s+(earnings|results|launch|announcement))\b'
        ]
        if any(re.search(pattern, name_lower) for pattern in temporal_patterns):
            return 'TEMPORAL'
        
        # Default to CONCEPT for anything else
        return 'CONCEPT'
    
    def _calculate_business_value_score(self, entity_name: str, entity_type: str) -> float:
        """Calculate business value score for an entity (0.0 to 1.0)"""
        name_lower = entity_name.lower().strip()
        score = 0.5  # Base score
        
        # HIGH VALUE PATTERNS (Add significant points)
        
        # Financial metrics with specific amounts
        if re.search(r'\$[\d,.]+(billion|million|k|thousand)', name_lower) or \
           re.search(r'\d+%|\d+\.\d+%', name_lower):
            score += 0.4  # Financial specificity is highly valuable
            
        # Named executives with titles
        if any(title in name_lower for title in ['ceo ', 'cto ', 'cfo ', 'coo ', 'cmo ', 'president ', 'chairman ']):
            if len(name_lower.split()) >= 3:  # Must have actual name, not just title
                score += 0.3
            else:
                score -= 0.2  # Generic title without name is low value
                
        # Specific company/organization names with qualifiers
        if any(org_type in name_lower for org_type in ['bank', 'corp', 'company', 'ltd', 'inc', 'group']):
            if len(name_lower.split()) >= 2:  # Must be specific, not just "bank"
                score += 0.3
                
        # Branded products/services
        if any(brand_indicator in name_lower for brand_indicator in ['app', 'platform', 'service']) and \
           len(name_lower.split()) >= 2:  # Must be specific like "PayLah! app"
            score += 0.2
            
        # Strategic initiatives with specific names
        if any(initiative in name_lower for initiative in ['project ', 'initiative', 'program', 'transformation']) and \
           len(name_lower.split()) >= 2:
            score += 0.2
            
        # Specific geographic markets
        specific_locations = ['singapore', 'hong kong', 'southeast asia', 'asia pacific', 'indonesia', 'thailand']
        if any(loc in name_lower for loc in specific_locations):
            score += 0.2
            
        # TECHNOLOGY VALUE BOOST PATTERNS (Critical for business strategy documents)
        
        # Specific technology names and databases (High business value)
        specific_tech_names = [
            'oceanbase', 'sofastack', 'tdsql', 'apache kafka', 'kafka', 'redis', 'mongodb', 'postgresql', 
            'mysql', 'oracle', 'elasticsearch', 'kubernetes', 'docker', 'terraform', 'ansible',
            'hadoop', 'spark', 'flink', 'cassandra', 'clickhouse', 'snowflake', 'databricks'
        ]
        if any(tech in name_lower for tech in specific_tech_names):
            score += 0.3  # Specific tech names are highly valuable
            
        # Strategic technology terms (Medium-high business value)
        strategic_tech_terms = [
            'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
            'blockchain', 'smart contract', 'cryptocurrency', 'digital identity', 'biometric',
            'quantum computing', 'edge computing', 'iot', 'internet of things', 'ar', 'vr',
            'augmented reality', 'virtual reality', 'natural language processing', 'computer vision'
        ]
        if any(term in name_lower for term in strategic_tech_terms):
            score += 0.2  # Strategic tech concepts are valuable
            
        # Business technology strategies (Medium business value)
        business_tech_strategies = [
            'cloud migration', 'digital transformation', 'devops', 'cicd', 'microservices',
            'api economy', 'api gateway', 'service mesh', 'containerization', 'serverless',
            'cloud native', 'hybrid cloud', 'multi-cloud', 'data lake', 'data warehouse',
            'real-time analytics', 'stream processing', 'event-driven architecture',
            'mainframe decommissioning', 'legacy modernization', 'innovation strategy'
        ]
        if any(strategy in name_lower for strategy in business_tech_strategies):
            score += 0.25  # Business tech strategies are valuable
            
        # Cloud platforms and major tech stacks
        cloud_platforms = [
            'aws', 'amazon web services', 'azure', 'microsoft azure', 'google cloud', 'gcp',
            'alibaba cloud', 'tencent cloud', 'openstack', 'vmware', 'citrix'
        ]
        if any(platform in name_lower for platform in cloud_platforms):
            score += 0.3  # Cloud platforms are highly strategic
            
        # Fintech and banking technologies
        fintech_tech = [
            'core banking', 'payment gateway', 'digital wallet', 'mobile banking', 
            'open banking', 'api banking', 'regtech', 'compliance automation',
            'risk management system', 'fraud detection', 'kyc', 'aml', 'pci dss'
        ]
        if any(tech in name_lower for tech in fintech_tech):
            score += 0.25  # Fintech tech is valuable for financial institutions
            
        # Programming languages and frameworks (when mentioned in business context)
        programming_tech = [
            'java', 'python', 'javascript', 'typescript', 'react', 'angular', 'vue',
            'spring boot', 'node.js', 'express', '.net', 'c#', 'go', 'rust', 'scala'
        ]
        # Only boost if it's part of a larger tech discussion (compound terms)
        if any(tech in name_lower for tech in programming_tech) and len(name_lower.split()) >= 2:
            score += 0.15  # Programming tech in business context gets modest boost
            
        # LOW VALUE PATTERNS (Subtract points)
        
        # Generic business terms without specificity
        generic_terms = [
            'system', 'platform', 'technology', 'solution', 'approach', 'way', 'method',
            'process', 'business', 'digital', 'innovation', 'growth', 'development',
            'management', 'operation', 'service', 'product', 'application'
        ]
        if name_lower in generic_terms:
            score -= 0.4
            
        # Pronouns and articles
        meaningless_terms = ['the', 'this', 'that', 'these', 'those', 'it', 'they', 'we', 'us']
        if name_lower in meaningless_terms:
            score -= 0.6
            
        # Single character or very short generic terms
        if len(name_lower) <= 2:
            score -= 0.4
            
        # Common stop words
        stop_words = ['and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        if name_lower in stop_words:
            score -= 0.6
            
        # REBALANCED: Enhanced entity type multipliers for business entities
        if entity_type in ['ORGANIZATION', 'ORG', 'COMPANY', 'BANK', 'CORPORATION', 'FINTECH']:
            score *= 1.3  # Higher boost for organization entities
        elif entity_type in ['EXECUTIVE', 'FINANCIAL_METRIC', 'PRODUCT']:
            score *= 1.2  # Boost business-relevant types
        elif entity_type in ['CONCEPT', 'GENERIC']:
            score *= 0.8  # Reduce generic concepts
            
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
    
    def _is_generic_business_term(self, entity_name: str) -> bool:
        """Check if entity is a generic business term that should be filtered out"""
        name_lower = entity_name.lower().strip()
        
        # Comprehensive list of generic business terms
        generic_business_terms = {
            # Generic roles without names
            'ceo', 'cto', 'cfo', 'coo', 'cmo', 'president', 'chairman', 'director', 'manager',
            'executive', 'leader', 'team', 'staff', 'employee', 'consultant', 'advisor',
            
            # Vague technology concepts
            'system', 'platform', 'technology', 'solution', 'software', 'hardware', 'application',
            'tool', 'framework', 'infrastructure', 'architecture', 'implementation',
            
            # Generic business concepts
            'business', 'company', 'organization', 'enterprise', 'corporation', 'firm',
            'strategy', 'initiative', 'program', 'project', 'approach', 'method', 'process',
            'operation', 'function', 'activity', 'service', 'product', 'offering',
            
            # Vague descriptors
            'digital', 'innovation', 'transformation', 'modernization', 'optimization',
            'improvement', 'enhancement', 'development', 'growth', 'expansion',
            'management', 'governance', 'compliance', 'quality', 'performance',
            
            # Common words that are never meaningful as entities
            'the', 'this', 'that', 'these', 'those', 'they', 'it', 'we', 'us', 'our',
            'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'way', 'thing', 'something', 'anything', 'everything', 'nothing',
            
            # Generic time/location without specificity
            'today', 'tomorrow', 'yesterday', 'now', 'then', 'here', 'there', 'where',
            'market', 'industry', 'sector', 'region', 'area', 'location', 'place'
        }
        
        # Check exact match
        if name_lower in generic_business_terms:
            return True
            
        # Check for very short non-specific terms
        if len(name_lower) <= 2 and name_lower not in ['ai', 'ml', 'db', 'ui', 'ux']:
            return True
            
        # REBALANCED: Less aggressive filtering of business terms - allow many business entity types
        if len(name_lower.split()) == 1:
            # Only filter extremely generic terms, keep business entity types
            extremely_generic_words = {
                'data', 'analytics', 'security', 'mobile', 'web'
                # REMOVED: 'bank', 'finance', 'payment', 'credit', 'loan', 'investment', 'cloud'
                # These are legitimate business entity types and should not be filtered
            }
            if name_lower in extremely_generic_words:
                return True
                
        return False
    
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
                # Handle multiple field name formats from different LLM responses
                source = (raw_rel.get('source_entity', '') or 
                         raw_rel.get('source', '') or
                         raw_rel.get('entity1', '') or
                         raw_rel.get('subject', '')).strip()
                
                target = (raw_rel.get('target_entity', '') or
                         raw_rel.get('target', '') or 
                         raw_rel.get('entity2', '') or
                         raw_rel.get('object', '')).strip()
                
                rel_type = (raw_rel.get('relationship_type', '') or
                           raw_rel.get('relation', '') or
                           raw_rel.get('type', '') or
                           raw_rel.get('relationship', '')).upper()
                
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
                    logger.debug(f"  ⚠️ Invalid relationship type '{rel_type}', using 'USES'")
                    rel_type = 'USES'  # CHANGED: Default to specific type instead of generic
                
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
        
        # Apply relationship filtering to reduce by 25% while preserving quality
        filtered_relationships = self._filter_relationships_for_quality(validated_relationships)
        
        if len(filtered_relationships) != len(validated_relationships):
            logger.info(f"Relationship filtering applied: {len(validated_relationships)} -> {len(filtered_relationships)} "
                       f"({len(validated_relationships) - len(filtered_relationships)} removed)")
        
        return filtered_relationships
    
    def _filter_relationships_for_quality(self, relationships: List[ExtractedRelationship]) -> List[ExtractedRelationship]:
        """Filter relationships to reduce count by ~25% while preserving quality"""
        if not relationships:
            return relationships
            
        # Calculate relationship quality scores
        scored_relationships = []
        for rel in relationships:
            score = self._calculate_relationship_quality_score(rel)
            scored_relationships.append((rel, score))
        
        # Sort by quality score (descending - highest quality first)
        scored_relationships.sort(key=lambda x: x[1], reverse=True)
        
        # Target reduction: keep top 75% (remove bottom 25%)
        target_count = max(1, int(len(relationships) * 0.75))
        
        # Keep the highest quality relationships
        filtered_relationships = [rel for rel, score in scored_relationships[:target_count]]
        
        return filtered_relationships
    
    def _calculate_relationship_quality_score(self, rel: ExtractedRelationship) -> float:
        """Calculate quality score for a relationship (0.0 to 1.0)"""
        score = rel.confidence  # Start with base confidence
        
        # Boost high-value relationship types
        high_value_types = [
            'IMPLEMENTS', 'USES', 'MANAGES', 'OPERATES', 'DEVELOPS', 'COLLABORATES_WITH',
            'PARTNERS_WITH', 'ACQUIRED_BY', 'INVESTS_IN', 'COMPETES_WITH', 'SUPPLIES_TO',
            'MIGRATES_TO', 'REPLACES', 'INTEGRATES_WITH', 'DEPENDS_ON'
        ]
        if rel.relationship_type in high_value_types:
            score += 0.2
            
        # Boost relationships with specific context/evidence
        if rel.context and len(rel.context.strip()) > 20:
            score += 0.15
            
        # Boost relationships between technology entities (strategic importance)
        tech_indicators = ['platform', 'system', 'service', 'technology', 'database', 'cloud', 'api']
        source_is_tech = any(indicator in rel.source_entity.lower() for indicator in tech_indicators)
        target_is_tech = any(indicator in rel.target_entity.lower() for indicator in tech_indicators)
        
        if source_is_tech and target_is_tech:
            score += 0.25  # Tech-to-tech relationships are highly valuable
        elif source_is_tech or target_is_tech:
            score += 0.15  # Mixed tech relationships are moderately valuable
            
        # Penalize generic relationship types
        generic_types = ['RELATED_TO', 'ASSOCIATED_WITH', 'CONNECTED_TO']
        if rel.relationship_type in generic_types:
            score -= 0.15
            
        # Penalize fuzzy matched relationships (less reliable)
        if rel.properties.get('fuzzy_matched', False):
            score -= 0.1
            
        # Boost relationships with temporal information
        if rel.properties.get('temporal_info'):
            score += 0.1
            
        # Boost relationships with detailed attributes
        if rel.properties.get('attributes') and len(rel.properties.get('attributes', {})) > 0:
            score += 0.1
            
        # Penalize very short entity names (likely generic)
        if len(rel.source_entity) <= 3 or len(rel.target_entity) <= 3:
            score -= 0.2
            
        return max(0.0, min(1.0, score))
    
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
        """Check if two entities are similar using STRICTER heuristics to preserve more entities"""
        # Only consider them similar if they are VERY close matches
        
        # Exact match after stripping whitespace
        if entity1.strip() == entity2.strip():
            return True
            
        # Check for abbreviations only if one is exactly contained in the other
        # AND the shorter one is at least 3 characters (to avoid false matches)
        if len(entity1) >= 3 and len(entity2) >= 3:
            if entity1 == entity2[:len(entity1)] or entity2 == entity1[:len(entity2)]:
                # Additional check: the longer one should be reasonably close in length
                if abs(len(entity1) - len(entity2)) <= 5:
                    return True
        
        # Very strict suffix check - only for exact company suffixes
        suffixes = [' inc.', ' ltd.', ' corp.', ' co.']
        for suffix in suffixes:
            if entity1.endswith(suffix) and entity2 == entity1[:-len(suffix)]:
                return True
            if entity2.endswith(suffix) and entity1 == entity2[:-len(suffix)]:
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
        
        # Default fallback - use specific type instead of generic
        return 'COLLABORATES_WITH'
    
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

    async def _extraction_pass_core_entities(self, text: str, context: Optional[Dict[str, Any]], 
                                           domain_hints: Optional[List[str]], dynamic_timeout: Optional[int] = None) -> Dict[str, Any]:
        """Pass 1: AGGRESSIVE core business entities extraction - 4x more comprehensive"""
        start_time = datetime.now()
        
        # Build SUPER AGGRESSIVE prompt for core entities
        from app.services.settings_prompt_service import get_prompt_service as get_settings_prompt_service
        prompt_service = get_settings_prompt_service()
        
        # Get dynamic schema focused on core entity types
        dynamic_schema = await dynamic_schema_manager.get_combined_schema()
        core_entity_types = [t for t in dynamic_schema['entity_types'] 
                           if any(core in t.upper() for core in ['ORGANIZATION', 'PERSON', 'TECHNOLOGY', 'PRODUCT', 'COMPANY', 'EXECUTIVE', 'BANK', 'FINTECH'])]
        
        # ULTRA-AGGRESSIVE extraction prompt
        specialized_prompt = f"""
🎯 ULTRA-AGGRESSIVE BUSINESS ENTITY EXTRACTION - TARGET: 60-70+ ENTITIES

YOU ARE A BUSINESS INTELLIGENCE EXPERT. Your task is to extract EVERY SINGLE business entity mentioned in this text.
DO NOT be conservative - extract aggressively and comprehensively.

PRIORITY ENTITY CATEGORIES (EXTRACT ALL):
{', '.join(core_entity_types)}

TEXT TO ANALYZE ({len(text):,} characters):
{text}

ULTRA-AGGRESSIVE EXTRACTION REQUIREMENTS:
✅ ORGANIZATIONS: Every company, bank, subsidiary, division, business unit, department, team
   - Full names AND abbreviations (DBS Bank, DBS, Development Bank of Singapore)
   - Parent companies AND subsidiaries (Ant Group → Alipay, Alibaba)
   - Regional entities (DBS Singapore, DBS Hong Kong)
   - Business divisions (Corporate Banking, Retail Banking, Wealth Management)

✅ PEOPLE & ROLES: Every person, executive, title, role mentioned
   - Full names (Piyush Gupta)
   - Titles (CEO, CTO, Chief Technology Officer, Head of Digital)
   - Roles (analysts, managers, customers, partners)
   - Generic roles (executives, leadership team, board members)

✅ TECHNOLOGIES & SYSTEMS: Every technology, platform, system, tool
   - Database systems (OceanBase, MariaDB, Oracle, PostgreSQL)
   - Platforms (cloud platforms, trading platforms, payment platforms)
   - Technologies (AI, machine learning, blockchain, automation)
   - Software (applications, APIs, frameworks, tools)

✅ PRODUCTS & SERVICES: Every product, service, offering mentioned
   - Banking products (loans, deposits, credit cards, investment products)
   - Digital services (mobile banking, internet banking, payment services)
   - Platform services (trading, wealth management, corporate banking)

✅ LOCATIONS & FACILITIES: Every geographic entity, office, facility
   - Countries (Singapore, Hong Kong, Indonesia, Thailand)
   - Cities (Singapore, Hong Kong, Jakarta, Bangkok)
   - Regions (Southeast Asia, Asia Pacific)
   - Facilities (headquarters, regional offices, data centers)

✅ BUSINESS CONCEPTS: Every strategic concept, initiative, program
   - Strategies (digital transformation, innovation strategy)
   - Initiatives (modernization programs, cost optimization)
   - Concepts (competitive advantage, market leadership)

EXTRACTION RULES:
1. Extract VARIANTS: "DBS" and "DBS Bank" and "Development Bank of Singapore" as separate entities
2. Extract HIERARCHIES: "Ant Group" and "Alipay" and "Alibaba" as separate entities
3. Extract ABBREVIATIONS: Both "CEO" and "Chief Executive Officer"
4. Extract NUMBERS/METRICS: Any percentage, cost, revenue, target mentioned
5. Extract TIMEFRAMES: Q1 2024, FY2023, next quarter, 2025 targets
6. NO FILTERING: Include everything that could remotely be a business entity
7. CONFIDENCE THRESHOLD: Include entities with confidence >= 0.3 (very aggressive)

🎯 TARGET: Extract 15-25 entities from this chunk alone (not 2-3 entities!)

Output COMPREHENSIVE JSON with:
- "entities": Array of ALL entities found with types and confidence
- "relationships": Basic relationships between entities
- "reasoning": Brief explanation of extraction strategy

BE AGGRESSIVE - Extract 4x more entities than you normally would!
"""
        
        try:
            llm_response = await self._call_llm_for_extraction(specialized_prompt, dynamic_timeout)
            parsed_result = self._parse_llm_response(llm_response)
            
            entities = self._enhance_entities_with_hierarchy(parsed_result.get('entities', []))
            relationships = self._validate_and_score_relationships(parsed_result.get('relationships', []), entities)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"✅ Pass 1 complete: {len(entities)} core entities, {len(relationships)} relationships")
            
            return {
                'entities': entities,
                'relationships': relationships,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Core entity extraction pass failed: {e}")
            return {'entities': [], 'relationships': [], 'processing_time_ms': 0}
    
    async def _extraction_pass_business_concepts(self, text: str, existing_entities: List[ExtractedEntity],
                                               context: Optional[Dict[str, Any]], domain_hints: Optional[List[str]],
                                               dynamic_timeout: Optional[int] = None) -> Dict[str, Any]:
        """Pass 2: ULTRA-AGGRESSIVE business concepts, strategies, metrics extraction"""
        start_time = datetime.now()
        
        # Build HYPER-AGGRESSIVE prompt for business concepts
        entity_context = [e.canonical_form for e in existing_entities[:30]]  # More context entities
        
        business_concepts_prompt = f"""
🚀 ULTRA-AGGRESSIVE BUSINESS CONCEPTS EXTRACTION - TARGET: 20-30+ MORE ENTITIES

You are extracting from a BUSINESS STRATEGY DOCUMENT. Be extremely aggressive in finding ALL business concepts.

KNOWN ENTITIES FROM PREVIOUS PASS:
{', '.join(entity_context)}

TEXT TO ANALYZE ({len(text):,} characters):
{text}

🎯 AGGRESSIVE EXTRACTION TARGETS:

✅ STRATEGIES & INITIATIVES (Extract ALL mentions):
   - Digital transformation, innovation strategy, growth strategy, competitive strategy
   - Modernization programs, optimization initiatives, transformation programs
   - Cost reduction initiatives, efficiency programs, automation projects
   - Market expansion, customer acquisition, product development
   - Technology adoption, system upgrades, platform migrations

✅ BUSINESS METRICS & KPIs (Extract ALL numbers/targets):
   - Revenue figures, growth percentages, market share numbers
   - Cost savings, efficiency gains, productivity improvements
   - ROI, ROE, profit margins, expense ratios
   - Customer metrics, satisfaction scores, NPS scores
   - Performance indicators, benchmarks, targets

✅ MARKETS & SEGMENTS (Extract ALL market references):
   - Customer segments (retail, corporate, SME, wealth)
   - Geographic markets (Singapore, Hong Kong, Southeast Asia)
   - Industry verticals (financial services, banking, fintech)
   - Market positions (leader, challenger, niche player)
   - Competitive landscapes, market dynamics

✅ FINANCIAL CONCEPTS (Extract ALL financial elements):
   - Investment rounds, funding sources, capital allocation
   - Budget allocations, cost structures, revenue streams
   - Risk factors, compliance requirements, regulatory capital
   - Profitability drivers, value creation, shareholder value

✅ OPERATIONAL CONCEPTS (Extract ALL operational elements):
   - Business processes, workflows, operational excellence
   - Customer journeys, user experiences, service delivery
   - Quality metrics, service levels, performance standards
   - Automation levels, digital adoption, technology utilization

✅ TEMPORAL & MILESTONE ELEMENTS (Extract ALL time references):
   - Project timelines, implementation phases, rollout schedules
   - Quarterly targets, annual goals, multi-year plans
   - Milestone dates, deadline commitments, launch schedules
   - Historical achievements, future projections, trend analysis

✅ COMPETITIVE & MARKET INTELLIGENCE:
   - Competitive advantages, differentiation factors, unique value propositions
   - Market opportunities, competitive threats, industry disruptions
   - Partnership strategies, alliance frameworks, ecosystem participation

AGGRESSIVE EXTRACTION RULES:
1. Extract EVERY percentage, dollar amount, timeline mentioned
2. Extract CONCEPTS even if implied ("improve efficiency" → "efficiency improvement")
3. Extract COMPOUND CONCEPTS ("digital transformation strategy" as separate entity)
4. Extract INDUSTRY JARGON and technical terms
5. Extract COMPARATIVE TERMS ("market-leading", "best-in-class")
6. CONFIDENCE THRESHOLD: Include concepts with confidence >= 0.2 (extremely aggressive)

🎯 TARGET: Extract 20-30 additional business concepts from this chunk!

Output COMPREHENSIVE JSON with:
- "entities": ALL business concepts found (aim for 20-30 new entities)
- "relationships": Strategic relationships between concepts and existing entities  
- "reasoning": Strategy used for aggressive extraction

BE ULTRA-AGGRESSIVE - This is a business strategy document, every concept matters!
"""
        
        try:
            llm_response = await self._call_llm_for_extraction(business_concepts_prompt, dynamic_timeout)
            parsed_result = self._parse_llm_response(llm_response)
            
            entities = self._enhance_entities_with_hierarchy(parsed_result.get('entities', []))
            relationships = self._validate_and_score_relationships(parsed_result.get('relationships', []), 
                                                                 existing_entities + entities)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"✅ Pass 2 complete: {len(entities)} business concepts, {len(relationships)} relationships")
            
            return {
                'entities': entities,
                'relationships': relationships,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Business concepts extraction pass failed: {e}")
            return {'entities': [], 'relationships': [], 'processing_time_ms': 0}
    
    async def _extraction_pass_deep_relationships(self, text: str, existing_entities: List[ExtractedEntity],
                                                context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Pass 3: Deep relationship analysis and inference between known entities"""
        start_time = datetime.now()
        
        # Focus on finding relationships between existing entities
        entity_names = [e.canonical_form for e in existing_entities]
        
        if len(entity_names) < 2:
            return {'relationships': [], 'processing_time_ms': 0}
        
        deep_relationships_prompt = f"""
SPECIALIZED EXTRACTION TASK: Deep Relationship Analysis

KNOWN ENTITIES:
{', '.join(entity_names)}

TEXT TO ANALYZE:
{text}

TASK: Find ALL relationships between the known entities listed above.

RELATIONSHIP CATEGORIES TO FOCUS ON:
1. ORGANIZATIONAL: owns, manages, reports_to, subsidiary_of, division_of, partners_with
2. BUSINESS: competes_with, serves, supplies, acquires, invests_in, collaborates_with
3. OPERATIONAL: uses, implements, integrates_with, enables, supports, powers
4. STRATEGIC: targets, focuses_on, prioritizes, addresses, transforms, disrupts
5. GEOGRAPHIC: located_in, operates_in, based_in, serves_region, expands_to
6. TEMPORAL: founded_in, launched_in, announced_in, planned_for, scheduled_for

INFERENCE RULES:
- If entities appear in same sentence/paragraph, infer relationships
- Use business context to determine relationship types
- Consider co-occurrence patterns for implicit relationships
- Identify supply chain, value chain, and ecosystem relationships

Output JSON focused on relationships between known entities.
"""
        
        try:
            llm_response = await self._call_llm_for_extraction(deep_relationships_prompt)
            parsed_result = self._parse_llm_response(llm_response)
            
            relationships = self._validate_and_score_relationships(parsed_result.get('relationships', []), existing_entities)
            
            # Add co-occurrence based relationships
            cooccurrence_rels = await self._infer_cooccurrence_relationships(text, existing_entities)
            relationships.extend(cooccurrence_rels)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"✅ Pass 3 complete: {len(relationships)} deep relationships")
            
            return {
                'relationships': relationships,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Deep relationships extraction pass failed: {e}")
            return {'relationships': [], 'processing_time_ms': 0}
    
    async def _extraction_pass_temporal_causal(self, text: str, existing_entities: List[ExtractedEntity],
                                             existing_relationships: List[ExtractedRelationship],
                                             context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Pass 4: Temporal sequences and causal relationships"""
        start_time = datetime.now()
        
        temporal_causal_prompt = f"""
SPECIALIZED EXTRACTION TASK: Temporal and Causal Analysis

KNOWN ENTITIES: {', '.join([e.canonical_form for e in existing_entities[:30]])}

TEXT TO ANALYZE:
{text}

FOCUS ON:
1. TEMPORAL ENTITIES: dates, years, quarters, periods, phases, timelines
2. CAUSAL RELATIONSHIPS: causes, leads_to, results_in, enables, triggers, drives
3. TEMPORAL SEQUENCES: before, after, during, followed_by, preceded_by
4. BUSINESS CAUSALITY: strategy leads to initiatives, investments enable growth
5. TIMELINE ANALYSIS: announcement -> launch -> adoption -> results

EXTRACT:
- Time-related entities (Q1 2024, FY2023, next quarter, etc.)
- Causal chains (investment -> technology -> competitive advantage)
- Temporal dependencies (Phase 1 -> Phase 2 -> Phase 3)
- Business impact relationships (initiative -> improvement -> results)

Output JSON with temporal entities and causal/temporal relationships.
"""
        
        try:
            llm_response = await self._call_llm_for_extraction(temporal_causal_prompt)
            parsed_result = self._parse_llm_response(llm_response)
            
            entities = self._enhance_entities_with_hierarchy(parsed_result.get('entities', []))
            relationships = self._validate_and_score_relationships(parsed_result.get('relationships', []), 
                                                                 existing_entities + entities)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"✅ Pass 4 complete: {len(entities)} temporal entities, {len(relationships)} causal relationships")
            
            return {
                'entities': entities,
                'relationships': relationships,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Temporal causal extraction pass failed: {e}")
            return {'entities': [], 'relationships': [], 'processing_time_ms': 0}
    
    def _consolidate_entities(self, all_entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Consolidate and deduplicate entities from multiple passes"""
        entity_map = {}
        
        for entity in all_entities:
            # Normalize entity name for matching
            normalized_name = entity.canonical_form.lower().strip()
            
            # Check for exact matches first
            if normalized_name in entity_map:
                # Keep entity with higher confidence
                if entity.confidence > entity_map[normalized_name].confidence:
                    entity_map[normalized_name] = entity
            else:
                # Check for fuzzy matches (similar entities)
                found_similar = False
                for existing_name, existing_entity in entity_map.items():
                    if self._entities_similar(normalized_name, existing_name):
                        # Merge with higher confidence entity
                        if entity.confidence > existing_entity.confidence:
                            # Replace with higher confidence version
                            del entity_map[existing_name]
                            entity_map[normalized_name] = entity
                        found_similar = True
                        break
                
                if not found_similar:
                    entity_map[normalized_name] = entity
        
        consolidated = list(entity_map.values())
        logger.info(f"🔧 Entity consolidation: {len(all_entities)} -> {len(consolidated)} entities")
        return consolidated
    
    def _consolidate_relationships(self, all_relationships: List[ExtractedRelationship], 
                                 final_entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Consolidate and deduplicate relationships, ensuring entities exist"""
        entity_names = {e.canonical_form.lower(): e.canonical_form for e in final_entities}
        relationship_set = set()
        valid_relationships = []
        
        for rel in all_relationships:
            # Normalize entity names
            source_norm = rel.source_entity.lower().strip()
            target_norm = rel.target_entity.lower().strip()
            
            # Find matching entities (with fuzzy matching)
            source_match = self._find_entity_match(source_norm, entity_names)
            target_match = self._find_entity_match(target_norm, entity_names)
            
            if source_match and target_match and source_match != target_match:
                # Create relationship key for deduplication
                rel_key = (source_match, target_match, rel.relationship_type)
                
                if rel_key not in relationship_set:
                    relationship_set.add(rel_key)
                    
                    # Create clean relationship with matched entity names
                    clean_rel = ExtractedRelationship(
                        source_entity=source_match,
                        target_entity=target_match,
                        relationship_type=rel.relationship_type,
                        confidence=rel.confidence,
                        context=rel.context,
                        properties=rel.properties or {}
                    )
                    valid_relationships.append(clean_rel)
        
        logger.info(f"🔧 Relationship consolidation: {len(all_relationships)} -> {len(valid_relationships)} relationships")
        
        # AGGRESSIVE FILTERING: Apply hard caps and quality filtering
        filtered_relationships = self._apply_aggressive_relationship_filtering(valid_relationships, final_entities)
        logger.info(f"🚨 AGGRESSIVE FILTERING: {len(valid_relationships)} -> {len(filtered_relationships)} relationships")
        
        return filtered_relationships
    
    def _find_entity_match(self, normalized_name: str, entity_names: Dict[str, str]) -> Optional[str]:
        """Find matching entity name with fuzzy matching"""
        # Exact match first
        if normalized_name in entity_names:
            return entity_names[normalized_name]
        
        # Fuzzy matching
        for existing_norm, canonical in entity_names.items():
            if self._entities_similar(normalized_name, existing_norm):
                return canonical
        
        return None
    
    def _apply_aggressive_relationship_filtering(self, relationships: List[ExtractedRelationship], 
                                               entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Apply aggressive filtering to reduce relationship count from 1238 to 150-300 maximum"""
        if not relationships:
            return relationships
            
        logger.info(f"🚨 Starting aggressive relationship filtering on {len(relationships)} relationships")
        
        # Step 1: Remove generic/low-value relationship types
        high_value_types = {
            'IMPLEMENTS', 'USES', 'MANAGES', 'OPERATES', 'DEVELOPS', 'WORKS_FOR',
            'LOCATED_IN', 'PART_OF', 'OWNS', 'COLLABORATES_WITH', 'COMPETES_WITH',
            'PROVIDES', 'REQUIRES', 'DEPENDS_ON', 'INTEGRATES_WITH', 'SUPPORTS',
            'LEADS', 'REPORTS_TO', 'SERVES', 'PARTNERS_WITH', 'ACQUIRED_BY'
        }
        
        filtered_rels = []
        for rel in relationships:
            # Remove generic relationship types
            if rel.relationship_type in ['RELATED_TO', 'ASSOCIATED_WITH', 'CONNECTED_TO', 'MENTIONED_WITH']:
                continue
                
            # Keep only high-value relationship types
            if rel.relationship_type in high_value_types:
                filtered_rels.append(rel)
                
        logger.info(f"🚨 After removing generic types: {len(relationships)} -> {len(filtered_rels)} relationships")
        
        # Step 2: Apply confidence threshold filtering (minimum 0.5 - ADJUSTED for balance)
        confidence_filtered = []
        for rel in filtered_rels:
            try:
                confidence = float(rel.confidence) if rel.confidence is not None else 0.0
                if confidence >= 0.5:  # ADJUSTED: Lowered from 0.7 to 0.5 for better balance
                    confidence_filtered.append(rel)
            except (TypeError, ValueError):
                # Skip relationships with invalid confidence values
                continue
        logger.info(f"🚨 After confidence filtering (>=0.5): {len(filtered_rels)} -> {len(confidence_filtered)} relationships")
        
        # Step 3: OPTIMIZATION - Limit relationships per entity (max 2 per entity for ≤4.0 ratio)
        entity_relationship_count = {}
        entity_limited_rels = []
        
        # Sort by confidence to prioritize highest quality relationships
        confidence_filtered.sort(key=lambda r: float(r.confidence) if r.confidence is not None else 0.0, reverse=True)
        
        for rel in confidence_filtered:
            source_count = entity_relationship_count.get(rel.source_entity, 0)
            target_count = entity_relationship_count.get(rel.target_entity, 0)
            
            # OPTIMIZATION: Maximum 2 relationships per entity to maintain ≤4.0 ratio
            if source_count < 2 and target_count < 2:
                entity_limited_rels.append(rel)
                entity_relationship_count[rel.source_entity] = source_count + 1
                entity_relationship_count[rel.target_entity] = target_count + 1
                
        logger.info(f"🚨 OPTIMIZATION - After per-entity limits (max 2): {len(confidence_filtered)} -> {len(entity_limited_rels)} relationships")
        
        # Step 4: OPTIMIZATION - Apply dynamic hard cap based on entity count (max 4 relationships per entity)
        HARD_CAP = min(200, len(entities) * 4)  # OPTIMIZATION: Reduced from entities * 6 to entities * 4
        if len(entity_limited_rels) > HARD_CAP:
            # Keep only the highest quality relationships up to the cap
            final_rels = entity_limited_rels[:HARD_CAP]
            logger.info(f"🚨 Applied hard cap: {len(entity_limited_rels)} -> {HARD_CAP} relationships")
        else:
            final_rels = entity_limited_rels
            
        # Step 5: Final optimization - If we have too many relationships per entity, reduce further
        total_entities = len(entities)
        final_count = len(final_rels)
        relationships_per_entity = final_count / max(total_entities, 1)
        
        # OPTIMIZATION: If ratio is still too high (>4 per entity), apply stricter per-entity limits
        if relationships_per_entity > 4.0:
            logger.info(f"🚨 OPTIMIZATION - Ratio too high ({relationships_per_entity:.1f}), applying stricter per-entity limits")
            logger.info(f"🚨 Entity analysis: {total_entities} entities, {final_count} relationships")
            
            # OPTIMIZATION: Calculate optimal relationships per entity for target ≤4.0 ratio
            target_ratio = min(4.0, max(2.0, 200 / total_entities))  # Changed from 6.0 to 4.0, 300 to 200
            max_rels_per_entity = int(target_ratio)
            
            # Re-apply per-entity limits with stricter caps
            entity_relationship_count = {}
            ultra_filtered_rels = []
            
            for rel in final_rels:
                source_count = entity_relationship_count.get(rel.source_entity, 0)
                target_count = entity_relationship_count.get(rel.target_entity, 0)
                
                if source_count < max_rels_per_entity and target_count < max_rels_per_entity:
                    ultra_filtered_rels.append(rel)
                    entity_relationship_count[rel.source_entity] = source_count + 1
                    entity_relationship_count[rel.target_entity] = target_count + 1
            
            final_rels = ultra_filtered_rels
            final_count = len(final_rels)
            relationships_per_entity = final_count / max(total_entities, 1)
            
            logger.info(f"🚨 Ultra-filtering applied: max {max_rels_per_entity} relationships per entity")
        
        logger.info(f"✅ OPTIMIZATION FILTERING COMPLETE:")
        logger.info(f"   📊 Final: {final_count} relationships for {total_entities} entities")
        logger.info(f"   📊 Ratio: {relationships_per_entity:.1f} relationships per entity")
        logger.info(f"   📊 Target achieved: {final_count <= 200 and relationships_per_entity <= 4.0}")
        
        # OPTIMIZATION: FINAL ENFORCEMENT - Ensure ratio never exceeds 4.0
        final_ratio = len(final_rels) / max(total_entities, 1)
        if final_ratio > 4.0:
            max_allowed_relationships = int(total_entities * 4)  # OPTIMIZATION: Changed from * 6 to * 4
            final_rels = final_rels[:max_allowed_relationships]
            logger.info(f"🚨 OPTIMIZATION - FINAL RATIO ENFORCEMENT: Cut to {max_allowed_relationships} relationships (4.0 per entity ratio)")
        
        return final_rels
    
    async def _generate_cooccurrence_relationships(self, text: str, entities: List[ExtractedEntity],
                                                 existing_relationships: List[ExtractedRelationship]) -> List[ExtractedRelationship]:
        """Generate relationships based on entity co-occurrence patterns"""
        cooccurrence_relationships = []
        
        # Track existing relationships to avoid duplicates
        existing_rel_keys = {(r.source_entity.lower(), r.target_entity.lower(), r.relationship_type) 
                           for r in existing_relationships}
        
        # Analyze entities that appear frequently together
        text_lower = text.lower()
        entity_positions = {}
        
        # Find all positions of each entity in text
        for entity in entities:
            positions = []
            entity_text = entity.canonical_form.lower()
            start = 0
            while True:
                pos = text_lower.find(entity_text, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            entity_positions[entity.canonical_form] = positions
        
        # AGGRESSIVE FILTERING: Much stricter proximity and co-occurrence requirements
        proximity_threshold = 200  # REDUCED from 500 to 200 characters for tighter proximity
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1.canonical_form == entity2.canonical_form:
                    continue
                
                # Check if entities appear near each other
                positions1 = entity_positions.get(entity1.canonical_form, [])
                positions2 = entity_positions.get(entity2.canonical_form, [])
                
                cooccurrences = 0
                for pos1 in positions1:
                    for pos2 in positions2:
                        if abs(pos1 - pos2) <= proximity_threshold:
                            cooccurrences += 1
                
                # NUCLEAR FIX: Require extremely frequent co-occurrence to prevent explosion
                if cooccurrences >= 100:  # OPTIMIZATION: Increased from 50 to 100 co-occurrences minimum for ≤4.0 ratio
                    rel_type = self._infer_cooccurrence_relationship_type(entity1, entity2)
                    rel_key = (entity1.canonical_form.lower(), entity2.canonical_form.lower(), rel_type)
                    
                    if rel_key not in existing_rel_keys:
                        cooccurrence_rel = ExtractedRelationship(
                            source_entity=entity1.canonical_form,
                            target_entity=entity2.canonical_form,
                            relationship_type=rel_type,
                            confidence=0.8 + (cooccurrences * 0.02),  # AGGRESSIVE: Start at 0.8 confidence minimum
                            context=f"Co-occurs {cooccurrences} times in document",
                            properties={
                                'inference_method': 'cooccurrence',
                                'cooccurrence_count': cooccurrences,
                                'proximity_threshold': proximity_threshold
                            }
                        )
                        cooccurrence_relationships.append(cooccurrence_rel)
                        
                        # NUCLEAR SAFETY: Hard cap on co-occurrence relationships to prevent explosion
                        if len(cooccurrence_relationships) >= 25:  # Maximum 25 co-occurrence relationships
                            logger.warning("🚑 NUCLEAR CAP: Stopping co-occurrence generation at 25 relationships")
                            return cooccurrence_relationships
        
        logger.info(f"💡 Generated {len(cooccurrence_relationships)} co-occurrence relationships")
        return cooccurrence_relationships
    
    def _infer_cooccurrence_relationship_type(self, entity1: ExtractedEntity, entity2: ExtractedEntity) -> str:
        """Infer relationship type based on entity types for co-occurrence"""
        type1, type2 = entity1.label, entity2.label
        
        # Business relationship patterns
        if type1 in ['ORGANIZATION', 'COMPANY'] and type2 in ['ORGANIZATION', 'COMPANY']:
            return 'COLLABORATES_WITH'
        elif type1 in ['PERSON', 'EXECUTIVE'] and type2 in ['ORGANIZATION', 'COMPANY']:
            return 'WORKS_FOR'  # CHANGED: More specific than ASSOCIATED_WITH
        elif type1 in ['TECHNOLOGY', 'SYSTEM'] and type2 in ['ORGANIZATION', 'COMPANY']:
            return 'USED_BY'
        elif type1 in ['PRODUCT', 'SERVICE'] and type2 in ['MARKET', 'CUSTOMER']:
            return 'SERVES'
        elif type1 == 'LOCATION' and type2 in ['ORGANIZATION', 'COMPANY']:
            return 'RELATED_REGION'
        else:
            return 'COLLABORATES_WITH'  # CHANGED: More specific than RELATED_TO
    
    async def _infer_cooccurrence_relationships(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Enhanced co-occurrence relationship inference"""
        relationships = []
        text_lower = text.lower()
        
        # Enhanced proximity analysis with contextual clues
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1.canonical_form == entity2.canonical_form:
                    continue
                
                # Find all occurrences where entities appear close together
                entity1_pos = [m.start() for m in re.finditer(re.escape(entity1.canonical_form.lower()), text_lower)]
                entity2_pos = [m.start() for m in re.finditer(re.escape(entity2.canonical_form.lower()), text_lower)]
                
                close_occurrences = 0
                for pos1 in entity1_pos:
                    for pos2 in entity2_pos:
                        if abs(pos1 - pos2) <= 100:  # AGGRESSIVE: Further reduced to 100 characters for ultra-tight proximity
                            close_occurrences += 1
                
                if close_occurrences >= 25:  # NUCLEAR FIX: Require 25+ close occurrences to prevent explosion
                    rel_type = self._infer_relationship_type(entity1, entity2, text)
                    
                    relationship = ExtractedRelationship(
                        source_entity=entity1.canonical_form,
                        target_entity=entity2.canonical_form,
                        relationship_type=rel_type,
                        confidence=0.8 + (close_occurrences * 0.02),  # AGGRESSIVE: Start at 0.8 confidence
                        context=f"Co-occurrence inference ({close_occurrences} times)",
                        properties={
                            'inference_method': 'enhanced_cooccurrence',
                            'close_occurrences': close_occurrences
                        }
                    )
                    relationships.append(relationship)
                    
                    # NUCLEAR SAFETY: Hard cap on co-occurrence relationships to prevent explosion
                    if len(relationships) >= 50:  # Maximum 50 co-occurrence relationships
                        logger.warning("🚑 NUCLEAR CAP: Stopping co-occurrence generation at 50 relationships")
                        return relationships
        
        return relationships
    
    def _calculate_multi_pass_confidence(self, entities: List[ExtractedEntity], 
                                       relationships: List[ExtractedRelationship],
                                       pass_metadata: Dict[str, Any]) -> float:
        """Calculate confidence score for multi-pass extraction"""
        if not entities and not relationships:
            return 0.0
        
        # Base confidence from individual entities and relationships
        entity_confidence = sum(e.confidence for e in entities) / len(entities) if entities else 0
        relationship_confidence = sum(r.confidence for r in relationships) / len(relationships) if relationships else 0
        
        # Bonus for comprehensive extraction (multiple passes yielding results)
        passes_successful = sum(1 for pass_data in pass_metadata.values() 
                              if pass_data.get('entities_found', 0) > 0 or pass_data.get('relationships_found', 0) > 0)
        pass_bonus = min(0.2, passes_successful * 0.05)  # Up to 20% bonus
        
        # Bonus for relationship density (well-connected graph)
        density_bonus = min(0.15, len(relationships) / max(len(entities), 1) * 0.1)
        
        final_confidence = (entity_confidence + relationship_confidence) / 2 + pass_bonus + density_bonus
        return min(1.0, final_confidence)


# Singleton instance
_llm_extractor: Optional[LLMKnowledgeExtractor] = None

def get_llm_knowledge_extractor() -> LLMKnowledgeExtractor:
    """Get or create LLM knowledge extractor singleton"""
    global _llm_extractor
    if _llm_extractor is None:
        _llm_extractor = LLMKnowledgeExtractor()
    return _llm_extractor