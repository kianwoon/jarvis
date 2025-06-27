"""
Intelligent Agent Selector for Multi-Agent Systems

Provides sophisticated agent selection based on:
- Question complexity analysis
- Semantic capability matching
- Tool requirement analysis
- Historical performance data
- Dynamic agent count optimization
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass

from app.core.langgraph_agents_cache import (
    get_agents_with_capabilities, 
    analyze_agent_capabilities,
    find_agents_by_domain,
    find_agents_by_skill
)
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.core.agent_performance_tracker import performance_tracker
from app.core.tool_requirement_analyzer import ToolRequirementAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class QuestionAnalysis:
    """Analysis results for a user question"""
    complexity: str  # simple, moderate, complex, advanced
    domain: str  # technical, business, research, creative, general
    required_skills: List[str]
    tool_requirements: List[str]
    collaboration_type: str  # sequential, parallel, hierarchical
    optimal_agent_count: int
    keywords: List[str]
    confidence: float

@dataclass
class AgentScore:
    """Scoring result for an agent"""
    agent_name: str
    total_score: float
    capability_score: float
    tool_score: float
    domain_score: float
    collaboration_score: float
    performance_score: float
    diversity_bonus: float
    reasons: List[str]

class IntelligentAgentSelector:
    """Advanced agent selection with semantic understanding"""
    
    def __init__(self):
        self.tool_analyzer = ToolRequirementAnalyzer()
        self.domain_keywords = {
            'technical': [
                'code', 'program', 'develop', 'software', 'api', 'database', 'system',
                'architecture', 'debug', 'algorithm', 'function', 'implementation',
                'server', 'deployment', 'framework', 'library', 'script'
            ],
            'business': [
                'strategy', 'revenue', 'profit', 'market', 'customer', 'sales',
                'business', 'commercial', 'roi', 'budget', 'cost', 'pricing',
                'competitive', 'stakeholder', 'management', 'executive'
            ],
            'research': [
                'research', 'study', 'analyze', 'investigate', 'explore', 'discover',
                'document', 'paper', 'article', 'literature', 'data', 'findings',
                'methodology', 'hypothesis', 'evidence', 'survey'
            ],
            'creative': [
                'create', 'design', 'content', 'write', 'story', 'creative',
                'brainstorm', 'innovative', 'artistic', 'visual', 'narrative',
                'campaign', 'brand', 'marketing', 'copywriting'
            ],
            'financial': [
                'financial', 'finance', 'money', 'investment', 'accounting',
                'budget', 'cost', 'revenue', 'profit', 'loss', 'cash', 'funding',
                'valuation', 'economics', 'banking'
            ],
            'security': [
                'security', 'risk', 'threat', 'vulnerability', 'compliance',
                'audit', 'privacy', 'protection', 'encryption', 'authentication',
                'authorization', 'secure', 'safety'
            ]
        }
        
        self.complexity_indicators = {
            'simple': [
                'simple', 'basic', 'quick', 'easy', 'straightforward',
                'brief', 'short', 'minimal', 'single'
            ],
            'moderate': [
                'detailed', 'comprehensive', 'thorough', 'complete',
                'analyze', 'compare', 'evaluate', 'multiple'
            ],
            'complex': [
                'complex', 'advanced', 'sophisticated', 'comprehensive',
                'integrate', 'coordinate', 'orchestrate', 'systematic',
                'multi-step', 'workflow', 'pipeline'
            ],
            'advanced': [
                'enterprise', 'large-scale', 'mission-critical', 'strategic',
                'transformation', 'optimization', 'architecture', 'framework'
            ]
        }
        
        self.tool_indicators = {
            'search': ['search', 'find', 'lookup', 'google', 'web', 'internet', 'query'],
            'document': ['document', 'pdf', 'file', 'read', 'extract', 'parse'],
            'email': ['email', 'mail', 'send', 'reply', 'message', 'communication'],
            'data': ['data', 'csv', 'excel', 'database', 'table', 'import', 'export'],
            'image': ['image', 'photo', 'picture', 'visual', 'diagram', 'chart'],
            'calendar': ['calendar', 'schedule', 'meeting', 'appointment', 'event'],
            'code': ['code', 'program', 'script', 'repository', 'github', 'git']
        }
        
        self.collaboration_patterns = {
            'sequential': [
                'step by step', 'then', 'after', 'next', 'process', 'workflow',
                'pipeline', 'sequence', 'order', 'follow'
            ],
            'parallel': [
                'simultaneously', 'at once', 'multiple', 'various', 'different',
                'independent', 'concurrent', 'parallel'
            ],
            'hierarchical': [
                'coordinate', 'manage', 'oversee', 'delegate', 'supervise',
                'organize', 'lead', 'direct', 'control'
            ]
        }
    
    def analyze_question(self, question: str) -> QuestionAnalysis:
        """Perform comprehensive analysis of the user question"""
        try:
            # Input validation
            if not question or not isinstance(question, str):
                logger.warning(f"Invalid question input: {type(question)}")
                question = "general assistance needed"
            
            question = question.strip()
            if not question:
                logger.warning("Empty question provided")
                question = "general assistance needed"
            
            question_lower = question.lower()
            words = re.findall(r'\b\w+\b', question_lower)
            word_count = len(question.split())
            
        except Exception as e:
            logger.error(f"Error in question preprocessing: {e}")
            question_lower = "general assistance"
            words = ["general", "assistance"]
            word_count = 2
        
        # Extract meaningful keywords (filter out common words)
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'
        }
        keywords = [w for w in words if w not in common_words and len(w) > 2]
        
        # Analyze domain
        domain_scores = defaultdict(int)
        for domain, domain_keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in domain_keywords:
                    domain_scores[domain] += 2
                elif any(keyword in dk for dk in domain_keywords):
                    domain_scores[domain] += 1
        
        try:
            primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if len(domain_scores) > 0 else 'general'
        except ValueError:
            primary_domain = 'general'
        
        # Analyze complexity
        complexity_scores = defaultdict(int)
        for complexity, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if indicator in question_lower:
                    complexity_scores[complexity] += 1
        
        # Additional complexity factors
        if word_count > 50:
            complexity_scores['complex'] += 2
        elif word_count > 30:
            complexity_scores['moderate'] += 1
        elif word_count > 15:
            complexity_scores['moderate'] += 1
        else:
            complexity_scores['simple'] += 1
            
        if question_lower.count('and') > 2:
            complexity_scores['complex'] += 1
        if any(word in question_lower for word in ['multiple', 'various', 'several']):
            complexity_scores['moderate'] += 1
            
        try:
            complexity = max(complexity_scores.items(), key=lambda x: x[1])[0] if len(complexity_scores) > 0 else 'simple'
        except ValueError:
            complexity = 'simple'
        
        # Determine required skills based on domain and keywords
        required_skills = self._extract_required_skills(keywords, primary_domain)
        
        # Analyze tool requirements using advanced analyzer
        try:
            tool_analysis = self.tool_analyzer.analyze_tool_requirements(question)
            tool_requirements = [req.tool_type for req in tool_analysis.requirements]
        except Exception as e:
            logger.warning(f"Tool analysis failed: {e}")
            tool_requirements = []
        
        # Determine collaboration type
        collab_scores = defaultdict(int)
        for collab_type, patterns in self.collaboration_patterns.items():
            for pattern in patterns:
                if pattern in question_lower:
                    collab_scores[collab_type] += 1
        
        try:
            collaboration_type = max(collab_scores.items(), key=lambda x: x[1])[0] if len(collab_scores) > 0 else 'sequential'
        except ValueError:
            collaboration_type = 'sequential'
        
        # Determine optimal agent count
        optimal_agent_count = self._calculate_optimal_agent_count(
            complexity, len(required_skills), len(tool_requirements), word_count
        )
        
        # Calculate confidence based on analysis strength
        confidence = self._calculate_confidence(
            domain_scores, complexity_scores, collab_scores, len(keywords)
        )
        
        return QuestionAnalysis(
            complexity=complexity,
            domain=primary_domain,
            required_skills=required_skills,
            tool_requirements=tool_requirements,
            collaboration_type=collaboration_type,
            optimal_agent_count=optimal_agent_count,
            keywords=keywords[:10],  # Top 10 keywords
            confidence=confidence
        )
    
    def _extract_required_skills(self, keywords: List[str], domain: str) -> List[str]:
        """Extract required skills based on keywords and domain"""
        skills = set()
        
        # Domain-specific skills
        if domain == 'technical':
            if any(k in keywords for k in ['code', 'program', 'develop']):
                skills.add('programming')
            if any(k in keywords for k in ['system', 'architecture']):
                skills.add('system_design')
            if any(k in keywords for k in ['debug', 'fix', 'error']):
                skills.add('debugging')
        elif domain == 'business':
            if any(k in keywords for k in ['strategy', 'plan']):
                skills.add('strategic_planning')
            if any(k in keywords for k in ['analysis', 'analyze']):
                skills.add('business_analysis')
            if any(k in keywords for k in ['market', 'customer']):
                skills.add('market_research')
        elif domain == 'research':
            skills.add('research')
            if any(k in keywords for k in ['document', 'paper']):
                skills.add('document_analysis')
            if any(k in keywords for k in ['data', 'statistics']):
                skills.add('data_analysis')
        
        # General skills based on keywords
        if any(k in keywords for k in ['write', 'create', 'compose']):
            skills.add('content_creation')
        if any(k in keywords for k in ['summarize', 'summary']):
            skills.add('summarization')
        if any(k in keywords for k in ['coordinate', 'manage']):
            skills.add('coordination')
            
        return list(skills)
    
    def _calculate_optimal_agent_count(self, complexity: str, skill_count: int, 
                                     tool_count: int, word_count: int) -> int:
        """Calculate optimal number of agents needed - prioritize diversity over complexity"""
        
        # Base count focused on diversity rather than complexity limitations
        # Most discussions benefit from multiple perspectives regardless of perceived complexity
        base_count = 4  # Good starting point for diverse perspectives
        
        # Encourage more agents for richer discussions
        if word_count > 50:  # Longer questions suggest need for comprehensive analysis
            base_count = 5
        if word_count > 100:  # Very detailed questions
            base_count = 6
            
        # Encourage diversity when multiple skills/tools are involved
        if skill_count > 2 or tool_count > 1:
            base_count = max(base_count, 5)
        
        return min(max(base_count, 3), 8)  # Cap between 3-8 agents for optimal diversity
    
    def _calculate_confidence(self, domain_scores: Dict, complexity_scores: Dict,
                            collab_scores: Dict, keyword_count: int) -> float:
        """Calculate confidence in the analysis"""
        confidence = 0.5  # Base confidence
        
        # Strong domain indicators
        if len(domain_scores) > 0 and max(domain_scores.values()) > 2:
            confidence += 0.2
        
        # Clear complexity indicators
        if len(complexity_scores) > 0 and max(complexity_scores.values()) > 1:
            confidence += 0.15
        
        # Clear collaboration patterns
        if len(collab_scores) > 0 and max(collab_scores.values()) > 0:
            confidence += 0.1
        
        # Sufficient keywords for analysis
        if keyword_count >= 5:
            confidence += 0.1
        elif keyword_count >= 3:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def select_agents(self, question: str, available_tools: Optional[Dict] = None) -> Tuple[List[str], QuestionAnalysis]:
        """Select optimal agents for the given question"""
        
        try:
            # Validate inputs
            if not question:
                logger.error("Empty question provided to select_agents")
                return [], QuestionAnalysis(
                    complexity="simple", domain="general", required_skills=[], 
                    tool_requirements=[], collaboration_type="sequential", 
                    optimal_agent_count=1, keywords=[], confidence=0.0
                )
            
            # Analyze the question
            analysis = self.analyze_question(question)
            logger.info(f"Question analysis: domain={analysis.domain}, complexity={analysis.complexity}, "
                       f"optimal_agents={analysis.optimal_agent_count}, confidence={analysis.confidence:.2f}")
            
            # Get all agents with capabilities
            try:
                agents_with_capabilities = get_agents_with_capabilities()
            except Exception as e:
                logger.error(f"Failed to get agents with capabilities: {e}")
                return [], analysis
            
            try:
                available_tools = available_tools or get_enabled_mcp_tools()
            except Exception as e:
                logger.warning(f"Failed to get available tools: {e}")
                available_tools = {}
            
            if not agents_with_capabilities:
                logger.warning("No agents available for selection")
                return [], analysis
                
        except Exception as e:
            logger.error(f"Critical error in select_agents initialization: {e}")
            return [], QuestionAnalysis(
                complexity="simple", domain="general", required_skills=[], 
                tool_requirements=[], collaboration_type="sequential", 
                optimal_agent_count=1, keywords=[], confidence=0.0
            )
        
        # Score each agent (initial scoring without diversity)
        agent_scores = []
        for agent_name, agent_data in agents_with_capabilities.items():
            try:
                score = self._score_agent(agent_data, analysis, available_tools)
                if score and score.total_score > 0:
                    agent_scores.append(score)
            except Exception as e:
                logger.warning(f"Failed to score agent {agent_name}: {e}")
                continue
        
        # Sort by base score first
        agent_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Intelligent selection with diversity optimization
        selected_agents = []
        selected_domains = set()
        selected_skills = set()
        final_scores = []
        
        # First pass: select top scorer
        if agent_scores:
            top_score = agent_scores[0]
            selected_agents.append(top_score.agent_name)
            agent_data = agents_with_capabilities[top_score.agent_name]
            capabilities = agent_data.get('capabilities', {})
            selected_domains.add(capabilities.get('primary_domain', 'general'))
            selected_skills.update(capabilities.get('skills', []))
            final_scores.append(top_score)
            
            logger.info(f"Selected top agent: {top_score.agent_name} (score: {top_score.total_score:.2f})")
        
        # Subsequent passes: optimize for diversity and performance
        for score in agent_scores[1:]:
            if len(selected_agents) >= analysis.optimal_agent_count:
                break
                
            agent_data = agents_with_capabilities[score.agent_name]
            capabilities = agent_data.get('capabilities', {})
            agent_domain = capabilities.get('primary_domain', 'general')
            agent_skills = set(capabilities.get('skills', []))
            
            # Calculate diversity bonus
            diversity_bonus = 0.0
            
            # Domain diversity bonus
            if agent_domain not in selected_domains:
                diversity_bonus += 0.15
            
            # Skill diversity bonus  
            new_skills = agent_skills - selected_skills
            if new_skills:
                diversity_bonus += min(0.10, len(new_skills) * 0.02)
            
            # Update score with diversity bonus
            adjusted_score = AgentScore(
                agent_name=score.agent_name,
                total_score=score.total_score + diversity_bonus,
                capability_score=score.capability_score,
                tool_score=score.tool_score,
                domain_score=score.domain_score,
                collaboration_score=score.collaboration_score,
                performance_score=score.performance_score,
                diversity_bonus=diversity_bonus,
                reasons=score.reasons + ([f"Diversity bonus: +{diversity_bonus:.2f}"] if diversity_bonus > 0 else [])
            )
            
            # Selection criteria: high adjusted score OR brings significant diversity
            if (adjusted_score.total_score > 0.6 or  # Lowered threshold with diversity
                diversity_bonus > 0.1 or  # Significant diversity contribution
                score.total_score > 0.8):  # Very high base score
                
                selected_agents.append(score.agent_name)
                selected_domains.add(agent_domain)
                selected_skills.update(agent_skills)
                final_scores.append(adjusted_score)
                
                logger.info(f"Selected agent: {score.agent_name} "
                           f"(base: {score.total_score:.2f}, diversity: +{diversity_bonus:.2f}, "
                           f"final: {adjusted_score.total_score:.2f})")
        
        # Intelligent fallback: ensure we have at least one agent
        if not selected_agents:
            if agent_scores:
                # Smart fallback: pick best available agents even with low scores
                selected_agents = [score.agent_name for score in agent_scores[:max(1, analysis.optimal_agent_count)]]
                logger.info(f"Intelligent fallback selection: {selected_agents}")
            else:
                # Last resort: capability-based fallback
                selected_agents = self._capability_based_fallback(analysis, agents_with_capabilities)
                logger.warning(f"Capability-based fallback selection: {selected_agents}")
        
        return selected_agents, analysis
    
    def _capability_based_fallback(self, analysis: QuestionAnalysis, 
                                 agents_with_capabilities: Dict) -> List[str]:
        """Intelligent fallback based on agent capabilities when scoring fails"""
        
        if not agents_with_capabilities:
            return []
        
        # Priority order for fallback selection
        fallback_agents = []
        
        # 1. Try to find agents matching the primary domain
        domain_agents = []
        for agent_name, agent_data in agents_with_capabilities.items():
            capabilities = agent_data.get('capabilities', {})
            if capabilities.get('primary_domain') == analysis.domain:
                domain_agents.append(agent_name)
        
        if domain_agents:
            fallback_agents.extend(domain_agents[:2])
        
        # 2. Add high-performance agents from history
        try:
            top_performers = performance_tracker.get_top_performers(
                domain=analysis.domain,
                complexity=analysis.complexity,
                limit=3
            )
            for agent in top_performers:
                if agent in agents_with_capabilities and agent not in fallback_agents:
                    fallback_agents.append(agent)
                    if len(fallback_agents) >= analysis.optimal_agent_count:
                        break
        except Exception as e:
            logger.warning(f"Failed to get top performers for fallback: {e}")
        
        # 3. Add agents with required skills
        if len(fallback_agents) < analysis.optimal_agent_count and analysis.required_skills:
            for agent_name, agent_data in agents_with_capabilities.items():
                if agent_name not in fallback_agents:
                    capabilities = agent_data.get('capabilities', {})
                    agent_skills = set(capabilities.get('skills', []))
                    required_skills = set(analysis.required_skills)
                    
                    if agent_skills.intersection(required_skills):
                        fallback_agents.append(agent_name)
                        if len(fallback_agents) >= analysis.optimal_agent_count:
                            break
        
        # 4. Fill remaining slots with any available agents (diversity preference)
        if len(fallback_agents) < analysis.optimal_agent_count:
            used_domains = set()
            for agent_name in fallback_agents:
                agent_data = agents_with_capabilities[agent_name]
                capabilities = agent_data.get('capabilities', {})
                used_domains.add(capabilities.get('primary_domain', 'general'))
            
            for agent_name, agent_data in agents_with_capabilities.items():
                if agent_name not in fallback_agents:
                    capabilities = agent_data.get('capabilities', {})
                    agent_domain = capabilities.get('primary_domain', 'general')
                    
                    # Prefer agents from unused domains
                    if agent_domain not in used_domains:
                        fallback_agents.append(agent_name)
                        used_domains.add(agent_domain)
                        if len(fallback_agents) >= analysis.optimal_agent_count:
                            break
            
            # If still need more, add any remaining agents
            if len(fallback_agents) < analysis.optimal_agent_count:
                for agent_name in agents_with_capabilities.keys():
                    if agent_name not in fallback_agents:
                        fallback_agents.append(agent_name)
                        if len(fallback_agents) >= analysis.optimal_agent_count:
                            break
        
        return fallback_agents[:analysis.optimal_agent_count]
    
    def _score_agent(self, agent_data: Dict, analysis: QuestionAnalysis, 
                    available_tools: Dict) -> Optional[AgentScore]:
        """Score an individual agent for the given question analysis"""
        
        try:
            # Validate agent data
            if not agent_data or not isinstance(agent_data, dict):
                logger.warning(f"Invalid agent data: {type(agent_data)}")
                return None
                
            agent_name = agent_data.get('name')
            if not agent_name:
                logger.warning("Agent data missing name field")
                return None
                
            capabilities = agent_data.get('capabilities', {})
            agent_tools = agent_data.get('tools', [])
            
            # Ensure agent_tools is a list
            if not isinstance(agent_tools, list):
                logger.warning(f"Agent {agent_name} has invalid tools format: {type(agent_tools)}")
                agent_tools = []
                
        except Exception as e:
            logger.error(f"Error validating agent data: {e}")
            return None
        
        # Initialize scores
        capability_score = 0.0
        tool_score = 0.0
        domain_score = 0.0
        collaboration_score = 0.0
        reasons = []
        
        # 1. Domain matching (25% weight)
        agent_domain = capabilities.get('primary_domain', 'general')
        if agent_domain == analysis.domain:
            domain_score = 1.0
            reasons.append(f"Perfect domain match ({analysis.domain})")
        elif analysis.domain == 'general':
            domain_score = 0.5
        else:
            domain_score = 0.2
        
        # 2. Capability/skill matching (35% weight)
        agent_skills = set(capabilities.get('skills', []))
        required_skills = set(analysis.required_skills)
        
        if required_skills and agent_skills:
            skill_overlap = len(agent_skills.intersection(required_skills))
            capability_score = skill_overlap / len(required_skills)
            if skill_overlap > 0:
                reasons.append(f"Has {skill_overlap}/{len(required_skills)} required skills")
        else:
            # Fallback: check expertise areas
            expertise_areas = set(capabilities.get('expertise_areas', []))
            if any(area in ['analysis', 'strategy', 'execution'] for area in expertise_areas):
                capability_score = 0.5
                reasons.append("Has relevant expertise areas")
        
        # 3. Tool matching (20% weight) - Enhanced with detailed analysis
        if analysis.tool_requirements:
            available_agent_tools = [tool for tool in agent_tools if tool in available_tools]
            tool_score = 0.0
            tool_matches = 0
            
            for required_tool in analysis.tool_requirements:
                best_match_score = 0.0
                
                # Check if agent has tools that could satisfy the requirement
                for agent_tool in available_agent_tools:
                    # Enhanced matching logic
                    match_score = 0.0
                    
                    # Direct name match
                    if required_tool in agent_tool.lower():
                        match_score = 1.0
                    elif agent_tool.lower() in required_tool:
                        match_score = 0.8
                    else:
                        # Check for semantic similarity
                        tool_info = available_tools.get(agent_tool, {})
                        tool_description = tool_info.get('description', '').lower()
                        
                        if required_tool in tool_description:
                            match_score = 0.6
                        elif any(word in tool_description for word in required_tool.split('_')):
                            match_score = 0.4
                    
                    best_match_score = max(best_match_score, match_score)
                
                if best_match_score > 0:
                    tool_matches += 1
                    tool_score += best_match_score
            
            if tool_matches > 0:
                tool_score = tool_score / len(analysis.tool_requirements)
                reasons.append(f"Has {tool_matches}/{len(analysis.tool_requirements)} tool types (score: {tool_score:.2f})")
            else:
                tool_score = 0.0
        else:
            tool_score = 0.5  # Neutral when no tools required
        
        # 4. Collaboration fit (10% weight)
        expertise_areas = capabilities.get('expertise_areas', [])
        if analysis.collaboration_type == 'hierarchical':
            if any(area in ['leadership', 'coordination'] for area in expertise_areas):
                collaboration_score = 1.0
                reasons.append("Good for hierarchical collaboration")
            else:
                collaboration_score = 0.3
        elif analysis.collaboration_type == 'parallel':
            if 'execution' in expertise_areas:
                collaboration_score = 0.8
                reasons.append("Good for parallel execution")
            else:
                collaboration_score = 0.5
        else:  # sequential
            collaboration_score = 0.7  # Most agents work well sequentially
        
        # 5. Performance history (20% weight)
        try:
            performance_score = performance_tracker.get_performance_score(
                agent_name, analysis.domain, analysis.complexity
            )
            if performance_score > 0.7:
                reasons.append(f"Strong performance history ({performance_score:.2f})")
            elif performance_score > 0.5:
                reasons.append(f"Good performance history ({performance_score:.2f})")
        except Exception as e:
            logger.warning(f"Failed to get performance score for {agent_name}: {e}")
            performance_score = 0.5  # Neutral score
        
        # 6. Diversity bonus (calculated separately)
        diversity_bonus = 0.0  # Will be calculated during selection
        
        # Calculate weighted total score (updated weights to include performance)
        base_score = (
            domain_score * 0.20 +           # Reduced from 0.25
            capability_score * 0.30 +       # Reduced from 0.35  
            tool_score * 0.20 +             # Reduced from 0.25
            collaboration_score * 0.10 +    # Reduced from 0.15
            performance_score * 0.20        # New factor
        )
        
        # Boost score for complexity matching
        complexity_level = capabilities.get('complexity_level', 'intermediate')
        if ((analysis.complexity in ['complex', 'advanced'] and complexity_level == 'advanced') or
            (analysis.complexity == 'simple' and complexity_level == 'basic') or
            (analysis.complexity == 'moderate' and complexity_level == 'intermediate')):
            base_score *= 1.1
            reasons.append(f"Complexity level match ({complexity_level})")
        
        # Final total score including diversity (will be updated during selection)
        total_score = base_score + diversity_bonus
        
        return AgentScore(
            agent_name=agent_name,
            total_score=total_score,
            capability_score=capability_score,
            tool_score=tool_score,
            domain_score=domain_score,
            collaboration_score=collaboration_score,
            performance_score=performance_score,
            diversity_bonus=diversity_bonus,
            reasons=reasons
        )
    
    def get_selection_explanation(self, selected_agents: List[str], 
                                analysis: QuestionAnalysis) -> Dict[str, any]:
        """Get detailed explanation of the agent selection"""
        return {
            "question_analysis": {
                "complexity": analysis.complexity,
                "domain": analysis.domain,
                "required_skills": analysis.required_skills,
                "tool_requirements": analysis.tool_requirements,
                "collaboration_type": analysis.collaboration_type,
                "optimal_agent_count": analysis.optimal_agent_count,
                "confidence": analysis.confidence
            },
            "selection_summary": {
                "agents_selected": selected_agents,
                "selection_rationale": f"Selected {len(selected_agents)} agents for {analysis.complexity} {analysis.domain} task requiring {analysis.collaboration_type} collaboration",
                "key_factors": [
                    f"Domain: {analysis.domain}",
                    f"Complexity: {analysis.complexity}",
                    f"Skills: {', '.join(analysis.required_skills) if analysis.required_skills else 'general'}",
                    f"Tools: {', '.join(analysis.tool_requirements) if analysis.tool_requirements else 'none specific'}"
                ]
            }
        }