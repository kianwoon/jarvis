"""
Smart agent recommendation endpoints for agentic pipelines
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import logging
from collections import Counter

from app.core.db import SessionLocal
from app.core.langgraph_agents_cache import get_langgraph_agents

logger = logging.getLogger(__name__)
router = APIRouter()

try:
    from app.langchain.smart_keyword_extractor import SmartKeywordExtractor
    KEYWORD_EXTRACTOR_AVAILABLE = True
except ImportError:
    logger.warning("SmartKeywordExtractor not available, using fallback")
    KEYWORD_EXTRACTOR_AVAILABLE = False

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class GoalAnalysisRequest(BaseModel):
    goal: str
    existing_agents: List[str] = []
    context: Optional[Dict[str, Any]] = {}

class AgentRecommendation(BaseModel):
    agent_name: str
    role: str
    relevance_score: float
    reason: str
    suggested_position: Optional[int] = None
    tools: List[str] = []
    complements: List[str] = []  # Other agents this works well with

class RecommendationResponse(BaseModel):
    recommendations: List[AgentRecommendation]
    suggested_mode: str
    analysis: Dict[str, Any]

# Agent capability keywords mapping
AGENT_CAPABILITIES = {
    "document_researcher": {
        "keywords": ["document", "pdf", "file", "research", "analyze", "extract", "read", "content"],
        "capabilities": ["document analysis", "information extraction", "research"],
        "complements": ["summarizer", "data_processor"]
    },
    "web_searcher": {
        "keywords": ["search", "google", "web", "internet", "find", "online", "lookup", "query"],
        "capabilities": ["web search", "online research", "information gathering"],
        "complements": ["summarizer", "fact_checker"]
    },
    "data_processor": {
        "keywords": ["data", "process", "transform", "clean", "format", "structure", "organize"],
        "capabilities": ["data transformation", "data cleaning", "formatting"],
        "complements": ["analyzer", "visualizer"]
    },
    "analyzer": {
        "keywords": ["analyze", "insight", "pattern", "trend", "statistics", "evaluate", "assess"],
        "capabilities": ["data analysis", "pattern recognition", "insights generation"],
        "complements": ["data_processor", "report_writer"]
    },
    "code_developer": {
        "keywords": ["code", "program", "develop", "script", "function", "api", "software", "debug"],
        "capabilities": ["code generation", "debugging", "API development"],
        "complements": ["code_reviewer", "documentation_writer"]
    },
    "summarizer": {
        "keywords": ["summary", "summarize", "brief", "overview", "key points", "tldr", "abstract"],
        "capabilities": ["text summarization", "key point extraction"],
        "complements": ["document_researcher", "report_writer"]
    },
    "report_writer": {
        "keywords": ["report", "write", "document", "compose", "draft", "article", "content"],
        "capabilities": ["report generation", "content writing", "documentation"],
        "complements": ["analyzer", "summarizer"]
    },
    "email_responder": {
        "keywords": ["email", "reply", "response", "message", "communication", "correspond"],
        "capabilities": ["email composition", "professional communication"],
        "complements": ["sentiment_analyzer", "summarizer"]
    },
    "planner": {
        "keywords": ["plan", "strategy", "organize", "schedule", "coordinate", "roadmap", "timeline"],
        "capabilities": ["strategic planning", "task organization", "scheduling"],
        "complements": ["task_executor", "progress_tracker"]
    },
    "fact_checker": {
        "keywords": ["verify", "fact", "check", "validate", "confirm", "truth", "accurate"],
        "capabilities": ["fact verification", "accuracy checking", "validation"],
        "complements": ["web_searcher", "analyzer"]
    },
    "sentiment_analyzer": {
        "keywords": ["sentiment", "emotion", "feeling", "opinion", "feedback", "mood", "tone"],
        "capabilities": ["sentiment analysis", "emotion detection", "feedback analysis"],
        "complements": ["analyzer", "report_writer"]
    },
    "translator": {
        "keywords": ["translate", "language", "localize", "multilingual", "international"],
        "capabilities": ["language translation", "localization"],
        "complements": ["summarizer", "report_writer"]
    }
}

# Task patterns for collaboration mode suggestions
TASK_PATTERNS = {
    "sequential": ["step by step", "then", "after", "process", "workflow", "pipeline"],
    "parallel": ["multiple", "various", "different", "independent", "simultaneously", "at once"],
    "hierarchical": ["coordinate", "manage", "oversee", "delegate", "supervise", "organize"]
}

@router.post("/recommend", response_model=RecommendationResponse)
async def get_agent_recommendations(
    request: GoalAnalysisRequest,
    db: Session = Depends(get_db)
) -> RecommendationResponse:
    """Get smart agent recommendations based on goal analysis"""
    
    # Extract keywords from goal
    if KEYWORD_EXTRACTOR_AVAILABLE:
        keyword_extractor = SmartKeywordExtractor()
        keywords = keyword_extractor.extract_keywords(request.goal.lower())
    else:
        # Fallback to simple keyword extraction
        import re
        words = re.findall(r'\b\w+\b', request.goal.lower())
        # Filter out common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once'}
        keywords = [w for w in words if w not in common_words and len(w) > 2][:10]
    
    # Get all available agents
    available_agents = get_langgraph_agents()
    
    # Score each agent based on keyword matching
    agent_scores = {}
    for agent_name, agent_data in available_agents.items():
        if agent_name in request.existing_agents:
            continue  # Skip already selected agents
            
        score = 0
        reasons = []
        
        # Check agent capabilities
        if agent_name in AGENT_CAPABILITIES:
            capability_data = AGENT_CAPABILITIES[agent_name]
            
            # Match keywords
            for keyword in keywords:
                if keyword in capability_data["keywords"]:
                    score += 2
                    reasons.append(f"Matches keyword '{keyword}'")
                elif any(keyword in cap for cap in capability_data["capabilities"]):
                    score += 1
                    reasons.append(f"Has capability for '{keyword}'")
        
        # Check agent description and role
        agent_text = f"{agent_data.get('role', '')} {agent_data.get('description', '')}".lower()
        for keyword in keywords:
            if keyword in agent_text:
                score += 1
                reasons.append(f"Role/description mentions '{keyword}'")
        
        if score > 0:
            agent_scores[agent_name] = {
                "score": score,
                "reasons": reasons,
                "data": agent_data
            }
    
    # Sort by score and get top recommendations
    sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    recommendations = []
    
    for i, (agent_name, score_data) in enumerate(sorted_agents[:5]):  # Top 5 recommendations
        agent_data = score_data["data"]
        
        # Determine complementary agents
        complements = []
        if agent_name in AGENT_CAPABILITIES:
            complements = AGENT_CAPABILITIES[agent_name]["complements"]
            # Filter to only include available agents not already selected
            complements = [c for c in complements if c in available_agents and c not in request.existing_agents]
        
        recommendation = AgentRecommendation(
            agent_name=agent_name,
            role=agent_data.get("role", ""),
            relevance_score=score_data["score"] / max(len(keywords), 1),  # Normalize score
            reason="; ".join(score_data["reasons"][:2]),  # Top 2 reasons
            suggested_position=i,
            tools=agent_data.get("tools", []),
            complements=complements
        )
        recommendations.append(recommendation)
    
    # Suggest collaboration mode
    suggested_mode = suggest_collaboration_mode(request.goal, keywords)
    
    # Analysis details
    analysis = {
        "keywords_extracted": keywords,
        "total_agents_analyzed": len(available_agents),
        "agents_recommended": len(recommendations),
        "goal_complexity": determine_goal_complexity(keywords, request.goal)
    }
    
    return RecommendationResponse(
        recommendations=recommendations,
        suggested_mode=suggested_mode,
        analysis=analysis
    )

def suggest_collaboration_mode(goal: str, keywords: List[str]) -> str:
    """Suggest the best collaboration mode based on goal analysis"""
    goal_lower = goal.lower()
    
    mode_scores = {
        "sequential": 0,
        "parallel": 0,
        "hierarchical": 0
    }
    
    # Check for pattern matches
    for mode, patterns in TASK_PATTERNS.items():
        for pattern in patterns:
            if pattern in goal_lower:
                mode_scores[mode] += 2
            elif any(pattern in keyword for keyword in keywords):
                mode_scores[mode] += 1
    
    # Default logic based on goal structure
    if "and" in goal_lower and "then" in goal_lower:
        mode_scores["sequential"] += 1
    elif goal_lower.count("and") > 2:
        mode_scores["parallel"] += 1
    elif any(word in goal_lower for word in ["manage", "coordinate", "oversee"]):
        mode_scores["hierarchical"] += 1
    
    # Return mode with highest score, default to sequential
    return max(mode_scores.items(), key=lambda x: x[1])[0] if max(mode_scores.values()) > 0 else "sequential"

def determine_goal_complexity(keywords: List[str], goal: str) -> str:
    """Determine the complexity of the goal"""
    word_count = len(goal.split())
    keyword_count = len(keywords)
    
    if word_count < 10 and keyword_count < 3:
        return "simple"
    elif word_count < 30 and keyword_count < 6:
        return "moderate"
    else:
        return "complex"

@router.get("/agent-compatibility/{agent_name}")
async def get_agent_compatibility(
    agent_name: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get compatibility information for a specific agent"""
    
    available_agents = get_langgraph_agents()
    
    if agent_name not in available_agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    # Get compatibility data
    compatibility = {
        "agent": agent_name,
        "works_well_with": [],
        "common_patterns": [],
        "suggested_tools": []
    }
    
    if agent_name in AGENT_CAPABILITIES:
        capability_data = AGENT_CAPABILITIES[agent_name]
        
        # Get complementary agents
        complements = capability_data["complements"]
        for complement in complements:
            if complement in available_agents:
                compatibility["works_well_with"].append({
                    "agent": complement,
                    "reason": f"Complements {agent_name}'s capabilities"
                })
        
        # Get common usage patterns from pipeline history
        query = text("""
            SELECT p.name, p.collaboration_mode, 
                   array_agg(pa.agent_name ORDER BY pa.execution_order) as agents
            FROM agentic_pipelines p
            JOIN pipeline_agents pa ON p.id = pa.pipeline_id
            WHERE pa.pipeline_id IN (
                SELECT pipeline_id FROM pipeline_agents WHERE agent_name = :agent_name
            )
            GROUP BY p.id, p.name, p.collaboration_mode
            LIMIT 5
        """)
        
        result = db.execute(query, {"agent_name": agent_name})
        patterns = []
        for row in result:
            patterns.append({
                "pipeline": row.name,
                "mode": row.collaboration_mode,
                "agents": row.agents
            })
        compatibility["common_patterns"] = patterns
        
        # Suggest tools based on agent type
        agent_data = available_agents[agent_name]
        current_tools = set(agent_data.get("tools", []))
        
        # Get tools used by similar agents
        similar_tools = Counter()
        for other_agent, other_data in available_agents.items():
            if other_agent != agent_name and other_data.get("role") == agent_data.get("role"):
                for tool in other_data.get("tools", []):
                    if tool not in current_tools:
                        similar_tools[tool] += 1
        
        # Get top suggested tools
        compatibility["suggested_tools"] = [
            {"tool": tool, "used_by_similar": count}
            for tool, count in similar_tools.most_common(5)
        ]
    
    return compatibility

@router.post("/validate-agent-combination")
async def validate_agent_combination(
    agents: List[str],
    mode: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Validate if a combination of agents works well together"""
    
    validation = {
        "is_valid": True,
        "warnings": [],
        "suggestions": [],
        "score": 0
    }
    
    # Check for duplicate agents
    if len(agents) != len(set(agents)):
        validation["warnings"].append("Duplicate agents detected")
        validation["score"] -= 10
    
    # Check agent availability
    available_agents = get_langgraph_agents()
    for agent in agents:
        if agent not in available_agents:
            validation["is_valid"] = False
            validation["warnings"].append(f"Agent '{agent}' not found")
            return validation
    
    # Mode-specific validations
    if mode == "sequential":
        # Check if agents can work in sequence
        for i in range(len(agents) - 1):
            current = agents[i]
            next_agent = agents[i + 1]
            
            # Check if there's a good handoff
            if current in AGENT_CAPABILITIES and next_agent in AGENT_CAPABILITIES:
                if next_agent in AGENT_CAPABILITIES[current]["complements"]:
                    validation["score"] += 10
                else:
                    validation["suggestions"].append(
                        f"Consider adding an intermediate agent between {current} and {next_agent}"
                    )
    
    elif mode == "parallel":
        # Check if agents can work independently
        overlapping_capabilities = set()
        for agent in agents:
            if agent in AGENT_CAPABILITIES:
                caps = set(AGENT_CAPABILITIES[agent]["capabilities"])
                if overlapping_capabilities & caps:
                    validation["warnings"].append(
                        f"Agent '{agent}' has overlapping capabilities with other agents"
                    )
                    validation["score"] -= 5
                overlapping_capabilities.update(caps)
    
    elif mode == "hierarchical":
        # Check if there's a good coordinator
        has_coordinator = any(
            agent in ["planner", "coordinator", "manager"] or 
            "plan" in agent or "manage" in agent
            for agent in agents
        )
        if not has_coordinator:
            validation["suggestions"].append(
                "Consider adding a coordinator agent for hierarchical mode"
            )
            validation["score"] -= 5
    
    # Check for missing complementary agents
    all_complements = set()
    for agent in agents:
        if agent in AGENT_CAPABILITIES:
            all_complements.update(AGENT_CAPABILITIES[agent]["complements"])
    
    missing_complements = all_complements - set(agents)
    if missing_complements:
        validation["suggestions"].append(
            f"Consider adding: {', '.join(list(missing_complements)[:3])}"
        )
    
    # Calculate final score
    validation["score"] = max(0, min(100, 50 + validation["score"]))
    
    return validation