"""
Pipeline validation endpoints for pre-execution checks
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import json

from app.core.db import SessionLocal
from app.core.langgraph_agents_cache import get_langgraph_agents
from app.core.mcp_tools_cache import get_enabled_mcp_tools

router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class ValidationError(BaseModel):
    type: str  # AGENT_NOT_FOUND, TOOL_NOT_FOUND, COMMUNICATION_MISSING, etc.
    severity: str  # error, warning, info
    message: str
    details: Dict[str, Any] = {}

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []
    info: List[ValidationError] = []
    suggestions: List[Dict[str, Any]] = []
    execution_estimate: Optional[Dict[str, Any]] = None

class PipelineValidationRequest(BaseModel):
    pipeline_id: Optional[int] = None
    pipeline_data: Optional[Dict[str, Any]] = None
    test_data: Optional[Dict[str, Any]] = None

@router.post("/validate", response_model=ValidationResult)
async def validate_pipeline(
    request: PipelineValidationRequest,
    db: Session = Depends(get_db)
) -> ValidationResult:
    """Validate a pipeline before execution"""
    errors = []
    warnings = []
    info = []
    suggestions = []
    
    # Get pipeline data
    if request.pipeline_id:
        # Fetch from database
        query = text("""
            SELECT p.*, 
                   array_agg(
                       json_build_object(
                           'agent_name', pa.agent_name,
                           'execution_order', pa.execution_order,
                           'parent_agent', pa.parent_agent,
                           'config', pa.config
                       ) ORDER BY pa.execution_order
                   ) as agents
            FROM agentic_pipelines p
            LEFT JOIN pipeline_agents pa ON p.id = pa.pipeline_id
            WHERE p.id = :pipeline_id
            GROUP BY p.id
        """)
        result = db.execute(query, {"pipeline_id": request.pipeline_id}).fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        pipeline = {
            "id": result.id,
            "name": result.name,
            "collaboration_mode": result.collaboration_mode,
            "agents": result.agents or [],
            "config": result.config or {}
        }
    else:
        pipeline = request.pipeline_data
        if not pipeline:
            raise HTTPException(status_code=400, detail="Either pipeline_id or pipeline_data must be provided")
    
    # Get available agents and tools
    available_agents = get_langgraph_agents()
    available_tools = get_enabled_mcp_tools()
    
    # 1. Validate agents exist
    agent_names = [agent["agent_name"] for agent in pipeline["agents"]]
    for agent in pipeline["agents"]:
        agent_name = agent["agent_name"]
        if agent_name not in available_agents:
            errors.append(ValidationError(
                type="AGENT_NOT_FOUND",
                severity="error",
                message=f"Agent '{agent_name}' not found in the system",
                details={"agent": agent_name}
            ))
    
    # 2. Validate tools availability
    for agent in pipeline["agents"]:
        agent_name = agent["agent_name"]
        agent_config = agent.get("config", {})
        tools = agent_config.get("tools", [])
        
        for tool in tools:
            if tool not in available_tools:
                errors.append(ValidationError(
                    type="TOOL_NOT_FOUND",
                    severity="error",
                    message=f"Tool '{tool}' not available for agent '{agent_name}'",
                    details={"agent": agent_name, "tool": tool}
                ))
    
    # 3. Check for empty pipeline
    if not pipeline["agents"]:
        errors.append(ValidationError(
            type="EMPTY_PIPELINE",
            severity="error",
            message="Pipeline has no agents configured",
            details={}
        ))
    
    # 4. Validate collaboration mode specific rules
    mode = pipeline["collaboration_mode"]
    
    if mode == "sequential":
        # Check for communication patterns between sequential agents
        for i in range(len(pipeline["agents"]) - 1):
            current = pipeline["agents"][i]
            next_agent = pipeline["agents"][i + 1]
            
            # Check if communication pattern exists
            pattern_query = text("""
                SELECT * FROM agent_communication_patterns 
                WHERE from_agent = :from_agent AND to_agent = :to_agent
            """)
            pattern = db.execute(pattern_query, {
                "from_agent": current["agent_name"],
                "to_agent": next_agent["agent_name"]
            }).fetchone()
            
            if not pattern:
                warnings.append(ValidationError(
                    type="COMMUNICATION_PATTERN_MISSING",
                    severity="warning",
                    message=f"No communication pattern defined between '{current['agent_name']}' and '{next_agent['agent_name']}'",
                    details={
                        "from_agent": current["agent_name"],
                        "to_agent": next_agent["agent_name"],
                        "position": i
                    }
                ))
                
                # Add suggestion
                suggestions.append({
                    "type": "CREATE_COMMUNICATION_PATTERN",
                    "message": f"Consider creating a communication pattern between '{current['agent_name']}' and '{next_agent['agent_name']}'",
                    "action": {
                        "from_agent": current["agent_name"],
                        "to_agent": next_agent["agent_name"]
                    }
                })
    
    elif mode == "hierarchical":
        # Check that at least one agent has no parent (lead agent)
        lead_agents = [a for a in pipeline["agents"] if not a.get("parent_agent")]
        if not lead_agents:
            errors.append(ValidationError(
                type="NO_LEAD_AGENT",
                severity="error",
                message="Hierarchical pipeline requires at least one lead agent (no parent)",
                details={}
            ))
        
        # Check for circular dependencies
        agent_map = {a["agent_name"]: a for a in pipeline["agents"]}
        for agent in pipeline["agents"]:
            visited = set()
            current = agent["agent_name"]
            while current:
                if current in visited:
                    errors.append(ValidationError(
                        type="CIRCULAR_DEPENDENCY",
                        severity="error",
                        message=f"Circular dependency detected involving agent '{agent['agent_name']}'",
                        details={"agent": agent["agent_name"]}
                    ))
                    break
                visited.add(current)
                parent = agent_map.get(current, {}).get("parent_agent")
                current = parent
    
    # 5. Check for duplicate agents
    seen_agents = set()
    for agent in pipeline["agents"]:
        if agent["agent_name"] in seen_agents:
            warnings.append(ValidationError(
                type="DUPLICATE_AGENT",
                severity="warning",
                message=f"Agent '{agent['agent_name']}' appears multiple times in the pipeline",
                details={"agent": agent["agent_name"]}
            ))
        seen_agents.add(agent["agent_name"])
    
    # 6. Check for output generation agent
    has_output_agent = any(
        'writer' in agent["agent_name"] or 
        'responder' in agent["agent_name"] or
        'synthesizer' in agent["agent_name"]
        for agent in pipeline["agents"]
    )
    
    if not has_output_agent:
        warnings.append(ValidationError(
            type="NO_OUTPUT_AGENT",
            severity="warning",
            message="Pipeline doesn't have an obvious output generation agent",
            details={}
        ))
        
        suggestions.append({
            "type": "ADD_OUTPUT_AGENT",
            "message": "Consider adding an output agent like 'synthesizer' or 'response_writer'",
            "agents": ["synthesizer", "response_writer", "email_responder"]
        })
    
    # 7. Estimate execution time
    execution_estimate = estimate_execution_time(pipeline)
    
    # 8. Check agent templates
    for agent in pipeline["agents"]:
        template_query = text("SELECT * FROM agent_templates WHERE name = :name")
        template = db.execute(template_query, {"name": agent["agent_name"]}).fetchone()
        if template:
            info.append(ValidationError(
                type="TEMPLATE_AVAILABLE",
                severity="info",
                message=f"Agent '{agent['agent_name']}' has a template available",
                details={"agent": agent["agent_name"], "has_template": True}
            ))
    
    is_valid = len(errors) == 0
    
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        info=info,
        suggestions=suggestions,
        execution_estimate=execution_estimate
    )

def estimate_execution_time(pipeline: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate pipeline execution time based on agents and mode"""
    base_time_per_agent = 30  # seconds
    mode = pipeline["collaboration_mode"]
    agents = pipeline["agents"]
    
    if mode == "sequential":
        total_time = len(agents) * base_time_per_agent
    elif mode == "parallel":
        # Parallel execution - time of slowest agent
        total_time = base_time_per_agent * 1.5
    else:  # hierarchical
        # Estimate based on depth
        max_depth = calculate_hierarchy_depth(agents)
        total_time = max_depth * base_time_per_agent
    
    return {
        "estimated_seconds": total_time,
        "estimated_minutes": round(total_time / 60, 1),
        "confidence": "medium",
        "factors": {
            "agent_count": len(agents),
            "mode": mode,
            "base_time_per_agent": base_time_per_agent
        }
    }

def calculate_hierarchy_depth(agents: List[Dict[str, Any]]) -> int:
    """Calculate the maximum depth of the hierarchy"""
    agent_map = {a["agent_name"]: a for a in agents}
    max_depth = 0
    
    for agent in agents:
        depth = 0
        current = agent["agent_name"]
        visited = set()
        
        while current and current not in visited:
            visited.add(current)
            parent = agent_map.get(current, {}).get("parent_agent")
            if parent:
                depth += 1
                current = parent
            else:
                break
        
        max_depth = max(max_depth, depth + 1)
    
    return max_depth

@router.post("/validate/dry-run")
async def dry_run_pipeline(
    request: PipelineValidationRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Perform a dry run of the pipeline with test data"""
    # First validate the pipeline
    validation_result = await validate_pipeline(request, db)
    
    if not validation_result.is_valid:
        return {
            "success": False,
            "validation": validation_result,
            "message": "Pipeline validation failed"
        }
    
    # TODO: Implement actual dry run logic
    # This would execute the pipeline with mock data and no side effects
    
    return {
        "success": True,
        "validation": validation_result,
        "dry_run_results": {
            "agents_tested": len(request.pipeline_data.get("agents", [])),
            "estimated_tokens": 1000,
            "estimated_cost": "$0.05",
            "test_output": "This is a simulated output from the pipeline dry run"
        }
    }