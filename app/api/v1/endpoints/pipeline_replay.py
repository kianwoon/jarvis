"""
Pipeline Replay API endpoints

Allows users to replay pipeline executions from specific agents with modified inputs.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from datetime import datetime
import json
import uuid

from app.core.db import get_db
from app.core.pipeline_multi_agent_bridge import PipelineMultiAgentBridge
from app.core.redis_client import get_redis_client
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/replay/{execution_id}/from/{agent_name}")
async def replay_pipeline_from_agent(
    execution_id: str,
    agent_name: str,
    modified_input: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Replay a pipeline execution from a specific agent with modified input.
    
    Args:
        execution_id: Original execution ID
        agent_name: Name of the agent to start replay from
        modified_input: Modified input data for the agent
    
    Returns:
        New execution ID for tracking the replay
    """
    
    # Verify original execution exists
    result = db.execute(
        text("""
        SELECT pe.*, pt.agent_sequence 
        FROM pipeline_executions pe
        JOIN pipeline_templates pt ON pe.pipeline_id = pt.id
        WHERE pe.id = :execution_id
        """),
        {"execution_id": execution_id}
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    pipeline_id = result.pipeline_id
    agent_sequence = json.loads(result.agent_sequence)
    
    # Verify agent exists in pipeline
    agent_found = False
    agent_index = -1
    for i, agent_info in enumerate(agent_sequence):
        if agent_info["agent"] == agent_name:
            agent_found = True
            agent_index = i
            break
    
    if not agent_found:
        raise HTTPException(
            status_code=404, 
            detail=f"Agent '{agent_name}' not found in pipeline"
        )
    
    # Create new execution record for replay
    replay_execution_id = f"replay_{uuid.uuid4().hex[:8]}_{execution_id}"
    
    db.execute(
        text("""
        INSERT INTO pipeline_executions 
        (id, pipeline_id, status, started_at, parent_execution_id, 
         replay_from_agent, is_replay, context)
        VALUES (:id, :pipeline_id, :status, :started_at, :parent_execution_id,
                :replay_from_agent, :is_replay, :context)
        """),
        {
            "id": replay_execution_id,
            "pipeline_id": pipeline_id,
            "status": "starting",
            "started_at": datetime.now(),
            "parent_execution_id": execution_id,
            "replay_from_agent": agent_name,
            "is_replay": True,
            "context": json.dumps({
                "original_execution": execution_id,
                "start_agent": agent_name,
                "start_index": agent_index,
                "modified_input": modified_input
            })
        }
    )
    db.commit()
    
    # Start replay execution in background
    background_tasks.add_task(
        execute_replay,
        replay_execution_id,
        pipeline_id,
        execution_id,
        agent_name,
        agent_index,
        modified_input
    )
    
    return {
        "replay_execution_id": replay_execution_id,
        "original_execution_id": execution_id,
        "starting_from_agent": agent_name,
        "agent_index": agent_index,
        "status": "started"
    }


async def execute_replay(
    replay_execution_id: str,
    pipeline_id: str,
    original_execution_id: str,
    start_agent: str,
    start_index: int,
    modified_input: Dict[str, Any]
):
    """Execute the replay in background"""
    
    try:
        # Get original execution data to reconstruct state
        db = get_db()
        
        # Get all agent outputs before the start agent
        previous_outputs = []
        if start_index > 0:
            results = db.execute(
                text("""
                SELECT agent_name, output, structured_output 
                FROM pipeline_execution_steps
                WHERE execution_id = :execution_id
                ORDER BY completed_at ASC
                LIMIT :limit
                """),
                {"execution_id": original_execution_id, "limit": start_index}
            ).fetchall()
            
            for result in results:
                previous_outputs.append({
                    "agent": result.agent_name,
                    "output": result.output,
                    "structured_data": json.loads(result.structured_output or "{}")
                })
        
        # Update modified input with previous outputs
        modified_input["previous_outputs"] = previous_outputs
        
        # Create bridge and execute from the specified agent
        bridge = PipelineMultiAgentBridge(pipeline_id, replay_execution_id)
        
        # Publish initial status
        redis_client = get_redis_client()
        if redis_client:
            channel = f"pipeline_execution:{replay_execution_id}"
            redis_client.publish(channel, json.dumps({
                "type": "status",
                "payload": {
                    "status": "running",
                    "progress": (start_index / len(previous_outputs + 1)) * 100,
                    "current_agent": start_agent,
                    "timestamp": datetime.now().isoformat()
                }
            }))
        
        # Execute pipeline from the specified point
        async for event in bridge.execute_pipeline(
            modified_input.get("query", ""),
            modified_input.get("context", {})
        ):
            # Events are automatically published by the bridge
            pass
            
    except Exception as e:
        logger.error(f"Replay execution failed: {e}")
        
        # Update execution status
        db = get_db()
        db.execute(
            text("""
            UPDATE pipeline_executions 
            SET status = 'failed', completed_at = :completed_at, error = :error
            WHERE id = :execution_id
            """),
            {
                "execution_id": replay_execution_id,
                "completed_at": datetime.now(),
                "error": str(e)
            }
        )
        db.commit()
        
        # Publish error
        if redis_client:
            redis_client.publish(channel, json.dumps({
                "type": "error",
                "payload": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }))


@router.get("/replay/{replay_execution_id}/status")
async def get_replay_status(
    replay_execution_id: str,
    db: Session = Depends(get_db)
):
    """Get the status of a replay execution"""
    
    result = db.execute(
        text("""
        SELECT * FROM pipeline_executions 
        WHERE id = :execution_id AND is_replay = true
        """),
        {"execution_id": replay_execution_id}
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Replay execution not found")
    
    return {
        "execution_id": result.id,
        "pipeline_id": result.pipeline_id,
        "status": result.status,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "parent_execution_id": result.parent_execution_id,
        "replay_from_agent": result.replay_from_agent,
        "error": result.error
    }


@router.post("/io-templates")
async def save_io_template(
    template_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Save an agent I/O data pair as a template for future use"""
    
    # Validate required fields
    required_fields = ["name", "agent_name", "input_template"]
    for field in required_fields:
        if field not in template_data:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required field: {field}"
            )
    
    # Insert template
    db.execute(
        text("""
        INSERT INTO pipeline_io_templates 
        (name, description, agent_name, input_template, expected_output, tags, created_by)
        VALUES (:name, :description, :agent_name, :input_template, 
                :expected_output, :tags, :created_by)
        """),
        {
            "name": template_data["name"],
            "description": template_data.get("description", ""),
            "agent_name": template_data["agent_name"],
            "input_template": json.dumps(template_data["input_template"]),
            "expected_output": json.dumps(template_data.get("expected_output", {})),
            "tags": template_data.get("tags", []),
            "created_by": template_data.get("created_by", "system")
        }
    )
    db.commit()
    
    return {"status": "success", "message": "Template saved successfully"}


@router.get("/io-templates/{agent_name}")
async def get_io_templates(
    agent_name: str,
    db: Session = Depends(get_db)
):
    """Get all I/O templates for a specific agent"""
    
    results = db.execute(
        text("""
        SELECT * FROM pipeline_io_templates 
        WHERE agent_name = :agent_name
        ORDER BY created_at DESC
        """),
        {"agent_name": agent_name}
    ).fetchall()
    
    templates = []
    for result in results:
        templates.append({
            "id": result.id,
            "name": result.name,
            "description": result.description,
            "agent_name": result.agent_name,
            "input_template": json.loads(result.input_template),
            "expected_output": json.loads(result.expected_output) if result.expected_output else None,
            "tags": result.tags,
            "created_by": result.created_by,
            "created_at": result.created_at
        })
    
    return templates