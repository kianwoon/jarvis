"""
API endpoints for Agentic Pipeline feature.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from app.core.pipeline_manager import PipelineManager
from app.core.pipeline_executor import PipelineExecutor
from app.core.pipeline_config import get_pipeline_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_pipeline_settings()

# Initialize managers
pipeline_manager = PipelineManager()
pipeline_executor = PipelineExecutor()


# Pydantic models
class AgentConfig(BaseModel):
    agent_name: str
    execution_order: Optional[int] = 0
    parent_agent: Optional[str] = None
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PipelineCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    goal: Optional[str] = ""
    collaboration_mode: str = Field(...)
    is_active: Optional[bool] = True
    agents: List[AgentConfig]
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_by: Optional[str] = "user"


class PipelineUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    goal: Optional[str] = None
    collaboration_mode: Optional[str] = Field(None)
    is_active: Optional[bool] = None
    agents: Optional[List[AgentConfig]] = None
    config: Optional[Dict[str, Any]] = None


class PipelineAgentUpdate(BaseModel):
    """Model for updating a specific agent within a pipeline"""
    config: Dict[str, Any] = Field(default_factory=dict)
    system_prompt: Optional[str] = None
    tools: Optional[List[str]] = None


class PipelineExecuteRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    trigger_type: Optional[str] = "manual"
    additional_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    direct_execution: Optional[bool] = False
    debug_mode: Optional[bool] = Field(default=False, description="Enable detailed I/O logging for debugging")


class ScheduleConfig(BaseModel):
    schedule_type: str = Field(...)
    schedule_config: Dict[str, Any]
    is_active: Optional[bool] = True


class CommunicationPattern(BaseModel):
    from_agent: str
    to_agent: str
    pattern_type: str = Field(...)
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AddAgentRequest(BaseModel):
    agent_type: str
    role: str = "output"
    configuration: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ValidationRequest(BaseModel):
    pipeline_id: Optional[int] = None
    pipeline_data: Optional[Dict[str, Any]] = None


@router.get("/")
async def list_pipelines(active_only: bool = False):
    """List all pipelines."""
    try:
        pipelines = await pipeline_manager.list_pipelines(active_only=active_only)
        return {
            "pipelines": pipelines,
            "total": len(pipelines)
        }
    except Exception as e:
        logger.error(f"Failed to list pipelines: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def create_pipeline(pipeline: PipelineCreate):
    """Create a new pipeline."""
    try:
        # Validate collaboration mode
        if pipeline.collaboration_mode not in settings.get_collaboration_modes():
            raise ValueError(f"Invalid collaboration mode: {pipeline.collaboration_mode}. Valid modes: {settings.get_collaboration_modes()}")
        
        # Validate agents exist
        # TODO: Add validation against available agents
        
        pipeline_data = pipeline.dict()
        created_pipeline = await pipeline_manager.create_pipeline(pipeline_data)
        
        return {
            "message": "Pipeline created successfully",
            "pipeline": created_pipeline
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_id}")
async def get_pipeline(pipeline_id: int):
    """Get pipeline details."""
    try:
        pipeline = await pipeline_manager.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        # Enrich agents with full details from langgraph
        try:
            # Import here to avoid circular dependencies
            from app.core.langgraph_agents_cache import get_langgraph_agents
            
            # Get all available agents
            langgraph_agents = get_langgraph_agents()
            langgraph_agents_map = {agent['name']: agent for agent in langgraph_agents.values()}
            
            # Enrich pipeline agents with langgraph ID only - DO NOT OVERWRITE CONFIG
            for agent in pipeline.get('agents', []):
                agent_name = agent.get('agent_name', '')
                if agent_name in langgraph_agents_map:
                    langgraph_agent = langgraph_agents_map[agent_name]
                    # ONLY add the langgraph ID for reference
                    # DO NOT overwrite any config values from the database!
                    agent['langgraph_agent_id'] = langgraph_agent.get('id')
                    
                    # If config is missing certain fields, use langgraph as fallback ONLY
                    if 'config' not in agent:
                        agent['config'] = {}
                    
                    # Only fill in missing values, never overwrite existing ones
                    if 'role' not in agent['config'] and not agent['config'].get('role'):
                        agent['config']['role'] = langgraph_agent.get('role', 'Agent')
                    if 'description' not in agent['config'] and not agent['config'].get('description'):
                        agent['config']['description'] = langgraph_agent.get('description', '')
        except Exception as e:
            logger.warning(f"Failed to enrich agents with langgraph details: {str(e)}")
            # Continue without enrichment if it fails
        
        return pipeline
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{pipeline_id}")
async def update_pipeline(pipeline_id: int, update: PipelineUpdate):
    """Update pipeline."""
    try:
        # Filter out None values
        update_data = {k: v for k, v in update.dict().items() if v is not None}
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No update data provided")
        
        # Validate collaboration mode if provided
        if update.collaboration_mode and update.collaboration_mode not in settings.get_collaboration_modes():
            raise ValueError(f"Invalid collaboration mode: {update.collaboration_mode}. Valid modes: {settings.get_collaboration_modes()}")
        
        updated_pipeline = await pipeline_manager.update_pipeline(
            pipeline_id,
            update_data
        )
        
        return {
            "message": "Pipeline updated successfully",
            "pipeline": updated_pipeline
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{pipeline_id}")
async def delete_pipeline(pipeline_id: int):
    """Delete pipeline."""
    try:
        # Remove from scheduler if scheduled
        from app.core.pipeline_scheduler import pipeline_scheduler
        pipeline_scheduler._unschedule_pipeline(pipeline_id)
        
        # Delete pipeline and related data
        success = await pipeline_manager.delete_pipeline(pipeline_id)
        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return {"message": "Pipeline deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_id}/agents/{agent_id}")
async def get_pipeline_agent_direct(
    pipeline_id: int,
    agent_id: int
):
    """Get a specific pipeline agent configuration directly from database (no cache)"""
    from app.core.db import SessionLocal
    from sqlalchemy import text
    import json
    
    db = SessionLocal()
    try:
        # Query database directly - NO CACHE
        result = db.execute(
            text("""
                SELECT id, pipeline_id, agent_name, execution_order, parent_agent, config 
                FROM pipeline_agents 
                WHERE id = :agent_id AND pipeline_id = :pipeline_id
            """),
            {"agent_id": agent_id, "pipeline_id": pipeline_id}
        ).fetchone()
        
        if not result:
            raise HTTPException(
                status_code=404, 
                detail=f"Pipeline agent {agent_id} not found in pipeline {pipeline_id}"
            )
        
        # Convert to dict
        agent_data = {
            "id": result.id,
            "pipeline_id": result.pipeline_id,
            "agent_name": result.agent_name,
            "execution_order": result.execution_order,
            "parent_agent": result.parent_agent,
            "config": result.config or {}
        }
        
        logger.info(f"Retrieved pipeline agent {agent_id} directly from database")
        return agent_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline agent directly: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.put("/{pipeline_id}/agents/{agent_id}")
async def update_pipeline_agent(
    pipeline_id: int,
    agent_id: int,
    agent_update: PipelineAgentUpdate
):
    """Update a specific agent configuration within a pipeline"""
    from app.core.db import SessionLocal
    from sqlalchemy import text
    import json
    
    db = SessionLocal()
    try:
        # Check if the pipeline agent exists
        result = db.execute(
            text("""
                SELECT id, config 
                FROM pipeline_agents 
                WHERE id = :agent_id AND pipeline_id = :pipeline_id
            """),
            {"agent_id": agent_id, "pipeline_id": pipeline_id}
        ).fetchone()
        
        if not result:
            raise HTTPException(
                status_code=404, 
                detail=f"Pipeline agent {agent_id} not found in pipeline {pipeline_id}"
            )
        
        # Get current config
        current_config = result.config or {}
        
        # Merge configurations
        updated_config = {**current_config, **agent_update.config}
        
        # Add system_prompt and tools if provided
        if agent_update.system_prompt is not None:
            updated_config["system_prompt"] = agent_update.system_prompt
        if agent_update.tools is not None:
            updated_config["tools"] = agent_update.tools
        
        # Update the pipeline agent (fix parameter syntax)
        db.execute(
            text("""
                UPDATE pipeline_agents 
                SET config = :config 
                WHERE id = :agent_id AND pipeline_id = :pipeline_id
            """),
            {
                "config": json.dumps(updated_config),
                "agent_id": agent_id,
                "pipeline_id": pipeline_id
            }
        )
        db.commit()
        
        # Clear pipeline cache if method exists
        if hasattr(pipeline_manager, 'clear_pipeline_cache'):
            await pipeline_manager.clear_pipeline_cache(pipeline_id)
        
        # Also clear langgraph agents cache to force fresh data loading
        try:
            from app.core.langgraph_agents_cache import reload_cache_from_db
            reload_cache_from_db()
            logger.info("Cleared langgraph agents cache after pipeline agent update")
        except Exception as e:
            logger.warning(f"Failed to clear langgraph agents cache: {e}")
        
        logger.info(f"Updated pipeline agent {agent_id} in pipeline {pipeline_id}")
        
        return {
            "message": "Pipeline agent updated successfully",
            "agent_id": agent_id,
            "pipeline_id": pipeline_id,
            "config": updated_config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update pipeline agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/{pipeline_id}/execute")
async def execute_pipeline(
    pipeline_id: int,
    request: PipelineExecuteRequest,
    background_tasks: BackgroundTasks
):
    """Execute a pipeline."""
    try:
        # Get pipeline details to access goal
        pipeline = await pipeline_manager.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        # Prepare input data
        input_data = {
            "query": request.query,
            "conversation_history": request.conversation_history,
            "direct_execution": request.direct_execution,
            "debug_mode": request.debug_mode,
            **request.additional_params
        }
        
        # If direct execution and no query, use pipeline goal as context
        if request.direct_execution and not request.query:
            input_data["pipeline_goal"] = pipeline.get("goal", pipeline.get("description", "Execute the pipeline tasks"))
        
        # Start pipeline execution in background and return execution_id immediately
        # Create execution record first to get execution_id
        execution_id = await pipeline_executor.pipeline_manager.record_execution(
            pipeline_id=pipeline_id,
            trigger_type=request.trigger_type,
            status="starting",
            input_data=input_data
        )
        
        # Start execution in background using asyncio.create_task
        import asyncio
        
        async def execute_pipeline_background():
            """Execute pipeline in background and handle completion"""
            try:
                logger.info(f"[BACKGROUND] Starting pipeline {pipeline_id} execution {execution_id}")
                result = await pipeline_executor.execute_pipeline(
                    pipeline_id=pipeline_id,
                    input_data=input_data,
                    trigger_type=request.trigger_type,
                    existing_execution_id=execution_id  # Pass existing execution_id
                )
                logger.info(f"[BACKGROUND] Pipeline {pipeline_id} execution {execution_id} completed successfully")
            except Exception as e:
                logger.error(f"[BACKGROUND] Pipeline {pipeline_id} execution {execution_id} failed: {e}")
                # Update execution status to failed
                try:
                    await pipeline_executor.pipeline_manager.update_execution_status(
                        execution_id=execution_id,
                        status="failed",
                        result={"error": str(e)}
                    )
                except Exception as update_error:
                    logger.error(f"[BACKGROUND] Failed to update execution status: {update_error}")
        
        # Start background execution (fire and forget)
        asyncio.create_task(execute_pipeline_background())
        
        # Return execution details immediately for frontend tracking
        return {
            "execution_id": execution_id,
            "pipeline_id": pipeline_id,
            "status": "starting",
            "message": "Pipeline execution started. Use WebSocket to monitor progress.",
            "websocket_url": f"/ws/execution/{execution_id}",
            "monitor_url": f"/api/v1/pipelines/executions/{execution_id}"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_id}/executions")
async def get_execution_history(pipeline_id: int, limit: int = None):
    """Get pipeline execution history."""
    try:
        executions = await pipeline_manager.get_execution_history(
            pipeline_id,
            limit=limit
        )
        
        return {
            "executions": executions,
            "total": len(executions)
        }
    except Exception as e:
        logger.error(f"Failed to get execution history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions/{execution_id}")
async def get_execution_details(execution_id: int):
    """Get full execution details including agent outputs."""
    try:
        # Get execution from database
        from app.core.db import SessionLocal
        from sqlalchemy import text
        
        db = SessionLocal()
        try:
            # Simple query without casting
            result = db.execute(
                text("""
                    SELECT * FROM pipeline_executions
                    WHERE id = :execution_id
                    LIMIT 1
                """),
                {"execution_id": str(execution_id)}
            ).fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Execution not found")
            
            # Skip the fucking agent_io table - we don't need it
            agent_io = []  # The data is already in the execution result
            
            # Format response dynamically based on available columns
            result_dict = dict(result._mapping) if hasattr(result, '_mapping') else dict(zip(result.keys(), result))
            
            # Build execution data
            execution_data = {
                "execution_id": str(result_dict.get('id', execution_id)),
                "pipeline_id": result_dict.get('pipeline_id'),
                "status": result_dict.get('status', 'unknown'),
                "trigger_type": result_dict.get('trigger_type', 'manual'),
                "input_data": result_dict.get('input_data') or {
                    "query": result_dict.get('initial_query', ''),
                    "context": result_dict.get('context', {})
                },
                "output_data": result_dict.get('output_data') or result_dict.get('result'),
                "error_message": result_dict.get('error_message') or result_dict.get('error'),
                "execution_metadata": result_dict.get('execution_metadata', {}),
                "created_at": None,
                "updated_at": None
            }
            
            # Handle date fields
            for date_field in ['created_at', 'started_at']:
                if result_dict.get(date_field):
                    execution_data["created_at"] = result_dict[date_field].isoformat()
                    break
                    
            for date_field in ['updated_at', 'completed_at']:
                if result_dict.get(date_field):
                    execution_data["updated_at"] = result_dict[date_field].isoformat()
                    break
            
            # Add empty agent_io list initially
            execution_data["agent_io"] = []
            
            # Try to get agent I/O data
            if agent_io:
                execution_data["agent_io"] = [
                    {
                        "agent_name": io.agent_name,
                        "status": io.status,
                        "input": io.input_data,
                        "output": io.output_data,
                        "tools_used": io.tools_used,
                        "metrics": io.metrics,
                        "error": io.error_message,
                        "started_at": io.started_at.isoformat() if io.started_at else None,
                        "completed_at": io.completed_at.isoformat() if io.completed_at else None,
                        "execution_time": (
                            (io.completed_at - io.started_at).total_seconds()
                            if io.completed_at and io.started_at else None
                        )
                    }
                    for io in agent_io
                ]
            
            # If no agent I/O records, try to extract from result
            if not agent_io and result.output_data:
                try:
                    output = result.output_data
                    if isinstance(output, dict):
                        # Check for agent_outputs in result
                        if 'agent_outputs' in output:
                            execution_data['agent_io'] = [
                                {
                                    "agent_name": agent.get('agent', 'Unknown'),
                                    "status": "completed",
                                    "input": {"query": result_dict.get('initial_query', '')},
                                    "output": {"response": agent.get('output', agent.get('content', ''))},
                                    "tools_used": agent.get('tools_used', []),
                                    "metrics": {},
                                    "error": None,
                                    "started_at": result.started_at.isoformat() if result.started_at else None,
                                    "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                                    "execution_time": agent.get('execution_time', 0)
                                }
                                for agent in output.get('agent_outputs', [])
                            ]
                        # Check for nested result structure
                        elif 'result' in output and isinstance(output['result'], dict):
                            nested_result = output['result']
                            if 'agent_outputs' in nested_result:
                                execution_data['agent_io'] = [
                                    {
                                        "agent_name": agent.get('agent', 'Unknown'),
                                        "status": "completed",
                                        "input": {"query": result_dict.get('initial_query', '')},
                                        "output": {"response": agent.get('output', agent.get('content', ''))},
                                        "tools_used": agent.get('tools_used', []),
                                        "metrics": {},
                                        "error": None,
                                        "started_at": result.started_at.isoformat() if result.started_at else None,
                                        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                                        "execution_time": agent.get('execution_time', 0)
                                    }
                                    for agent in nested_result.get('agent_outputs', [])
                                ]
                except Exception as e:
                    logger.warning(f"Failed to extract agent data from result: {e}")
            
            return execution_data
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/executions/{execution_id}/progress")
async def get_execution_progress(execution_id: int):
    """Get execution progress."""
    try:
        progress = await pipeline_executor.get_execution_progress(execution_id)
        return {
            "execution_id": execution_id,
            "progress": progress
        }
    except Exception as e:
        logger.error(f"Failed to get execution progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: int):
    """Cancel a running execution."""
    try:
        success = await pipeline_executor.cancel_execution(execution_id)
        if success:
            return {"message": "Execution cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel execution")
    except Exception as e:
        logger.error(f"Failed to cancel execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_id}/preview")
async def preview_pipeline(pipeline_id: int):
    """Preview pipeline flow."""
    try:
        pipeline = await pipeline_manager.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        # Generate flow preview
        preview = {
            "name": pipeline["name"],
            "mode": pipeline["collaboration_mode"],
            "agents": []
        }
        
        if pipeline["collaboration_mode"] == "sequential":
            # Sort by execution order
            sorted_agents = sorted(
                pipeline["agents"],
                key=lambda x: x.get("execution_order", 0)
            )
            preview["agents"] = [
                {
                    "name": agent["agent_name"],
                    "order": agent.get("execution_order", 0)
                }
                for agent in sorted_agents
            ]
        elif pipeline["collaboration_mode"] == "hierarchical":
            # Build hierarchy
            hierarchy = {}
            for agent in pipeline["agents"]:
                if agent.get("parent_agent"):
                    parent = agent["parent_agent"]
                    if parent not in hierarchy:
                        hierarchy[parent] = []
                    hierarchy[parent].append(agent["agent_name"])
            
            preview["hierarchy"] = hierarchy
            preview["agents"] = [
                {
                    "name": agent["agent_name"],
                    "parent": agent.get("parent_agent"),
                    "children": hierarchy.get(agent["agent_name"], [])
                }
                for agent in pipeline["agents"]
            ]
        else:  # parallel
            preview["agents"] = [
                {"name": agent["agent_name"]}
                for agent in pipeline["agents"]
            ]
        
        return preview
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to preview pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{pipeline_id}/schedule")
async def create_or_update_schedule(pipeline_id: int, schedule: ScheduleConfig):
    """Create or update pipeline schedule."""
    try:
        # Validate schedule type
        if schedule.schedule_type not in settings.get_schedule_types():
            raise ValueError(f"Invalid schedule type: {schedule.schedule_type}. Valid types: {settings.get_schedule_types()}")
        
        # Check if pipeline exists
        pipeline = await pipeline_manager.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        # Create or update schedule
        created_schedule = await pipeline_manager.create_or_update_schedule(
            pipeline_id=pipeline_id,
            schedule_type=schedule.schedule_type,
            schedule_config=schedule.schedule_config,
            is_active=schedule.is_active
        )
        
        return {
            "message": "Schedule created/updated successfully",
            "schedule": created_schedule
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create/update schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_id}/schedule")
async def get_pipeline_schedule(pipeline_id: int):
    """Get pipeline schedule."""
    try:
        schedule = await pipeline_manager.get_pipeline_schedule(pipeline_id)
        if not schedule:
            # Return 404 for no schedule (not an error, just not found)
            raise HTTPException(status_code=404, detail="No schedule configured for this pipeline")
        
        return schedule
    except HTTPException:
        # Re-raise HTTP exceptions without logging as errors
        raise
    except Exception as e:
        logger.error(f"Failed to get schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{pipeline_id}/schedule")
async def delete_pipeline_schedule(pipeline_id: int):
    """Delete pipeline schedule."""
    try:
        success = await pipeline_manager.delete_pipeline_schedule(pipeline_id)
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return {"message": "Schedule deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scheduler/status")
async def get_scheduler_status():
    """Get scheduler service status."""
    try:
        from app.core.pipeline_scheduler import pipeline_scheduler
        
        active_jobs = []
        for job in pipeline_scheduler.scheduler.get_jobs():
            active_jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
        
        return {
            "status": "running" if pipeline_scheduler.scheduler.running else "stopped",
            "active_jobs": len(active_jobs),
            "jobs": active_jobs
        }
    except Exception as e:
        logger.error(f"Failed to get scheduler status: {str(e)}")
        return {"status": "error", "error": str(e)}


@router.get("/config/options")
async def get_configuration_options():
    """Get available configuration options for pipelines."""
    try:
        return {
            "collaboration_modes": settings.get_collaboration_modes(),
            "communication_patterns": settings.get_communication_patterns(),
            "schedule_types": settings.get_schedule_types(),
            "output_agent_patterns": settings.PIPELINE_OUTPUT_AGENT_PATTERNS,
            "limits": {
                "max_agents": settings.PIPELINE_MAX_AGENTS,
                "max_execution_time": settings.PIPELINE_MAX_EXECUTION_TIME,
                "max_history": settings.PIPELINE_MAX_HISTORY
            },
            "features": {
                "approvals_enabled": settings.PIPELINE_ENABLE_APPROVALS,
                "audit_trail_enabled": settings.PIPELINE_ENABLE_AUDIT_TRAIL,
                "cost_tracking_enabled": settings.PIPELINE_ENABLE_COST_TRACKING,
                "sla_monitoring_enabled": settings.PIPELINE_ENABLE_SLA_MONITORING
            }
        }
    except Exception as e:
        logger.error(f"Failed to get configuration options: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_pipeline(request: ValidationRequest):
    """Validate a pipeline configuration."""
    try:
        # Get pipeline data either from ID or direct data
        pipeline = None
        if request.pipeline_id:
            pipeline = await pipeline_manager.get_pipeline(request.pipeline_id)
            if not pipeline:
                raise HTTPException(status_code=404, detail="Pipeline not found")
        elif request.pipeline_data:
            pipeline = request.pipeline_data
        else:
            raise HTTPException(status_code=400, detail="Either pipeline_id or pipeline_data must be provided")
        
        # Perform validation
        errors = []
        warnings = []
        info = []
        suggestions = []
        
        # Check for required fields
        if not pipeline.get('name'):
            errors.append({
                'type': 'MISSING_FIELD',
                'severity': 'error',
                'message': 'Pipeline name is required',
                'details': {'field': 'name'}
            })
        
        if not pipeline.get('agents') or len(pipeline['agents']) == 0:
            errors.append({
                'type': 'NO_AGENTS',
                'severity': 'error',
                'message': 'Pipeline must have at least one agent',
                'details': {}
            })
        
        # Check for output agent - but be smart about it
        agents = pipeline.get('agents', [])
        
        # Use configured output agent patterns
        output_producing_agents = settings.PIPELINE_OUTPUT_AGENT_PATTERNS
        
        # Check if any agent has output capabilities using settings
        has_output_capability = False
        for agent in agents:
            if settings.is_output_agent(agent.get('agent_name', ''), agent.get('config', {})):
                has_output_capability = True
                break
                
            # Check if it's the last agent in a sequential pipeline (likely the output)
            if pipeline.get('collaboration_mode') == 'sequential' and agent == agents[-1]:
                # Last agent in sequential mode is usually the output
                has_output_capability = True
                break
        
        # Only warn if there's truly no output capability
        if not has_output_capability and len(agents) > 1:
            warnings.append({
                'type': 'NO_OUTPUT_AGENT',
                'severity': 'warning',
                'message': 'Pipeline may need an agent to handle final output',
                'details': {
                    'suggestion': 'Consider if your last agent produces the desired output'
                }
            })
        
        # Check communication patterns for sequential mode
        if pipeline.get('collaboration_mode') == 'sequential' and len(agents) > 1:
            # Check if agents have proper communication patterns
            config = pipeline.get('config', {})
            communication_patterns = config.get('communication_patterns', [])
            
            # For sequential, check if each agent communicates with the next
            for i in range(len(agents) - 1):
                from_agent = agents[i].get('agent_name')
                to_agent = agents[i + 1].get('agent_name')
                
                has_pattern = any(
                    p.get('from_agent') == from_agent and p.get('to_agent') == to_agent
                    for p in communication_patterns
                )
                
                if not has_pattern:
                    warnings.append({
                        'type': 'COMMUNICATION_PATTERN_MISSING',
                        'severity': 'warning',
                        'message': f'No communication pattern between {from_agent} and {to_agent}',
                        'details': {
                            'from_agent': from_agent,
                            'to_agent': to_agent
                        }
                    })
        
        # Add execution estimate
        execution_estimate = {
            'estimated_seconds': len(agents) * 10,  # Rough estimate
            'estimated_minutes': (len(agents) * 10) / 60,
            'confidence': 'medium',
            'factors': {
                'agent_count': len(agents),
                'mode': pipeline.get('collaboration_mode', 'unknown')
            }
        }
        
        # Provide suggestions
        if pipeline.get('collaboration_mode') == 'parallel' and len(agents) > 5:
            suggestions.append({
                'message': 'Consider using hierarchical mode for better organization with many agents',
                'action': 'change_mode'
            })
        
        is_valid = len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'info': info,
            'suggestions': suggestions,
            'execution_estimate': execution_estimate
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{pipeline_id}/communication-patterns")
async def create_communication_pattern(pipeline_id: int, pattern: CommunicationPattern):
    """Create a communication pattern between agents in a pipeline."""
    try:
        # Validate pattern type
        if pattern.pattern_type not in settings.get_communication_patterns():
            raise ValueError(f"Invalid pattern type: {pattern.pattern_type}. Valid types: {settings.get_communication_patterns()}")
        
        # Get pipeline to validate it exists
        pipeline = await pipeline_manager.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        # Validate agents exist in pipeline
        agent_names = [agent.get('agent_name') for agent in pipeline.get('agents', [])]
        if pattern.from_agent not in agent_names:
            raise HTTPException(status_code=400, detail=f"Agent '{pattern.from_agent}' not found in pipeline")
        if pattern.to_agent not in agent_names:
            raise HTTPException(status_code=400, detail=f"Agent '{pattern.to_agent}' not found in pipeline")
        
        # Update pipeline configuration with communication pattern
        config = pipeline.get('config', {})
        if 'communication_patterns' not in config:
            config['communication_patterns'] = []
        
        # Add the new pattern
        new_pattern = {
            'from_agent': pattern.from_agent,
            'to_agent': pattern.to_agent,
            'pattern_type': pattern.pattern_type,
            'parameters': pattern.parameters
        }
        config['communication_patterns'].append(new_pattern)
        
        # Update pipeline
        updated_pipeline = await pipeline_manager.update_pipeline(
            pipeline_id,
            {'config': config}
        )
        
        return {
            "message": "Communication pattern created successfully",
            "pattern": new_pattern
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create communication pattern: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{pipeline_id}/agents")
async def add_agent_to_pipeline(pipeline_id: int, request: AddAgentRequest):
    """Add an agent to a pipeline."""
    try:
        # Get pipeline to validate it exists
        pipeline = await pipeline_manager.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        # Get current agents
        agents = pipeline.get('agents', [])
        
        # Create new agent configuration
        new_agent = {
            'agent_name': request.agent_type,
            'execution_order': len(agents),  # Add at the end
            'config': {
                'role': request.role,
                **request.configuration
            }
        }
        
        # Add the new agent
        agents.append(new_agent)
        
        # Update pipeline
        updated_pipeline = await pipeline_manager.update_pipeline(
            pipeline_id,
            {'agents': agents}
        )
        
        return {
            "message": "Agent added successfully",
            "agent": new_agent
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))