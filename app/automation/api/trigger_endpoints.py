"""
External Trigger API Endpoints for TriggerNode
Allows external systems to trigger workflow execution via public endpoints
"""
from fastapi import APIRouter, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import uuid
import secrets
import json
import time
import asyncio
from datetime import datetime

from app.automation.core.automation_cache import (
    get_automation_workflows, 
    get_workflow_by_id,
    cache_workflow_execution
)
from app.automation.integrations.postgres_bridge import postgres_bridge
from app.automation.core.automation_executor import AutomationExecutor

logger = logging.getLogger(__name__)
router = APIRouter()

# Rate limiting storage (in production, use Redis)
rate_limit_storage = {}

# Pydantic models for trigger endpoints
class TriggerRequest(BaseModel):
    """Request model for external trigger"""
    data: Optional[Dict[str, Any]] = Field(None, description="Payload data")

class TriggerResponse(BaseModel):
    """Response model for external trigger"""
    success: bool
    execution_id: str
    status: str
    output: Optional[Any] = None
    message: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: str

class TriggerConfig(BaseModel):
    """Configuration model for trigger setup"""
    workflow_id: int
    trigger_name: str
    http_methods: List[str] = ["POST"]
    authentication_type: str = "api_key"
    auth_header_name: Optional[str] = "X-API-Key"
    auth_token: Optional[str] = None
    rate_limit: int = 60
    timeout: int = 300
    response_format: str = "workflow_output"
    custom_response_template: Optional[str] = None
    cors_enabled: bool = True
    cors_origins: str = "*"
    log_requests: bool = True

def generate_secure_token() -> str:
    """Generate a secure API token"""
    return secrets.token_urlsafe(32)

def validate_authentication(
    trigger_config: Dict[str, Any],
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
) -> bool:
    """Validate authentication based on trigger configuration"""
    auth_type = trigger_config.get("authentication_type", "none")
    
    if auth_type == "none":
        return True
    
    expected_token = trigger_config.get("auth_token")
    if not expected_token:
        return False
    
    if auth_type == "api_key":
        header_name = trigger_config.get("auth_header_name", "X-API-Key")
        if header_name.lower() == "x-api-key":
            provided_token = x_api_key
        else:
            provided_token = request.headers.get(header_name)
        return provided_token == expected_token
    
    elif auth_type == "bearer_token":
        if not authorization or not authorization.startswith("Bearer "):
            return False
        provided_token = authorization[7:]  # Remove "Bearer " prefix
        return provided_token == expected_token
    
    elif auth_type == "basic_auth":
        # Basic auth validation would go here
        return False  # TODO: Implement basic auth
    
    elif auth_type == "custom_header":
        header_name = trigger_config.get("auth_header_name", "X-Custom-Auth")
        provided_token = request.headers.get(header_name)
        return provided_token == expected_token
    
    return False

def check_rate_limit(trigger_id: str, rate_limit: int) -> bool:
    """Check if request is within rate limit"""
    if rate_limit <= 0:  # Unlimited
        return True
    
    current_time = time.time()
    minute_window = int(current_time // 60)
    
    if trigger_id not in rate_limit_storage:
        rate_limit_storage[trigger_id] = {}
    
    # Clean old entries
    rate_limit_storage[trigger_id] = {
        window: count for window, count in rate_limit_storage[trigger_id].items()
        if current_time - (window * 60) < 60
    }
    
    # Check current minute
    current_count = rate_limit_storage[trigger_id].get(minute_window, 0)
    if current_count >= rate_limit:
        return False
    
    # Increment counter
    rate_limit_storage[trigger_id][minute_window] = current_count + 1
    return True

def format_response(
    response_format: str,
    custom_template: Optional[str],
    execution_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Format response based on configuration"""
    output_data = execution_data.get("output", {})
    
    # Ensure output_data is always a dict
    if not isinstance(output_data, dict):
        output_data = {"result": str(output_data)}
    
    if response_format == "workflow_output":
        return output_data
    
    elif response_format == "status_only":
        return {
            "status": execution_data.get("status", "unknown"),
            "execution_id": execution_data.get("execution_id", "")
        }
    
    elif response_format == "detailed":
        return {
            "success": execution_data.get("status") == "completed",
            "execution_id": execution_data.get("execution_id", ""),
            "status": execution_data.get("status", "unknown"),
            "output": output_data,
            "execution_time": execution_data.get("execution_time", 0),
            "timestamp": execution_data.get("completed_at", datetime.utcnow().isoformat())
        }
    
    elif response_format == "custom" and custom_template:
        try:
            # Simple template substitution
            template = custom_template
            template = template.replace("{{output}}", json.dumps(output_data))
            template = template.replace("{{status}}", execution_data.get("status", "unknown"))
            template = template.replace("{{execution_time}}", str(execution_data.get("execution_time", 0)))
            template = template.replace("{{timestamp}}", execution_data.get("completed_at", datetime.utcnow().isoformat()))
            template = template.replace("{{execution_id}}", execution_data.get("execution_id", ""))
            return json.loads(template)
        except Exception as e:
            logger.error(f"Error formatting custom response: {e}")
            return {"error": "Invalid response template"}
    
    # Default to workflow output
    return output_data

async def find_trigger_node_in_workflow(workflow_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find TriggerNode in workflow configuration"""
    logger.debug(f"Finding trigger node in workflow: type={type(workflow_config)}")
    
    # Handle case where workflow_config might be a string
    if isinstance(workflow_config, str):
        logger.error(f"Workflow config is string instead of dict: {workflow_config[:100]}...")
        return None
    
    if not isinstance(workflow_config, dict):
        logger.error(f"Workflow config is not dict: type={type(workflow_config)}")
        return None
    
    nodes = workflow_config.get("langflow_config", {}).get("nodes", [])
    logger.debug(f"Found {len(nodes)} nodes in workflow")
    
    for node in nodes:
        node_data = node.get("data", {})
        if node_data.get("type") == "TriggerNode":
            logger.debug(f"Found TriggerNode: {node_data.get('node', {})}")
            return node_data.get("node", {})
    
    return None

@router.post("/trigger/{trigger_id}")
async def execute_external_trigger(
    trigger_id: str,
    request: Request,
    trigger_request: Optional[TriggerRequest] = None,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
):
    """External endpoint for triggering workflow execution"""
    start_time = time.time()
    
    try:
        # Get request data
        request_data = {}
        if trigger_request:
            request_data = trigger_request.data or {}
        
        # Get query parameters
        query_params = dict(request.query_params)
        
        # Get headers (excluding sensitive ones)
        request_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ["authorization", "x-api-key", "cookie"]
        }
        
        # Find workflow with matching trigger
        logger.info(f"Getting automation workflows for trigger: {trigger_id}")
        workflows = get_automation_workflows()
        logger.info(f"Found {len(workflows)} total workflows")
        
        target_workflow = None
        trigger_config = None
        
        for workflow_id, workflow_data in workflows.items():
            logger.info(f"Checking workflow {workflow_id}: type={type(workflow_data)}")
            logger.info(f"Workflow structure keys: {list(workflow_data.keys())}")
            
            try:
                trigger_node_config = await find_trigger_node_in_workflow(workflow_data)
                logger.info(f"Trigger node config for workflow {workflow_id}: {trigger_node_config}")
                
                if trigger_node_config and trigger_node_config.get("trigger_name") == trigger_id:
                    logger.info(f"Found matching trigger in workflow {workflow_id}")
                    target_workflow = workflow_data
                    target_workflow["id"] = workflow_id
                    trigger_config = trigger_node_config
                    
                    # Debug the workflow structure
                    logger.info(f"Target workflow keys: {list(target_workflow.keys())}")
                    if "langflow_config" in target_workflow:
                        langflow_config = target_workflow["langflow_config"]
                        logger.info(f"Langflow config keys: {list(langflow_config.keys())}")
                        if "nodes" in langflow_config:
                            nodes = langflow_config["nodes"]
                            logger.info(f"Found {len(nodes)} nodes in langflow_config")
                            for i, node in enumerate(nodes[:3]):  # Show first 3 nodes
                                logger.info(f"Node {i}: type={node.get('data', {}).get('type')}, id={node.get('id')}")
                    break
            except Exception as e:
                logger.error(f"Error checking workflow {workflow_id}: {e}")
                continue
        
        if not target_workflow or not trigger_config:
            raise HTTPException(status_code=404, detail="Trigger not found")
        
        logger.info(f"Target workflow active status: {target_workflow.get('is_active', False)}")
        if not target_workflow.get("is_active", False):
            raise HTTPException(status_code=400, detail="Workflow is not active")
        
        # Validate HTTP method
        allowed_methods = trigger_config.get("http_methods", ["POST"])
        logger.info(f"Request method: {request.method}, allowed: {allowed_methods}")
        if request.method not in allowed_methods:
            raise HTTPException(
                status_code=405, 
                detail=f"Method {request.method} not allowed. Allowed: {', '.join(allowed_methods)}"
            )
        
        # Validate authentication
        logger.info(f"Validating authentication for trigger: {trigger_id}")
        if not validate_authentication(trigger_config, request, authorization, x_api_key):
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        # Check rate limiting
        rate_limit = trigger_config.get("rate_limit", 60)
        logger.info(f"Checking rate limit: {rate_limit} for trigger: {trigger_id}")
        if not check_rate_limit(trigger_id, rate_limit):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Log request if enabled
        if trigger_config.get("log_requests", True):
            logger.info(f"Trigger {trigger_id} called: {request.method} from {request.client.host}")
        
        # Prepare workflow input
        logger.info(f"Preparing workflow input for trigger: {trigger_id}")
        workflow_input = {
            "trigger_data": request_data,
            "query_params": query_params,
            "headers": request_headers,
            "method": request.method,
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.info(f"Workflow input prepared: {workflow_input}")
        
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        logger.info(f"Generated execution ID: {execution_id}")
        
        # Create execution record
        logger.info(f"Creating execution record for workflow: {target_workflow['id']}")
        execution_data = {
            "workflow_id": target_workflow["id"],
            "execution_id": execution_id,
            "status": "running",
            "input_data": workflow_input,
            "message": f"External trigger: {trigger_id}",
            "execution_log": []
        }
        logger.info(f"Execution data created: {execution_data}")
        
        # Execute workflow
        logger.info(f"Creating AutomationExecutor for workflow: {target_workflow['id']}")
        executor = AutomationExecutor()
        timeout = trigger_config.get("timeout", 300)
        logger.info(f"About to execute workflow stream with timeout: {timeout}")
        
        try:
            # Execute with timeout
            result = None
            # Convert workflow_id to integer
            workflow_id_int = int(target_workflow["id"])
            logger.info(f"Starting workflow execution: id={workflow_id_int}, type={type(workflow_id_int)}")
            
            async for update in executor.execute_workflow_stream(
                workflow_id_int,
                execution_id,
                target_workflow["langflow_config"],  # Extract just the langflow_config
                workflow_input,   # input_data
                f"External trigger: {trigger_id}"  # message
            ):
                logger.info(f"Received workflow update: type={type(update)}, value={update}")
                
                # Handle case where update might be a string instead of dict
                if isinstance(update, dict):
                    update_type = update.get("type")
                    logger.info(f"Processing dict update with type: {update_type}")
                    
                    if update_type == "workflow_result" or update_type == "final":
                        result = update.get("response") or update.get("data", {})
                        logger.info(f"Found final result: type={type(result)}, value={result}")
                        break
                    elif update_type == "workflow_complete":
                        # Handle workflow completion without explicit result
                        result = update.get("result", "Workflow completed successfully")
                        logger.info(f"Found complete result: type={type(result)}, value={result}")
                        break
                else:
                    # If update is not a dict, treat it as the final result
                    logger.info(f"Received non-dict update: {type(update)}, value: {update}")
                    result = update
                    break
            
            if result is None:
                raise HTTPException(status_code=500, detail="Workflow execution failed")
            
            # Handle different result types - sometimes result is a string, sometimes a dict
            logger.info(f"Workflow result type: {type(result)}, value: {result}")
            if isinstance(result, str):
                output_data = {"result": result}
                logger.info(f"Handling string result: {output_data}")
            elif isinstance(result, dict):
                output_data = result.get("output", result)  # Try to get 'output' key, fallback to entire result
                logger.info(f"Handling dict result: {output_data}")
            else:
                output_data = {"result": str(result)}  # Convert to string for other types
                logger.info(f"Handling other result type: {output_data}")
            
            # Update execution data
            execution_data.update({
                "status": "completed",
                "output": output_data,
                "execution_time": time.time() - start_time,
                "completed_at": datetime.utcnow().isoformat()
            })
            
        except asyncio.TimeoutError:
            execution_data.update({
                "status": "timeout",
                "error_message": f"Execution timed out after {timeout} seconds"
            })
            raise HTTPException(status_code=408, detail="Workflow execution timed out")
        
        except Exception as e:
            execution_data.update({
                "status": "error",
                "error_message": str(e)
            })
            logger.error(f"Workflow execution error: {e}")
            raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")
        
        # Format response
        response_format = trigger_config.get("response_format", "workflow_output")
        custom_template = trigger_config.get("custom_response_template")
        
        # Ensure execution_data is always a dict before passing to format_response
        if not isinstance(execution_data, dict):
            execution_data = {
                "status": "completed",
                "output": {"result": str(execution_data)},
                "execution_time": time.time() - start_time,
                "completed_at": datetime.utcnow().isoformat(),
                "execution_id": execution_id
            }
        
        formatted_response = format_response(response_format, custom_template, execution_data)
        
        # Handle CORS
        response = JSONResponse(content=formatted_response)
        if trigger_config.get("cors_enabled", True):
            cors_origins = trigger_config.get("cors_origins", "*")
            response.headers["Access-Control-Allow-Origin"] = cors_origins
            response.headers["Access-Control-Allow-Methods"] = ", ".join(allowed_methods)
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"External trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.options("/trigger/{trigger_id}")
async def trigger_options(trigger_id: str, request: Request):
    """Handle CORS preflight requests"""
    try:
        # Find workflow with matching trigger
        workflows = get_automation_workflows()
        trigger_config = None
        
        for workflow_id, workflow_data in workflows.items():
            trigger_node_config = await find_trigger_node_in_workflow(workflow_data)
            if trigger_node_config and trigger_node_config.get("trigger_name") == trigger_id:
                trigger_config = trigger_node_config
                break
        
        if not trigger_config:
            raise HTTPException(status_code=404, detail="Trigger not found")
        
        response = JSONResponse(content={})
        
        if trigger_config.get("cors_enabled", True):
            allowed_methods = trigger_config.get("http_methods", ["POST"])
            cors_origins = trigger_config.get("cors_origins", "*")
            
            response.headers["Access-Control-Allow-Origin"] = cors_origins
            response.headers["Access-Control-Allow-Methods"] = ", ".join(allowed_methods + ["OPTIONS"])
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
            response.headers["Access-Control-Max-Age"] = "3600"
        
        return response
        
    except Exception as e:
        logger.error(f"CORS preflight error: {e}")
        return JSONResponse(content={}, status_code=200)

@router.get("/triggers")
async def list_triggers():
    """List all configured triggers"""
    try:
        workflows = get_automation_workflows()
        triggers = []
        
        for workflow_id, workflow_data in workflows.items():
            trigger_config = await find_trigger_node_in_workflow(workflow_data)
            if trigger_config:
                trigger_info = {
                    "trigger_id": trigger_config.get("trigger_name"),
                    "workflow_id": workflow_id,
                    "workflow_name": workflow_data.get("name", "Unnamed"),
                    "http_methods": trigger_config.get("http_methods", ["POST"]),
                    "authentication_type": trigger_config.get("authentication_type", "api_key"),
                    "rate_limit": trigger_config.get("rate_limit", 60),
                    "is_active": workflow_data.get("is_active", False),
                    "webhook_url": f"/api/v1/automation/external/trigger/{trigger_config.get('trigger_name')}"
                }
                triggers.append(trigger_info)
        
        return {"triggers": triggers, "total": len(triggers)}
        
    except Exception as e:
        logger.error(f"Error listing triggers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/triggers/{trigger_id}/test")
async def test_trigger(trigger_id: str, test_data: Dict[str, Any] = None):
    """Test a trigger endpoint with sample data"""
    try:
        # This would simulate a trigger call for testing
        # Implementation would call the actual trigger endpoint
        return {
            "message": f"Test endpoint for trigger {trigger_id}",
            "test_data": test_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error testing trigger: {e}")
        raise HTTPException(status_code=500, detail=str(e))