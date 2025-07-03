"""
API endpoint to expose node schema with input/output handle metadata
This allows the backend to understand node handle definitions from the frontend
Supports both legacy custom nodes and new agent-based workflows
"""
from fastapi import APIRouter
from typing import Dict, List, Any
from app.automation.flows.agent_nodes import get_all_node_schemas, get_node_categories

router = APIRouter()

# AI-focused node schema for unstructured data intelligence
NODE_SCHEMA = {
    'start': {
        'label': 'Start',
        'description': 'Workflow starting point',
        'category': 'Workflow Control',
        'color': '#4caf50',
        'inputs': [],
        'outputs': [
            {'id': 'output', 'label': 'Output', 'type': 'any', 'description': 'Initial workflow data'}
        ],
        'backendType': 'JarvisStartNode'
    },
    'end': {
        'label': 'End',
        'description': 'Workflow endpoint',
        'category': 'Workflow Control',
        'color': '#f44336',
        'inputs': [
            {'id': 'input', 'label': 'Input', 'type': 'any', 'description': 'Final workflow result'}
        ],
        'outputs': [],
        'backendType': 'JarvisEndNode'
    },
    'condition': {
        'label': 'Condition',
        'description': 'Basic conditional logic branching',
        'category': 'Workflow Control',
        'color': '#795548',
        'inputs': [
            {'id': 'input', 'label': 'Input', 'type': 'any', 'description': 'Data to evaluate'},
            {'id': 'condition', 'label': 'Condition', 'type': 'any', 'description': 'Condition value'}
        ],
        'outputs': [
            {'id': 'true', 'label': 'True', 'type': 'any', 'description': 'When condition is true'},
            {'id': 'false', 'label': 'False', 'type': 'any', 'description': 'When condition is false'}
        ],
        'backendType': 'JarvisConditionNode'
    },
    'loop': {
        'label': 'Loop',
        'description': 'Iterate over document batches or data collections',
        'category': 'Workflow Control',
        'color': '#673ab7',
        'inputs': [
            {'id': 'array', 'label': 'Array', 'type': 'array', 'description': 'Array to iterate over'},
            {'id': 'input', 'label': 'Input', 'type': 'any', 'description': 'Data for each iteration'}
        ],
        'outputs': [
            {'id': 'item', 'label': 'Current Item', 'type': 'any', 'description': 'Current array item'},
            {'id': 'index', 'label': 'Index', 'type': 'number', 'description': 'Current index'},
            {'id': 'output', 'label': 'Final Result', 'type': 'array', 'description': 'Loop results'}
        ],
        'backendType': 'JarvisLoopNode'
    },
    'llm': {
        'label': 'LLM',
        'description': 'Direct LLM access for custom prompts and experimentation',
        'category': 'AI Processing',
        'color': '#2196f3',
        'inputs': [
            {'id': 'text', 'label': 'Prompt', 'type': 'string', 'description': 'Text prompt for LLM'},
            {'id': 'context', 'label': 'Context', 'type': 'any', 'description': 'Additional context data'}
        ],
        'outputs': [
            {'id': 'output', 'label': 'Response', 'type': 'string', 'description': 'LLM generated response'},
            {'id': 'metadata', 'label': 'Metadata', 'type': 'object', 'description': 'Response metadata'}
        ],
        'backendType': 'JarvisLLMNode'
    },
    'agent': {
        'label': 'AI Agent',
        'description': 'Structured AI workflows with predefined agents and MCP tools',
        'category': 'AI Processing',
        'color': '#9c27b0',
        'inputs': [
            {'id': 'query', 'label': 'Query', 'type': 'string', 'description': 'Query for the agent'},
            {'id': 'context', 'label': 'Context', 'type': 'any', 'description': 'Context data'},
            {'id': 'documents', 'label': 'Documents', 'type': 'array', 'description': 'Documents to process'}
        ],
        'outputs': [
            {'id': 'output', 'label': 'Result', 'type': 'any', 'description': 'Agent execution result'},
            {'id': 'reasoning', 'label': 'Reasoning', 'type': 'string', 'description': 'Agent reasoning steps'},
            {'id': 'analysis', 'label': 'Analysis', 'type': 'object', 'description': 'Detailed analysis results'}
        ],
        'backendType': 'JarvisAgentNode'
    },
    'aiDecision': {
        'label': 'AI Decision',
        'description': 'Content-aware intelligent branching based on AI analysis',
        'category': 'AI Processing',
        'color': '#ff9800',
        'inputs': [
            {'id': 'input', 'label': 'Input', 'type': 'any', 'description': 'Data to analyze for decision'},
            {'id': 'criteria', 'label': 'Criteria', 'type': 'string', 'description': 'Decision criteria prompt'}
        ],
        'outputs': [
            {'id': 'decision', 'label': 'Decision', 'type': 'string', 'description': 'AI decision result'},
            {'id': 'confidence', 'label': 'Confidence', 'type': 'number', 'description': 'Decision confidence score'},
            {'id': 'reasoning', 'label': 'Reasoning', 'type': 'string', 'description': 'Decision reasoning'}
        ],
        'backendType': 'JarvisAIDecisionNode'
    },
    'documentProcessor': {
        'label': 'Document Processor',
        'description': 'Process and analyze PDF, Word, and text documents',
        'category': 'Document Intelligence',
        'color': '#1976d2',
        'inputs': [
            {'id': 'document', 'label': 'Document', 'type': 'any', 'description': 'Document file or content'},
            {'id': 'task', 'label': 'Analysis Task', 'type': 'string', 'description': 'What to extract or analyze'}
        ],
        'outputs': [
            {'id': 'content', 'label': 'Extracted Content', 'type': 'string', 'description': 'Extracted text content'},
            {'id': 'analysis', 'label': 'Analysis', 'type': 'object', 'description': 'Document analysis results'},
            {'id': 'metadata', 'label': 'Metadata', 'type': 'object', 'description': 'Document metadata'}
        ],
        'backendType': 'JarvisDocumentProcessorNode'
    },
    'imageProcessor': {
        'label': 'Image Processor',
        'description': 'OCR, image analysis, and visual understanding',
        'category': 'Document Intelligence',
        'color': '#e91e63',
        'inputs': [
            {'id': 'image', 'label': 'Image', 'type': 'any', 'description': 'Image file or data'},
            {'id': 'task', 'label': 'Analysis Task', 'type': 'string', 'description': 'OCR, description, or specific analysis'}
        ],
        'outputs': [
            {'id': 'text', 'label': 'Extracted Text', 'type': 'string', 'description': 'OCR text extraction'},
            {'id': 'description', 'label': 'Description', 'type': 'string', 'description': 'Image description'},
            {'id': 'analysis', 'label': 'Analysis', 'type': 'object', 'description': 'Detailed image analysis'}
        ],
        'backendType': 'JarvisImageProcessorNode'
    },
    'audioProcessor': {
        'label': 'Audio Processor',
        'description': 'Audio transcription and analysis',
        'category': 'Document Intelligence',
        'color': '#ff5722',
        'inputs': [
            {'id': 'audio', 'label': 'Audio', 'type': 'any', 'description': 'Audio file or data'},
            {'id': 'task', 'label': 'Analysis Task', 'type': 'string', 'description': 'Transcription or audio analysis'}
        ],
        'outputs': [
            {'id': 'transcript', 'label': 'Transcript', 'type': 'string', 'description': 'Audio transcription'},
            {'id': 'analysis', 'label': 'Analysis', 'type': 'object', 'description': 'Audio content analysis'},
            {'id': 'metadata', 'label': 'Metadata', 'type': 'object', 'description': 'Audio metadata'}
        ],
        'backendType': 'JarvisAudioProcessorNode'
    },
    'multiModalFusion': {
        'label': 'Multi-modal Fusion',
        'description': 'Combine and analyze text, images, and audio together',
        'category': 'Document Intelligence',
        'color': '#9c27b0',
        'inputs': [
            {'id': 'text', 'label': 'Text Data', 'type': 'string', 'description': 'Text content'},
            {'id': 'images', 'label': 'Images', 'type': 'array', 'description': 'Image files'},
            {'id': 'audio', 'label': 'Audio', 'type': 'any', 'description': 'Audio files'},
            {'id': 'task', 'label': 'Fusion Task', 'type': 'string', 'description': 'How to combine and analyze'}
        ],
        'outputs': [
            {'id': 'analysis', 'label': 'Unified Analysis', 'type': 'object', 'description': 'Combined multi-modal analysis'},
            {'id': 'summary', 'label': 'Summary', 'type': 'string', 'description': 'Unified content summary'},
            {'id': 'insights', 'label': 'Insights', 'type': 'array', 'description': 'Cross-modal insights'}
        ],
        'backendType': 'JarvisMultiModalFusionNode'
    },
    'contextMemory': {
        'label': 'Context Memory',
        'description': 'Maintain workflow context for multi-turn AI processes',
        'category': 'AI Memory & Storage',
        'color': '#607d8b',
        'inputs': [
            {'id': 'data', 'label': 'Data', 'type': 'any', 'description': 'Data to store in context'},
            {'id': 'key', 'label': 'Context Key', 'type': 'string', 'description': 'Memory key identifier'}
        ],
        'outputs': [
            {'id': 'context', 'label': 'Context', 'type': 'object', 'description': 'Current context state'},
            {'id': 'history', 'label': 'History', 'type': 'array', 'description': 'Context history'}
        ],
        'backendType': 'JarvisContextMemoryNode'
    },
    'variable': {
        'label': 'Variable Store',
        'description': 'Store workflow variables and intermediate results',
        'category': 'AI Memory & Storage',
        'color': '#8bc34a',
        'inputs': [
            {'id': 'value', 'label': 'Value', 'type': 'any', 'description': 'Value to store'}
        ],
        'outputs': [
            {'id': 'value', 'label': 'Value', 'type': 'any', 'description': 'Variable value'},
            {'id': 'name', 'label': 'Name', 'type': 'string', 'description': 'Variable name'}
        ],
        'backendType': 'JarvisVariableNode'
    },
    'dataMapper': {
        'label': 'Data Mapper',
        'description': 'Transform data structures for AI workflow compatibility',
        'category': 'AI Memory & Storage',
        'color': '#e91e63',
        'inputs': [
            {'id': 'data', 'label': 'Input Data', 'type': 'any', 'description': 'Data to transform'},
            {'id': 'mapping', 'label': 'Mapping', 'type': 'object', 'description': 'Transformation mapping'}
        ],
        'outputs': [
            {'id': 'data', 'label': 'Output Data', 'type': 'any', 'description': 'Transformed data'},
            {'id': 'metadata', 'label': 'Metadata', 'type': 'object', 'description': 'Transformation metadata'}
        ],
        'backendType': 'JarvisDataMapperNode'
    }
}

@router.get("/node-schema")
async def get_node_schema() -> Dict[str, Any]:
    """Get complete node schema with input/output handle metadata"""
    return {
        "schema": NODE_SCHEMA,
        "version": "1.0",
        "total_nodes": len(NODE_SCHEMA)
    }

@router.get("/node-schema/{node_type}")
async def get_node_type_schema(node_type: str) -> Dict[str, Any]:
    """Get schema for a specific node type"""
    if node_type not in NODE_SCHEMA:
        return {"error": f"Node type '{node_type}' not found"}
    
    return NODE_SCHEMA[node_type]

@router.get("/node-schema/{node_type}/inputs")
async def get_node_inputs(node_type: str) -> List[Dict[str, Any]]:
    """Get input handles for a specific node type"""
    if node_type not in NODE_SCHEMA:
        return []
    
    return NODE_SCHEMA[node_type].get('inputs', [])

@router.get("/node-schema/{node_type}/outputs")
async def get_node_outputs(node_type: str) -> List[Dict[str, Any]]:
    """Get output handles for a specific node type"""
    if node_type not in NODE_SCHEMA:
        return []
    
    return NODE_SCHEMA[node_type].get('outputs', [])

@router.get("/handle-mappings")
async def get_handle_mappings() -> Dict[str, str]:
    """Get common handle to parameter mappings for input resolution"""
    return {
        'input': 'input',  # Generic input
        'text': 'prompt',  # For LLM nodes
        'content': 'prompt',  # Alternative for LLM nodes
        'query': 'query',  # For agent and tool nodes
        'data': 'data',  # For data processing nodes
        'value': 'value',  # For variable nodes
        'array': 'array_input',  # For loop nodes
        'condition': 'condition',  # For condition nodes
        'url': 'url',  # For HTTP nodes
        'key': 'key',  # For Redis nodes
        'sql': 'sql_query',  # For database nodes
        'template': 'template',  # For email nodes
        'path': 'path',  # For file nodes
    }

@router.get("/agent-nodes")
async def get_agent_node_schemas() -> Dict[str, Any]:
    """Get agent-based node schemas for visual editor"""
    try:
        agent_schemas = get_all_node_schemas()
        categories = get_node_categories()
        
        # Convert to format expected by visual editor
        schema_dict = {}
        for schema in agent_schemas:
            node_type = schema["type"]
            schema_dict[node_type.lower()] = {
                "label": schema["name"],
                "description": schema["description"],
                "category": schema["category"],
                "color": schema["color"],
                "inputs": schema["inputs"],
                "outputs": schema["outputs"],
                "backendType": node_type,
                "properties": schema["properties"],
                "icon": schema.get("icon", "🔧")
            }
        
        return {
            "schema": schema_dict,
            "categories": categories,
            "version": "2.0",
            "type": "agent-based",
            "total_nodes": len(schema_dict)
        }
    except Exception as e:
        return {
            "error": f"Failed to load agent node schemas: {str(e)}",
            "schema": {},
            "categories": [],
            "version": "2.0",
            "type": "agent-based"
        }

@router.get("/node-types")
async def get_available_node_types() -> Dict[str, Any]:
    """Get available node types (legacy and agent-based)"""
    try:
        legacy_types = list(NODE_SCHEMA.keys())
        agent_schemas = get_all_node_schemas()
        agent_types = [schema["type"].lower() for schema in agent_schemas]
        
        return {
            "legacy_nodes": legacy_types,
            "agent_nodes": agent_types,
            "total_legacy": len(legacy_types),
            "total_agent": len(agent_types),
            "recommended": "agent_nodes"  # Recommend agent-based approach
        }
    except Exception as e:
        return {
            "error": f"Failed to get node types: {str(e)}",
            "legacy_nodes": [],
            "agent_nodes": [],
            "recommended": "legacy_nodes"
        }

@router.post("/validate-workflow")
async def validate_workflow(workflow_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate workflow configuration before execution"""
    try:
        from app.automation.flows.agent_nodes import validate_agent_workflow
        
        # Detect workflow type
        nodes = workflow_config.get("nodes", [])
        agent_node_types = {"AgentNode", "InputNode", "OutputNode", "ConditionNode", "ParallelNode"}
        
        has_agent_nodes = any(
            node.get("data", {}).get("type", "") in agent_node_types 
            for node in nodes
        )
        
        if has_agent_nodes:
            # Validate agent-based workflow
            validation_result = validate_agent_workflow(workflow_config)
            validation_result["workflow_type"] = "agent_based"
            
            # Perform pre-execution cache existence check if cache nodes exist
            cache_info = validation_result.get("cache_info")
            if cache_info and cache_info.get("requires_cache_check"):
                try:
                    from app.automation.integrations.redis_bridge import workflow_redis
                    
                    cache_status = []
                    for cache_node_info in cache_info["cache_nodes"]:
                        node_id = cache_node_info["node_id"]
                        
                        # Generate cache key for pre-execution check
                        cache_key = _generate_pre_execution_cache_key(
                            cache_node_info, workflow_config.get("id")
                        )
                        
                        # Check if cache exists
                        cache_exists = workflow_redis.exists(cache_key) if cache_key else False
                        
                        cache_status.append({
                            "node_id": node_id,
                            "cache_key": cache_key,
                            "cache_exists": cache_exists,
                            "cache_key_pattern": cache_node_info["cache_key_pattern"],
                            "estimated_hit": cache_exists
                        })
                    
                    validation_result["cache_status"] = {
                        "total_cache_nodes": len(cache_status),
                        "cache_nodes_with_data": len([c for c in cache_status if c["cache_exists"]]),
                        "estimated_cache_hits": len([c for c in cache_status if c["estimated_hit"]]),
                        "cache_details": cache_status
                    }
                    
                except Exception as e:
                    validation_result["cache_status"] = {
                        "error": f"Failed to check cache status: {str(e)}",
                        "cache_check_failed": True
                    }
        else:
            # Basic validation for legacy workflows
            validation_result = {
                "valid": len(nodes) > 0,
                "errors": [] if len(nodes) > 0 else ["Workflow must have at least one node"],
                "warnings": [],
                "workflow_type": "legacy"
            }
        
        return validation_result
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Validation failed: {str(e)}"],
            "warnings": [],
            "workflow_type": "unknown"
        }

def _generate_pre_execution_cache_key(cache_node_info: dict, workflow_id: int = None) -> str:
    """Generate cache key for pre-execution validation"""
    try:
        cache_key_pattern = cache_node_info.get("cache_key_pattern", "auto")
        custom_key = cache_node_info.get("custom_key", "")
        node_id = cache_node_info["node_id"]
        
        if cache_key_pattern == "custom" and custom_key:
            return custom_key
        elif cache_key_pattern == "node_only":
            return f"cache_{node_id}"
        elif cache_key_pattern == "input_hash":
            # For pre-execution, we can't know the exact input hash
            # Return a pattern that would match any input hash for this node
            return f"cache_{node_id}_*"  # This is a pattern, not exact key
        else:  # auto pattern
            if workflow_id:
                return f"cache_{workflow_id}_{node_id}"
            else:
                return f"cache_{node_id}"
                
    except Exception:
        return f"cache_{cache_node_info.get('node_id', 'unknown')}"