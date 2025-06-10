"""
WebSocket endpoints for real-time pipeline execution monitoring
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
import json
import asyncio
from datetime import datetime
import logging

from app.core.db import SessionLocal
from app.core.redis_client import get_redis_client
from app.core.pipeline_multi_agent_bridge import execute_pipeline_with_agents

logger = logging.getLogger(__name__)

router = APIRouter()

# Store active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.execution_status: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, execution_id: str):
        await websocket.accept()
        if execution_id not in self.active_connections:
            self.active_connections[execution_id] = []
        self.active_connections[execution_id].append(websocket)
        
        # Send current status if available
        if execution_id in self.execution_status:
            await websocket.send_json({
                "type": "status",
                "data": self.execution_status[execution_id]
            })

    def disconnect(self, websocket: WebSocket, execution_id: str):
        if execution_id in self.active_connections:
            self.active_connections[execution_id].remove(websocket)
            if not self.active_connections[execution_id]:
                del self.active_connections[execution_id]

    async def send_to_execution(self, execution_id: str, message: dict):
        if execution_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[execution_id]:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn, execution_id)

    async def broadcast_status(self, execution_id: str, status: dict):
        self.execution_status[execution_id] = status
        await self.send_to_execution(execution_id, {
            "type": "status",
            "data": status
        })

    async def broadcast_log(self, execution_id: str, log_entry: dict):
        await self.send_to_execution(execution_id, {
            "type": "log",
            "data": log_entry
        })

    async def broadcast_metrics(self, execution_id: str, metrics: dict):
        await self.send_to_execution(execution_id, {
            "type": "metrics",
            "data": metrics
        })

    async def broadcast_agent_update(self, execution_id: str, agent_name: str, update: dict):
        await self.send_to_execution(execution_id, {
            "type": "agent_update",
            "data": {
                "agent": agent_name,
                "update": update
            }
        })
    
    async def broadcast_agent_io_update(self, execution_id: str, agent_name: str, io_data: dict):
        """Broadcast detailed agent I/O update"""
        await self.send_to_execution(execution_id, {
            "type": "agent_io_update",
            "data": {
                "agent": agent_name,
                "io_data": io_data
            }
        })

    def cleanup_execution(self, execution_id: str):
        if execution_id in self.execution_status:
            del self.execution_status[execution_id]

manager = ConnectionManager()

@router.websocket("/ws/execution/{execution_id}")
async def websocket_endpoint(websocket: WebSocket, execution_id: str):
    await manager.connect(websocket, execution_id)
    try:
        # Start monitoring the execution
        await monitor_execution(execution_id, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket, execution_id)
    except Exception as e:
        logger.error(f"WebSocket error for execution {execution_id}: {e}")
        manager.disconnect(websocket, execution_id)

async def monitor_execution(execution_id: str, websocket: WebSocket):
    """Monitor execution progress from Redis and send updates"""
    redis_client = get_redis_client()
    if not redis_client:
        await websocket.send_json({
            "type": "error",
            "data": {"message": "Redis not available for monitoring"}
        })
        return

    # Subscribe to execution channel
    pubsub = redis_client.pubsub()
    channel = f"pipeline_execution:{execution_id}"
    pubsub.subscribe(channel)

    try:
        while True:
            # Check for messages
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    
                    # Route different types of updates
                    if data['type'] == 'status':
                        await manager.broadcast_status(execution_id, data['payload'])
                    elif data['type'] == 'log':
                        await manager.broadcast_log(execution_id, data['payload'])
                    elif data['type'] == 'metrics':
                        await manager.broadcast_metrics(execution_id, data['payload'])
                    elif data['type'] == 'agent_update':
                        await manager.broadcast_agent_update(
                            execution_id, 
                            data['payload']['agent'],
                            data['payload']['update']
                        )
                    elif data['type'] == 'agent_io_update':
                        await manager.broadcast_agent_io_update(
                            execution_id,
                            data['payload']['agent'],
                            data['payload']['update']
                        )
                    elif data['type'] == 'complete':
                        await websocket.send_json({
                            "type": "complete",
                            "data": data['payload']
                        })
                        # Don't break immediately - let client close connection
                        # break
                    elif data['type'] == 'error':
                        await websocket.send_json({
                            "type": "error",
                            "data": data['payload']
                        })
                        break
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in message: {message['data']}")
            
            # Also check WebSocket is still alive
            try:
                await websocket.send_json({"type": "ping"})
            except:
                break
                
            await asyncio.sleep(0.1)
    finally:
        pubsub.unsubscribe(channel)
        pubsub.close()

# Helper functions to publish updates from the execution side
async def publish_execution_update(execution_id: str, update_type: str, payload: dict):
    """Publish execution updates to Redis for WebSocket distribution"""
    redis_client = get_redis_client()
    if redis_client:
        channel = f"pipeline_execution:{execution_id}"
        message = json.dumps({
            "type": update_type,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat()
        })
        redis_client.publish(channel, message)

async def publish_status_update(execution_id: str, status: str, progress: float, current_agent: Optional[str] = None):
    await publish_execution_update(execution_id, "status", {
        "status": status,
        "progress": progress,
        "current_agent": current_agent,
        "timestamp": datetime.utcnow().isoformat()
    })

async def publish_agent_start(execution_id: str, agent_name: str):
    await publish_execution_update(execution_id, "agent_update", {
        "agent": agent_name,
        "update": {
            "status": "running",
            "started_at": datetime.utcnow().isoformat()
        }
    })

async def publish_agent_complete(execution_id: str, agent_name: str, output: Any, execution_time: float):
    await publish_execution_update(execution_id, "agent_update", {
        "agent": agent_name,
        "update": {
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "execution_time": execution_time,
            "output_preview": str(output)[:200] if output else None
        }
    })

async def publish_agent_error(execution_id: str, agent_name: str, error: str):
    await publish_execution_update(execution_id, "agent_update", {
        "agent": agent_name,
        "update": {
            "status": "error",
            "error": error,
            "failed_at": datetime.utcnow().isoformat()
        }
    })

async def publish_log_entry(execution_id: str, level: str, message: str, agent: Optional[str] = None):
    await publish_execution_update(execution_id, "log", {
        "level": level,
        "message": message,
        "agent": agent,
        "timestamp": datetime.utcnow().isoformat()
    })

async def publish_metrics_update(execution_id: str, metrics: dict):
    await publish_execution_update(execution_id, "metrics", metrics)

async def publish_execution_complete(execution_id: str, result: Any, total_time: float):
    await publish_execution_update(execution_id, "complete", {
        "result": result,
        "total_time": total_time,
        "completed_at": datetime.utcnow().isoformat()
    })

async def publish_execution_error(execution_id: str, error: str):
    await publish_execution_update(execution_id, "error", {
        "error": error,
        "failed_at": datetime.utcnow().isoformat()
    })


@router.websocket("/ws/execute/{pipeline_id}")
async def execute_pipeline_websocket(
    websocket: WebSocket,
    pipeline_id: str,
    db: Session = Depends(SessionLocal)
):
    """Execute a pipeline with real-time updates via WebSocket"""
    
    import uuid
    execution_id = str(uuid.uuid4())
    
    await manager.connect(websocket, execution_id)
    
    try:
        # Receive initial query from client
        data = await websocket.receive_json()
        query = data.get("query", "")
        context = data.get("context", {})
        
        # Send execution started event
        await websocket.send_json({
            "type": "execution_started",
            "data": {
                "execution_id": execution_id,
                "pipeline_id": pipeline_id,
                "query": query
            }
        })
        
        # Execute pipeline with enhanced multi-agent system
        start_time = datetime.now()
        
        async for event in execute_pipeline_with_agents(
            pipeline_id=pipeline_id,
            execution_id=execution_id,
            query=query,
            context=context
        ):
            # Forward events to WebSocket
            await websocket.send_json(event)
            
            # Also publish to Redis for other subscribers
            if event["type"] == "agent_token":
                # Don't publish every token to Redis to avoid overload
                pass
            elif event["type"] == "agent_complete":
                agent_data = event["data"]
                await publish_agent_complete(
                    execution_id,
                    agent_data["agent"],
                    agent_data.get("response", ""),
                    agent_data.get("duration", 0)
                )
            elif event["type"] == "pipeline_complete":
                total_time = (datetime.now() - start_time).total_seconds()
                await publish_execution_complete(
                    execution_id,
                    event["data"].get("summary", {}),
                    total_time
                )
            elif event["type"] == "error":
                await publish_execution_error(
                    execution_id,
                    event["data"].get("error", "Unknown error")
                )
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for execution {execution_id}")
    except Exception as e:
        logger.error(f"Pipeline execution error: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "data": {
                "execution_id": execution_id,
                "error": str(e)
            }
        })
    finally:
        manager.disconnect(websocket, execution_id)