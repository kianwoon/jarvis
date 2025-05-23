import redis
import json
from app.core.db import SessionLocal, MCPTool

REDIS_HOST = 'redis'
REDIS_PORT = 6379
MCP_TOOLS_KEY = 'mcp_tools_cache'

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def get_enabled_mcp_tools():
    cached = r.get(MCP_TOOLS_KEY)
    if cached:
        return json.loads(cached)
    return reload_enabled_mcp_tools()

def reload_enabled_mcp_tools():
    db = SessionLocal()
    try:
        tools = db.query(MCPTool).filter(MCPTool.is_active == True).all()
        enabled_tools = {
            tool.name: {
                "name": tool.name,
                "description": tool.description,
                "endpoint": tool.endpoint,
                "method": tool.method,
                "parameters": tool.parameters,
                "headers": tool.headers,
                "manifest_id": tool.manifest_id
            }
            for tool in tools
        }
        r.set(MCP_TOOLS_KEY, json.dumps(enabled_tools))
        return enabled_tools
    finally:
        db.close() 