"""
Debug the dynamic router issue
"""

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem

async def test_router():
    """Test just the router"""
    
    system = DynamicMultiAgentSystem()
    
    query = "help me to evaluate migrate mariaDB to oceanbase"
    
    print(f"Testing router with query: {query}\n")
    
    try:
        result = await system.route_query(query)
        print(f"Routing result: {result}")
    except Exception as e:
        print(f"Router failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_router())