#!/usr/bin/env python3
"""
Emergency monitoring script to track relationship counts
"""
import sys
import os
import time
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.services.neo4j_service import get_neo4j_service

def monitor_counts():
    """Monitor relationship counts in real-time"""
    neo4j_service = get_neo4j_service()
    
    while True:
        try:
            with neo4j_service.driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    OPTIONAL MATCH (n)-[r]-()
                    RETURN count(DISTINCT n) as nodes, count(r) as relationships
                """)
                record = result.single()
                nodes = record['nodes']
                relationships = record['relationships']
                
                ratio = relationships / nodes if nodes > 0 else 0
                status = "🚨 CRITICAL" if relationships > 100 else "✅ OK"
                
                print(f"{status} | Nodes: {nodes}, Relationships: {relationships}, Ratio: {ratio:.1f}:1")
                
                if relationships > 100:
                    print("🚨 RELATIONSHIP EXPLOSION DETECTED!")
                
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        
        time.sleep(5)

if __name__ == "__main__":
    monitor_counts()