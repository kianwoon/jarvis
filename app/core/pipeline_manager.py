"""
Pipeline Manager for Agentic Pipeline feature.
Handles pipeline CRUD operations and orchestration.
"""
import json
import logging
import redis
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.db import SessionLocal
from app.core.config import get_settings
from app.core.redis_client import get_redis_client
from app.core.pipeline_config import get_pipeline_settings

logger = logging.getLogger(__name__)
settings = get_pipeline_settings()


class PipelineManager:
    """Manages agentic pipeline operations."""
    
    def __init__(self):
        self.redis_client = None
        self._pipeline_cache_key = "agentic_pipelines"
        self._pipeline_agents_cache_key = "pipeline_agents"
    
    def _get_redis(self):
        """Get Redis client (lazy initialization)"""
        if self.redis_client is None:
            self.redis_client = get_redis_client()
        return self.redis_client
    
    def get_db(self) -> Session:
        """Get database session."""
        return SessionLocal()
    
    async def create_pipeline(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new pipeline."""
        db = self.get_db()
        try:
            # Create pipeline
            query = text("""
                INSERT INTO agentic_pipelines 
                (name, description, goal, collaboration_mode, is_active, created_by, config)
                VALUES (:name, :description, :goal, :collaboration_mode, :is_active, :created_by, :config)
                RETURNING *
            """)
            
            result = db.execute(
                query,
                {
                    "name": pipeline_data["name"],
                    "description": pipeline_data.get("description", ""),
                    "goal": pipeline_data.get("goal", ""),
                    "collaboration_mode": pipeline_data["collaboration_mode"],
                    "is_active": pipeline_data.get("is_active", True),
                    "created_by": pipeline_data.get("created_by", "system"),
                    "config": json.dumps(pipeline_data.get("config", {}))
                }
            )
            
            row = result.fetchone()
            if row is None:
                raise ValueError("Failed to create pipeline")
            
            # Convert row to dict properly
            pipeline = dict(zip(result.keys(), row))
            # Handle config field - it might already be a dict from PostgreSQL
            if isinstance(pipeline["config"], str):
                pipeline["config"] = json.loads(pipeline["config"])
            elif pipeline["config"] is None:
                pipeline["config"] = {}
            
            # Add agents to pipeline
            if "agents" in pipeline_data:
                await self._add_agents_to_pipeline(
                    db, 
                    pipeline["id"], 
                    pipeline_data["agents"]
                )
            
            db.commit()
            
            # Clear cache
            redis_client = self._get_redis()
            if redis_client:
                redis_client.delete(self._pipeline_cache_key)
            
            return pipeline
            
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Pipeline creation failed: {e}")
            raise ValueError(f"Pipeline with name '{pipeline_data['name']}' already exists")
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()
    
    async def get_pipeline(self, pipeline_id: int) -> Optional[Dict[str, Any]]:
        """Get pipeline by ID."""
        db = self.get_db()
        try:
            # Get pipeline
            query = text("SELECT * FROM agentic_pipelines WHERE id = :id")
            result = db.execute(query, {"id": pipeline_id})
            pipeline = result.fetchone()
            
            if not pipeline:
                return None
            
            # Convert row to dict properly
            pipeline = dict(zip(result.keys(), pipeline))
            # Handle config field
            if isinstance(pipeline["config"], str):
                pipeline["config"] = json.loads(pipeline["config"])
            elif pipeline["config"] is None:
                pipeline["config"] = {}
            
            # Get agents
            agents_query = text("""
                SELECT * FROM pipeline_agents 
                WHERE pipeline_id = :pipeline_id
                ORDER BY execution_order
            """)
            agents_result = db.execute(agents_query, {"pipeline_id": pipeline_id})
            
            pipeline["agents"] = []
            for row in agents_result:
                agent_dict = dict(zip(agents_result.keys(), row))
                # Handle config field
                if isinstance(agent_dict["config"], str):
                    agent_dict["config"] = json.loads(agent_dict["config"])
                elif agent_dict["config"] is None:
                    agent_dict["config"] = {}
                pipeline["agents"].append(agent_dict)
            
            return pipeline
        finally:
            db.close()
    
    async def list_pipelines(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """List all pipelines."""
        # Try cache first
        cache_key = f"{self._pipeline_cache_key}:{'active' if active_only else 'all'}"
        redis_client = self._get_redis()
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        db = self.get_db()
        try:
            query = text("""
                SELECT p.*, 
                       COUNT(DISTINCT pa.id) as agent_count,
                       COUNT(DISTINCT pe.id) as execution_count,
                       ps.schedule_type,
                       ps.schedule_config,
                       ps.next_run,
                       ps.is_active as schedule_active
                FROM agentic_pipelines p
                LEFT JOIN pipeline_agents pa ON p.id = pa.pipeline_id
                LEFT JOIN pipeline_executions pe ON p.id = pe.pipeline_id
                LEFT JOIN pipeline_schedules ps ON p.id = ps.pipeline_id
                WHERE (:active_only = false OR p.is_active = true)
                GROUP BY p.id, p.name, p.description, p.goal, p.collaboration_mode, 
                         p.is_active, p.created_by, p.config, p.created_at, p.updated_at,
                         ps.schedule_type, ps.schedule_config, ps.next_run, ps.is_active
                ORDER BY p.created_at DESC
            """)
            
            result = db.execute(query, {"active_only": active_only})
            pipelines = []
            for row in result:
                # Convert row to dict properly
                pipeline_dict = dict(zip(result.keys(), row))
                # Handle config field - it might already be a dict from PostgreSQL
                if isinstance(pipeline_dict["config"], str):
                    pipeline_dict["config"] = json.loads(pipeline_dict["config"])
                elif pipeline_dict["config"] is None:
                    pipeline_dict["config"] = {}
                
                # Handle schedule info
                if pipeline_dict.get("schedule_type") and pipeline_dict.get("schedule_active"):
                    schedule_config = pipeline_dict.get("schedule_config")
                    if isinstance(schedule_config, str):
                        schedule_config = json.loads(schedule_config)
                    
                    pipeline_dict["schedule"] = {
                        "type": pipeline_dict["schedule_type"],
                        "config": schedule_config,
                        "next_run": pipeline_dict["next_run"],
                        "is_active": pipeline_dict["schedule_active"]
                    }
                
                # Remove individual schedule fields
                for field in ["schedule_type", "schedule_config", "next_run", "schedule_active"]:
                    pipeline_dict.pop(field, None)
                
                pipelines.append(pipeline_dict)
            
            # Cache with configurable TTL
            if redis_client:
                redis_client.setex(
                    cache_key,
                    settings.PIPELINE_LIST_CACHE_TTL,
                    json.dumps(pipelines, default=str)
                )
            
            return pipelines
        finally:
            db.close()
    
    async def update_pipeline(
        self, 
        pipeline_id: int, 
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update pipeline."""
        db = self.get_db()
        try:
            # Build update query
            update_fields = []
            params = {"id": pipeline_id}
            
            for field in ["name", "description", "goal", "collaboration_mode", "is_active"]:
                if field in update_data:
                    update_fields.append(f"{field} = :{field}")
                    params[field] = update_data[field]
            
            if "config" in update_data:
                update_fields.append("config = :config")
                params["config"] = json.dumps(update_data["config"])
            
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            
            query = text(f"""
                UPDATE agentic_pipelines 
                SET {', '.join(update_fields)}
                WHERE id = :id
                RETURNING *
            """)
            
            result = db.execute(query, params)
            row = result.fetchone()
            if row is None:
                raise ValueError("Pipeline not found")
            
            # Convert row to dict properly
            pipeline = dict(zip(result.keys(), row))
            # Handle config field - it might already be a dict from PostgreSQL
            if isinstance(pipeline["config"], str):
                pipeline["config"] = json.loads(pipeline["config"])
            elif pipeline["config"] is None:
                pipeline["config"] = {}
            
            # Update agents if provided
            if "agents" in update_data:
                # Get existing agents
                existing_agents_query = text("""
                    SELECT id, agent_name, execution_order, config 
                    FROM pipeline_agents 
                    WHERE pipeline_id = :pipeline_id
                """)
                existing_agents_result = db.execute(existing_agents_query, {"pipeline_id": pipeline_id})
                existing_agents = {row.agent_name: row for row in existing_agents_result}
                
                # Track which agents to keep (don't delete agents that are being updated)
                agents_to_keep = set()
                
                # Update or create agents
                for idx, agent_data in enumerate(update_data["agents"]):
                    agent_name = agent_data["agent_name"]
                    agents_to_keep.add(agent_name)
                    
                    if agent_name in existing_agents:
                        # Update existing agent
                        existing_agent = existing_agents[agent_name]
                        # Merge existing config with new config
                        existing_config = existing_agent.config or {}
                        new_config = agent_data.get("config", {})
                        merged_config = {**existing_config, **new_config}
                        
                        db.execute(
                            text("""
                                UPDATE pipeline_agents 
                                SET execution_order = :execution_order, config = :config
                                WHERE id = :agent_id AND pipeline_id = :pipeline_id
                            """),
                            {
                                "execution_order": idx,
                                "config": json.dumps(merged_config),
                                "agent_id": existing_agent.id,
                                "pipeline_id": pipeline_id
                            }
                        )
                    else:
                        # Create new agent
                        db.execute(
                            text("""
                                INSERT INTO pipeline_agents (pipeline_id, agent_name, execution_order, config)
                                VALUES (:pipeline_id, :agent_name, :execution_order, :config)
                            """),
                            {
                                "pipeline_id": pipeline_id,
                                "agent_name": agent_name,
                                "execution_order": idx,
                                "config": json.dumps(agent_data.get("config", {}))
                            }
                        )
                
                # Remove agents that are no longer in the pipeline
                for agent_name, agent_row in existing_agents.items():
                    if agent_name not in agents_to_keep:
                        db.execute(
                            text("DELETE FROM pipeline_agents WHERE id = :agent_id"),
                            {"agent_id": agent_row.id}
                        )
            
            db.commit()
            
            # Clear cache
            redis_client = self._get_redis()
            if redis_client:
                redis_client.delete(self._pipeline_cache_key)
            
            return await self.get_pipeline(pipeline_id)
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()
    
    async def delete_pipeline(self, pipeline_id: int) -> bool:
        """Delete pipeline and all related data."""
        db = self.get_db()
        try:
            # Delete in order due to foreign key constraints
            # 1. Delete executions
            db.execute(
                text("DELETE FROM pipeline_executions WHERE pipeline_id = :id"),
                {"id": pipeline_id}
            )
            
            # 2. Delete schedules
            db.execute(
                text("DELETE FROM pipeline_schedules WHERE pipeline_id = :id"),
                {"id": pipeline_id}
            )
            
            # 3. Delete agents
            db.execute(
                text("DELETE FROM pipeline_agents WHERE pipeline_id = :id"),
                {"id": pipeline_id}
            )
            
            # 4. Finally delete the pipeline
            result = db.execute(
                text("DELETE FROM agentic_pipelines WHERE id = :id"),
                {"id": pipeline_id}
            )
            
            db.commit()
            
            # Clear cache
            redis_client = self._get_redis()
            if redis_client:
                redis_client.delete(self._pipeline_cache_key)
            
            return result.rowcount > 0
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete pipeline {pipeline_id}: {str(e)}")
            raise
        finally:
            db.close()
    
    async def _add_agents_to_pipeline(
        self, 
        db: Session,
        pipeline_id: int, 
        agents: List[Dict[str, Any]]
    ):
        """Add agents to pipeline."""
        for idx, agent in enumerate(agents):
            query = text("""
                INSERT INTO pipeline_agents
                (pipeline_id, agent_name, execution_order, parent_agent, config)
                VALUES (:pipeline_id, :agent_name, :execution_order, :parent_agent, :config)
            """)
            
            db.execute(
                query,
                {
                    "pipeline_id": pipeline_id,
                    "agent_name": agent["agent_name"],
                    "execution_order": agent.get("execution_order", idx),
                    "parent_agent": agent.get("parent_agent"),
                    "config": json.dumps(agent.get("config", {}))
                }
            )
    
    async def clear_pipeline_cache(self, pipeline_id: int):
        """Clear pipeline cache when individual agents are updated."""
        redis_client = self._get_redis()
        if redis_client:
            # Clear the general pipeline list cache
            redis_client.delete(self._pipeline_cache_key)
            # Clear specific pipeline cache if it exists
            cache_keys = [
                f"{self._pipeline_cache_key}:all",
                f"{self._pipeline_cache_key}:active"
            ]
            for key in cache_keys:
                redis_client.delete(key)
            logger.info(f"Cleared pipeline cache for pipeline {pipeline_id}")
    
    async def record_execution(
        self,
        pipeline_id: int,
        trigger_type: str,
        status: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        execution_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Record pipeline execution."""
        db = self.get_db()
        try:
            query = text("""
                INSERT INTO pipeline_executions
                (pipeline_id, trigger_type, status, input_data, output_data, 
                 error_message, execution_metadata)
                VALUES (:pipeline_id, :trigger_type, :status, :input_data, 
                        :output_data, :error_message, :execution_metadata)
                RETURNING id
            """)
            
            result = db.execute(
                query,
                {
                    "pipeline_id": pipeline_id,
                    "trigger_type": trigger_type,
                    "status": status,
                    "input_data": json.dumps(input_data or {}),
                    "output_data": json.dumps(output_data or {}),
                    "error_message": error_message,
                    "execution_metadata": json.dumps(execution_metadata or {})
                }
            )
            
            execution_id = result.fetchone()[0]
            db.commit()
            return execution_id
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()
    
    async def update_execution_status(
        self,
        execution_id: int,
        status: str,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        execution_metadata: Optional[Dict[str, Any]] = None
    ):
        """Update execution status."""
        db = self.get_db()
        try:
            update_fields = ["status = :status"]
            params = {"id": execution_id, "status": status}
            
            if status in ["completed", "failed"]:
                update_fields.append("completed_at = CURRENT_TIMESTAMP")
            
            if output_data is not None:
                update_fields.append("output_data = :output_data")
                params["output_data"] = json.dumps(output_data)
            
            if error_message is not None:
                update_fields.append("error_message = :error_message")
                params["error_message"] = error_message
            
            if execution_metadata is not None:
                update_fields.append("execution_metadata = :execution_metadata")
                params["execution_metadata"] = json.dumps(execution_metadata)
            
            query = text(f"""
                UPDATE pipeline_executions
                SET {', '.join(update_fields)}
                WHERE id = :id
            """)
            
            db.execute(query, params)
            db.commit()
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()
    
    async def get_execution_history(
        self,
        pipeline_id: int,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """Get pipeline execution history."""
        db = self.get_db()
        try:
            query = text("""
                SELECT * FROM pipeline_executions
                WHERE pipeline_id = :pipeline_id
                ORDER BY started_at DESC
                LIMIT :limit
            """)
            
            # Use configured default if limit not provided
            if limit is None:
                limit = settings.PIPELINE_MAX_HISTORY
            
            result = db.execute(
                query,
                {"pipeline_id": pipeline_id, "limit": limit}
            )
            
            executions = []
            for row in result:
                exec_dict = dict(zip(result.keys(), row))
                # Handle JSON fields
                for field in ["input_data", "output_data", "execution_metadata"]:
                    if isinstance(exec_dict[field], str):
                        exec_dict[field] = json.loads(exec_dict[field])
                    elif exec_dict[field] is None:
                        exec_dict[field] = {}
                executions.append(exec_dict)
            return executions
        finally:
            db.close()
    
    async def create_or_update_schedule(
        self,
        pipeline_id: int,
        schedule_type: str,
        schedule_config: Dict[str, Any],
        is_active: bool = True
    ) -> Dict[str, Any]:
        """Create or update pipeline schedule."""
        db = self.get_db()
        try:
            # Calculate next run time based on schedule type
            next_run = self._calculate_next_run(schedule_type, schedule_config)
            
            # Check if schedule exists
            check_query = text("""
                SELECT id FROM pipeline_schedules 
                WHERE pipeline_id = :pipeline_id
            """)
            existing = db.execute(check_query, {"pipeline_id": pipeline_id}).fetchone()
            
            if existing:
                # Update existing schedule
                query = text("""
                    UPDATE pipeline_schedules
                    SET schedule_type = :schedule_type,
                        schedule_config = :schedule_config,
                        next_run = :next_run,
                        is_active = :is_active
                    WHERE pipeline_id = :pipeline_id
                    RETURNING *
                """)
            else:
                # Create new schedule
                query = text("""
                    INSERT INTO pipeline_schedules
                    (pipeline_id, schedule_type, schedule_config, next_run, is_active)
                    VALUES (:pipeline_id, :schedule_type, :schedule_config, :next_run, :is_active)
                    RETURNING *
                """)
            
            result = db.execute(
                query,
                {
                    "pipeline_id": pipeline_id,
                    "schedule_type": schedule_type,
                    "schedule_config": json.dumps(schedule_config),
                    "next_run": next_run,
                    "is_active": is_active
                }
            )
            
            row = result.fetchone()
            if row is None:
                raise ValueError("Failed to create/update schedule")
            
            schedule = dict(zip(result.keys(), row))
            # Handle JSON field
            if isinstance(schedule["schedule_config"], str):
                schedule["schedule_config"] = json.loads(schedule["schedule_config"])
            
            db.commit()
            return schedule
            
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()
    
    async def get_pipeline_schedule(self, pipeline_id: int) -> Optional[Dict[str, Any]]:
        """Get pipeline schedule."""
        db = self.get_db()
        try:
            query = text("""
                SELECT * FROM pipeline_schedules
                WHERE pipeline_id = :pipeline_id
            """)
            
            result = db.execute(query, {"pipeline_id": pipeline_id})
            row = result.fetchone()
            
            if not row:
                return None
            
            schedule = dict(zip(result.keys(), row))
            # Handle JSON field
            if isinstance(schedule["schedule_config"], str):
                schedule["schedule_config"] = json.loads(schedule["schedule_config"])
            
            return schedule
            
        finally:
            db.close()
    
    async def delete_pipeline_schedule(self, pipeline_id: int) -> bool:
        """Delete pipeline schedule."""
        db = self.get_db()
        try:
            query = text("""
                DELETE FROM pipeline_schedules
                WHERE pipeline_id = :pipeline_id
            """)
            
            result = db.execute(query, {"pipeline_id": pipeline_id})
            db.commit()
            
            return result.rowcount > 0
            
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()
    
    def _calculate_next_run(self, schedule_type: str, schedule_config: Dict[str, Any]) -> Optional[datetime]:
        """Calculate next run time based on schedule type."""
        from datetime import datetime, timedelta
        import croniter
        
        now = datetime.now()
        
        if schedule_type == "interval":
            minutes = schedule_config.get("minutes", 60)
            return now + timedelta(minutes=minutes)
            
        elif schedule_type == "cron":
            cron_expr = schedule_config.get("expression", "0 * * * *")
            try:
                cron = croniter.croniter(cron_expr, now)
                return cron.get_next(datetime)
            except Exception:
                # Default to 1 hour if cron expression is invalid
                return now + timedelta(hours=1)
                
        elif schedule_type == "one_time":
            scheduled_time = schedule_config.get("datetime")
            if scheduled_time:
                # Parse ISO format datetime
                return datetime.fromisoformat(scheduled_time.replace('Z', '+00:00').replace('T', ' '))
            return None
            
        return None