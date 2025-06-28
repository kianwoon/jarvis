"""
PostgreSQL Bridge for Langflow Integration
Provides database access for Langflow workflows using existing infrastructure
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.core.db import SessionLocal, AutomationWorkflow, AutomationExecution, AutomationTrigger

logger = logging.getLogger(__name__)

class PostgresBridge:
    """Bridge between Langflow and PostgreSQL database"""
    
    def __init__(self):
        pass
    
    def get_workflow(self, workflow_id: int) -> Optional[Dict[str, Any]]:
        """Get workflow by ID"""
        db = SessionLocal()
        try:
            workflow = db.query(AutomationWorkflow).filter(AutomationWorkflow.id == workflow_id).first()
            if not workflow:
                return None
            
            return {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "langflow_config": workflow.langflow_config,
                "trigger_config": workflow.trigger_config,
                "is_active": workflow.is_active,
                "created_by": workflow.created_by,
                "created_at": workflow.created_at.isoformat() if workflow.created_at else None,
                "updated_at": workflow.updated_at.isoformat() if workflow.updated_at else None
            }
        except Exception as e:
            logger.error(f"[POSTGRES BRIDGE] Error getting workflow {workflow_id}: {e}")
            return None
        finally:
            db.close()
    
    def create_workflow(self, workflow_data: Dict[str, Any]) -> Optional[int]:
        """Create new workflow"""
        db = SessionLocal()
        try:
            workflow = AutomationWorkflow(
                name=workflow_data["name"],
                description=workflow_data.get("description"),
                langflow_config=workflow_data["langflow_config"],
                trigger_config=workflow_data.get("trigger_config"),
                is_active=workflow_data.get("is_active", True),
                created_by=workflow_data.get("created_by", "system")
            )
            
            db.add(workflow)
            db.commit()
            db.refresh(workflow)
            
            logger.info(f"[POSTGRES BRIDGE] Created workflow: {workflow.id}")
            return workflow.id
        except Exception as e:
            logger.error(f"[POSTGRES BRIDGE] Error creating workflow: {e}")
            db.rollback()
            return None
        finally:
            db.close()
    
    def update_workflow(self, workflow_id: int, updates: Dict[str, Any]) -> bool:
        """Update workflow"""
        db = SessionLocal()
        try:
            workflow = db.query(AutomationWorkflow).filter(AutomationWorkflow.id == workflow_id).first()
            if not workflow:
                return False
            
            for key, value in updates.items():
                if hasattr(workflow, key):
                    setattr(workflow, key, value)
            
            workflow.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"[POSTGRES BRIDGE] Updated workflow: {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"[POSTGRES BRIDGE] Error updating workflow {workflow_id}: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def delete_workflow(self, workflow_id: int) -> bool:
        """Delete workflow"""
        db = SessionLocal()
        try:
            workflow = db.query(AutomationWorkflow).filter(AutomationWorkflow.id == workflow_id).first()
            if not workflow:
                return False
            
            db.delete(workflow)
            db.commit()
            
            logger.info(f"[POSTGRES BRIDGE] Deleted workflow: {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"[POSTGRES BRIDGE] Error deleting workflow {workflow_id}: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def create_execution(self, execution_data: Dict[str, Any]) -> Optional[int]:
        """Create new execution record"""
        db = SessionLocal()
        try:
            execution = AutomationExecution(
                workflow_id=execution_data["workflow_id"],
                execution_id=execution_data["execution_id"],
                status=execution_data.get("status", "running"),
                input_data=execution_data.get("input_data"),
                output_data=execution_data.get("output_data"),
                execution_log=execution_data.get("execution_log", [])
            )
            
            db.add(execution)
            db.commit()
            db.refresh(execution)
            
            logger.info(f"[POSTGRES BRIDGE] Created execution: {execution.id}")
            return execution.id
        except Exception as e:
            logger.error(f"[POSTGRES BRIDGE] Error creating execution: {e}")
            db.rollback()
            return None
        finally:
            db.close()
    
    def update_execution(self, execution_id: str, updates: Dict[str, Any]) -> bool:
        """Update execution record"""
        db = SessionLocal()
        try:
            execution = db.query(AutomationExecution).filter(
                AutomationExecution.execution_id == execution_id
            ).first()
            
            if not execution:
                return False
            
            for key, value in updates.items():
                if hasattr(execution, key):
                    setattr(execution, key, value)
            
            # Set completion time if status is completed or failed
            if updates.get("status") in ["completed", "failed", "cancelled"] and not execution.completed_at:
                execution.completed_at = datetime.utcnow()
            
            db.commit()
            
            logger.info(f"[POSTGRES BRIDGE] Updated execution: {execution_id}")
            return True
        except Exception as e:
            logger.error(f"[POSTGRES BRIDGE] Error updating execution {execution_id}: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution by execution_id"""
        db = SessionLocal()
        try:
            execution = db.query(AutomationExecution).filter(
                AutomationExecution.execution_id == execution_id
            ).first()
            
            if not execution:
                return None
            
            return {
                "id": execution.id,
                "workflow_id": execution.workflow_id,
                "execution_id": execution.execution_id,
                "status": execution.status,
                "input_data": execution.input_data,
                "output_data": execution.output_data,
                "execution_log": execution.execution_log,
                "error_message": execution.error_message,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None
            }
        except Exception as e:
            logger.error(f"[POSTGRES BRIDGE] Error getting execution {execution_id}: {e}")
            return None
        finally:
            db.close()
    
    def get_workflow_executions(self, workflow_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent executions for a workflow"""
        db = SessionLocal()
        try:
            executions = db.query(AutomationExecution).filter(
                AutomationExecution.workflow_id == workflow_id
            ).order_by(AutomationExecution.started_at.desc()).limit(limit).all()
            
            result = []
            for execution in executions:
                result.append({
                    "id": execution.id,
                    "execution_id": execution.execution_id,
                    "status": execution.status,
                    "started_at": execution.started_at.isoformat() if execution.started_at else None,
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "error_message": execution.error_message
                })
            
            return result
        except Exception as e:
            logger.error(f"[POSTGRES BRIDGE] Error getting executions for workflow {workflow_id}: {e}")
            return []
        finally:
            db.close()
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get all active workflows"""
        db = SessionLocal()
        try:
            workflows = db.query(AutomationWorkflow).filter(
                AutomationWorkflow.is_active == True
            ).all()
            
            result = []
            for workflow in workflows:
                result.append({
                    "id": workflow.id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "created_by": workflow.created_by,
                    "created_at": workflow.created_at.isoformat() if workflow.created_at else None
                })
            
            return result
        except Exception as e:
            logger.error(f"[POSTGRES BRIDGE] Error getting active workflows: {e}")
            return []
        finally:
            db.close()

# Global instance
postgres_bridge = PostgresBridge()