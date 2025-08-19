"""
Meta-Task Workflow Orchestrator
Manages workflow creation, execution, and state tracking
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
from sqlalchemy.orm import Session
from app.core.db import (
    MetaTaskWorkflow, MetaTaskNode, MetaTaskEdge, MetaTaskExecution,
    MetaTaskTemplate, get_db_session
)
from app.agents.task_decomposer import TaskDecomposer, TaskChunk
from app.agents.continuity_manager import ContinuityManager

logger = logging.getLogger(__name__)

class MetaTaskWorkflowOrchestrator:
    """Orchestrates meta-task workflow execution"""
    
    def __init__(self):
        self.logger = logger
        self.task_decomposer = TaskDecomposer()
        self.continuity_manager = ContinuityManager()
    
    async def create_workflow(
        self, 
        template_id: str, 
        name: str, 
        input_data: Dict[str, Any],
        description: str = None
    ) -> Optional[Dict[str, Any]]:
        """Create a new workflow from a template"""
        try:
            with get_db_session() as db:
                # Get template
                template = db.query(MetaTaskTemplate).filter(
                    MetaTaskTemplate.id == template_id
                ).first()
                
                if not template:
                    self.logger.error(f"Template {template_id} not found")
                    return None
                
                # Create workflow
                workflow = MetaTaskWorkflow(
                    template_id=template_id,
                    name=name,
                    description=description,
                    workflow_config=template.template_config,
                    input_data=input_data,
                    progress={'current_phase': 0, 'total_phases': len(template.template_config.get('phases', []))}
                )
                
                db.add(workflow)
                db.commit()
                db.refresh(workflow)
                
                # Create nodes from template phases
                await self._create_nodes_from_template(db, workflow, template)
                
                self.logger.info(f"Created workflow: {workflow.name}")
                return self._workflow_to_dict(workflow)
        except Exception as e:
            self.logger.error(f"Error creating workflow: {e}")
            return None
    
    async def execute_workflow(
        self, 
        workflow_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a workflow and yield progress events"""
        try:
            with get_db_session() as db:
                workflow = db.query(MetaTaskWorkflow).filter(
                    MetaTaskWorkflow.id == workflow_id
                ).first()
                
                if not workflow:
                    yield {"type": "error", "message": f"Workflow {workflow_id} not found"}
                    return
                
                # Update workflow status
                workflow.status = "running"
                workflow.started_at = datetime.utcnow()
                db.commit()
                
                yield {
                    "type": "workflow_started",
                    "workflow_id": workflow_id,
                    "workflow_name": workflow.name,
                    "total_phases": len(workflow.workflow_config.get('phases', []))
                }
                
                # Get nodes in execution order
                nodes = db.query(MetaTaskNode).filter(
                    MetaTaskNode.workflow_id == workflow_id
                ).order_by(MetaTaskNode.execution_order).all()
                
                # Execute nodes sequentially
                workflow_output = {}
                for i, node in enumerate(nodes):
                    try:
                        yield {
                            "type": "phase_started",
                            "phase_number": i + 1,
                            "phase_name": node.name,
                            "phase_type": node.node_type
                        }
                        
                        # Execute node
                        node_result = await self._execute_node(db, node, workflow_output)
                        
                        if node_result['status'] == 'completed':
                            workflow_output.update(node_result.get('output_data', {}))
                            
                            yield {
                                "type": "phase_completed",
                                "phase_number": i + 1,
                                "phase_name": node.name,
                                "output_preview": str(node_result.get('output_data', {}))[:200] + "..."
                            }
                        else:
                            yield {
                                "type": "phase_failed",
                                "phase_number": i + 1,
                                "phase_name": node.name,
                                "error": node_result.get('error_message', 'Unknown error')
                            }
                            
                            # Mark workflow as failed
                            workflow.status = "failed"
                            workflow.error_message = f"Phase '{node.name}' failed: {node_result.get('error_message')}"
                            workflow.completed_at = datetime.utcnow()
                            db.commit()
                            return
                    
                    except Exception as e:
                        self.logger.error(f"Error executing node {node.id}: {e}")
                        yield {
                            "type": "phase_failed",
                            "phase_number": i + 1,
                            "phase_name": node.name,
                            "error": str(e)
                        }
                        
                        workflow.status = "failed"
                        workflow.error_message = f"Phase '{node.name}' failed: {str(e)}"
                        workflow.completed_at = datetime.utcnow()
                        db.commit()
                        return
                
                # Mark workflow as completed
                workflow.status = "completed"
                workflow.output_data = workflow_output
                workflow.completed_at = datetime.utcnow()
                db.commit()
                
                yield {
                    "type": "workflow_completed",
                    "workflow_id": workflow_id,
                    "total_execution_time": (workflow.completed_at - workflow.started_at).total_seconds(),
                    "output_data": workflow_output
                }
        
        except Exception as e:
            self.logger.error(f"Error executing workflow {workflow_id}: {e}")
            yield {"type": "error", "message": str(e)}
    
    async def _create_nodes_from_template(
        self, 
        db: Session, 
        workflow: MetaTaskWorkflow, 
        template: MetaTaskTemplate
    ):
        """Create nodes from template phases"""
        phases = template.template_config.get('phases', [])
        
        for i, phase in enumerate(phases):
            node = MetaTaskNode(
                workflow_id=workflow.id,
                name=phase['name'],
                node_type=phase['type'],
                node_config=phase,
                execution_order=i,
                position_x=100 + (i * 200),
                position_y=100
            )
            db.add(node)
        
        db.commit()
    
    async def _execute_node(
        self, 
        db: Session, 
        node: MetaTaskNode, 
        workflow_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single node"""
        try:
            # Update node status
            node.status = "running"
            node.started_at = datetime.utcnow()
            db.commit()
            
            # Create execution record
            execution = MetaTaskExecution(
                node_id=node.id,
                execution_order=1,
                status="running",
                input_data=workflow_context,
                started_at=datetime.utcnow()
            )
            db.add(execution)
            db.commit()
            
            # Execute based on node type
            result = await self._execute_node_by_type(node, workflow_context)
            
            # Update execution
            execution.status = "completed" if result['success'] else "failed"
            execution.output_data = result.get('output_data', {})
            execution.error_message = result.get('error_message')
            execution.execution_time_ms = result.get('execution_time_ms', 0)
            execution.completed_at = datetime.utcnow()
            db.commit()
            
            # Update node
            node.status = "completed" if result['success'] else "failed"
            node.output_data = result.get('output_data', {})
            node.error_message = result.get('error_message')
            node.completed_at = datetime.utcnow()
            db.commit()
            
            return {
                'status': 'completed' if result['success'] else 'failed',
                'output_data': result.get('output_data', {}),
                'error_message': result.get('error_message')
            }
        
        except Exception as e:
            self.logger.error(f"Error executing node {node.id}: {e}")
            
            # Update records as failed
            node.status = "failed"
            node.error_message = str(e)
            node.completed_at = datetime.utcnow()
            db.commit()
            
            return {
                'status': 'failed',
                'error_message': str(e)
            }
    
    async def _execute_node_by_type(
        self, 
        node: MetaTaskNode, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute node based on its type"""
        node_type = node.node_type
        config = node.node_config
        
        if node_type == "analyzer":
            return await self._execute_analyzer_node(node, context, config)
        elif node_type == "generator":
            return await self._execute_generator_node(node, context, config)
        elif node_type == "reviewer":
            return await self._execute_reviewer_node(node, context, config)
        elif node_type == "assembler":
            return await self._execute_assembler_node(node, context, config)
        else:
            return {
                'success': False,
                'error_message': f"Unknown node type: {node_type}"
            }
    
    async def _execute_analyzer_node(
        self, 
        node: MetaTaskNode, 
        context: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute analyzer node"""
        # Mock implementation - in reality would use LLM
        await asyncio.sleep(1)  # Simulate processing
        
        analysis = {
            'requirements': context.get('topic', 'Unknown topic'),
            'structure': ['introduction', 'main_content', 'conclusion'],
            'estimated_length': context.get('target_length', 10),
            'key_points': ['point1', 'point2', 'point3']
        }
        
        return {
            'success': True,
            'output_data': {'analysis': analysis},
            'execution_time_ms': 1000
        }
    
    async def _execute_generator_node(
        self, 
        node: MetaTaskNode, 
        context: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute generator node"""
        await asyncio.sleep(2)  # Simulate processing
        
        content = f"Generated content for {node.name} based on: {context.get('topic', 'Unknown')}"
        
        return {
            'success': True,
            'output_data': {'generated_content': content, 'section': node.name},
            'execution_time_ms': 2000
        }
    
    async def _execute_reviewer_node(
        self, 
        node: MetaTaskNode, 
        context: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute reviewer node"""
        await asyncio.sleep(1)
        
        review = {
            'quality_score': 0.85,
            'suggestions': ['Improve clarity', 'Add more examples'],
            'approved': True
        }
        
        return {
            'success': True,
            'output_data': {'review': review},
            'execution_time_ms': 1000
        }
    
    async def _execute_assembler_node(
        self, 
        node: MetaTaskNode, 
        context: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute assembler node"""
        await asyncio.sleep(1)
        
        final_document = f"Final assembled document with all sections from context: {json.dumps(context, indent=2)[:500]}..."
        
        return {
            'success': True,
            'output_data': {'final_document': final_document, 'word_count': 1500},
            'execution_time_ms': 1000
        }
    
    def _workflow_to_dict(self, workflow: MetaTaskWorkflow) -> Dict[str, Any]:
        """Convert workflow model to dictionary"""
        return {
            'id': workflow.id,
            'template_id': workflow.template_id,
            'name': workflow.name,
            'description': workflow.description,
            'status': workflow.status,
            'progress': workflow.progress,
            'input_data': workflow.input_data,
            'output_data': workflow.output_data,
            'error_message': workflow.error_message,
            'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
            'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None,
            'created_at': workflow.created_at.isoformat() if workflow.created_at else None
        }