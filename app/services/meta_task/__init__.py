"""
Meta-Task Service Module
Handles complex, multi-part deliverables that exceed token limits
"""

from .template_manager import MetaTaskTemplateManager
from .workflow_orchestrator import MetaTaskWorkflowOrchestrator
from .execution_engine import MetaTaskExecutionEngine

__all__ = [
    "MetaTaskTemplateManager",
    "MetaTaskWorkflowOrchestrator", 
    "MetaTaskExecutionEngine"
]