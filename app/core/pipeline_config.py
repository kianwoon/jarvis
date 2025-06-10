"""
Pipeline Configuration Management

This module provides centralized configuration for the pipeline system,
eliminating hardcoded values and supporting environment-specific settings.
"""
import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
import json


class PipelineSettings(BaseSettings):
    """Pipeline system configuration with environment variable support"""
    
    # Execution settings
    PIPELINE_REDIS_TTL: int = 3600  # Redis state TTL in seconds
    PIPELINE_PROGRESS_TTL: int = 3600  # Progress cache TTL
    PIPELINE_MAX_CONCURRENT: int = int(os.getenv("PIPELINE_MAX_CONCURRENT", "1"))
    PIPELINE_DEFAULT_TIMEOUT: int = 300  # Default execution timeout
    
    # Cache settings
    PIPELINE_LIST_CACHE_TTL: int = 300  # Pipeline list cache TTL
    PIPELINE_AGENT_CACHE_TTL: int = 600  # Agent cache TTL
    
    # Scheduler settings
    PIPELINE_MONITOR_INTERVAL: int = 60  # Schedule monitor interval
    PIPELINE_DEFAULT_SCHEDULE_INTERVAL: int = 60  # Default schedule interval
    
    # Limits
    PIPELINE_MAX_HISTORY: int = 50  # Max execution history records
    PIPELINE_MAX_AGENTS: int = 20  # Max agents per pipeline
    PIPELINE_MAX_EXECUTION_TIME: int = 3600  # Max execution time (1 hour)
    
    # Collaboration modes (extensible)
    PIPELINE_COLLABORATION_MODES: list = [
        "sequential",
        "parallel", 
        "hierarchical",
        "conditional",  # New: branching logic
        "approval_gate",  # New: human approval
        "event_driven",  # New: reactive pipelines
        "hybrid"  # New: mixed modes
    ]
    
    # Communication patterns (extensible)
    PIPELINE_COMMUNICATION_PATTERNS: list = [
        "direct",
        "broadcast",
        "request_response",
        "pub_sub",
        "stream",  # New: streaming data
        "batch",  # New: batch processing
        "webhook"  # New: external callbacks
    ]
    
    # Schedule types (extensible)
    PIPELINE_SCHEDULE_TYPES: list = [
        "cron",
        "interval",
        "one_time",
        "event_triggered",  # New: event-based
        "business_hours",  # New: business hours only
        "sla_based"  # New: SLA-driven scheduling
    ]
    
    # Agent output detection (configurable)
    PIPELINE_OUTPUT_AGENT_PATTERNS: list = [
        "synthesizer", "response_writer", "report_writer", "email_responder",
        "summarizer", "gmail_send", "send_email", "reply", "responder",
        "writer", "generator", "composer", "creator", "publisher",
        "notifier", "alerter", "dashboard_updater", "api_caller"
    ]
    
    # Corporate workflow settings
    PIPELINE_ENABLE_APPROVALS: bool = True
    PIPELINE_ENABLE_AUDIT_TRAIL: bool = True
    PIPELINE_ENABLE_COST_TRACKING: bool = False
    PIPELINE_ENABLE_SLA_MONITORING: bool = True
    
    # Resource management
    PIPELINE_RESOURCE_LIMITS: Dict[str, Any] = {
        "cpu_threshold": 80,  # CPU usage threshold percentage
        "memory_threshold": 80,  # Memory usage threshold percentage
        "max_queue_size": 100,  # Max queued pipelines
        "priority_levels": ["low", "normal", "high", "critical"]
    }
    
    # Retry configuration
    PIPELINE_RETRY_CONFIG: Dict[str, Any] = {
        "max_retries": 3,
        "retry_delay": 60,  # seconds
        "exponential_backoff": True,
        "retriable_errors": ["timeout", "resource_unavailable", "rate_limit"]
    }
    
    class Config:
        env_prefix = "JARVIS_"
        case_sensitive = False
        
    @classmethod
    def from_env(cls) -> "PipelineSettings":
        """Load settings from environment variables"""
        return cls()
    
    @classmethod
    def from_file(cls, file_path: str) -> "PipelineSettings":
        """Load settings from JSON file"""
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def get_collaboration_modes(self) -> list:
        """Get available collaboration modes"""
        return self.PIPELINE_COLLABORATION_MODES
    
    def get_communication_patterns(self) -> list:
        """Get available communication patterns"""
        return self.PIPELINE_COMMUNICATION_PATTERNS
    
    def get_schedule_types(self) -> list:
        """Get available schedule types"""
        return self.PIPELINE_SCHEDULE_TYPES
    
    def is_output_agent(self, agent_name: str, agent_config: Dict[str, Any]) -> bool:
        """Determine if an agent is an output agent"""
        agent_name_lower = agent_name.lower()
        
        # Check explicit role
        if agent_config.get('role', '').lower() == 'output':
            return True
            
        # Check against patterns
        for pattern in self.PIPELINE_OUTPUT_AGENT_PATTERNS:
            if pattern in agent_name_lower:
                return True
                
        # Check tools
        tools = agent_config.get('tools', [])
        output_tool_keywords = ['send', 'publish', 'write', 'notify', 'alert', 'export']
        for tool in tools:
            tool_str = str(tool).lower()
            if any(keyword in tool_str for keyword in output_tool_keywords):
                return True
                
        return False


# Singleton instance
_settings: Optional[PipelineSettings] = None


def get_pipeline_settings() -> PipelineSettings:
    """Get pipeline settings singleton"""
    global _settings
    if _settings is None:
        _settings = PipelineSettings.from_env()
    return _settings


def reload_pipeline_settings():
    """Reload pipeline settings (useful for testing or runtime updates)"""
    global _settings
    _settings = PipelineSettings.from_env()
    return _settings


# Corporate workflow templates
CORPORATE_WORKFLOW_TEMPLATES = {
    "approval_workflow": {
        "name": "Multi-Level Approval Workflow",
        "description": "Pipeline with multiple approval stages",
        "collaboration_mode": "approval_gate",
        "template_agents": [
            {"type": "document_processor", "role": "process"},
            {"type": "validator", "role": "validate"},
            {"type": "approver_l1", "role": "approve", "level": 1},
            {"type": "approver_l2", "role": "approve", "level": 2},
            {"type": "notifier", "role": "output"}
        ]
    },
    "data_processing": {
        "name": "ETL Data Processing Pipeline",
        "description": "Extract, Transform, Load data pipeline",
        "collaboration_mode": "sequential",
        "template_agents": [
            {"type": "data_extractor", "role": "extract"},
            {"type": "data_validator", "role": "validate"},
            {"type": "data_transformer", "role": "transform"},
            {"type": "data_loader", "role": "load"},
            {"type": "report_generator", "role": "output"}
        ]
    },
    "incident_response": {
        "name": "Incident Response Workflow",
        "description": "Automated incident detection and response",
        "collaboration_mode": "event_driven",
        "template_agents": [
            {"type": "monitor", "role": "detect"},
            {"type": "analyzer", "role": "analyze"},
            {"type": "responder", "role": "respond"},
            {"type": "escalator", "role": "escalate"},
            {"type": "reporter", "role": "output"}
        ]
    },
    "compliance_audit": {
        "name": "Compliance Audit Pipeline",
        "description": "Automated compliance checking and reporting",
        "collaboration_mode": "sequential",
        "template_agents": [
            {"type": "data_collector", "role": "collect"},
            {"type": "compliance_checker", "role": "check"},
            {"type": "risk_assessor", "role": "assess"},
            {"type": "report_writer", "role": "output"},
            {"type": "notifier", "role": "notify"}
        ]
    }
}