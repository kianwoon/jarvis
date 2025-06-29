"""
Automation Resource Monitor
Tracks and monitors resource usage for automation workflows to prevent exhaustion
"""
import asyncio
import logging
import psutil
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResourceUsage:
    """Resource usage tracking for a workflow execution"""
    workflow_id: int
    execution_id: str
    started_at: datetime = field(default_factory=datetime.now)
    
    # Memory tracking
    memory_start_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_current_mb: float = 0.0
    
    # Process tracking
    subprocess_count: int = 0
    subprocess_peak: int = 0
    
    # Connection tracking
    db_connections: int = 0
    redis_connections: int = 0
    
    # Agent tracking
    agents_executed: int = 0
    agents_active: int = 0
    
    # Performance metrics
    execution_duration_ms: Optional[float] = None
    avg_agent_duration_ms: Optional[float] = None
    
    # Resource limits exceeded
    limits_exceeded: List[str] = field(default_factory=list)


@dataclass
class ResourceLimits:
    """Resource limits for workflow execution"""
    max_memory_mb: float = 1024  # 1GB per workflow
    max_subprocesses: int = 15   # Max MCP subprocesses per workflow
    max_agents: int = 25         # Max agents per workflow
    max_execution_time_ms: float = 1800000  # 30 minutes
    max_db_connections: int = 10  # Max DB connections per workflow
    max_redis_connections: int = 10  # Max Redis connections per workflow


class AutomationResourceMonitor:
    """Monitor and track resource usage for automation workflows"""
    
    def __init__(self, limits: ResourceLimits = None):
        self.limits = limits or ResourceLimits()
        self.active_workflows: Dict[str, WorkflowResourceUsage] = {}
        self.workflow_history: deque = deque(maxlen=100)  # Keep last 100 completed workflows
        self.resource_alerts: List[Dict[str, Any]] = []
        self._monitoring_task = None
        
        # System resource baseline
        self.system_baseline = self._get_system_baseline()
        
        # Start monitoring task
        self._start_monitoring()
    
    def _get_system_baseline(self) -> Dict[str, Any]:
        """Get system resource baseline"""
        try:
            process = psutil.Process()
            return {
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files()),
                "num_threads": process.num_threads(),
                "timestamp": datetime.now()
            }
        except Exception as e:
            logger.warning(f"Failed to get system baseline: {e}")
            return {"error": str(e), "timestamp": datetime.now()}
    
    def _start_monitoring(self):
        """Start background monitoring task"""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._periodic_monitoring())
    
    async def _periodic_monitoring(self):
        """Periodic monitoring of active workflows"""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                await self._update_workflow_metrics()
                await self._check_resource_limits()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
    
    async def _update_workflow_metrics(self):
        """Update metrics for all active workflows"""
        current_memory = self._get_current_memory_mb()
        current_subprocesses = self._get_subprocess_count()
        
        for workflow_key, usage in self.active_workflows.items():
            # Update current metrics
            usage.memory_current_mb = current_memory
            usage.subprocess_count = current_subprocesses
            
            # Update peaks
            if current_memory > usage.memory_peak_mb:
                usage.memory_peak_mb = current_memory
            
            if current_subprocesses > usage.subprocess_peak:
                usage.subprocess_peak = current_subprocesses
    
    async def _check_resource_limits(self):
        """Check if any workflows are exceeding resource limits"""
        for workflow_key, usage in self.active_workflows.items():
            alerts = []
            
            # Check memory limit
            if usage.memory_current_mb > self.limits.max_memory_mb:
                alerts.append(f"Memory limit exceeded: {usage.memory_current_mb:.1f}MB > {self.limits.max_memory_mb}MB")
            
            # Check subprocess limit
            if usage.subprocess_count > self.limits.max_subprocesses:
                alerts.append(f"Subprocess limit exceeded: {usage.subprocess_count} > {self.limits.max_subprocesses}")
            
            # Check agent limit
            if usage.agents_executed > self.limits.max_agents:
                alerts.append(f"Agent limit exceeded: {usage.agents_executed} > {self.limits.max_agents}")
            
            # Check execution time
            if usage.execution_duration_ms and usage.execution_duration_ms > self.limits.max_execution_time_ms:
                alerts.append(f"Execution time limit exceeded: {usage.execution_duration_ms:.1f}ms > {self.limits.max_execution_time_ms}ms")
            
            # Record alerts
            for alert in alerts:
                if alert not in usage.limits_exceeded:
                    usage.limits_exceeded.append(alert)
                    self._record_alert(usage.workflow_id, usage.execution_id, alert)
    
    def _record_alert(self, workflow_id: int, execution_id: str, alert: str):
        """Record a resource limit alert"""
        alert_record = {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "alert": alert,
            "timestamp": datetime.now(),
            "severity": "warning"
        }
        
        self.resource_alerts.append(alert_record)
        logger.warning(f"Resource alert: Workflow {workflow_id}:{execution_id} - {alert}")
        
        # Keep only last 50 alerts
        if len(self.resource_alerts) > 50:
            self.resource_alerts = self.resource_alerts[-50:]
    
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_subprocess_count(self) -> int:
        """Get current subprocess count"""
        try:
            process = psutil.Process()
            return len(process.children(recursive=True))
        except Exception:
            return 0
    
    def start_workflow_monitoring(self, workflow_id: int, execution_id: str) -> WorkflowResourceUsage:
        """Start monitoring a workflow execution"""
        workflow_key = f"{workflow_id}:{execution_id}"
        
        usage = WorkflowResourceUsage(
            workflow_id=workflow_id,
            execution_id=execution_id,
            memory_start_mb=self._get_current_memory_mb(),
            subprocess_count=self._get_subprocess_count()
        )
        
        self.active_workflows[workflow_key] = usage
        logger.info(f"Started resource monitoring for workflow {workflow_key}")
        
        return usage
    
    def update_agent_count(self, workflow_id: int, execution_id: str, executed: int = 0, active: int = 0):
        """Update agent execution counts"""
        workflow_key = f"{workflow_id}:{execution_id}"
        if workflow_key in self.active_workflows:
            usage = self.active_workflows[workflow_key]
            if executed > 0:
                usage.agents_executed += executed
            usage.agents_active = active
    
    def update_connection_counts(self, workflow_id: int, execution_id: str, db_connections: int = 0, redis_connections: int = 0):
        """Update connection counts"""
        workflow_key = f"{workflow_id}:{execution_id}"
        if workflow_key in self.active_workflows:
            usage = self.active_workflows[workflow_key]
            usage.db_connections = db_connections
            usage.redis_connections = redis_connections
    
    def complete_workflow_monitoring(self, workflow_id: int, execution_id: str) -> Optional[WorkflowResourceUsage]:
        """Complete monitoring for a workflow execution"""
        workflow_key = f"{workflow_id}:{execution_id}"
        
        if workflow_key not in self.active_workflows:
            return None
        
        usage = self.active_workflows[workflow_key]
        
        # Calculate final metrics
        usage.execution_duration_ms = (datetime.now() - usage.started_at).total_seconds() * 1000
        if usage.agents_executed > 0:
            usage.avg_agent_duration_ms = usage.execution_duration_ms / usage.agents_executed
        
        # Move to history
        self.workflow_history.append(usage)
        del self.active_workflows[workflow_key]
        
        logger.info(f"Completed resource monitoring for workflow {workflow_key}")
        logger.info(f"Final metrics - Memory: {usage.memory_peak_mb:.1f}MB, Agents: {usage.agents_executed}, Duration: {usage.execution_duration_ms:.1f}ms")
        
        return usage
    
    def get_workflow_stats(self, workflow_id: int = None, execution_id: str = None) -> Dict[str, Any]:
        """Get resource statistics"""
        if workflow_id and execution_id:
            # Get specific workflow stats
            workflow_key = f"{workflow_id}:{execution_id}"
            if workflow_key in self.active_workflows:
                usage = self.active_workflows[workflow_key]
                return {
                    "workflow_id": usage.workflow_id,
                    "execution_id": usage.execution_id,
                    "status": "active",
                    "memory_current_mb": usage.memory_current_mb,
                    "memory_peak_mb": usage.memory_peak_mb,
                    "subprocess_count": usage.subprocess_count,
                    "agents_executed": usage.agents_executed,
                    "agents_active": usage.agents_active,
                    "execution_duration_ms": (datetime.now() - usage.started_at).total_seconds() * 1000,
                    "limits_exceeded": usage.limits_exceeded
                }
            else:
                return {"error": "Workflow not found in active monitoring"}
        
        # Get overall stats
        active_workflows = len(self.active_workflows)
        total_memory = sum(usage.memory_current_mb for usage in self.active_workflows.values())
        total_subprocesses = sum(usage.subprocess_count for usage in self.active_workflows.values())
        total_agents = sum(usage.agents_active for usage in self.active_workflows.values())
        
        # Historical averages
        if self.workflow_history:
            avg_duration = sum(w.execution_duration_ms for w in self.workflow_history if w.execution_duration_ms) / len(self.workflow_history)
            avg_memory_peak = sum(w.memory_peak_mb for w in self.workflow_history) / len(self.workflow_history)
            avg_agents = sum(w.agents_executed for w in self.workflow_history) / len(self.workflow_history)
        else:
            avg_duration = avg_memory_peak = avg_agents = 0
        
        return {
            "active_workflows": active_workflows,
            "total_memory_mb": total_memory,
            "total_subprocesses": total_subprocesses,
            "total_active_agents": total_agents,
            "resource_limits": {
                "max_memory_mb": self.limits.max_memory_mb,
                "max_subprocesses": self.limits.max_subprocesses,
                "max_agents": self.limits.max_agents,
                "max_execution_time_ms": self.limits.max_execution_time_ms
            },
            "system_utilization": {
                "memory_utilization": total_memory / (self.limits.max_memory_mb * active_workflows) if active_workflows > 0 else 0,
                "subprocess_utilization": total_subprocesses / (self.limits.max_subprocesses * active_workflows) if active_workflows > 0 else 0
            },
            "historical_averages": {
                "avg_duration_ms": avg_duration,
                "avg_memory_peak_mb": avg_memory_peak,
                "avg_agents_per_workflow": avg_agents,
                "completed_workflows": len(self.workflow_history)
            },
            "recent_alerts": self.resource_alerts[-10:],  # Last 10 alerts
            "timestamp": datetime.now().isoformat()
        }
    
    async def cleanup(self):
        """Clean up monitoring resources"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Resource monitor cleanup completed")


# Global resource monitor instance
automation_resource_monitor = AutomationResourceMonitor()


def get_resource_monitor() -> AutomationResourceMonitor:
    """Get the global resource monitor instance"""
    return automation_resource_monitor