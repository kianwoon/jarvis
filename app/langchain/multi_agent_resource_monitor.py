"""
Multi-Agent Resource Monitor
Tracks and monitors resource usage for multi-agent workflows to prevent exhaustion
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
class MultiAgentResourceUsage:
    """Resource usage tracking for a multi-agent execution"""
    conversation_id: str
    mode: str  # 'sequential', 'parallel', 'hierarchical', etc.
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
    agent_failures: int = 0
    
    # Performance metrics
    execution_duration_ms: Optional[float] = None
    avg_agent_duration_ms: Optional[float] = None
    llm_calls_count: int = 0
    tool_calls_count: int = 0
    
    # Resource limits exceeded
    limits_exceeded: List[str] = field(default_factory=list)
    
    # Agent-specific metrics
    agent_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class MultiAgentResourceLimits:
    """Resource limits for multi-agent execution"""
    max_memory_mb: float = 2048  # 2GB for multi-agent workflows
    max_subprocesses: int = 30   # Max MCP subprocesses for multi-agent
    max_agents: int = 50         # Max agents per conversation
    max_execution_time_ms: float = 3600000  # 60 minutes for multi-agent
    max_db_connections: int = 20  # Max DB connections per multi-agent session
    max_redis_connections: int = 20  # Max Redis connections per multi-agent session
    max_llm_calls: int = 200     # Max LLM calls per multi-agent session
    max_tool_calls: int = 100    # Max tool calls per multi-agent session


class MultiAgentResourceMonitor:
    """Monitor and track resource usage for multi-agent workflows"""
    
    def __init__(self, limits: MultiAgentResourceLimits = None):
        self.limits = limits or MultiAgentResourceLimits()
        self.active_sessions: Dict[str, MultiAgentResourceUsage] = {}
        self.session_history: deque = deque(maxlen=50)  # Keep last 50 completed sessions
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
        """Periodic monitoring of active multi-agent sessions"""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                await self._update_session_metrics()
                await self._check_resource_limits()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Multi-agent resource monitoring error: {e}")
    
    async def _update_session_metrics(self):
        """Update metrics for all active multi-agent sessions"""
        current_memory = self._get_current_memory_mb()
        current_subprocesses = self._get_subprocess_count()
        
        for conversation_id, usage in self.active_sessions.items():
            # Update current metrics
            usage.memory_current_mb = current_memory
            usage.subprocess_count = current_subprocesses
            
            # Update peaks
            if current_memory > usage.memory_peak_mb:
                usage.memory_peak_mb = current_memory
            
            if current_subprocesses > usage.subprocess_peak:
                usage.subprocess_peak = current_subprocesses
    
    async def _check_resource_limits(self):
        """Check if any multi-agent sessions are exceeding resource limits"""
        for conversation_id, usage in self.active_sessions.items():
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
            
            # Check LLM calls limit
            if usage.llm_calls_count > self.limits.max_llm_calls:
                alerts.append(f"LLM calls limit exceeded: {usage.llm_calls_count} > {self.limits.max_llm_calls}")
            
            # Check tool calls limit
            if usage.tool_calls_count > self.limits.max_tool_calls:
                alerts.append(f"Tool calls limit exceeded: {usage.tool_calls_count} > {self.limits.max_tool_calls}")
            
            # Check execution time
            if usage.execution_duration_ms and usage.execution_duration_ms > self.limits.max_execution_time_ms:
                alerts.append(f"Execution time limit exceeded: {usage.execution_duration_ms:.1f}ms > {self.limits.max_execution_time_ms}ms")
            
            # Record alerts
            for alert in alerts:
                if alert not in usage.limits_exceeded:
                    usage.limits_exceeded.append(alert)
                    self._record_alert(conversation_id, usage.mode, alert)
    
    def _record_alert(self, conversation_id: str, mode: str, alert: str):
        """Record a resource limit alert"""
        alert_record = {
            "conversation_id": conversation_id,
            "mode": mode,
            "alert": alert,
            "timestamp": datetime.now(),
            "severity": "warning"
        }
        
        self.resource_alerts.append(alert_record)
        logger.warning(f"Multi-agent resource alert: {conversation_id} ({mode}) - {alert}")
        
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
    
    def start_session_monitoring(self, conversation_id: str, mode: str = "sequential") -> MultiAgentResourceUsage:
        """Start monitoring a multi-agent session"""
        usage = MultiAgentResourceUsage(
            conversation_id=conversation_id,
            mode=mode,
            memory_start_mb=self._get_current_memory_mb(),
            subprocess_count=self._get_subprocess_count()
        )
        
        self.active_sessions[conversation_id] = usage
        logger.info(f"Started multi-agent resource monitoring for session {conversation_id} (mode: {mode})")
        
        return usage
    
    def record_agent_start(self, conversation_id: str, agent_name: str):
        """Record that an agent has started"""
        if conversation_id in self.active_sessions:
            usage = self.active_sessions[conversation_id]
            usage.agents_active += 1
            
            # Initialize agent metrics if not exists
            if agent_name not in usage.agent_metrics:
                usage.agent_metrics[agent_name] = {
                    "start_time": datetime.now(),
                    "llm_calls": 0,
                    "tool_calls": 0,
                    "duration_ms": None,
                    "success": None
                }
    
    def record_agent_complete(self, conversation_id: str, agent_name: str, success: bool = True):
        """Record that an agent has completed"""
        if conversation_id in self.active_sessions:
            usage = self.active_sessions[conversation_id]
            usage.agents_active = max(0, usage.agents_active - 1)
            usage.agents_executed += 1
            
            if not success:
                usage.agent_failures += 1
            
            # Update agent metrics
            if agent_name in usage.agent_metrics:
                agent_metrics = usage.agent_metrics[agent_name]
                if "start_time" in agent_metrics:
                    duration = (datetime.now() - agent_metrics["start_time"]).total_seconds() * 1000
                    agent_metrics["duration_ms"] = duration
                agent_metrics["success"] = success
    
    def record_llm_call(self, conversation_id: str, agent_name: str = None):
        """Record an LLM call"""
        if conversation_id in self.active_sessions:
            usage = self.active_sessions[conversation_id]
            usage.llm_calls_count += 1
            
            if agent_name and agent_name in usage.agent_metrics:
                usage.agent_metrics[agent_name]["llm_calls"] += 1
    
    def record_tool_call(self, conversation_id: str, agent_name: str = None):
        """Record a tool call"""
        if conversation_id in self.active_sessions:
            usage = self.active_sessions[conversation_id]
            usage.tool_calls_count += 1
            
            if agent_name and agent_name in usage.agent_metrics:
                usage.agent_metrics[agent_name]["tool_calls"] += 1
    
    def update_connection_counts(self, conversation_id: str, db_connections: int = 0, redis_connections: int = 0):
        """Update connection counts"""
        if conversation_id in self.active_sessions:
            usage = self.active_sessions[conversation_id]
            usage.db_connections = db_connections
            usage.redis_connections = redis_connections
    
    def complete_session_monitoring(self, conversation_id: str) -> Optional[MultiAgentResourceUsage]:
        """Complete monitoring for a multi-agent session"""
        if conversation_id not in self.active_sessions:
            return None
        
        usage = self.active_sessions[conversation_id]
        
        # Calculate final metrics
        usage.execution_duration_ms = (datetime.now() - usage.started_at).total_seconds() * 1000
        if usage.agents_executed > 0:
            usage.avg_agent_duration_ms = usage.execution_duration_ms / usage.agents_executed
        
        # Move to history
        self.session_history.append(usage)
        del self.active_sessions[conversation_id]
        
        logger.info(f"Completed multi-agent resource monitoring for session {conversation_id}")
        logger.info(f"Final metrics - Memory: {usage.memory_peak_mb:.1f}MB, Agents: {usage.agents_executed}, "
                   f"LLM calls: {usage.llm_calls_count}, Tool calls: {usage.tool_calls_count}, "
                   f"Duration: {usage.execution_duration_ms:.1f}ms")
        
        return usage
    
    def get_session_stats(self, conversation_id: str = None) -> Dict[str, Any]:
        """Get multi-agent resource statistics"""
        if conversation_id:
            # Get specific session stats
            if conversation_id in self.active_sessions:
                usage = self.active_sessions[conversation_id]
                return {
                    "conversation_id": usage.conversation_id,
                    "mode": usage.mode,
                    "status": "active",
                    "memory_current_mb": usage.memory_current_mb,
                    "memory_peak_mb": usage.memory_peak_mb,
                    "subprocess_count": usage.subprocess_count,
                    "agents_executed": usage.agents_executed,
                    "agents_active": usage.agents_active,
                    "agent_failures": usage.agent_failures,
                    "llm_calls_count": usage.llm_calls_count,
                    "tool_calls_count": usage.tool_calls_count,
                    "execution_duration_ms": (datetime.now() - usage.started_at).total_seconds() * 1000,
                    "limits_exceeded": usage.limits_exceeded,
                    "agent_metrics": usage.agent_metrics
                }
            else:
                return {"error": "Session not found in active monitoring"}
        
        # Get overall stats
        active_sessions = len(self.active_sessions)
        total_memory = sum(usage.memory_current_mb for usage in self.active_sessions.values())
        total_subprocesses = sum(usage.subprocess_count for usage in self.active_sessions.values())
        total_agents = sum(usage.agents_active for usage in self.active_sessions.values())
        total_llm_calls = sum(usage.llm_calls_count for usage in self.active_sessions.values())
        total_tool_calls = sum(usage.tool_calls_count for usage in self.active_sessions.values())
        
        # Historical averages
        if self.session_history:
            avg_duration = sum(s.execution_duration_ms for s in self.session_history if s.execution_duration_ms) / len(self.session_history)
            avg_memory_peak = sum(s.memory_peak_mb for s in self.session_history) / len(self.session_history)
            avg_agents = sum(s.agents_executed for s in self.session_history) / len(self.session_history)
            avg_llm_calls = sum(s.llm_calls_count for s in self.session_history) / len(self.session_history)
            failure_rate = sum(s.agent_failures for s in self.session_history) / max(1, sum(s.agents_executed for s in self.session_history))
        else:
            avg_duration = avg_memory_peak = avg_agents = avg_llm_calls = failure_rate = 0
        
        return {
            "active_sessions": active_sessions,
            "total_memory_mb": total_memory,
            "total_subprocesses": total_subprocesses,
            "total_active_agents": total_agents,
            "total_llm_calls": total_llm_calls,
            "total_tool_calls": total_tool_calls,
            "resource_limits": {
                "max_memory_mb": self.limits.max_memory_mb,
                "max_subprocesses": self.limits.max_subprocesses,
                "max_agents": self.limits.max_agents,
                "max_execution_time_ms": self.limits.max_execution_time_ms,
                "max_llm_calls": self.limits.max_llm_calls,
                "max_tool_calls": self.limits.max_tool_calls
            },
            "system_utilization": {
                "memory_utilization": total_memory / (self.limits.max_memory_mb * active_sessions) if active_sessions > 0 else 0,
                "subprocess_utilization": total_subprocesses / (self.limits.max_subprocesses * active_sessions) if active_sessions > 0 else 0,
                "llm_call_utilization": total_llm_calls / (self.limits.max_llm_calls * active_sessions) if active_sessions > 0 else 0
            },
            "historical_averages": {
                "avg_duration_ms": avg_duration,
                "avg_memory_peak_mb": avg_memory_peak,
                "avg_agents_per_session": avg_agents,
                "avg_llm_calls_per_session": avg_llm_calls,
                "agent_failure_rate": failure_rate,
                "completed_sessions": len(self.session_history)
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
        
        logger.info("Multi-agent resource monitor cleanup completed")


# Global multi-agent resource monitor instance
multi_agent_resource_monitor = MultiAgentResourceMonitor()


def get_multi_agent_resource_monitor() -> MultiAgentResourceMonitor:
    """Get the global multi-agent resource monitor instance"""
    return multi_agent_resource_monitor