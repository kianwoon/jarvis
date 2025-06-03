"""
MCP Process Management Service
Handles spawning, monitoring, and managing MCP server processes for command-based configurations.
"""
import asyncio
import subprocess
import psutil
import json
import os
import signal
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.core.db import get_db, MCPServer
from app.core.redis_base import RedisCache
from app.core.mcp_security import mcp_security
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessInfo:
    pid: int
    command: str
    args: List[str]
    env: Dict[str, str]
    working_dir: str
    start_time: float
    restart_count: int

class MCPProcessManager:
    """Manages MCP server processes for command-based configurations"""
    
    def __init__(self):
        self.processes: Dict[int, ProcessInfo] = {}  # server_id -> ProcessInfo
        self.redis_cache = RedisCache(key_prefix="mcp_process:")
        self.health_check_interval = 30  # seconds
        self.max_process_age = 3600  # 1 hour before restart
        
    async def start_server(self, server_id: int, db: Session) -> Tuple[bool, str]:
        """Start an MCP server process"""
        try:
            server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
            if not server:
                return False, f"Server {server_id} not found"
            
            if server.config_type != "command":
                return False, f"Server {server_id} is not command-based"
            
            if server.is_running and server.process_id:
                if self._is_process_alive(server.process_id):
                    return True, f"Server {server_id} already running (PID: {server.process_id})"
            
            # Validate command and security using the security validator
            is_valid, error_msg = mcp_security.validate_command(server.command, server.args or [])
            if not is_valid:
                return False, f"Security validation failed: {error_msg}"
            
            # Validate environment
            is_env_valid, env_error = mcp_security.validate_environment(server.env or {})
            if not is_env_valid:
                return False, f"Environment validation failed: {env_error}"
            
            # Validate working directory
            is_dir_valid, dir_error = mcp_security.validate_working_directory(server.working_directory)
            if not is_dir_valid:
                return False, f"Working directory validation failed: {dir_error}"
            
            # Prepare secure environment
            env = mcp_security.create_secure_environment(server.env)
            
            # Get secure working directory
            working_dir = mcp_security.get_secure_working_directory(server.working_directory)
            
            # Ensure working directory exists
            os.makedirs(working_dir, exist_ok=True)
            
            # Start the process
            cmd_args = [server.command] + (server.args or [])
            
            logger.info(f"Starting MCP server {server_id}: {' '.join(cmd_args)}")
            
            process = subprocess.Popen(
                cmd_args,
                env=env,
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Create new process group for easier cleanup
            )
            
            # Store process info
            process_info = ProcessInfo(
                pid=process.pid,
                command=server.command,
                args=server.args or [],
                env=server.env or {},
                working_dir=working_dir,
                start_time=time.time(),
                restart_count=server.restart_count
            )
            
            self.processes[server_id] = process_info
            
            # Update database
            server.process_id = process.pid
            server.is_running = True
            server.health_status = "starting"
            db.commit()
            
            # Cache process info
            self.redis_cache.set(
                f"process_{server_id}",
                {
                    "pid": process.pid,
                    "start_time": process_info.start_time,
                    "status": "running"
                },
                expire=3600
            )
            
            # Wait a moment to check if process started successfully
            await asyncio.sleep(2)
            if not self._is_process_alive(process.pid):
                self._cleanup_dead_process(server_id, db)
                return False, f"Process failed to start or died immediately"
            
            return True, f"Server {server_id} started successfully (PID: {process.pid})"
            
        except Exception as e:
            logger.error(f"Error starting server {server_id}: {e}")
            return False, f"Failed to start server: {str(e)}"
    
    async def stop_server(self, server_id: int, db: Session, force: bool = False) -> Tuple[bool, str]:
        """Stop an MCP server process"""
        try:
            server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
            if not server:
                return False, f"Server {server_id} not found"
            
            if not server.is_running or not server.process_id:
                return True, f"Server {server_id} is not running"
            
            pid = server.process_id
            
            if not self._is_process_alive(pid):
                self._cleanup_dead_process(server_id, db)
                return True, f"Server {server_id} was already stopped"
            
            # Attempt graceful shutdown first
            if not force:
                try:
                    os.kill(pid, signal.SIGTERM)
                    # Wait up to 10 seconds for graceful shutdown
                    for _ in range(10):
                        await asyncio.sleep(1)
                        if not self._is_process_alive(pid):
                            break
                    else:
                        # Still running, force kill
                        logger.warning(f"Graceful shutdown failed for server {server_id}, force killing")
                        force = True
                except ProcessLookupError:
                    pass  # Process already dead
            
            # Force kill if needed
            if force and self._is_process_alive(pid):
                try:
                    # Kill entire process group
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            
            self._cleanup_dead_process(server_id, db)
            return True, f"Server {server_id} stopped successfully"
            
        except Exception as e:
            logger.error(f"Error stopping server {server_id}: {e}")
            return False, f"Failed to stop server: {str(e)}"
    
    async def restart_server(self, server_id: int, db: Session) -> Tuple[bool, str]:
        """Restart an MCP server process"""
        stop_success, stop_msg = await self.stop_server(server_id, db, force=True)
        if not stop_success:
            return False, f"Failed to stop server before restart: {stop_msg}"
        
        # Increment restart count
        server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
        if server:
            server.restart_count += 1
            db.commit()
        
        await asyncio.sleep(1)  # Brief pause between stop and start
        return await self.start_server(server_id, db)
    
    async def health_check_server(self, server_id: int, db: Session) -> Tuple[bool, str]:
        """Perform health check on an MCP server"""
        try:
            server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
            if not server:
                return False, "Server not found"
            
            if not server.is_running or not server.process_id:
                return False, "Server not running"
            
            # Check if process is alive
            if not self._is_process_alive(server.process_id):
                self._cleanup_dead_process(server_id, db)
                return False, "Process is dead"
            
            # For command-based servers, we can only check process liveness
            # TODO: Implement actual MCP protocol health checks
            server.health_status = "healthy"
            server.last_health_check = func.now() if hasattr(func, 'now') else None
            db.commit()
            
            return True, "Server is healthy"
            
        except Exception as e:
            logger.error(f"Health check failed for server {server_id}: {e}")
            return False, f"Health check error: {str(e)}"
    
    async def monitor_processes(self):
        """Background task to monitor all running processes"""
        while True:
            try:
                with next(get_db()) as db:
                    servers = db.query(MCPServer).filter(
                        MCPServer.is_running == True,
                        MCPServer.config_type == "command"
                    ).all()
                    
                    for server in servers:
                        await self._monitor_single_process(server, db)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in process monitoring: {e}")
                await asyncio.sleep(10)  # Brief pause before retry
    
    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is alive"""
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return False
    
    def _cleanup_dead_process(self, server_id: int, db: Session):
        """Clean up records for a dead process"""
        server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
        if server:
            server.process_id = None
            server.is_running = False
            server.health_status = "stopped"
            db.commit()
        
        # Remove from local tracking
        if server_id in self.processes:
            del self.processes[server_id]
        
        # Remove from cache
        self.redis_cache.delete(f"process_{server_id}")
    
    
    async def _monitor_single_process(self, server: MCPServer, db: Session):
        """Monitor a single process and handle restarts if needed"""
        try:
            if not self._is_process_alive(server.process_id):
                logger.warning(f"Process {server.process_id} for server {server.id} is dead")
                
                # Check restart policy
                if (server.restart_policy == "always" or 
                    (server.restart_policy == "on-failure" and server.restart_count < server.max_restarts)):
                    
                    logger.info(f"Attempting to restart server {server.id}")
                    success, msg = await self.restart_server(server.id, db)
                    if success:
                        logger.info(f"Successfully restarted server {server.id}")
                    else:
                        logger.error(f"Failed to restart server {server.id}: {msg}")
                else:
                    logger.info(f"Not restarting server {server.id} due to restart policy")
                    self._cleanup_dead_process(server.id, db)
            
            # Check if process is too old (for periodic restarts)
            elif server_id in self.processes:
                process_info = self.processes[server_id]
                if time.time() - process_info.start_time > self.max_process_age:
                    logger.info(f"Restarting server {server.id} due to age limit")
                    await self.restart_server(server.id, db)
                    
        except Exception as e:
            logger.error(f"Error monitoring server {server.id}: {e}")

# Global instance
mcp_process_manager = MCPProcessManager()

async def start_process_monitor():
    """Start the background process monitor"""
    asyncio.create_task(mcp_process_manager.monitor_processes())