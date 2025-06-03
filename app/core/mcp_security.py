"""
Security validation and sandboxing for MCP command execution
"""
import os
import re
import shlex
from typing import List, Dict, Optional, Set
from pathlib import Path

class MCPSecurityValidator:
    """Security validator for MCP command execution"""
    
    def __init__(self):
        # Whitelist of allowed commands (expandable based on needs)
        self.allowed_commands = {
            'python', 'python3', 'node', 'npm', 'pip', 'pip3',
            'java', 'go', 'rust', 'cargo', 'dotnet',
            'mcp-server', 'uvicorn', 'gunicorn'
        }
        
        # Dangerous commands that should never be allowed
        self.dangerous_commands = {
            'rm', 'rmdir', 'dd', 'mkfs', 'fdisk', 'mount', 'umount',
            'passwd', 'su', 'sudo', 'chmod', 'chown', 'systemctl',
            'service', 'init', 'reboot', 'shutdown', 'halt', 'killall',
            'crontab', 'at', 'batch', 'wall', 'write', 'mesg',
            'finger', 'who', 'w', 'last', 'lastb', 'users'
        }
        
        # Dangerous path patterns
        self.dangerous_paths = {
            '/etc/', '/bin/', '/sbin/', '/usr/bin/', '/usr/sbin/',
            '/boot/', '/dev/', '/proc/', '/sys/', '/root/',
            '/var/log/', '/var/run/', '/var/lib/dpkg/',
        }
        
        # Dangerous argument patterns
        self.dangerous_patterns = [
            r'\.\./', r'\$\(', r'`', r'\|', r'&&', r'\|\|', 
            r'>', r'>>', r'<', r'&', r'\*', r'\?', r'\[', r'\]',
            r'~/', r'/etc', r'/bin', r'/usr/bin', r'/sbin'
        ]
    
    def validate_command(self, command: str, args: List[str] = None) -> tuple[bool, str]:
        """
        Validate a command for security
        Returns: (is_valid, error_message)
        """
        if not command:
            return False, "Command cannot be empty"
        
        # Extract command name
        cmd_name = os.path.basename(command.strip())
        
        # Check if command is in dangerous list
        if cmd_name in self.dangerous_commands:
            return False, f"Command '{cmd_name}' is not allowed for security reasons"
        
        # For now, only allow whitelisted commands
        if cmd_name not in self.allowed_commands:
            return False, f"Command '{cmd_name}' is not in the allowed command list"
        
        # Validate arguments
        if args:
            for arg in args:
                if not self._validate_argument(arg):
                    return False, f"Argument '{arg}' contains dangerous patterns"
        
        # Validate command path
        if not self._validate_path(command):
            return False, f"Command path '{command}' is not allowed"
        
        return True, "Command validation passed"
    
    def validate_environment(self, env: Dict[str, str]) -> tuple[bool, str]:
        """
        Validate environment variables
        Returns: (is_valid, error_message)
        """
        if not env:
            return True, "No environment variables to validate"
        
        dangerous_env_vars = {
            'PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH',
            'NODE_PATH', 'JAVA_HOME', 'HOME'
        }
        
        for key, value in env.items():
            # Check for dangerous environment variable names
            if key in dangerous_env_vars:
                return False, f"Environment variable '{key}' is not allowed"
            
            # Check for dangerous values
            if any(pattern in value for pattern in ['./', '../', '/etc/', '/bin/']):
                return False, f"Environment variable '{key}' contains dangerous path"
        
        return True, "Environment validation passed"
    
    def validate_working_directory(self, working_dir: str) -> tuple[bool, str]:
        """
        Validate working directory
        Returns: (is_valid, error_message)
        """
        if not working_dir:
            return True, "No working directory specified"
        
        # Resolve absolute path
        try:
            abs_path = os.path.abspath(working_dir)
        except Exception:
            return False, "Invalid working directory path"
        
        # Check against dangerous paths
        for dangerous_path in self.dangerous_paths:
            if abs_path.startswith(dangerous_path):
                return False, f"Working directory '{abs_path}' is in a restricted area"
        
        # Ensure it's not a system directory
        if abs_path in ['/', '/etc', '/bin', '/sbin', '/usr', '/var']:
            return False, f"Working directory '{abs_path}' is a system directory"
        
        return True, "Working directory validation passed"
    
    def _validate_argument(self, arg: str) -> bool:
        """Validate a single argument for dangerous patterns"""
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, arg):
                return False
        
        # Check for absolute paths to dangerous directories
        for dangerous_path in self.dangerous_paths:
            if arg.startswith(dangerous_path):
                return False
        
        return True
    
    def _validate_path(self, path: str) -> bool:
        """Validate a command path"""
        # Allow relative paths and paths in safe directories
        if not os.path.isabs(path):
            return True
        
        # Check against dangerous paths
        for dangerous_path in self.dangerous_paths:
            if path.startswith(dangerous_path):
                return False
        
        return True
    
    def create_secure_environment(self, base_env: Dict[str, str] = None) -> Dict[str, str]:
        """Create a secure environment for process execution"""
        # Start with minimal environment
        secure_env = {
            'PATH': '/usr/local/bin:/usr/bin:/bin',
            'LANG': 'en_US.UTF-8',
            'LC_ALL': 'en_US.UTF-8',
        }
        
        # Add user-specified environment variables (after validation)
        if base_env:
            is_valid, error = self.validate_environment(base_env)
            if is_valid:
                secure_env.update(base_env)
        
        return secure_env
    
    def get_secure_working_directory(self, requested_dir: str = None) -> str:
        """Get a secure working directory"""
        if requested_dir:
            is_valid, error = self.validate_working_directory(requested_dir)
            if is_valid:
                return requested_dir
        
        # Default to a safe temporary directory
        return "/tmp/mcp_workspace"

# Global security validator instance
mcp_security = MCPSecurityValidator()