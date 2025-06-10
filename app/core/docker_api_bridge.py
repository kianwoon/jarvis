"""
Docker API Bridge - Direct Docker API communication without CLI
"""

import docker
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DockerAPIBridge:
    """Direct Docker API communication without needing docker CLI"""
    
    def __init__(self):
        try:
            # Connect to Docker daemon via socket
            self.client = docker.from_env()
            logger.info("Docker API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Docker API client: {e}")
            self.client = None
    
    def exec_in_container(self, container_name: str, command: list, stdin_data: Optional[str] = None) -> Dict[str, Any]:
        """Execute command in a container using Docker API"""
        if not self.client:
            return {
                "error": "Docker API client not initialized"
            }
        
        try:
            # Get container
            container = self.client.containers.get(container_name)
            
            # If stdin_data is provided, it should be the last element of command
            if stdin_data and command and command[-1] == stdin_data:
                # Remove stdin_data from command
                command = command[:-1]
            
            # Create exec instance with stdin if needed
            if stdin_data:
                # Create exec with stdin attached
                exec_instance = self.client.api.exec_create(
                    container.id,
                    command,
                    stdin=True,
                    stdout=True,
                    stderr=True,
                    tty=False
                )
                
                # Start exec and get socket
                socket = self.client.api.exec_start(
                    exec_instance['Id'],
                    socket=True
                )
                
                # Send stdin data
                socket._sock.sendall(stdin_data.encode('utf-8'))
                socket._sock.shutdown(1)  # Shutdown write side
                
                # Read output with Docker stream protocol handling
                import struct
                output = b''
                stderr = b''
                
                while True:
                    # Docker stream protocol: 8-byte header followed by payload
                    header = socket._sock.recv(8)
                    if not header or len(header) < 8:
                        break
                    
                    # Parse header: stream_type(1) + reserved(3) + size(4)
                    stream_type = header[0]
                    size = struct.unpack('>I', header[4:8])[0]
                    
                    # Read payload
                    payload = b''
                    while len(payload) < size:
                        chunk = socket._sock.recv(min(size - len(payload), 4096))
                        if not chunk:
                            break
                        payload += chunk
                    
                    # Stream type 1 = stdout, 2 = stderr
                    if stream_type == 1:
                        output += payload
                    elif stream_type == 2:
                        stderr += payload
                
                socket.close()
                
                # Get exit code
                exec_info = self.client.api.exec_inspect(exec_instance['Id'])
                exit_code = exec_info.get('ExitCode', 0)
                
                stdout = output
                stderr = b''
            else:
                # Simple exec without stdin
                result = container.exec_run(
                    command,
                    stdout=True,
                    stderr=True,
                    stdin=False,
                    tty=False,
                    demux=True
                )
                
                stdout, stderr = result.output
                exit_code = result.exit_code
            
            # Decode outputs
            stdout_str = stdout.decode('utf-8') if stdout else ""
            stderr_str = stderr.decode('utf-8') if stderr else ""
            
            # Parse JSON output if possible
            try:
                if stdout_str:
                    output = json.loads(stdout_str)
                else:
                    output = {}
            except json.JSONDecodeError:
                output = {"raw_output": stdout_str}
            
            return {
                "success": exit_code == 0,
                "exit_code": exit_code,
                "output": output,
                "stdout": stdout_str,
                "stderr": stderr_str
            }
            
        except docker.errors.NotFound:
            return {
                "error": f"Container '{container_name}' not found"
            }
        except Exception as e:
            logger.error(f"Error executing command in container: {e}")
            return {
                "error": str(e)
            }
    
    def is_container_running(self, container_name: str) -> bool:
        """Check if a container is running"""
        if not self.client:
            return False
        
        try:
            container = self.client.containers.get(container_name)
            return container.status == "running"
        except:
            return False


# Global instance
_docker_api = None


def get_docker_api() -> DockerAPIBridge:
    """Get or create Docker API bridge instance"""
    global _docker_api
    if _docker_api is None:
        _docker_api = DockerAPIBridge()
    return _docker_api