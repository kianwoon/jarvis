"""
API Executor Service for APINode
Handles HTTP requests with authentication, retry logic, and response processing
"""
import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import aiohttp
import base64
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)

class APIExecutor:
    """Executes API calls with comprehensive error handling and retry logic"""
    
    def __init__(self):
        self.session = None
        self.rate_limiters = {}  # Track rate limits per API endpoint
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def execute_api_call(
        self,
        node_config: Dict[str, Any],
        parameters: Dict[str, Any],
        workflow_id: int,
        execution_id: str,
        node_id: str
    ) -> Dict[str, Any]:
        """
        Execute API call based on node configuration and parameters
        
        Args:
            node_config: APINode configuration
            parameters: Parameters from LLM or previous nodes
            workflow_id: Workflow identifier
            execution_id: Execution identifier
            node_id: Node identifier
            
        Returns:
            Dict containing response, status, headers, and metadata
        """
        start_time = time.time()
        
        try:
            # Build request URL
            url = self._build_url(node_config, parameters)
            
            # Prepare headers
            headers = self._build_headers(node_config)
            
            # Prepare request body (for POST/PUT/PATCH)
            body = self._build_request_body(node_config, parameters)
            
            # Check rate limiting
            if not self._check_rate_limit(node_config, node_id):
                raise Exception("Rate limit exceeded")
            
            # Execute request with retry logic
            response_data = await self._execute_with_retry(
                url=url,
                method=node_config.get('http_method', 'GET'),
                headers=headers,
                body=body,
                timeout=node_config.get('timeout', 30),
                retry_count=node_config.get('retry_count', 3)
            )
            
            # Transform response if transformation is configured
            transformed_response = self._transform_response(
                response_data['response'],
                node_config.get('response_transformation', '')
            )
            
            # Calculate duration
            duration = int((time.time() - start_time) * 1000)
            
            # Return structured response
            return {
                'response': transformed_response,
                'status': response_data['status'],
                'headers': response_data['headers'],
                'metadata': {
                    'duration': duration,
                    'url': url,
                    'method': node_config.get('http_method', 'GET'),
                    'node_id': node_id,
                    'workflow_id': workflow_id,
                    'execution_id': execution_id,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            # Handle errors based on configuration
            error_handling = node_config.get('error_handling', 'throw')
            duration = int((time.time() - start_time) * 1000)
            
            error_response = {
                'response': None,
                'status': getattr(e, 'status', 500),
                'headers': {},
                'metadata': {
                    'duration': duration,
                    'error': str(e),
                    'node_id': node_id,
                    'workflow_id': workflow_id,
                    'execution_id': execution_id,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            if error_handling == 'throw':
                raise e
            elif error_handling == 'return_null':
                error_response['response'] = None
                return error_response
            elif error_handling == 'return_error':
                error_response['response'] = {'error': str(e)}
                return error_response
            else:  # retry handled in _execute_with_retry
                raise e
                
    def _build_url(self, node_config: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Build the full API URL with parameters"""
        base_url = node_config.get('base_url', '').rstrip('/')
        endpoint_path = node_config.get('endpoint_path', '').lstrip('/')
        
        if not base_url:
            raise ValueError("Base URL is required")
        if not endpoint_path:
            raise ValueError("Endpoint path is required")
            
        # Combine base URL and endpoint path
        url = f"{base_url}/{endpoint_path}"
        
        # Add query parameters for GET requests
        if node_config.get('http_method', 'GET') == 'GET' and parameters:
            query_params = []
            for key, value in parameters.items():
                if value is not None:
                    query_params.append(f"{key}={value}")
            
            if query_params:
                separator = '&' if '?' in url else '?'
                url += separator + '&'.join(query_params)
        
        return url
        
    def _build_headers(self, node_config: Dict[str, Any]) -> Dict[str, str]:
        """Build HTTP headers including authentication"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Jarvis-APINode/1.0'
        }
        
        # Add custom headers
        custom_headers = node_config.get('custom_headers', {})
        if custom_headers:
            headers.update(custom_headers)
        
        # Add authentication headers
        auth_type = node_config.get('authentication_type', 'none')
        
        if auth_type == 'api_key':
            header_name = node_config.get('auth_header_name', 'X-API-Key')
            auth_token = node_config.get('auth_token')
            if auth_token:
                headers[header_name] = auth_token
                
        elif auth_type == 'bearer_token':
            auth_token = node_config.get('auth_token')
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
                
        elif auth_type == 'basic_auth':
            username = node_config.get('basic_auth_username')
            password = node_config.get('basic_auth_password')
            if username and password:
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers['Authorization'] = f'Basic {credentials}'
                
        elif auth_type == 'custom_header':
            header_name = node_config.get('auth_header_name', 'X-Custom-Auth')
            auth_token = node_config.get('auth_token')
            if auth_token:
                headers[header_name] = auth_token
        
        return headers
        
    def _build_request_body(self, node_config: Dict[str, Any], parameters: Dict[str, Any]) -> Optional[str]:
        """Build request body for POST/PUT/PATCH requests"""
        method = node_config.get('http_method', 'GET')
        
        if method in ['POST', 'PUT', 'PATCH'] and parameters:
            # Validate against request schema if provided
            request_schema = node_config.get('request_schema')
            if request_schema:
                # Basic validation - could be enhanced with jsonschema
                required_fields = request_schema.get('required', [])
                for field in required_fields:
                    if field not in parameters:
                        raise ValueError(f"Required field '{field}' missing in parameters")
            
            return json.dumps(parameters)
        
        return None
        
    def _check_rate_limit(self, node_config: Dict[str, Any], node_id: str) -> bool:
        """Check if request is within rate limit"""
        rate_limit = node_config.get('rate_limit', 60)  # requests per minute
        
        if rate_limit <= 0:  # Unlimited
            return True
            
        current_time = time.time()
        minute_window = int(current_time // 60)
        
        if node_id not in self.rate_limiters:
            self.rate_limiters[node_id] = {}
        
        # Clean old entries
        self.rate_limiters[node_id] = {
            window: count for window, count in self.rate_limiters[node_id].items()
            if current_time - (window * 60) < 60
        }
        
        # Check current minute
        current_count = self.rate_limiters[node_id].get(minute_window, 0)
        if current_count >= rate_limit:
            return False
        
        # Increment counter
        self.rate_limiters[node_id][minute_window] = current_count + 1
        return True
        
    async def _execute_with_retry(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        body: Optional[str],
        timeout: int,
        retry_count: int
    ) -> Dict[str, Any]:
        """Execute HTTP request with retry logic"""
        last_exception = None
        
        for attempt in range(retry_count + 1):
            try:
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                timeout_config = aiohttp.ClientTimeout(total=timeout)
                
                async with self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=body,
                    timeout=timeout_config
                ) as response:
                    # Read response
                    response_text = await response.text()
                    
                    # Try to parse as JSON
                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        response_data = response_text
                    
                    # Check if response is successful
                    if 200 <= response.status < 300:
                        return {
                            'response': response_data,
                            'status': response.status,
                            'headers': dict(response.headers)
                        }
                    else:
                        # Create exception with status code
                        error = Exception(f"HTTP {response.status}: {response_text}")
                        error.status = response.status
                        raise error
                        
            except Exception as e:
                last_exception = e
                logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                
                # Don't retry on certain errors
                if hasattr(e, 'status') and e.status in [400, 401, 403, 404]:
                    break
                
                # Wait before retry (exponential backoff)
                if attempt < retry_count:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
        
        # All retries failed
        raise last_exception or Exception("API call failed after all retries")
        
    def _transform_response(self, response: Any, transformation_code: str) -> Any:
        """Transform API response using JavaScript-like code"""
        if not transformation_code or not transformation_code.strip():
            return response
            
        try:
            # Simple transformation support
            # In a production system, you'd want to use a proper JS engine
            # For now, we'll support basic transformations
            
            # Replace common JavaScript patterns with Python equivalents
            python_code = transformation_code.replace('return ', '')
            python_code = python_code.replace('response.', 'response.')
            
            # Create a safe environment for execution
            safe_globals = {
                'response': response,
                'json': json,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict
            }
            
            # Execute the transformation
            result = eval(python_code, safe_globals)
            return result
            
        except Exception as e:
            logger.warning(f"Response transformation failed: {str(e)}")
            return response
            
    def create_mcp_tool_definition(self, node_config: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        """
        Create MCP tool definition from APINode configuration
        This is used by the MCP integration to register the API as a tool
        """
        if not node_config.get('enable_mcp_tool', True):
            return None
            
        tool_name = f"api_{node_id}"
        tool_description = node_config.get('tool_description', '')
        
        if not tool_description:
            base_url = node_config.get('base_url', '')
            endpoint_path = node_config.get('endpoint_path', '')
            tool_description = f"API call to {base_url}{endpoint_path}"
        
        # Extract parameters from request schema
        request_schema = node_config.get('request_schema', {})
        parameters = request_schema.get('properties', {})
        required = request_schema.get('required', [])
        
        return {
            'name': tool_name,
            'description': tool_description,
            'parameters': {
                'type': 'object',
                'properties': parameters,
                'required': required
            },
            'node_id': node_id,
            'node_config': node_config
        }

# Global instance for reuse
_api_executor = None

async def get_api_executor() -> APIExecutor:
    """Get or create the global API executor instance"""
    global _api_executor
    if _api_executor is None:
        _api_executor = APIExecutor()
        await _api_executor.__aenter__()
    return _api_executor

async def cleanup_api_executor():
    """Clean up the global API executor instance"""
    global _api_executor
    if _api_executor:
        await _api_executor.__aexit__(None, None, None)
        _api_executor = None