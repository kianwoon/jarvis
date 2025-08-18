"""
MCP Parameter Injector

A proper MCP-compliant solution for dynamically injecting parameters based on 
the tool's inputSchema. This module inspects the tool's schema to determine
what parameters it accepts and only adds parameters that are actually supported.

Key features:
- Dynamic parameter injection based on tool's inputSchema
- Support for temporal parameters (dateRestrict, sort) for search tools
- Works with any MCP tool regardless of name
- Schema-driven approach following MCP protocol
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MCPParameterInjector:
    """
    Dynamically injects parameters into MCP tool calls based on the tool's inputSchema.
    This follows the MCP protocol specification where tools declare their capabilities
    through their inputSchema.
    """
    
    def __init__(self):
        # MCP Parameter Injector is now purely schema-driven
        # No hardcoded mappings - all capabilities come from tool's inputSchema
        pass
    
    def inject_parameters(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any], 
        tool_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main entry point for parameter injection.
        
        Args:
            tool_name: Name of the MCP tool
            parameters: Original parameters from the user/LLM
            tool_info: Tool information including inputSchema
            
        Returns:
            Enhanced parameters with injected values based on schema
        """
        try:
            # Get the tool's inputSchema
            input_schema = self._get_input_schema(tool_info)
            if not input_schema:
                logger.debug(f"No inputSchema found for tool {tool_name}, returning original parameters")
                return parameters
            
            # Create a copy to avoid modifying the original
            enhanced_params = parameters.copy() if parameters else {}
            
            # ALWAYS inject temporal parameters for fresh results
            # We want the latest, most recent information regardless of query keywords
            enhanced_params = self._inject_temporal_parameters(
                tool_name, enhanced_params, input_schema
            )
            
            # Add other smart parameter injections based on schema
            enhanced_params = self._inject_smart_defaults(
                tool_name, enhanced_params, input_schema
            )
            
            return enhanced_params
            
        except Exception as e:
            logger.error(f"Error injecting parameters for {tool_name}: {e}")
            return parameters
    
    def _get_input_schema(self, tool_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract the inputSchema from tool information.
        The schema can be in different locations depending on how the tool was registered.
        """
        # First check if there's a direct inputSchema field
        if 'inputSchema' in tool_info:
            return tool_info['inputSchema']
        
        # Check the parameters field (where MCPTool stores inputSchema)
        if 'parameters' in tool_info:
            params = tool_info['parameters']
            # If parameters is already a schema-like dict with properties
            if isinstance(params, dict):
                if 'properties' in params or 'type' in params:
                    return params
                # Sometimes the schema is nested
                if 'inputSchema' in params:
                    return params['inputSchema']
        
        return None
    
    
    def _find_query_parameter(self, parameters: Dict[str, Any]) -> Optional[str]:
        """
        Find the query parameter in the parameters dict.
        Uses common parameter names without hardcoded mappings.
        """
        # Common query parameter names to check
        query_params = ['query', 'q', 'search', 'searchQuery', 'search_query', 'text']
        for param_name in query_params:
            if param_name in parameters:
                value = parameters[param_name]
                return str(value) if value else None
        return None
    
    def _inject_temporal_parameters(
        self, 
        tool_name: str,
        parameters: Dict[str, Any], 
        input_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        ALWAYS inject temporal parameters to ensure fresh, recent results.
        Uses ONLY the tool's inputSchema to determine what parameters are supported.
        No hardcoded mappings or keyword checking - purely schema-driven.
        """
        # Get the schema properties
        properties = input_schema.get('properties', {})
        
        # ALWAYS add date restriction for fresh results if tool supports it
        if 'date_restrict' in properties and 'date_restrict' not in parameters:
            # Default to last 6 months for general freshness
            parameters['date_restrict'] = 'm6'
            logger.info(f"[MCP Parameter Injector] ALWAYS adding date_restrict='m6' to {tool_name} for fresh results")
        elif 'dateRestrict' in properties and 'dateRestrict' not in parameters:
            # Handle alternative naming
            parameters['dateRestrict'] = 'm6'
            logger.info(f"[MCP Parameter Injector] ALWAYS adding dateRestrict='m6' to {tool_name} for fresh results")
        
        # ALWAYS enable sort_by_date for chronological ordering if tool supports it
        if 'sort_by_date' in properties and 'sort_by_date' not in parameters:
            # Check if the parameter type is boolean
            param_schema = properties.get('sort_by_date', {})
            if param_schema.get('type') == 'boolean':
                # ALWAYS enable date sorting for fresh results first
                parameters['sort_by_date'] = True
                logger.info(f"[MCP Parameter Injector] ALWAYS adding sort_by_date=True to {tool_name} for chronological ordering")
        
        # ALWAYS set sort parameter to date if available
        if 'sort' in properties and 'sort' not in parameters:
            # Always prefer date sorting for fresh results
            sort_value = self._determine_sort_value(properties.get('sort', {}))
            if sort_value:
                parameters['sort'] = sort_value
                logger.info(f"[MCP Parameter Injector] ALWAYS adding sort='{sort_value}' to {tool_name} for fresh results")
        
        return parameters
    
    def _find_schema_parameter(self, properties: Dict[str, Any], parameter_name: str) -> bool:
        """
        Check if a specific parameter exists in the schema properties.
        """
        return parameter_name in properties
    
    def _get_default_date_restriction(self) -> str:
        """
        Get the default date restriction for fresh results.
        Always returns a sensible default for getting recent information.
        """
        # Default to last 6 months for general freshness
        return 'm6'
    
    def _determine_sort_value(self, sort_schema: Dict[str, Any]) -> Optional[str]:
        """
        Determine the appropriate sort value based on the schema.
        Checks if 'date' is a valid enum value or returns appropriate default.
        """
        # Check if schema defines enum values
        enum_values = sort_schema.get('enum', [])
        if enum_values:
            # Prefer date-based sorting for temporal queries
            for value in ['date', 'Date', 'DATE', 'recent', 'newest']:
                if value in enum_values:
                    return value
            # Fall back to relevance if date sorting not available
            for value in ['relevance', 'Relevance', 'RELEVANCE']:
                if value in enum_values:
                    return value
        
        # If no enum constraints, use common sort values
        schema_type = sort_schema.get('type', 'string')
        if schema_type == 'string':
            return 'date'  # Default to date sorting for temporal queries
        
        return None
    
    def _inject_smart_defaults(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        input_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Inject other smart defaults based on the schema and query context.
        """
        properties = input_schema.get('properties', {})
        
        # Check for common result count parameters if they exist in schema
        count_params = ['num_results', 'numResults', 'count', 'limit', 'max_results', 'maxResults', 'size']
        for count_param in count_params:
            if count_param in properties and count_param not in parameters:
                # Get the maximum allowed value from schema
                count_schema = properties.get(count_param, {})
                max_value = count_schema.get('maximum', 10)
                default_value = count_schema.get('default', 5)
                
                # Use a reasonable default for better results
                parameters[count_param] = min(10, max_value)
                logger.info(f"[MCP Parameter Injector] Set {count_param}={parameters[count_param]} for comprehensive results")
                break  # Only set one count parameter
        
        return parameters
    
    
    def get_tool_capabilities(self, tool_info: Dict[str, Any]) -> Dict[str, bool]:
        """
        Analyze a tool's inputSchema to determine its capabilities.
        Useful for understanding what parameters a tool supports.
        
        Returns:
            Dict with capability flags like supports_date_filtering, supports_sorting, etc.
        """
        capabilities = {
            'supports_date_filtering': False,
            'supports_sorting': False,
            'supports_sort_by_date': False,  # Added specific capability for boolean date sorting
            'supports_pagination': False,
            'supports_filtering': False,
            'accepts_query': False
        }
        
        input_schema = self._get_input_schema(tool_info)
        if not input_schema:
            return capabilities
        
        properties = input_schema.get('properties', {})
        
        # Check for date filtering support
        if 'date_restrict' in properties or 'dateRestrict' in properties:
            capabilities['supports_date_filtering'] = True
        
        # Check for sorting support
        if 'sort' in properties:
            capabilities['supports_sorting'] = True
        
        # Check for sort_by_date boolean parameter support
        if 'sort_by_date' in properties:
            capabilities['supports_sort_by_date'] = True
        
        # Check for query support
        if any(param in properties for param in ['query', 'q', 'search', 'searchQuery', 'search_query', 'text']):
            capabilities['accepts_query'] = True
        
        # Check for pagination support
        if any(param in properties for param in ['num_results', 'numResults', 'count', 'limit', 'max_results', 'maxResults', 'size']):
            capabilities['supports_pagination'] = True
        
        # Check for filtering support (generic)
        filter_params = ['filter', 'filters', 'where', 'criteria']
        if any(param in properties for param in filter_params):
            capabilities['supports_filtering'] = True
        
        return capabilities


# Global instance for easy access
mcp_parameter_injector = MCPParameterInjector()


def inject_mcp_parameters(
    tool_name: str,
    parameters: Dict[str, Any],
    tool_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to inject parameters using the global injector.
    
    Args:
        tool_name: Name of the MCP tool
        parameters: Original parameters
        tool_info: Tool information including inputSchema
        
    Returns:
        Enhanced parameters with injected values
    """
    return mcp_parameter_injector.inject_parameters(tool_name, parameters, tool_info)


def analyze_tool_capabilities(tool_info: Dict[str, Any]) -> Dict[str, bool]:
    """
    Convenience function to analyze tool capabilities.
    
    Args:
        tool_info: Tool information including inputSchema
        
    Returns:
        Dict with capability flags
    """
    return mcp_parameter_injector.get_tool_capabilities(tool_info)