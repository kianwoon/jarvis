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
        # Common temporal keywords that indicate a query needs recent information
        self.temporal_keywords = [
            'latest', 'recent', 'current', 'newest', 'today', 'now',
            'this week', 'this month', 'this year', 'update', 'news',
            'breaking', 'trending', 'new', 'fresh', 'live'
        ]
        
        # Mapping of common parameter names across different tools
        # This helps us identify similar parameters even if named differently
        self.parameter_mappings = {
            # Date restriction parameters
            'date_restrict': ['dateRestrict', 'date_restrict', 'dateRange', 'date_range', 'timeRange', 'time_range'],
            # Sort parameters
            'sort': ['sort', 'sortBy', 'sort_by', 'orderBy', 'order_by', 'sortOrder', 'sort_order'],
            # Query parameters
            'query': ['query', 'q', 'search', 'searchQuery', 'search_query', 'text'],
            # Result count parameters
            'count': ['num_results', 'numResults', 'count', 'limit', 'max_results', 'maxResults', 'size']
        }
    
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
            
            # Check if this query needs temporal enhancement
            if self._needs_temporal_enhancement(enhanced_params):
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
    
    def _needs_temporal_enhancement(self, parameters: Dict[str, Any]) -> bool:
        """
        Determine if the query needs temporal parameters based on the query text.
        """
        # Check if there's a query parameter
        query = self._find_query_parameter(parameters)
        if not query:
            return False
        
        query_lower = query.lower()
        
        # Check for temporal keywords
        return any(keyword in query_lower for keyword in self.temporal_keywords)
    
    def _find_query_parameter(self, parameters: Dict[str, Any]) -> Optional[str]:
        """
        Find the query parameter in the parameters dict.
        Handles different naming conventions.
        """
        for param_name in self.parameter_mappings['query']:
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
        Inject temporal parameters if the tool's schema supports them.
        """
        # Get the schema properties
        properties = input_schema.get('properties', {})
        
        # Check for date restriction parameter support
        date_param = self._find_schema_parameter(properties, self.parameter_mappings['date_restrict'])
        if date_param and date_param not in parameters:
            # Analyze the query to determine appropriate date range
            query = self._find_query_parameter(parameters)
            date_value = self._determine_date_restriction(query)
            if date_value:
                parameters[date_param] = date_value
                logger.info(f"[MCP Parameter Injector] Added {date_param}='{date_value}' to {tool_name} based on schema")
        
        # Check for sort parameter support
        sort_param = self._find_schema_parameter(properties, self.parameter_mappings['sort'])
        if sort_param and sort_param not in parameters:
            # For temporal queries, sort by date/relevance
            sort_value = self._determine_sort_value(properties.get(sort_param, {}))
            if sort_value:
                parameters[sort_param] = sort_value
                logger.info(f"[MCP Parameter Injector] Added {sort_param}='{sort_value}' to {tool_name} based on schema")
        
        return parameters
    
    def _find_schema_parameter(self, properties: Dict[str, Any], possible_names: list) -> Optional[str]:
        """
        Find if any of the possible parameter names exist in the schema properties.
        Returns the actual parameter name used in the schema.
        """
        for name in possible_names:
            if name in properties:
                return name
        return None
    
    def _determine_date_restriction(self, query: Optional[str]) -> Optional[str]:
        """
        Determine the appropriate date restriction based on the query.
        Returns values compatible with Google's dateRestrict format.
        """
        if not query:
            return None
        
        query_lower = query.lower()
        
        # Very recent (last day)
        if any(term in query_lower for term in ['today', 'breaking', 'just', 'now']):
            return 'd1'
        
        # Last week
        if any(term in query_lower for term in ['this week', 'past week', 'recent']):
            return 'w1'
        
        # Last month
        if any(term in query_lower for term in ['this month', 'past month', 'latest']):
            return 'm1'
        
        # Last 3 months
        if any(term in query_lower for term in ['recent months', 'quarterly']):
            return 'm3'
        
        # Last 6 months
        if any(term in query_lower for term in ['past six months', 'half year']):
            return 'm6'
        
        # Last year
        if any(term in query_lower for term in ['this year', 'past year', 'current year']):
            return 'y1'
        
        # Default for general temporal queries
        if any(term in query_lower for term in self.temporal_keywords):
            return 'w1'  # Default to last week for general "recent" queries
        
        return None
    
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
        
        # Check for result count parameter
        count_param = self._find_schema_parameter(properties, self.parameter_mappings['count'])
        if count_param and count_param not in parameters:
            # Determine if we need more results for comprehensive queries
            query = self._find_query_parameter(parameters)
            if query and self._is_comprehensive_query(query):
                # Get the maximum allowed value from schema
                count_schema = properties.get(count_param, {})
                max_value = count_schema.get('maximum', 10)
                default_value = count_schema.get('default', 5)
                
                # Use a higher value for comprehensive queries
                parameters[count_param] = min(10, max_value)
                logger.info(f"[MCP Parameter Injector] Set {count_param}={parameters[count_param]} for comprehensive query")
        
        return parameters
    
    def _is_comprehensive_query(self, query: str) -> bool:
        """
        Determine if a query requires comprehensive results.
        """
        comprehensive_keywords = [
            'comprehensive', 'detailed', 'complete', 'full', 'all',
            'everything', 'thorough', 'extensive', 'in-depth'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in comprehensive_keywords)
    
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
            'supports_pagination': False,
            'supports_filtering': False,
            'accepts_query': False
        }
        
        input_schema = self._get_input_schema(tool_info)
        if not input_schema:
            return capabilities
        
        properties = input_schema.get('properties', {})
        
        # Check for date filtering support
        if self._find_schema_parameter(properties, self.parameter_mappings['date_restrict']):
            capabilities['supports_date_filtering'] = True
        
        # Check for sorting support
        if self._find_schema_parameter(properties, self.parameter_mappings['sort']):
            capabilities['supports_sorting'] = True
        
        # Check for query support
        if self._find_schema_parameter(properties, self.parameter_mappings['query']):
            capabilities['accepts_query'] = True
        
        # Check for pagination support
        if self._find_schema_parameter(properties, self.parameter_mappings['count']):
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