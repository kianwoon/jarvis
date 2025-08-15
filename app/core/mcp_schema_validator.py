"""
MCP Schema Validator

Utilities for validating and inspecting MCP tool schemas.
This helps ensure tools are properly configured and their schemas are complete.
"""

import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MCPSchemaValidator:
    """Validates and inspects MCP tool schemas"""
    
    def __init__(self):
        # JSON Schema types as per specification
        self.valid_types = ['string', 'number', 'integer', 'boolean', 'array', 'object', 'null']
        
        # Common schema properties
        self.common_properties = [
            'type', 'description', 'default', 'enum', 'minimum', 'maximum',
            'minLength', 'maxLength', 'pattern', 'items', 'properties', 'required'
        ]
    
    def validate_tool_schema(self, tool_name: str, tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a tool's schema and return validation results.
        
        Args:
            tool_name: Name of the tool
            tool_info: Tool information including inputSchema
            
        Returns:
            Validation results with warnings and errors
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'schema_info': {}
        }
        
        # Check for inputSchema
        input_schema = self._extract_input_schema(tool_info)
        if not input_schema:
            results['errors'].append("No inputSchema found")
            results['valid'] = False
            return results
        
        # Validate schema structure
        if not isinstance(input_schema, dict):
            results['errors'].append("inputSchema must be an object")
            results['valid'] = False
            return results
        
        # Check for required schema fields
        if 'type' not in input_schema:
            results['warnings'].append("Schema missing 'type' field (should be 'object' for parameters)")
        elif input_schema.get('type') != 'object':
            results['warnings'].append(f"Schema type is '{input_schema.get('type')}', expected 'object'")
        
        # Check properties
        properties = input_schema.get('properties', {})
        if not properties:
            results['warnings'].append("No properties defined in schema")
        else:
            results['schema_info']['parameter_count'] = len(properties)
            results['schema_info']['parameters'] = list(properties.keys())
            
            # Validate each property
            for prop_name, prop_schema in properties.items():
                self._validate_property(prop_name, prop_schema, results)
        
        # Check required fields
        required = input_schema.get('required', [])
        if required:
            results['schema_info']['required_parameters'] = required
            # Check that required fields exist in properties
            for req_field in required:
                if req_field not in properties:
                    results['errors'].append(f"Required field '{req_field}' not defined in properties")
                    results['valid'] = False
        
        # Add suggestions for common patterns
        self._add_suggestions(tool_name, input_schema, results)
        
        return results
    
    def _extract_input_schema(self, tool_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract inputSchema from various possible locations"""
        # Direct inputSchema field
        if 'inputSchema' in tool_info:
            return tool_info['inputSchema']
        
        # In parameters field (common in database storage)
        if 'parameters' in tool_info:
            params = tool_info['parameters']
            if isinstance(params, dict):
                # Check if it's already a schema
                if 'properties' in params or 'type' in params:
                    return params
                # Check if schema is nested
                if 'inputSchema' in params:
                    return params['inputSchema']
        
        return None
    
    def _validate_property(self, name: str, schema: Dict[str, Any], results: Dict[str, Any]):
        """Validate a single property schema"""
        if not isinstance(schema, dict):
            results['errors'].append(f"Property '{name}' schema must be an object")
            return
        
        # Check type
        prop_type = schema.get('type')
        if not prop_type:
            results['warnings'].append(f"Property '{name}' missing 'type' field")
        elif prop_type not in self.valid_types:
            results['errors'].append(f"Property '{name}' has invalid type '{prop_type}'")
        
        # Check description
        if 'description' not in schema:
            results['suggestions'].append(f"Consider adding a description for property '{name}'")
        
        # Type-specific validations
        if prop_type == 'string':
            if 'enum' in schema:
                enum_values = schema['enum']
                if not isinstance(enum_values, list) or not enum_values:
                    results['errors'].append(f"Property '{name}' enum must be a non-empty array")
        
        elif prop_type == 'integer' or prop_type == 'number':
            if 'minimum' in schema and 'maximum' in schema:
                if schema['minimum'] > schema['maximum']:
                    results['errors'].append(f"Property '{name}' minimum > maximum")
        
        elif prop_type == 'array':
            if 'items' not in schema:
                results['warnings'].append(f"Array property '{name}' should define 'items' schema")
    
    def _add_suggestions(self, tool_name: str, schema: Dict[str, Any], results: Dict[str, Any]):
        """Add suggestions for improving the schema"""
        properties = schema.get('properties', {})
        
        # Check for search tools
        if 'search' in tool_name.lower() or 'find' in tool_name.lower():
            # Check for temporal parameters
            has_date_param = any(
                'date' in prop.lower() or 'time' in prop.lower() 
                for prop in properties.keys()
            )
            if not has_date_param:
                results['suggestions'].append(
                    "Search tool could benefit from date filtering parameters (e.g., dateRestrict, dateRange)"
                )
            
            # Check for sort parameter
            has_sort_param = any(
                'sort' in prop.lower() or 'order' in prop.lower()
                for prop in properties.keys()
            )
            if not has_sort_param:
                results['suggestions'].append(
                    "Search tool could benefit from sorting parameters (e.g., sort, orderBy)"
                )
        
        # Check for pagination
        if len(properties) > 0:
            has_pagination = any(
                'limit' in prop.lower() or 'count' in prop.lower() or 'max' in prop.lower()
                for prop in properties.keys()
            )
            if not has_pagination and 'list' in tool_name.lower() or 'search' in tool_name.lower():
                results['suggestions'].append(
                    "Consider adding pagination parameters (e.g., limit, max_results)"
                )
    
    def generate_schema_report(self, tool_name: str, tool_info: Dict[str, Any]) -> str:
        """Generate a human-readable schema report"""
        validation = self.validate_tool_schema(tool_name, tool_info)
        
        report = []
        report.append(f"Schema Validation Report for: {tool_name}")
        report.append("=" * 50)
        
        # Status
        status = "✓ VALID" if validation['valid'] else "✗ INVALID"
        report.append(f"Status: {status}")
        report.append("")
        
        # Schema info
        if validation['schema_info']:
            report.append("Schema Information:")
            for key, value in validation['schema_info'].items():
                report.append(f"  {key}: {value}")
            report.append("")
        
        # Errors
        if validation['errors']:
            report.append("Errors:")
            for error in validation['errors']:
                report.append(f"  ✗ {error}")
            report.append("")
        
        # Warnings
        if validation['warnings']:
            report.append("Warnings:")
            for warning in validation['warnings']:
                report.append(f"  ⚠ {warning}")
            report.append("")
        
        # Suggestions
        if validation['suggestions']:
            report.append("Suggestions:")
            for suggestion in validation['suggestions']:
                report.append(f"  💡 {suggestion}")
        
        return "\n".join(report)


def validate_all_tools() -> Dict[str, Any]:
    """Validate all registered MCP tools"""
    try:
        from app.core.mcp_tools_cache import get_enabled_mcp_tools
        
        validator = MCPSchemaValidator()
        tools = get_enabled_mcp_tools()
        
        results = {
            'total_tools': len(tools),
            'valid_tools': 0,
            'invalid_tools': 0,
            'tools_with_warnings': 0,
            'detailed_results': {}
        }
        
        for tool_name, tool_info in tools.items():
            validation = validator.validate_tool_schema(tool_name, tool_info)
            results['detailed_results'][tool_name] = validation
            
            if validation['valid']:
                results['valid_tools'] += 1
            else:
                results['invalid_tools'] += 1
            
            if validation['warnings']:
                results['tools_with_warnings'] += 1
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to validate tools: {e}")
        return {'error': str(e)}


def print_validation_summary():
    """Print a summary of all tool validations"""
    results = validate_all_tools()
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print("\nMCP Tool Schema Validation Summary")
    print("=" * 50)
    print(f"Total tools: {results['total_tools']}")
    print(f"Valid schemas: {results['valid_tools']}")
    print(f"Invalid schemas: {results['invalid_tools']}")
    print(f"Tools with warnings: {results['tools_with_warnings']}")
    print("")
    
    # Show details for invalid tools
    if results['invalid_tools'] > 0:
        print("Invalid Tools:")
        for tool_name, validation in results['detailed_results'].items():
            if not validation['valid']:
                print(f"  ✗ {tool_name}:")
                for error in validation['errors']:
                    print(f"    - {error}")
        print("")
    
    # Show tools with warnings
    if results['tools_with_warnings'] > 0:
        print("Tools with Warnings:")
        for tool_name, validation in results['detailed_results'].items():
            if validation['warnings']:
                print(f"  ⚠ {tool_name}:")
                for warning in validation['warnings']:
                    print(f"    - {warning}")


# Global validator instance
mcp_schema_validator = MCPSchemaValidator()


if __name__ == "__main__":
    # Run validation on all tools when script is executed directly
    print_validation_summary()