import React, { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Switch,
  FormControlLabel,
  Typography,
  Chip,
  Button,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Paper,
  Divider,
  Alert,
  Grid,
  Tabs,
  Tab,
  Card,
  CardContent,
  CardHeader,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  CircularProgress,
  Tooltip,
  Radio,
  RadioGroup
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Refresh as RefreshIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Cached as CacheIcon,
  CheckCircle as CheckCircleIcon,
  HelpOutline as HelpOutlineIcon
} from '@mui/icons-material';
import YamlEditor from './YamlEditor';
import DatabaseTableManager from './DatabaseTableManager';
import MCPServerManager from './MCPServerManager';
import MCPToolManager from './MCPToolManager';
import VectorDatabaseManager from './VectorDatabaseManager';
import KnowledgeGraphSettings from './KnowledgeGraphSettings';
import PromptManagement from './PromptManagement';

interface SettingsFormRendererProps {
  category: string;
  data: any;
  onChange: (field: string, value: any) => void;
  onRefresh?: () => void;
  isYamlBased?: boolean;
  onShowSuccess?: (message?: string) => void;
}

const SettingsFormRenderer: React.FC<SettingsFormRendererProps> = ({
  category,
  data,
  onChange,
  onRefresh,
  isYamlBased = false,
  onShowSuccess
}) => {
  console.log('[DEBUG] SettingsFormRenderer - category:', category, 'data:', data);
  if (category === 'knowledge_graph') {
    console.log('[DEBUG] SettingsFormRenderer - KG data.model_config:', data?.model_config);
    console.log('[DEBUG] SettingsFormRenderer - KG data keys:', Object.keys(data || {}));
  }
  // Move all hooks to the top level - they must always be called in the same order
  const [activeTab, setActiveTab] = React.useState(() => {
    if (category === 'rag') return 'retrieval';
    if (category === 'storage') return 'vector';
    if (category === 'overflow') return 'thresholds';
    return 'settings';
  });
  const [passwordVisibility, setPasswordVisibility] = React.useState<Record<string, boolean>>({});
  const [icebergPasswordVisibility, setIcebergPasswordVisibility] = React.useState<Record<string, boolean>>({});
  const [mcpTab, setMcpTab] = React.useState(0);
  const [testingConnection, setTestingConnection] = React.useState(false);
  const [connectionTestResult, setConnectionTestResult] = React.useState<{success: boolean, message?: string, error?: string} | null>(null);
  
  // Update activeTab when category changes
  React.useEffect(() => {
    if (category === 'rag') setActiveTab('retrieval');
    else if (category === 'storage') setActiveTab('vector');
    else if (category === 'overflow') setActiveTab('thresholds');
    else setActiveTab('settings');
  }, [category]);

  // Test Neo4j connection
  const testNeo4jConnection = async () => {
    setTestingConnection(true);
    setConnectionTestResult(null);
    
    try {
      const response = await fetch('/api/v1/settings/knowledge-graph/test-connection', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      const result = await response.json();
      setConnectionTestResult(result);
      
      if (result.success && onShowSuccess) {
        onShowSuccess('Neo4j connection test successful');
      }
    } catch (error) {
      console.error('Neo4j connection test failed:', error);
      setConnectionTestResult({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      });
    } finally {
      setTestingConnection(false);
    }
  };

  // Special handling for YAML-based configurations
  if (isYamlBased || category === 'self_reflection' || category === 'query_patterns') {
    const yamlValue = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
    
    return (
      <YamlEditor
        value={yamlValue}
        onChange={(value) => {
          try {
            // Try to parse as JSON first, then as YAML structure
            const parsed = JSON.parse(value);
            onChange('config', parsed);
          } catch {
            // If JSON parsing fails, store as string (raw YAML)
            onChange('config', value);
          }
        }}
        label={`${category.replace(/_/g, ' ')} Configuration`}
      />
    );
  }

  // Special handling for environment variables - use card-based layout
  if (category === 'environment') {
    // Convert environment_variables to flat structure for card rendering
    const envVars = data?.environment_variables || {};
    const flattenedFields = Object.entries(envVars).map(([key, value]) => ({ key, value }));
    
    if (flattenedFields.length === 0) {
      // Show the old editor if no variables exist, so user can add new ones
      return renderEnvironmentEditor(data, onChange);
    }
    
    // Use the new card-based layout for existing variables
    return renderEnvironmentFieldsWithCards(flattenedFields, 'settings', (field, value) => {
      // Handle both regular field updates and deletions
      if (value === undefined) {
        // Delete the environment variable
        const updated = { ...envVars };
        delete updated[field];
        onChange('environment_variables', updated);
      } else {
        // Update the environment variable
        onChange('environment_variables', {
          ...envVars,
          [field]: value
        });
      }
    }, onShowSuccess, (key, value, depth, customOnChange, fieldCategory, onShowSuccessCallback) => {
      // Simple field renderer for environment variables
      return (
        <Box key={key} sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <Box sx={{ flex: 1 }}>
            <TextField
              label={key}
              value={value || ''}
              onChange={(e) => customOnChange?.(key, e.target.value)}
              fullWidth
              variant="outlined"
              size="small"
              type={key.toLowerCase().includes('password') || key.toLowerCase().includes('secret') || key.toLowerCase().includes('key') ? 'password' : 'text'}
            />
          </Box>
          <IconButton
            size="small"
            onClick={() => {
              if (confirm(`Delete environment variable "${key}"?`)) {
                customOnChange?.(key, undefined);
              }
            }}
            sx={{ color: 'error.main' }}
          >
            <DeleteIcon />
          </IconButton>
        </Box>
      );
    });
  }

  // Special handling for MCP configuration
  if (category === 'mcp') {
    const mcpData = data || { servers: [], tools: [], settings: {} };
    return renderMCPConfiguration(mcpData, onChange, onRefresh, mcpTab, setMcpTab);
  }

  // Special handling for database-backed settings
  if (category === 'collection_registry') {
    const records = Array.isArray(data) ? data : data?.collections || [];
    return (
      <DatabaseTableManager
        category={category as 'collection_registry' | 'langgraph_agents'}
        data={records}
        onChange={(newData) => onChange('records', newData)}
        onRefresh={onRefresh || (() => {})}
      />
    );
  }

  // Special handling for LangGraph agents
  if (category === 'langgraph_agents') {
    const records = Array.isArray(data) ? data : data?.agents || [];
    return (
      <DatabaseTableManager
        category={category as 'collection_registry' | 'langgraph_agents'}
        data={records}
        onChange={(newData) => onChange('agents', newData)}
        onRefresh={onRefresh || (() => {})}
      />
    );
  }

  // Special handling for timeout configuration
  if (category === 'timeout') {
    return <TimeoutConfiguration data={data} onChange={onChange} onShowSuccess={onShowSuccess} />;
  }

  // Knowledge graph settings are now consolidated under LLM category

  // Default form rendering for regular settings
  return renderStandardForm(data, onChange, category, activeTab, setActiveTab, passwordVisibility, setPasswordVisibility, icebergPasswordVisibility, setIcebergPasswordVisibility, onShowSuccess, testNeo4jConnection, testingConnection, connectionTestResult);
};

const renderEnvironmentEditor = (data: any, onChange: (field: string, value: any) => void) => {
  const envVars = data?.environment_variables || {};
  
  const addEnvVar = () => {
    const newVar = prompt('Enter environment variable name:');
    if (newVar) {
      onChange('environment_variables', {
        ...envVars,
        [newVar]: ''
      });
    }
  };

  const updateEnvVar = (key: string, value: string) => {
    onChange('environment_variables', {
      ...envVars,
      [key]: value
    });
  };

  const deleteEnvVar = (key: string) => {
    const updated = { ...envVars };
    delete updated[key];
    onChange('environment_variables', updated);
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Environment Variables</Typography>
        <Button startIcon={<AddIcon />} onClick={addEnvVar}>
          Add Variable
        </Button>
      </Box>
      
      <Paper variant="outlined" sx={{ p: 2 }}>
        {Object.keys(envVars).length === 0 ? (
          <Alert severity="info">No environment variables configured</Alert>
        ) : (
          <List>
            {Object.entries(envVars).map(([key, value], index) => (
              <React.Fragment key={key}>
                <ListItem>
                  <Box sx={{ display: 'flex', gap: 2, width: '100%', alignItems: 'center' }}>
                    <TextField
                      label="Variable Name"
                      value={key}
                      disabled
                      size="small"
                      sx={{ minWidth: 200 }}
                    />
                    <TextField
                      label="Value"
                      value={value as string}
                      onChange={(e) => updateEnvVar(key, e.target.value)}
                      size="small"
                      fullWidth
                      type={key.includes('PASSWORD') || key.includes('SECRET') || key.includes('KEY') ? 'password' : 'text'}
                    />
                    <IconButton 
                      onClick={() => deleteEnvVar(key)}
                      color="error"
                      size="small"
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </ListItem>
                {index < Object.keys(envVars).length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        )}
      </Paper>
    </Box>
  );
};

const renderMCPConfiguration = (
  data: any, 
  onChange: (field: string, value: any) => void, 
  onRefresh: (() => void) | undefined,
  mcpTab: number,
  setMcpTab: React.Dispatch<React.SetStateAction<number>>
) => {

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setMcpTab(newValue);
  };

  const loadMCPData = async (type: 'servers' | 'tools') => {
    try {
      const endpoint = type === 'servers' ? '/api/v1/mcp/servers/' : '/api/v1/mcp/tools/';
      const response = await fetch(endpoint);
      if (response.ok) {
        const result = await response.json();
        //console.log(`Loaded MCP ${type}:`, result.length, 'items');
        onChange(type, Array.isArray(result) ? result : result.data || []);
      } else {
        //console.error(`Failed to load MCP ${type}: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      //console.error(`Error loading MCP ${type}:`, error);
    }
  };

  return (
    <Box>
      <Box sx={{ 
        borderBottom: 1, 
        borderColor: 'divider', 
        mb: 2,
        position: 'sticky',
        top: 0,
        bgcolor: 'background.paper',
        zIndex: 10,
        pt: 1,
        mt: -1
      }}>
        <Tabs value={mcpTab} onChange={handleTabChange} aria-label="mcp tabs">
          <Tab label="MCP Servers" />
          <Tab label="MCP Tools" />
          <Tab label="Tool Configuration" />
          <Tab label="Cache Management" />
        </Tabs>
      </Box>

      {mcpTab === 0 && (
        <MCPServerManager
          data={data.servers || []}
          tools={data.tools || []}
          onChange={(servers) => onChange('servers', servers)}
          onRefresh={() => {
            loadMCPData('servers');
            loadMCPData('tools');
            if (onRefresh) onRefresh();
          }}
        />
      )}

      {mcpTab === 1 && (
        <MCPToolManager
          data={data.tools || []}
          servers={data.servers || []}
          onChange={(tools) => onChange('tools', tools)}
          onRefresh={() => {
            loadMCPData('tools');
            if (onRefresh) onRefresh();
          }}
        />
      )}

      {mcpTab === 2 && (
        <Box>
          <Typography variant="h6" gutterBottom>Tool Execution Configuration</Typography>
          
          <Card>
            <CardHeader 
              title="Tool Call Limits" 
              subheader="Configure limits to prevent excessive tool calls and infinite loops"
            />
            <CardContent>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Maximum Tool Calls per Request"
                    type="number"
                    value={data.settings?.max_tool_calls || 3}
                    onChange={(e) => {
                      const newSettings = { ...data.settings, max_tool_calls: parseInt(e.target.value) || 3 };
                      onChange('settings', newSettings);
                    }}
                    helperText="Maximum number of tool calls allowed in a single request (prevents infinite loops)"
                    fullWidth
                    inputProps={{ min: 1, max: 10 }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Tool Timeout (seconds)"
                    type="number"
                    value={data.settings?.tool_timeout_seconds || 30}
                    onChange={(e) => {
                      const newSettings = { ...data.settings, tool_timeout_seconds: parseInt(e.target.value) || 30 };
                      onChange('settings', newSettings);
                    }}
                    helperText="Maximum time to wait for each tool execution"
                    fullWidth
                    inputProps={{ min: 5, max: 300 }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={data.settings?.enable_tool_retries !== false}
                        onChange={(e) => {
                          const newSettings = { ...data.settings, enable_tool_retries: e.target.checked };
                          onChange('settings', newSettings);
                        }}
                      />
                    }
                    label="Enable Tool Retries"
                  />
                  <Typography variant="caption" display="block" color="textSecondary">
                    Allow automatic retries for failed tool calls
                  </Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Maximum Retries per Tool"
                    type="number"
                    value={data.settings?.max_tool_retries || 2}
                    onChange={(e) => {
                      const newSettings = { ...data.settings, max_tool_retries: parseInt(e.target.value) || 2 };
                      onChange('settings', newSettings);
                    }}
                    helperText="Maximum number of retry attempts for each tool"
                    fullWidth
                    disabled={data.settings?.enable_tool_retries === false}
                    inputProps={{ min: 0, max: 5 }}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Box>
      )}

      {mcpTab === 3 && (
        <Box>
          <Typography variant="h6" gutterBottom>Cache Management</Typography>
          
          {/* Database to Cache Sync Section */}
          <Card>
            <CardHeader 
              title="Redis Cache Synchronization" 
              subheader="Manage the Redis cache used by MCP tools for improved runtime performance"
            />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<CacheIcon />}
                    onClick={async () => {
                      try {
                        const response = await fetch('/api/v1/mcp/tools/cache/reload', { method: 'POST' });
                        const result = await response.json();
                        alert(`Tools cache synced successfully! ${result.tools_count || 0} tools loaded to Redis.`);
                      } catch (error) {
                        alert('Failed to sync tools cache');
                      }
                    }}
                  >
                    Sync Tools to Cache
                  </Button>
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    Updates the Redis cache with all active MCP tools from the database for faster runtime access
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Information Alert */}
          <Alert severity="info" sx={{ mt: 3 }}>
            <Typography variant="body2">
              <strong>About Cache Synchronization:</strong>
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              The Redis cache stores MCP tool definitions for fast access during chat operations. 
              Use this button after making changes to tools to ensure the cache is up-to-date.
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              To refresh the data shown in the UI, use the "Refresh Data" buttons on the respective Tools and Servers pages.
            </Typography>
          </Alert>
        </Box>
      )}
    </Box>
  );
};

const renderDatabaseTableEditor = (category: string, data: any, onChange: (field: string, value: any) => void) => {
  const records = Array.isArray(data) ? data : data?.records || [];
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          {category === 'automation' ? 'Automation Workflows' : 'Collection Registry'}
        </Typography>
        <Button startIcon={<AddIcon />}>
          Add {category === 'automation' ? 'Workflow' : 'Collection'}
        </Button>
      </Box>
      
      <Paper variant="outlined" sx={{ p: 2 }}>
        {records.length === 0 ? (
          <Alert severity="info">
            No {category === 'automation' ? 'workflows' : 'collections'} configured
          </Alert>
        ) : (
          <List>
            {records.map((record: any, index: number) => (
              <React.Fragment key={record.id || index}>
                <ListItem>
                  <ListItemText
                    primary={record.name || record.title || `Item ${index + 1}`}
                    secondary={record.description || record.status || 'No description'}
                  />
                  <ListItemSecondaryAction>
                    <IconButton size="small">
                      <EditIcon />
                    </IconButton>
                    <IconButton size="small" color="error">
                      <DeleteIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
                {index < records.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        )}
      </Paper>
    </Box>
  );
};

// Model Selector Component - proper React component to avoid hooks rule violations
const ModelSelector: React.FC<{
  fieldKey: string;
  value: string;
  onChangeHandler: (key: string, value: any) => void;
  depth: number;
  onShowSuccess?: (message?: string) => void;
  customOnChange?: (field: string, value: any) => void;
}> = ({ fieldKey, value, onChangeHandler, depth, onShowSuccess, customOnChange }) => {
  const [models, setModels] = React.useState<Array<{name: string, id: string, size: string, modified: string, context_length: string}>>([]);
  const [loading, setLoading] = React.useState(false);

  React.useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    setLoading(true);
    try {
      // Call API to get available models
      const response = await fetch('/api/v1/ollama/models');
      if (response.ok) {
        const data = await response.json();
        
        if (data.success) {
          // Ollama is available - use actual models
          setModels(data.models || []);
        } else {
          // Ollama not available - use fallback models from API if provided
          if (data.fallback_models && data.fallback_models.length > 0) {
            setModels(data.fallback_models);
          } else {
            // Last resort fallback
            setModels([
              { name: 'llama3.1:8b', id: 'fallback-01', size: '4.7 GB', modified: 'N/A', context_length: '128,000' },
              { name: 'deepseek-r1:8b', id: 'fallback-04', size: '4.9 GB', modified: 'N/A', context_length: '65,536' }
            ]);
          }
        }
      } else {
        // HTTP error - try to parse response for fallback models
        try {
          const data = await response.json();
          if (data.fallback_models && data.fallback_models.length > 0) {
            setModels(data.fallback_models);
          } else {
            throw new Error('No fallback models in error response');
          }
        } catch {
          // Last resort fallback
          setModels([
            { name: 'llama3.1:8b', id: 'fallback-01', size: '4.7 GB', modified: 'N/A', context_length: '128,000' },
            { name: 'deepseek-r1:8b', id: 'fallback-04', size: '4.9 GB', modified: 'N/A', context_length: '65,536' }
          ]);
        }
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
      // Last resort fallback
      setModels([
        { name: 'llama3.1:8b', id: 'fallback-01', size: '4.7 GB', modified: 'N/A', context_length: '128,000' },
        { name: 'deepseek-r1:8b', id: 'fallback-04', size: '4.9 GB', modified: 'N/A', context_length: '65,536' }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const selectedModel = models.find(m => m.name === value);

  return (
    <div className="jarvis-form-group full-width" style={{ marginLeft: `${depth * 20}px`, gridColumn: '1 / -1' }}>
      {/* Unified Model Configuration Section */}
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardHeader 
          title="LLM Model Configuration"
          subheader="Select and configure your language model"
          action={
            <Tooltip title="Update available models and reload LLM cache with current settings">
              <Button 
                size="small" 
                onClick={async () => {
                  // First fetch available models
                  await fetchAvailableModels();
                  
                  // Then reload cache
                  try {
                    const response = await fetch('/api/v1/settings/llm/cache/reload', { method: 'POST' });
                    if (response.ok) {
                      const result = await response.json();
                      console.log('LLM cache reloaded:', result);
                      if (onShowSuccess) {
                        onShowSuccess('Models updated and cache reloaded successfully!');
                      }
                    } else {
                      console.error('Failed to reload cache');
                      if (onShowSuccess) {
                        onShowSuccess('Models updated but cache reload failed');
                      }
                    }
                  } catch (error) {
                    console.error('Error reloading cache:', error);
                    if (onShowSuccess) {
                      onShowSuccess('Models updated but cache reload failed');
                    }
                  }
                }}
                startIcon={loading ? <CircularProgress size={16} /> : <RefreshIcon />}
                disabled={loading}
                variant="contained"
              >
                {loading ? 'Updating...' : 'Update Models & Cache'}
              </Button>
            </Tooltip>
          }
        />
        <CardContent>
          {/* Model Selector */}
          <FormControl fullWidth sx={{ mb: 3 }}>
            <InputLabel id={`${fieldKey}-label`}>Select Model</InputLabel>
            <Select
              labelId={`${fieldKey}-label`}
              value={value}
              label="Select Model"
              onChange={(e) => {
                const newModelName = e.target.value;
                const newSelectedModel = models.find(m => m.name === newModelName);
                
                // Update the model field
                console.log('[DEBUG] ModelSelector onChange:', { fieldKey, newModelName, previousValue: value });
                onChangeHandler(fieldKey, newModelName);
                
                // Auto-update for both Settings and Query Classifier (now flattened)
                if (newSelectedModel && newSelectedModel.context_length !== 'Unknown') {
                  const contextLength = parseInt(newSelectedModel.context_length.replace(/,/g, ''));
                  if (!isNaN(contextLength)) {
                    
                    if (fieldKey === 'model' || fieldKey === 'main_llm.model') {
                      // Settings tab
                      onChangeHandler(fieldKey.includes('main_llm') ? 'main_llm.context_length' : 'context_length', contextLength);
                      const suggestedMaxTokens = Math.floor(contextLength * 0.75);
                      onChangeHandler(fieldKey.includes('main_llm') ? 'main_llm.max_tokens' : 'max_tokens', suggestedMaxTokens);
                    } else if (fieldKey === 'second_llm.model') {
                      // Second LLM tab
                      onChangeHandler('second_llm.context_length', contextLength);
                      const suggestedMaxTokens = Math.floor(contextLength * 0.75);
                      onChangeHandler('second_llm.max_tokens', suggestedMaxTokens);
                    } else if (fieldKey === 'knowledge_graph.model') {
                      // Knowledge Graph tab
                      onChangeHandler('knowledge_graph.context_length', contextLength);
                      const suggestedMaxTokens = Math.floor(contextLength * 0.75);
                      onChangeHandler('knowledge_graph.max_tokens', suggestedMaxTokens);
                    } else if (fieldKey === 'query_classifier.model') {
                      // Query Classifier tab - exactly same as Settings tab
                      onChangeHandler('query_classifier.context_length', contextLength);
                      const suggestedMaxTokens = Math.floor(contextLength * 0.75);
                      onChangeHandler('query_classifier.max_tokens', suggestedMaxTokens);
                    }
                  }
                }
              }}
              disabled={loading}
              endAdornment={loading && <CircularProgress size={20} />}
            >
              {models.map((model) => (
                <MenuItem key={model.id} value={model.name}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography>{model.name}</Typography>
                      {value === model.name && (
                        <Chip 
                          label="Active" 
                          size="small" 
                          color="success" 
                          icon={<CheckCircleIcon />}
                        />
                      )}
                    </Box>
                    <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                      <Typography variant="caption" color="text.secondary">
                        {model.size}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Context: {model.context_length}
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
              ))}
              <Divider />
              <MenuItem value="custom">
                <Typography color="primary">Enter custom model name...</Typography>
              </MenuItem>
            </Select>
          </FormControl>

          {/* Current Model Information Panel */}
          {selectedModel && (
            <Paper sx={{ p: 3, backgroundColor: 'action.hover', border: '1px solid', borderColor: 'primary.light' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <CheckCircleIcon color="success" />
                <Typography variant="h6" color="primary">
                  Current Model: {selectedModel.name}
                </Typography>
              </Box>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                      Model Size
                    </Typography>
                    <Typography variant="h6" color="primary">
                      {selectedModel.size}
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                      Context Length
                    </Typography>
                    <Typography variant="h6" color="primary">
                      {selectedModel.context_length}
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                      Last Modified
                    </Typography>
                    <Typography variant="body2" color="text.primary">
                      {selectedModel.modified}
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                      Model ID
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {selectedModel.id}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
              
              {selectedModel.context_length !== 'Unknown' && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    This model supports up to <strong>{selectedModel.context_length}</strong> tokens in context. 
                    Larger contexts allow for more detailed conversations but may increase processing time and costs.
                  </Typography>
                </Alert>
              )}
            </Paper>
          )}
          
          {!selectedModel && value && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              Model "{value}" not found in available models. Please refresh the model list or select a different model.
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

const renderRAGFieldsWithCards = (
  fields: Array<{key: string, value: any}>,
  categoryKey: string,
  onChange: (field: string, value: any) => void,
  onShowSuccess?: (message?: string) => void,
  renderFieldFn: (key: string, value: any, depth: number, customOnChange: (field: string, value: any) => void, fieldCategory: string, onShowSuccessCallback?: (message?: string) => void) => React.ReactNode
) => {
  // Define card configurations for each tab
  const cardConfigurations: Record<string, Array<{title: string, subtitle: string, fields: string[]}>> = {
    retrieval: [
      {
        title: 'Performance & Caching',
        subtitle: 'Connection timeouts, execution timeouts, and cache settings',
        fields: ['performance.enable_caching', 'performance.cache_ttl_hours', 'performance.connection_timeout_s', 
                'performance.execution_timeout_ms', 'document_retrieval.cache_max_size']
      },
      {
        title: 'Search Configuration',
        subtitle: 'Vector search parameters and similarity thresholds',
        fields: ['search_strategy.top_k_vector_search', 'performance.vector_search_nprobe', 
                'document_retrieval.similarity_threshold', 'document_retrieval.num_docs_retrieve',
                'document_retrieval.max_documents_mcp']
      },
      {
        title: 'Collection Auto-Detection',
        subtitle: 'Automatic collection selection and limits',
        fields: ['agent_settings.enable_collection_auto_detection', 'agent_settings.max_results_per_collection',
                'collection_selection.max_collections', 'collection_selection.cache_selections',
                'collection_selection.enable_llm_selection', 'document_retrieval.enable_query_expansion']
      },
      {
        title: 'Collection Management',
        subtitle: 'Default and fallback collections configuration',
        fields: ['document_retrieval.default_collections', 'collection_selection.fallback_collections',
                'collection_selection.selection_prompt_template']
      }
    ],
    reranking: [
      {
        title: 'Reranking Configuration',
        subtitle: 'Advanced reranking and scoring parameters',
        fields: ['reranking.enable_advanced_reranking', 'reranking.enable_qwen_reranker',
                'reranking.rerank_weight', 'reranking.rerank_threshold', 'reranking.num_to_rerank',
                'reranking.batch_size']
      },
      {
        title: 'BM25 Scoring',
        subtitle: 'BM25 algorithm parameters and corpus settings',
        fields: ['bm25_scoring.enable_bm25', 'bm25_scoring.k1', 'bm25_scoring.b',
                'bm25_scoring.bm25_weight', 'bm25_scoring.corpus_batch_size']
      },
      {
        title: 'Relevance Thresholds',
        subtitle: 'Agent confidence and relevance scoring',
        fields: ['agent_settings.min_relevance_score', 'agent_settings.confidence_threshold',
                'agent_settings.complex_query_threshold', 'agent_settings.collection_size_threshold']
      },
      {
        title: 'Search Weight Distribution',
        subtitle: 'Balance between keyword and semantic search',
        fields: ['search_strategy.keyword_weight', 'search_strategy.semantic_weight',
                'search_strategy.hybrid_threshold']
      }
    ],
    search: [
      {
        title: 'Strategy Configuration',
        subtitle: 'Search strategy and performance settings',
        fields: ['search_strategy.search_strategy', 'agent_settings.default_query_strategy',
                'search_strategy.enable_focused_search', 'search_strategy.default_max_results',
                'performance.max_concurrent_searches']
      },
      {
        title: 'Query Classification',
        subtitle: 'Query analysis and classification settings',
        fields: ['query_processing.enable_query_classification', 'query_processing.max_query_length',
                'query_processing.window_size']
      },
      {
        title: 'Text Processing',
        subtitle: 'Text preprocessing and normalization',
        fields: ['query_processing.enable_stemming', 'query_processing.enable_stop_word_removal']
      },
      {
        title: 'Query Expansion',
        subtitle: 'Query enhancement and expansion methods',
        fields: ['query_processing.query_expansion_methods']
      }
    ]
  };

  const cards = cardConfigurations[categoryKey] || [];
  
  // Track which fields have been assigned to cards
  const assignedFields = new Set<string>();
  
  const cardComponents = cards.map((card, index) => {
    const cardFields = fields.filter(field => 
      card.fields.some(fieldPattern => {
        const fieldKey = field.key.toLowerCase();
        const pattern = fieldPattern.toLowerCase();
        
        // Direct match
        if (fieldKey === pattern) return true;
        
        // Pattern matching - check if field contains all parts of pattern
        const patternParts = pattern.split('.');
        return patternParts.every(part => fieldKey.includes(part));
      })
    );

    // Mark these fields as assigned
    cardFields.forEach(field => assignedFields.add(field.key));

    if (cardFields.length === 0) return null;

    return (
      <Card key={index} variant="outlined" sx={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column' }}>
        <CardHeader 
          title={card.title}
          subheader={card.subtitle}
          sx={{ pb: 1, flexShrink: 0 }}
        />
        <CardContent sx={{ flexGrow: 1 }}>
          <div className="jarvis-form-grid single-column">
            {cardFields.map(({ key, value }) => 
              renderFieldFn(key, value, 0, (fieldKey, fieldValue) => {
                onChange(fieldKey, fieldValue);
              }, 'rag', onShowSuccess)
            )}
          </div>
        </CardContent>
      </Card>
    );
  }).filter(Boolean);

  // Create an "Other Settings" card for unassigned fields
  const unassignedFields = fields.filter(field => !assignedFields.has(field.key));
  if (unassignedFields.length > 0) {
    cardComponents.push(
      <Card key="other" variant="outlined" sx={{ height: 'fit-content' }}>
        <CardHeader 
          title="Other Settings"
          subheader="Additional configuration options"
          sx={{ pb: 1 }}
        />
        <CardContent>
          <div className="jarvis-form-grid single-column">
            {unassignedFields.map(({ key, value }) => 
              renderFieldFn(key, value, 0, (fieldKey, fieldValue) => {
                onChange(fieldKey, fieldValue);
              }, 'rag', onShowSuccess)
            )}
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div style={{ width: '100%', maxWidth: 'none', display: 'block', boxSizing: 'border-box' }}>
      <div className="rag-cards-grid">
        {cardComponents}
      </div>
    </div>
  );
};

const renderPerformanceFieldsWithCards = (
  fields: Array<{key: string, value: any}>,
  categoryKey: string,
  onChange: (field: string, value: any) => void,
  onShowSuccess?: (message?: string) => void,
  renderFieldFn: (key: string, value: any, depth: number, customOnChange: (field: string, value: any) => void, fieldCategory: string, onShowSuccessCallback?: (message?: string) => void) => React.ReactNode
) => {
  // Debug logging for Performance Optimization
  console.log('[DEBUG] Performance Optimization - categoryKey:', categoryKey);
  console.log('[DEBUG] Performance Optimization - fields:', fields.map(f => f.key));
  
  // Define card configurations for Performance Optimization tabs based on actual data structure
  const cardConfigurations: Record<string, Array<{title: string, subtitle: string, fields: string[]}>> = {
    detection: [
      {
        title: 'Detection Thresholds',
        subtitle: 'Numeric thresholds for detecting large output patterns',
        fields: ['strong_number_threshold', 'medium_number_threshold', 'small_number_threshold', 'min_items_for_chunking']
      },
      {
        title: 'Scoring Parameters', 
        subtitle: 'Scoring weights and multipliers for detection algorithms',
        fields: ['numeric_score_weight', 'keyword_score_weight', 'pattern_score_weight', 'score_multiplier_for_chunks']
      },
      {
        title: 'Confidence & Timing',
        subtitle: 'Confidence scoring and time-based calculations',
        fields: ['confidence_threshold', 'base_time']
      }
    ],
    processing: [
      {
        title: 'Chunk Processing',
        subtitle: 'Parameters for breaking large outputs into manageable chunks',
        fields: ['items_per_chunk', 'target_chunk_count', 'time_per_chunk']
      },
      {
        title: 'Performance Optimization',
        subtitle: 'Advanced settings for processing performance and efficiency',
        fields: ['chunking_bonus_multiplier']
      }
    ],
    memory: [
      {
        title: 'Redis Configuration',
        subtitle: 'Redis cache settings and conversation management',
        fields: ['redis_ttl', 'max_messages', 'max_history_display']
      },
      {
        title: 'Memory Management',
        subtitle: 'Memory optimization and resource allocation',
        fields: ['enable_memory_optimization']
      }
    ],
    patterns: [
      {
        title: 'Large Output Indicators',
        subtitle: 'Keywords that indicate requests for large comprehensive outputs',
        fields: ['keywords']
      },
      {
        title: 'Pattern Matching',
        subtitle: 'Regular expression patterns for detecting large output requests',
        fields: ['regex_patterns']
      }
    ]
  };

  const cards = cardConfigurations[categoryKey] || [];
  
  // Track which fields have been assigned to cards
  const assignedFields = new Set<string>();
  
  const cardComponents = cards.map((card, index) => {
    const cardFields = fields.filter(field => 
      card.fields.some(fieldPattern => {
        const fieldKey = field.key.toLowerCase();
        const pattern = fieldPattern.toLowerCase();
        // Try exact match first
        if (fieldKey === pattern) return true;
        // Try partial matches
        if (fieldKey.includes(pattern) || pattern.includes(fieldKey)) return true;
        // Try removing dots and underscores
        const cleanFieldKey = fieldKey.replace(/[._]/g, '');
        const cleanPattern = pattern.replace(/[._]/g, '');
        return cleanFieldKey === cleanPattern || cleanFieldKey.includes(cleanPattern) || cleanPattern.includes(cleanFieldKey);
      })
    );
    
    // Mark these fields as assigned
    cardFields.forEach(field => assignedFields.add(field.key));
    
    if (cardFields.length === 0) return null;
    
    return (
      <Card key={`${categoryKey}-${index}`} variant="outlined" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <CardHeader 
          title={card.title}
          subheader={card.subtitle}
          sx={{ pb: 1 }}
        />
        <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="jarvis-form-grid single-column">
            {cardFields.map(({ key, value }) => 
              renderFieldFn(key, value, 0, (fieldKey, fieldValue) => {
                onChange(fieldKey, fieldValue);
              }, 'large_generation', onShowSuccess)
            )}
          </div>
        </CardContent>
      </Card>
    );
  }).filter(Boolean);

  // Create an "Other Settings" card for unassigned fields
  const unassignedFields = fields.filter(field => !assignedFields.has(field.key));
  if (unassignedFields.length > 0) {
    cardComponents.push(
      <Card key="other" variant="outlined" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <CardHeader 
          title="Other Settings"
          subheader="Additional performance optimization options"
          sx={{ pb: 1 }}
        />
        <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="jarvis-form-grid single-column">
            {unassignedFields.map(({ key, value }) => 
              renderFieldFn(key, value, 0, (fieldKey, fieldValue) => {
                onChange(fieldKey, fieldValue);
              }, 'large_generation', onShowSuccess)
            )}
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div style={{ width: '100%', maxWidth: 'none', display: 'block', boxSizing: 'border-box' }}>
      <div className="rag-cards-grid">
        {cardComponents}
      </div>
    </div>
  );
};

const renderLangfuseFieldsWithCards = (
  fields: Array<{key: string, value: any}>,
  categoryKey: string,
  onChange: (field: string, value: any) => void,
  onShowSuccess?: (message?: string) => void,
  renderFieldFn: (key: string, value: any, depth: number, customOnChange: (field: string, value: any) => void, fieldCategory: string, onShowSuccessCallback?: (message?: string) => void) => React.ReactNode
) => {
  // Debug logging for Langfuse
  console.log('[DEBUG] Langfuse - categoryKey:', categoryKey);
  console.log('[DEBUG] Langfuse - fields:', fields.map(f => f.key));
  
  // Define card configurations for Langfuse/Monitoring settings
  const cardConfigurations: Record<string, Array<{title: string, subtitle: string, fields: string[]}>> = {
    settings: [
      {
        title: 'Connection Settings',
        subtitle: 'Langfuse server connection and authentication configuration',
        fields: ['host', 'public_key', 'secret_key', 'enabled', 'enable_langfuse', 'langfuse_enabled']
      },
      {
        title: 'Monitoring Configuration', 
        subtitle: 'Monitoring features and data collection settings',
        fields: ['sample_rate', 'sampling_rate', 'trace_enabled', 'enable_tracing', 'debug_mode', 'log_level']
      },
      {
        title: 'Performance Settings',
        subtitle: 'Performance optimization and resource management',
        fields: ['batch_size', 'flush_interval', 'timeout', 'max_retries', 'queue_size', 'async_enabled']
      },
      {
        title: 'Cost Tracking',
        subtitle: 'Cost monitoring and budget management settings',
        fields: ['cost_tracking', 'enable_cost_tracking', 'budget_limit', 'cost_threshold', 'currency']
      }
    ]
  };

  const cards = cardConfigurations[categoryKey] || [];
  
  // Track which fields have been assigned to cards
  const assignedFields = new Set<string>();
  
  const cardComponents = cards.map((card, index) => {
    const cardFields = fields.filter(field => 
      card.fields.some(fieldPattern => {
        const fieldKey = field.key.toLowerCase();
        const pattern = fieldPattern.toLowerCase();
        // Try exact match first
        if (fieldKey === pattern) return true;
        // Try partial matches
        if (fieldKey.includes(pattern) || pattern.includes(fieldKey)) return true;
        // Try removing dots and underscores
        const cleanFieldKey = fieldKey.replace(/[._]/g, '');
        const cleanPattern = pattern.replace(/[._]/g, '');
        return cleanFieldKey === cleanPattern || cleanFieldKey.includes(cleanPattern) || cleanPattern.includes(cleanFieldKey);
      })
    );
    
    // Mark these fields as assigned
    cardFields.forEach(field => assignedFields.add(field.key));
    
    if (cardFields.length === 0) return null;
    
    return (
      <Card key={`${categoryKey}-${index}`} variant="outlined" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <CardHeader 
          title={card.title}
          subheader={card.subtitle}
          sx={{ pb: 1 }}
        />
        <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="jarvis-form-grid single-column">
            {cardFields.map(({ key, value }) => 
              renderFieldFn(key, value, 0, (fieldKey, fieldValue) => {
                onChange(fieldKey, fieldValue);
              }, 'langfuse', onShowSuccess)
            )}
          </div>
        </CardContent>
      </Card>
    );
  }).filter(Boolean);

  // Create an "Other Settings" card for unassigned fields
  const unassignedFields = fields.filter(field => !assignedFields.has(field.key));
  if (unassignedFields.length > 0) {
    cardComponents.push(
      <Card key="other" variant="outlined" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <CardHeader 
          title="Other Settings"
          subheader="Additional monitoring and tracing options"
          sx={{ pb: 1 }}
        />
        <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="jarvis-form-grid single-column">
            {unassignedFields.map(({ key, value }) => 
              renderFieldFn(key, value, 0, (fieldKey, fieldValue) => {
                onChange(fieldKey, fieldValue);
              }, 'langfuse', onShowSuccess)
            )}
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div style={{ width: '100%', maxWidth: 'none', display: 'block', boxSizing: 'border-box' }}>
      <div className="rag-cards-grid">
        {cardComponents}
      </div>
    </div>
  );
};

const renderEnvironmentFieldsWithCards = (
  fields: Array<{key: string, value: any}>,
  categoryKey: string,
  onChange: (field: string, value: any) => void,
  onShowSuccess?: (message?: string) => void,
  renderFieldFn: (key: string, value: any, depth: number, customOnChange: (field: string, value: any) => void, fieldCategory: string, onShowSuccessCallback?: (message?: string) => void) => React.ReactNode
) => {
  // Debug logging for Environment
  console.log('[DEBUG] Environment - categoryKey:', categoryKey);
  console.log('[DEBUG] Environment - fields:', fields.map(f => f.key));
  
  // Define card configurations for Environment & Runtime settings
  const cardConfigurations: Record<string, Array<{title: string, subtitle: string, fields: string[]}>> = {
    settings: [
      {
        title: 'API Keys & Authentication',
        subtitle: 'API keys, tokens, and authentication credentials',
        fields: ['api_key', 'secret_key', 'access_token', 'auth_token', 'openai_api_key', 'anthropic_api_key', 'google_api_key']
      },
      {
        title: 'Database Configuration', 
        subtitle: 'Database connection strings and credentials',
        fields: ['database_url', 'db_host', 'db_port', 'db_user', 'db_password', 'redis_url', 'postgres_url', 'mysql_url']
      },
      {
        title: 'Service URLs & Endpoints',
        subtitle: 'External service URLs and API endpoints',
        fields: ['base_url', 'endpoint', 'webhook_url', 'callback_url', 'service_url', 'api_url']
      },
      {
        title: 'Runtime Configuration',
        subtitle: 'Environment settings and runtime parameters',
        fields: ['environment', 'debug', 'log_level', 'port', 'host', 'timeout', 'max_workers', 'workers']
      }
    ]
  };

  const cards = cardConfigurations[categoryKey] || [];
  
  // Track which fields have been assigned to cards
  const assignedFields = new Set<string>();
  
  const cardComponents = cards.map((card, index) => {
    const cardFields = fields.filter(field => 
      card.fields.some(fieldPattern => {
        const fieldKey = field.key.toLowerCase();
        const pattern = fieldPattern.toLowerCase();
        // Try exact match first
        if (fieldKey === pattern) return true;
        // Try partial matches
        if (fieldKey.includes(pattern) || pattern.includes(fieldKey)) return true;
        // Try removing dots and underscores
        const cleanFieldKey = fieldKey.replace(/[._]/g, '');
        const cleanPattern = pattern.replace(/[._]/g, '');
        return cleanFieldKey === cleanPattern || cleanFieldKey.includes(cleanPattern) || cleanPattern.includes(cleanFieldKey);
      })
    );
    
    // Mark these fields as assigned
    cardFields.forEach(field => assignedFields.add(field.key));
    
    if (cardFields.length === 0) return null;
    
    return (
      <Card key={`${categoryKey}-${index}`} variant="outlined" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <CardHeader 
          title={card.title}
          subheader={card.subtitle}
          sx={{ pb: 1 }}
        />
        <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="jarvis-form-grid single-column">
            {cardFields.map(({ key, value }) => 
              renderFieldFn(key, value, 0, (fieldKey, fieldValue) => {
                onChange(fieldKey, fieldValue);
              }, 'environment', onShowSuccess)
            )}
          </div>
        </CardContent>
      </Card>
    );
  }).filter(Boolean);

  // Create an "Other Environment Variables" card for unassigned fields
  const unassignedFields = fields.filter(field => !assignedFields.has(field.key));
  if (unassignedFields.length > 0) {
    cardComponents.push(
      <Card key="other" variant="outlined" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <CardHeader 
          title="Other Environment Variables"
          subheader="Additional environment settings and custom variables"
          sx={{ pb: 1 }}
        />
        <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="jarvis-form-grid single-column">
            {unassignedFields.map(({ key, value }) => 
              renderFieldFn(key, value, 0, (fieldKey, fieldValue) => {
                onChange(fieldKey, fieldValue);
              }, 'environment', onShowSuccess)
            )}
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div style={{ width: '100%', maxWidth: 'none', display: 'block', boxSizing: 'border-box' }}>
      {/* Add Variable Button */}
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            const newVar = prompt('Enter environment variable name:');
            if (newVar && newVar.trim()) {
              onChange('environment_variables', {
                ...fields.reduce((acc, field) => ({ ...acc, [field.key]: field.value }), {}),
                [newVar.trim()]: ''
              });
            }
          }}
          size="small"
        >
          Add Variable
        </Button>
      </Box>
      
      <div className="rag-cards-grid">
        {cardComponents}
      </div>
    </div>
  );
};

const renderStandardForm = (
  data: any, 
  onChange: (field: string, value: any) => void, 
  category: string | undefined,
  activeTab: string,
  setActiveTab: (tab: string) => void,
  passwordVisibility: Record<string, boolean>,
  setPasswordVisibility: React.Dispatch<React.SetStateAction<Record<string, boolean>>>,
  icebergPasswordVisibility: Record<string, boolean>,
  setIcebergPasswordVisibility: React.Dispatch<React.SetStateAction<Record<string, boolean>>>,
  onShowSuccess?: (message?: string) => void,
  testNeo4jConnection?: () => Promise<void>,
  testingConnection?: boolean,
  connectionTestResult?: {success: boolean, message?: string, error?: string} | null
) => {

  // Flatten nested objects and categorize fields into domain-intelligent tabs
  // Known nested structures that should be preserved (not flattened)
  const preserveNested = [
    'main_llm',
    'second_llm',
    'knowledge_graph',
    'thinking_mode',
    'agent_config',
    'conversation_memory',
    'error_recovery',
    'response_generation',
    'vector_db',
    'iceberg'
  ];

  const categorizeFields = (data: any, category?: string) => {
    let categories: Record<string, { title: string; fields: Record<string, any> }>;
    
    // Define category-specific tab structure
    if (category === 'rag') {
      categories = {
        retrieval: { title: 'Retrieval Settings', fields: {} },
        reranking: { title: 'Reranking & Scoring', fields: {} },
        search: { title: 'Search Strategy', fields: {} },
        processing: { title: 'Document Processing', fields: {} }
      };
    } else if (category === 'storage') {
      categories = {
        vector: { title: 'Vector Databases (Unstructured)', fields: {} },
        structured: { title: 'Iceberg (Structured)', fields: {} }
      };
    } else if (category === 'overflow') {
      categories = {
        thresholds: { title: 'Thresholds', fields: {} },
        storage_tiers: { title: 'Storage Tiers', fields: {} },
        retrieval: { title: 'Retrieval', fields: {} },
        auto_promotion: { title: 'Auto-Promotion', fields: {} }
      };
    } else if (category === 'large_generation') {
      categories = {
        detection: { title: 'Detection & Scoring', fields: {} },
        processing: { title: 'Processing & Performance', fields: {} },
        memory: { title: 'Memory & Caching', fields: {} },
        patterns: { title: 'Keywords & Patterns', fields: {} }
      };
    } else if (category === 'llm') {
      // LLM category structure - consolidated context into settings
      categories = {
        settings: { title: 'Main LLM', fields: {} },
        second_llm: { title: 'Second LLM', fields: {} },
        knowledge_graph: { title: 'Knowledge Graph', fields: {} },
        classifier: { title: 'Query Classifier', fields: {} },
        search_optimization: { title: 'Search Optimization', fields: {} },
        thinking: { title: 'Thinking Mode', fields: {} }
      };
    } else {
      // Default single-tab structure for other categories (monitoring, mcp, etc.)
      categories = {
        settings: { title: 'Settings', fields: {} }
      };
    }

    // Flatten nested objects recursively, but preserve certain known structures
    const flattenObject = (obj: any, prefix: string = ''): Record<string, any> => {
      const flattened: Record<string, any> = {};
      const seenKeys = new Set<string>(); // Track keys to prevent duplicates
      
      // Special debug for storage category
      if (category === 'storage' && !prefix) {
        console.log('[SettingsFormRenderer] Input to flattenObject:', obj);
        console.log('[SettingsFormRenderer] Input keys:', Object.keys(obj));
      }
      
      Object.entries(obj).forEach(([key, value]) => {
        // Skip if this key starts with "settings." - it's likely a duplicate
        if (key.startsWith('settings.')) {
          return;
        }
        
        const fullKey = prefix ? `${prefix}.${key}` : key;
        
        // If this is a known nested structure at the top level, keep it as-is
        if (!prefix && preserveNested.includes(key) && typeof value === 'object' && value !== null) {
          if (category === 'storage' && key === 'vector_db') {
            console.log('[SettingsFormRenderer] Preserving vector_db:', value);
          }
          if (!seenKeys.has(key)) {
            seenKeys.add(key);
            flattened[key] = value;
          }
        } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
          // Don't flatten if this is a preserved nested structure at any level
          const shouldPreserve = preserveNested.some(preserved => 
            key === preserved || fullKey.endsWith(`.${preserved}`)
          );
          
          if (shouldPreserve) {
            if (category === 'storage' && key === 'vector_db') {
              console.log('[SettingsFormRenderer] Preserving nested vector_db:', value);
            }
            if (!seenKeys.has(fullKey)) {
              seenKeys.add(fullKey);
              flattened[fullKey] = value;
            }
          } else {
            // Recursively flatten other nested objects
            const nestedFlattened = flattenObject(value, fullKey);
            Object.entries(nestedFlattened).forEach(([nestedKey, nestedValue]) => {
              if (!seenKeys.has(nestedKey)) {
                seenKeys.add(nestedKey);
                flattened[nestedKey] = nestedValue;
              }
            });
          }
        } else {
          if (!seenKeys.has(fullKey)) {
            seenKeys.add(fullKey);
            flattened[fullKey] = value;
          }
        }
      });
      
      return flattened;
    };

    const flattenedData = flattenObject(data);
    
    // Debug logging for storage category
    if (category === 'storage') {
      console.log('[SettingsFormRenderer] Flattened data:', flattenedData);
      console.log('[SettingsFormRenderer] Flattened keys:', Object.keys(flattenedData));
      if (flattenedData.vector_db) {
        console.log('[SettingsFormRenderer] vector_db in flattened data:', typeof flattenedData.vector_db, flattenedData.vector_db);
      }
    }

    Object.entries(flattenedData).forEach(([key, value]) => {
      const lowerKey = key.toLowerCase();
      if (category === 'llm') {
        console.log('[DEBUG] Processing flattened key:', key, 'lowerKey:', lowerKey);
      }
      
      // Skip keys that are already part of a preserved nested object
      const isPartOfNestedObject = key.includes('.') && 
        preserveNested.some(nested => key.startsWith(nested + '.'));
      
      if (isPartOfNestedObject) {
        return; // Skip this field as it's part of a preserved nested structure
      }
      
      if (category === 'rag') {
        // RAG-specific field categorization with card grouping
        
        // Retrieval Settings Tab
        if (lowerKey.includes('performance') && 
            (lowerKey.includes('cache') || lowerKey.includes('timeout') || lowerKey.includes('connection'))) {
          categories.retrieval.fields[key] = value;
        }
        else if (lowerKey.includes('search') && lowerKey.includes('top_k') ||
                 lowerKey.includes('vector') && lowerKey.includes('nprobe') ||
                 lowerKey.includes('similarity') && lowerKey.includes('threshold') ||
                 lowerKey.includes('retrieval') && (lowerKey.includes('num_docs') || lowerKey.includes('max_documents'))) {
          categories.retrieval.fields[key] = value;
        }
        else if (lowerKey.includes('collection') || lowerKey.includes('agent') && lowerKey.includes('max_results') ||
                 lowerKey.includes('detection') || lowerKey.includes('default') || lowerKey.includes('fallback')) {
          categories.retrieval.fields[key] = value;
        }
        
        // Reranking & Scoring Tab
        else if (lowerKey.includes('rerank') || lowerKey.includes('bm25') || lowerKey.includes('score') || 
                 lowerKey.includes('weight') || lowerKey.includes('threshold') || lowerKey.includes('relevance')) {
          categories.reranking.fields[key] = value;
        }
        
        // Search Strategy Tab
        else if (lowerKey.includes('search') || lowerKey.includes('query') || lowerKey.includes('strategy') || 
                 lowerKey.includes('hybrid') || lowerKey.includes('processing') || lowerKey.includes('expansion')) {
          categories.search.fields[key] = value;
        }
        
        // Anything else goes to retrieval as default
        else {
          categories.retrieval.fields[key] = value;
        }
      } else if (category === 'storage') {
        // Storage-specific field categorization
        console.log('[DEBUG] Processing storage field:', key, 'value type:', typeof value);
        
        // Handle preserved nested structures (vector_db, iceberg)
        if (key === 'vector_db' || key === 'iceberg') {
          console.log('[DEBUG] Found preserved structure:', key);
          if (key === 'vector_db') {
            categories.vector.fields[key] = value;
          } else {
            categories.structured.fields[key] = value;
          }
        }
        // Skip embedding fields - they're handled by VectorDatabaseManager
        else if (key === 'embedding_model' || key === 'embedding_endpoint') {
          // Don't add to any category - will be handled specially
          return;
        }
        // For other flattened fields, categorize by keywords
        else if (lowerKey.includes('milvus') || lowerKey.includes('qdrant') || lowerKey.includes('vector') || 
            lowerKey.includes('pinecone') || lowerKey.includes('weaviate') ||
            lowerKey.includes('chroma') || lowerKey.includes('faiss')) {
          categories.vector.fields[key] = value;
        }
        // Iceberg (Structured)
        else if (lowerKey.includes('iceberg') || lowerKey.includes('spark') || lowerKey.includes('hive') || 
                 lowerKey.includes('parquet') || lowerKey.includes('table') || lowerKey.includes('catalog') ||
                 lowerKey.includes('warehouse') || lowerKey.includes('schema')) {
          categories.structured.fields[key] = value;
        }
        // Default: put general storage settings in vector category
        else {
          categories.vector.fields[key] = value;
        }
      } else if (category === 'overflow') {
        // Overflow-specific field categorization
        
        // Thresholds Tab - token-related settings
        if (lowerKey.includes('threshold') || lowerKey.includes('token') || lowerKey.includes('chunk_size') || 
            lowerKey.includes('chunk_overlap') || lowerKey.includes('max_overflow_context_ratio')) {
          categories.thresholds.fields[key] = value;
        }
        // Storage Tiers Tab - TTL and tier settings  
        else if (lowerKey.includes('ttl') || lowerKey.includes('l1_') || lowerKey.includes('l2_') || 
                 lowerKey.includes('hour') || lowerKey.includes('day')) {
          categories.storage_tiers.fields[key] = value;
        }
        // Retrieval Tab - search and retrieval settings
        else if (lowerKey.includes('retrieval') || lowerKey.includes('top_k') || lowerKey.includes('semantic') || 
                 lowerKey.includes('search') || lowerKey.includes('keyword')) {
          categories.retrieval.fields[key] = value;
        }
        // Auto-Promotion Tab - promotion and access settings
        else if (lowerKey.includes('promote') || lowerKey.includes('auto') || lowerKey.includes('access') || 
                 lowerKey.includes('threshold_accesses')) {
          categories.auto_promotion.fields[key] = value;
        }
        // Default: put other overflow settings in thresholds
        else {
          categories.thresholds.fields[key] = value;
        }
      } else if (category === 'large_generation') {
        // Performance Optimization field categorization
        
        // Detection & Scoring
        if (lowerKey.includes('threshold') || lowerKey.includes('score') && lowerKey.includes('weight') ||
            lowerKey.includes('confidence') || lowerKey.includes('min_items')) {
          categories.detection.fields[key] = value;
        }
        // Processing & Performance  
        else if (lowerKey.includes('chunk') || lowerKey.includes('time') || lowerKey.includes('multiplier') ||
                 lowerKey.includes('target') || lowerKey.includes('items_per')) {
          categories.processing.fields[key] = value;
        }
        // Memory & Caching
        else if (lowerKey.includes('redis') || lowerKey.includes('ttl') || lowerKey.includes('memory') ||
                 lowerKey.includes('max_messages') || lowerKey.includes('history') || lowerKey.includes('optimization')) {
          categories.memory.fields[key] = value;
        }
        // Keywords & Patterns
        else if (lowerKey.includes('keyword') || lowerKey.includes('pattern') || lowerKey.includes('regex')) {
          categories.patterns.fields[key] = value;
        }
        // Default to detection category
        else {
          categories.detection.fields[key] = value;
        }
      } else if (category === 'llm') {
        // LLM-specific field categorization
        // Search Optimization Tab - All search_optimization-related settings
        if ((lowerKey.includes('search_optimization') && key !== 'search_optimization') ||
            key === 'enable_search_optimization' || key === 'optimization_timeout' || 
            key === 'optimization_prompt' || key === 'top_p') {
          console.log('[DEBUG] Adding to search_optimization tab:', key, value);
          categories.search_optimization.fields[key] = value;
        }
        // Second LLM Tab - All second_llm-related settings
        else if (lowerKey.includes('second_llm')) {
          categories.second_llm.fields[key] = value;
        }
        // Knowledge Graph Tab - Standard form like Main LLM and Second LLM  
        else if (lowerKey.includes('knowledge_graph') || lowerKey.includes('kg_') || 
                 lowerKey.includes('entity_') || lowerKey.includes('relationship_') || 
                 lowerKey.includes('neo4j') || lowerKey.includes('graph_') ||
                 (lowerKey.includes('extraction') && lowerKey.includes('prompt')) ||
                 lowerKey.includes('entity_types') || lowerKey.includes('relationship_types') ||
                 lowerKey.includes('max_entities_per_chunk') || lowerKey.includes('coreference_resolution') ||
                 lowerKey.includes('entity_discovery') || lowerKey.includes('relationship_discovery') ||
                 lowerKey.includes('knowledge_extraction') || lowerKey.includes('discovery') ||
                 (lowerKey.includes('prompt') && (lowerKey.includes('entity') || lowerKey.includes('relationship') || lowerKey.includes('knowledge'))) ||
                 (key === 'prompts' && Array.isArray(value) && value.some((item: any) => 
                   item?.name?.includes('entity_discovery') || 
                   item?.name?.includes('relationship_discovery') || 
                   item?.name?.includes('knowledge_extraction') ||
                   item?.prompt_type?.includes('entity') ||
                   item?.prompt_type?.includes('relationship') ||
                   item?.prompt_type?.includes('knowledge')
                 ))) {
          categories.knowledge_graph.fields[key] = value;
        }
        // Query Classifier Tab - All classifier-related settings and conflict_prevention settings
        else if (lowerKey.includes('query_classifier') || lowerKey.includes('classifier') || 
                 lowerKey.includes('conflict_prevention')) {
          categories.classifier.fields[key] = value;
        }
        // Thinking Mode Tab - Include thinking mode related fields
        else if (lowerKey.includes('thinking_mode') || lowerKey.includes('thinking') || lowerKey.includes('non_thinking')) {
          categories.thinking.fields[key] = value;
        }
        // Settings Tab - Core model configuration and context-related fields (everything else)
        else {
          categories.settings.fields[key] = value;
        }
      } else {
        // Default single-tab categorization for other categories (monitoring, mcp, etc.)
        categories.settings.fields[key] = value;
      }
    });

    // Remove empty categories
    const filteredCategories: Record<string, { title: string; fields: Record<string, any> }> = {};
    Object.entries(categories).forEach(([key, category]) => {
      if (Object.keys(category.fields).length > 0) {
        filteredCategories[key] = category;
      }
    });
    
    return filteredCategories;
  };

  const togglePasswordVisibility = (fieldKey: string) => {
    setPasswordVisibility(prev => ({ ...prev, [fieldKey]: !prev[fieldKey] }));
  };

  const getRAGHelpText = (key: string): string | null => {
    const helpTexts: Record<string, string> = {
      // Performance & Caching
      'performance.cache_ttl_hours': 'How long to cache results in hours. Recommended: 2-24 hours for balance between performance and freshness.',
      'performance.connection_timeout_s': 'Maximum seconds to wait for database connections. Recommended: 30-300 seconds.',
      'performance.execution_timeout_ms': 'Maximum milliseconds for query execution. Recommended: 10000-60000ms.',
      'performance.enable_caching': 'Enable caching to improve performance by storing frequently accessed results.',
      'document_retrieval.cache_max_size': 'Maximum number of cached queries. Recommended: 100-1000 based on memory.',
      
      // Search Configuration
      'search_strategy.top_k_vector_search': 'Number of documents to retrieve from vector search. Higher values = more comprehensive but slower. Recommended: 20-100.',
      'performance.vector_search_nprobe': 'Number of probes for vector search. Higher values = more accurate but slower. Recommended: 8-20.',
      'document_retrieval.similarity_threshold': 'Minimum cosine similarity for document relevance. Higher values = more precise but fewer results. Recommended: 0.6-0.8.',
      'document_retrieval.num_docs_retrieve': 'Total number of documents to retrieve across all collections. Recommended: 10-50.',
      'document_retrieval.max_documents_mcp': 'Maximum documents to return to MCP tools. Lower values prevent overwhelming tools. Recommended: 5-15.',
      
      // Collection Management
      'agent_settings.enable_collection_auto_detection': 'Automatically detect which collections to search based on query content.',
      'agent_settings.max_results_per_collection': 'Maximum results from each collection. Helps distribute results evenly. Recommended: 5-20.',
      'collection_selection.max_collections': 'Maximum number of collections to search simultaneously. More = comprehensive but slower. Recommended: 3-10.',
      'collection_selection.cache_selections': 'Cache collection selection decisions to improve performance for similar queries.',
      'collection_selection.enable_llm_selection': 'Use LLM to intelligently select relevant collections instead of searching all.',
      
      // BM25 Scoring
      'bm25_scoring.k1': 'Controls term frequency saturation. Higher values give more weight to term frequency. Recommended: 1.2-2.0.',
      'bm25_scoring.b': 'Controls document length normalization. Higher values penalize longer documents more. Recommended: 0.7-0.8.',
      'bm25_scoring.bm25_weight': 'Weight of BM25 score in hybrid search. Higher values favor keyword matching. Recommended: 0.2-0.5.',
      'bm25_scoring.enable_bm25': 'Enable BM25 keyword search to complement vector search for better results.',
      
      // Reranking
      'reranking.rerank_weight': 'Weight of reranking score in final ranking. Higher values trust reranker more. Recommended: 0.5-0.8.',
      'reranking.rerank_threshold': 'Minimum reranking score threshold. Higher values = more selective. Recommended: 0.5-0.8.',
      'reranking.enable_advanced_reranking': 'Enable advanced reranking algorithms for better result quality.',
      'reranking.enable_qwen_reranker': 'Use Qwen model for reranking. Provides better semantic understanding.',
      
      // Search Strategy
      'search_strategy.keyword_weight': 'Weight of keyword search in hybrid search. Higher values favor exact matches. Recommended: 0.2-0.4.',
      'search_strategy.semantic_weight': 'Weight of semantic search in hybrid search. Higher values favor meaning. Recommended: 0.6-0.8.',
      'search_strategy.hybrid_threshold': 'Threshold for hybrid search activation. Lower values activate hybrid more often. Recommended: 0.5-0.8.',
      'search_strategy.search_strategy': 'Search strategy: auto (adaptive), vector (semantic), hybrid (both), or keyword (exact).',
      
      // Relevance & Thresholds
      'agent_settings.min_relevance_score': 'Minimum relevance score for including results. Higher values = more selective. Recommended: 0.3-0.6.',
      'agent_settings.confidence_threshold': 'Confidence threshold for agent decisions. Higher values = more conservative. Recommended: 0.5-0.8.',
      'agent_settings.complex_query_threshold': 'Threshold for detecting complex queries requiring special handling. Recommended: 0.1-0.3.',
      
      // Query Processing
      'query_processing.enable_query_classification': 'Classify queries to apply appropriate search strategies.',
      'query_processing.enable_stemming': 'Reduce words to their root forms for better matching (e.g., "running" → "run").',
      'query_processing.enable_stop_word_removal': 'Remove common words like "the", "and" to focus on important terms.',
      'query_processing.max_query_length': 'Maximum query length in characters. Longer queries are truncated. Recommended: 2000-8000.',
      'query_processing.window_size': 'Size of context window for query processing. Recommended: 50-200.'
    };
    
    const lowerKey = key.toLowerCase();
    
    // Try exact match first
    if (helpTexts[lowerKey]) {
      return helpTexts[lowerKey];
    }
    
    // Try partial matches
    for (const [helpKey, helpText] of Object.entries(helpTexts)) {
      if (lowerKey.includes(helpKey.toLowerCase()) || helpKey.toLowerCase().includes(lowerKey)) {
        return helpText;
      }
    }
    
    return null;
  };

  const getOverflowHelpText = (key: string): string | null => {
    const helpTexts: Record<string, string> = {
      // Threshold Settings
      'overflow_threshold_tokens': 'Maximum tokens before triggering overflow storage (default: 8000). When conversation context exceeds this limit, older content is automatically chunked and moved to storage tiers. Higher values keep more content in active memory but use more resources. Lower values trigger overflow sooner but may break conversation continuity. Recommended: 6000-12000 tokens depending on available memory.',
      
      'chunk_size_tokens': 'Size of each text chunk in tokens when content overflows (default: 2000). Smaller chunks provide more precise retrieval but may lose semantic context across chunk boundaries. Larger chunks preserve context better but may include irrelevant content during retrieval. Balance between retrieval precision and context preservation. Recommended: 1500-3000 tokens.',
      
      'chunk_overlap_tokens': 'Number of overlapping tokens between consecutive chunks (default: 200). Prevents important information from being lost at chunk boundaries and maintains conversation flow. Higher overlap improves context continuity but increases storage usage and processing time. Too low may cause context gaps. Recommended: 10-15% of chunk size (150-400 tokens).',
      
      // Storage Tier Settings  
      'l1_ttl_hours': 'Time-to-live for hot storage (L1) in hours (default: 24). L1 stores recently accessed or frequently used chunks for fastest retrieval. Longer TTL keeps content readily available but uses more memory. Shorter TTL frees resources faster but may cause performance hits on re-access. Recommended: 12-48 hours based on conversation patterns.',
      
      'l2_ttl_days': 'Time-to-live for warm storage (L2) in days (default: 7). L2 stores older content that may still be referenced. After TTL expires, content is permanently deleted. Longer retention supports longer conversations but increases storage costs. Consider user privacy and storage limitations. Recommended: 3-14 days depending on use case.',
      
      // Retrieval Settings
      'max_overflow_context_ratio': 'Maximum percentage of total context window to allocate for retrieved overflow content (default: 0.3). Controls the balance between current conversation and historical context. Higher ratios provide more historical context but leave less room for new content. Lower ratios prioritize current conversation but may lose important background. Recommended: 0.2-0.5 (20-50%).',
      
      'retrieval_top_k': 'Number of most relevant chunks to retrieve when querying overflow storage (default: 5). More chunks provide broader context but may include less relevant information and consume more tokens. Fewer chunks are more focused but might miss important details. Performance impact increases with higher values. Recommended: 3-8 chunks based on context window size.',
      
      'enable_semantic_search': 'Use AI embeddings to find semantically similar chunks (default: true). Provides much better relevance matching by understanding meaning rather than just keywords. Requires more computational resources and embedding generation time. Disable only if performance is critical and keyword matching is sufficient. Strongly recommended to keep enabled for better user experience.',
      
      'enable_keyword_extraction': 'Extract and index keywords from chunks for faster text-based searching (default: true). Complements semantic search with traditional keyword matching. Minimal performance impact with significant search speed improvements. Helps find specific terms, names, or technical concepts. Should typically be enabled alongside semantic search.',
      
      // Auto-Promotion Settings
      'auto_promote_to_l1': 'Automatically promote frequently accessed chunks from L2 to L1 storage (default: true). Improves performance by moving popular content to faster storage tier. Creates adaptive performance optimization based on usage patterns. Disable if you want manual control over storage tiers or have limited L1 capacity. Recommended to keep enabled for optimal performance.',
      
      'promotion_threshold_accesses': 'Number of times a chunk must be accessed before auto-promoting to L1 (default: 3). Lower values promote content more aggressively, improving performance but potentially filling L1 with less critical content. Higher values are more conservative but may delay performance improvements. Balance between responsiveness and resource utilization. Recommended: 2-5 accesses depending on conversation frequency.'
    };
    
    const lowerKey = key.toLowerCase();
    
    // Try exact match first
    if (helpTexts[lowerKey]) {
      return helpTexts[lowerKey];
    }
    
    // Try partial matches
    for (const [helpKey, helpText] of Object.entries(helpTexts)) {
      if (lowerKey.includes(helpKey.toLowerCase()) || helpKey.toLowerCase().includes(lowerKey)) {
        return helpText;
      }
    }
    
    return null;
  };

  const getPerformanceHelpText = (key: string): string | null => {
    const helpTexts: Record<string, string> = {
      // Detection Thresholds
      'strong_number_threshold': 'Threshold for detecting strong numeric patterns in queries. Higher values = more selective detection. Recommended: 20-50.',
      'medium_number_threshold': 'Threshold for detecting medium numeric patterns. Should be lower than strong threshold. Recommended: 10-30.',
      'small_number_threshold': 'Threshold for detecting small numeric patterns. Lowest detection threshold. Recommended: 5-20.',
      'min_items_for_chunking': 'Minimum number of items before chunking is considered. Lower values = more aggressive chunking. Recommended: 10-50.',
      
      // Scoring Parameters
      'min_score_for_keywords': 'Minimum score required for keyword-based detection. Higher values = more selective. Recommended: 2-5.',
      'min_score_for_medium_numbers': 'Minimum score for medium number detection. Usually lower than keyword score. Recommended: 1-3.',
      'score_multiplier': 'Global score multiplier for detection algorithms. Higher values increase sensitivity. Recommended: 10-20.',
      'pattern_score_weight': 'Weight given to pattern matching in scoring. Higher values prioritize patterns. Recommended: 1-3.',
      
      // Confidence Calculation
      'max_score_for_confidence': 'Maximum score used for confidence calculation. Higher values = more nuanced confidence. Recommended: 3-10.',
      'max_number_for_confidence': 'Maximum number threshold for confidence scoring. Used for normalization. Recommended: 50-200.',
      'default_comprehensive_items': 'Default number of items for comprehensive requests. Recommended: 20-50.',
      'min_estimated_items': 'Minimum estimated items to trigger large generation mode. Recommended: 5-15.',
      
      // Chunk Processing
      'default_chunk_size': 'Default number of items per chunk. Lower values = more chunks, better streaming. Recommended: 10-25.',
      'max_target_count': 'Maximum number of chunks to create. Higher values = more granular processing. Recommended: 100-1000.',
      'estimated_seconds_per_chunk': 'Estimated processing time per chunk in seconds. Used for time estimation. Recommended: 30-120.',
      
      // Performance Optimization
      'chunking_bonus_multiplier': 'Bonus multiplier for chunking decisions. Higher values make chunking more attractive. Recommended: 1.1-1.5.',
      'base_time_estimation': 'Base time estimation for processing (seconds). Used as baseline for calculations. Recommended: 5-30.',
      'optimization_threshold': 'Threshold for triggering performance optimizations. Lower values = more aggressive. Recommended: 0.3-0.7.',
      
      // Redis Configuration
      'redis_conversation_ttl': 'Time-to-live for Redis conversation cache in seconds. Higher values = longer retention. Recommended: 3600-604800.',
      'max_redis_messages': 'Maximum number of messages to store in Redis. Higher values = more context. Recommended: 20-100.',
      'conversation_history_display': 'Maximum number of history items to display in UI. Recommended: 5-20.',
      
      // Memory Management
      'max_memory_messages': 'Maximum number of messages to keep in memory. Higher values = more context but more memory. Recommended: 10-50.',
      'memory_optimization_enabled': 'Enable memory optimization features to reduce memory usage during large generations.',
      'cleanup_interval': 'Interval for memory cleanup operations in seconds. Lower values = more frequent cleanup. Recommended: 60-300.',
      
      // Keywords & Indicators
      'large_output_indicators': 'Keywords that indicate large output requests (e.g., "generate", "create", "list", "comprehensive").',
      'comprehensive_keywords': 'Keywords that specifically indicate comprehensive requests (e.g., "comprehensive", "detailed", "all").',
      
      // Pattern Matching
      'large_patterns': 'Regular expression patterns for detecting large output requests in queries.',
      'pattern_weights': 'Weights for different pattern types. Higher weights = more influence on detection.'
    };
    
    const lowerKey = key.toLowerCase();
    
    // Try exact match first
    if (helpTexts[lowerKey]) {
      return helpTexts[lowerKey];
    }
    
    // Try partial matches
    for (const [helpKey, helpText] of Object.entries(helpTexts)) {
      if (lowerKey.includes(helpKey.toLowerCase()) || helpKey.toLowerCase().includes(lowerKey)) {
        return helpText;
      }
    }
    
    return null;
  };

  const getRAGDropdownOptions = (key: string): Array<{value: string, label: string}> | null => {
    const lowerKey = key.toLowerCase();
    
    if (lowerKey.includes('search') && lowerKey.includes('strategy')) {
      return [
        { value: 'auto', label: 'Auto (Adaptive)' },
        { value: 'vector', label: 'Vector (Semantic)' },
        { value: 'hybrid', label: 'Hybrid (Vector + Keyword)' },
        { value: 'keyword', label: 'Keyword (Exact Match)' }
      ];
    }
    
    if (lowerKey.includes('query') && lowerKey.includes('strategy')) {
      return [
        { value: 'auto', label: 'Auto (Adaptive)' },
        { value: 'simple', label: 'Simple' },
        { value: 'complex', label: 'Complex' },
        { value: 'focused', label: 'Focused' }
      ];
    }
    
    return null;
  };

  const validateRAGField = (key: string, value: any, category?: string): string | null => {
    if (category !== 'rag' && category !== 'large_generation') return null;
    
    const lowerKey = key.toLowerCase();
    
    if (typeof value === 'number') {
      // Validate numeric ranges (but skip Performance Optimization fields)
      if ((lowerKey.includes('threshold') || lowerKey.includes('similarity')) && category !== 'large_generation') {
        if (value < 0 || value > 1) {
          return 'Value must be between 0 and 1';
        }
      } else if (lowerKey.includes('weight') && category !== 'large_generation') {
        if (value < 0 || value > 1) {
          return 'Weight must be between 0 and 1';
        }
      } else if (lowerKey.includes('k1')) {
        if (value < 1.0 || value > 3.0) {
          return 'K1 parameter should be between 1.0 and 3.0';
        }
      } else if ((lowerKey.includes('bm25_b') || lowerKey === 'b') && lowerKey.includes('bm25')) {
        if (value < 0.0 || value > 1.0) {
          return 'B parameter should be between 0.0 and 1.0';
        }
      } else if (lowerKey.includes('batch') && lowerKey.includes('size')) {
        if (value < 1 || value > 10000) {
          return 'Batch size should be between 1 and 10000';
        }
      } else if (lowerKey.includes('top_k')) {
        if (value < 1 || value > 200) {
          return 'Top K should be between 1 and 200';
        }
      } else if (lowerKey.includes('timeout')) {
        if (value < 1) {
          return 'Timeout must be greater than 0';
        }
      } else if (lowerKey.includes('cache') && lowerKey.includes('size')) {
        if (value < 10 || value > 10000) {
          return 'Cache size should be between 10 and 10000';
        }
      }
      
      // Performance Optimization specific validations
      if (category === 'large_generation') {
        if (lowerKey.includes('strong_number_threshold')) {
          if (value < 1 || value > 100) {
            return 'Strong number threshold should be between 1 and 100';
          }
        } else if (lowerKey.includes('medium_number_threshold')) {
          if (value < 1 || value > 100) {
            return 'Medium number threshold should be between 1 and 100';
          }
        } else if (lowerKey.includes('small_number_threshold')) {
          if (value < 1 || value > 100) {
            return 'Small number threshold should be between 1 and 100';
          }
        } else if (lowerKey.includes('min_items_for_chunking')) {
          if (value < 1 || value > 100) {
            return 'Min items for chunking should be between 1 and 100';
          }
        } else if (lowerKey.includes('min_score_for_keywords')) {
          if (value < 1 || value > 10) {
            return 'Min score for keywords should be between 1 and 10';
          }
        } else if (lowerKey.includes('min_score_for_medium_numbers')) {
          if (value < 1 || value > 10) {
            return 'Min score for medium numbers should be between 1 and 10';
          }
        } else if (lowerKey.includes('score_multiplier')) {
          if (value < 1 || value > 50) {
            return 'Score multiplier should be between 1 and 50';
          }
        } else if (lowerKey.includes('pattern_score_weight')) {
          if (value < 1 || value > 10) {
            return 'Pattern score weight should be between 1 and 10';
          }
        } else if (lowerKey.includes('max_score_for_confidence')) {
          if (value < 1 || value > 20) {
            return 'Max score for confidence should be between 1 and 20';
          }
        } else if (lowerKey.includes('max_number_for_confidence')) {
          if (value < 10 || value > 500) {
            return 'Max number for confidence should be between 10 and 500';
          }
        } else if (lowerKey.includes('default_comprehensive_items')) {
          if (value < 5 || value > 100) {
            return 'Default comprehensive items should be between 5 and 100';
          }
        } else if (lowerKey.includes('min_estimated_items')) {
          if (value < 1 || value > 50) {
            return 'Min estimated items should be between 1 and 50';
          }
        } else if (lowerKey.includes('default_chunk_size')) {
          if (value < 5 || value > 50) {
            return 'Default chunk size should be between 5 and 50';
          }
        } else if (lowerKey.includes('max_target_count')) {
          if (value < 10 || value > 2000) {
            return 'Max target count should be between 10 and 2000';
          }
        } else if (lowerKey.includes('estimated_seconds_per_chunk')) {
          if (value < 1 || value > 300) {
            return 'Estimated seconds per chunk should be between 1 and 300';
          }
        } else if (lowerKey.includes('redis_conversation_ttl')) {
          if (value < 3600 || value > 604800) {
            return 'Redis conversation TTL should be between 1 hour and 7 days';
          }
        } else if (lowerKey.includes('max_redis_messages')) {
          if (value < 10 || value > 500) {
            return 'Max Redis messages should be between 10 and 500';
          }
        } else if (lowerKey.includes('max_memory_messages')) {
          if (value < 5 || value > 200) {
            return 'Max memory messages should be between 5 and 200';
          }
        } else if (lowerKey.includes('conversation_history_display')) {
          if (value < 1 || value > 50) {
            return 'Conversation history display should be between 1 and 50';
          }
        }
      }
    }
    
    return null;
  };

  const getLangfuseHelpText = (key: string): string | null => {
    const helpTexts: Record<string, string> = {
      // Connection Settings
      'host': 'Langfuse server URL (e.g., https://cloud.langfuse.com or your self-hosted instance)',
      'public_key': 'Public API key for Langfuse authentication - safe to expose in client-side code',
      'secret_key': 'Secret API key for Langfuse authentication - keep secure and private',
      'enabled': 'Enable or disable Langfuse monitoring globally',
      'enable_langfuse': 'Enable Langfuse integration for monitoring and tracing',
      'langfuse_enabled': 'Toggle Langfuse monitoring on/off',
      
      // Monitoring Configuration
      'sample_rate': 'Percentage of requests to monitor (0.0-1.0). Lower values reduce overhead.',
      'sampling_rate': 'Rate at which to sample requests for monitoring (0.0-1.0)',
      'trace_enabled': 'Enable detailed tracing of AI model calls and responses',
      'enable_tracing': 'Enable distributed tracing for request flow monitoring',
      'debug_mode': 'Enable debug mode for verbose logging and troubleshooting',
      'log_level': 'Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL',
      
      // Performance Settings
      'batch_size': 'Number of events to batch before sending to Langfuse (1-1000)',
      'flush_interval': 'Interval in seconds to flush batched events (1-300)',
      'timeout': 'Request timeout in seconds for Langfuse API calls (1-60)',
      'max_retries': 'Maximum number of retry attempts for failed requests (0-10)',
      'queue_size': 'Maximum size of the event queue before dropping events (100-10000)',
      'async_enabled': 'Enable asynchronous processing to reduce request latency',
      
      // Cost Tracking
      'cost_tracking': 'Enable cost tracking and budget monitoring',
      'enable_cost_tracking': 'Track costs for AI model usage and API calls',
      'budget_limit': 'Monthly budget limit in USD (0 = no limit)',
      'cost_threshold': 'Cost threshold for warnings (percentage of budget)',
      'currency': 'Currency code for cost tracking (USD, EUR, etc.)'
    };
    
    const lowerKey = key.toLowerCase();
    
    // Try exact match first
    if (helpTexts[lowerKey]) {
      return helpTexts[lowerKey];
    }
    
    // Try partial matches
    for (const [helpKey, helpText] of Object.entries(helpTexts)) {
      if (lowerKey.includes(helpKey.toLowerCase()) || helpKey.toLowerCase().includes(lowerKey)) {
        return helpText;
      }
    }
    
    return null;
  };

  const getEnvironmentHelpText = (key: string): string | null => {
    const helpTexts: Record<string, string> = {
      // API Keys & Authentication
      'openai_api_key': 'OpenAI API key for accessing GPT models and other OpenAI services',
      'anthropic_api_key': 'Anthropic API key for accessing Claude models',
      'google_api_key': 'Google API key for accessing Google Cloud services',
      'api_key': 'General API key for authentication with external services',
      'secret_key': 'Secret key used for signing and encryption operations',
      'access_token': 'Access token for API authentication',
      'auth_token': 'Authentication token for service access',
      
      // Database Configuration
      'database_url': 'Full database connection URL with credentials',
      'db_host': 'Database server hostname or IP address',
      'db_port': 'Database server port number (default: 5432 for PostgreSQL, 3306 for MySQL)',
      'db_user': 'Database username for authentication',
      'db_password': 'Database password for authentication',
      'redis_url': 'Redis server connection URL for caching and sessions',
      'postgres_url': 'PostgreSQL database connection URL',
      'mysql_url': 'MySQL database connection URL',
      
      // Service URLs & Endpoints
      'base_url': 'Base URL for the application or API service',
      'endpoint': 'API endpoint URL for external service integration',
      'webhook_url': 'Webhook URL for receiving callbacks from external services',
      'callback_url': 'Callback URL for OAuth and authentication flows',
      'service_url': 'External service URL for API calls',
      'api_url': 'API base URL for making requests',
      
      // Runtime Configuration
      'environment': 'Runtime environment (development, staging, production)',
      'debug': 'Enable debug mode for verbose logging (true/false)',
      'log_level': 'Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL',
      'port': 'Server port number (default: 8000)',
      'host': 'Server hostname or IP address (default: 0.0.0.0)',
      'timeout': 'Request timeout in seconds',
      'max_workers': 'Maximum number of worker processes',
      'workers': 'Number of worker processes to spawn'
    };
    
    const lowerKey = key.toLowerCase();
    
    // Try exact match first
    if (helpTexts[lowerKey]) {
      return helpTexts[lowerKey];
    }
    
    // Try partial matches
    for (const [helpKey, helpText] of Object.entries(helpTexts)) {
      if (lowerKey.includes(helpKey.toLowerCase()) || helpKey.toLowerCase().includes(lowerKey)) {
        return helpText;
      }
    }
    
    return null;
  };

  const getLLMHelpText = (key: string): string | null => {
    const helpTexts: Record<string, string> = {
      // Search Optimization
      'enable_search_optimization': 'Enable AI-powered search query optimization to improve search results by transforming user queries into better search terms.',
      'optimization_timeout': 'Maximum time in seconds to wait for query optimization. Lower values = faster response but may timeout on complex queries. Recommended: 8-15 seconds.',
      'optimization_prompt': 'Template prompt used by the LLM to optimize search queries. Includes guidelines and examples for query transformation.',
      'search_optimization.enable_search_optimization': 'Enable AI-powered search query optimization to improve search results by transforming user queries into better search terms.',
      'search_optimization.optimization_timeout': 'Maximum time in seconds to wait for query optimization. Lower values = faster response but may timeout on complex queries. Recommended: 8-15 seconds.',
      'search_optimization.optimization_prompt': 'Template prompt used by the LLM to optimize search queries. Includes guidelines and examples for query transformation.',
      
      // Conflict Prevention Settings
      'conflict_prevention.enabled': 'Enable conflict prevention to detect and handle conflicting or potentially harmful queries.',
      'conflict_prevention.disclaimer_patterns': 'List of patterns that trigger disclaimer messages when detected in queries.',
      'conflict_prevention.severity_thresholds.high': 'Threshold score for high severity conflicts that require special handling.',
      'conflict_prevention.severity_thresholds.medium': 'Threshold score for medium severity conflicts that trigger warnings.',
      'conflict_prevention.severity_thresholds.critical': 'Threshold score for critical conflicts that may block query execution.',
      'conflict_prevention.allow_with_disclaimers': 'Allow queries to proceed with disclaimers even when conflicts are detected.',
      
      // Main LLM Settings
      'main_llm.model': 'Primary LLM model used for main chat responses and reasoning tasks.',
      'main_llm.max_tokens': 'Maximum number of tokens the main LLM can generate in a single response.',
      'main_llm.context_length': 'Maximum context window size for the main LLM in tokens.',
      'main_llm.system_prompt': 'System prompt that defines the behavior and personality of the main LLM.',
      'main_llm.model_server': 'Server endpoint URL for the main LLM model.',
      'main_llm.repeat_penalty': 'Penalty applied to repeated tokens to encourage diverse responses. Higher values reduce repetition.',
      
      // Second LLM Settings
      'second_llm.model': 'Secondary LLM model used for specialized tasks or fallback scenarios.',
      'second_llm.max_tokens': 'Maximum number of tokens the second LLM can generate in a single response.',
      'second_llm.context_length': 'Maximum context window size for the second LLM in tokens.',
      'second_llm.system_prompt': 'System prompt that defines the behavior and personality of the second LLM.',
      'second_llm.model_server': 'Server endpoint URL for the second LLM model.',
      'second_llm.repeat_penalty': 'Penalty applied to repeated tokens to encourage diverse responses. Higher values reduce repetition.',
      
      // Knowledge Graph Settings
      'knowledge_graph.model': 'LLM model used for knowledge graph entity and relationship extraction.',
      'knowledge_graph.max_tokens': 'Maximum number of tokens the knowledge graph LLM can generate for extraction tasks.',
      'knowledge_graph.context_length': 'Maximum context window size for the knowledge graph LLM in tokens.',
      'knowledge_graph.system_prompt': 'System prompt that defines the behavior for knowledge graph extraction tasks.',
      'knowledge_graph.model_server': 'Server endpoint URL for the knowledge graph LLM model.',
      'knowledge_graph.repeat_penalty': 'Penalty applied to repeated tokens during knowledge graph extraction.',
      'knowledge_graph.temperature': 'Temperature setting for knowledge graph extraction. Lower values = more deterministic extraction.',
      'knowledge_graph.extraction_prompt': 'Template prompt used for extracting entities and relationships from text.',
      'knowledge_graph.entity_types': 'List of entity types to extract (e.g., Person, Organization, Location, Event).',
      'knowledge_graph.relationship_types': 'List of relationship types to identify (e.g., works_for, located_in, part_of).',
      'knowledge_graph.max_entities_per_chunk': 'Maximum number of entities to extract from each text chunk.',
      'knowledge_graph.enable_coreference_resolution': 'Enable coreference resolution to link pronouns and references to entities.',
      
      // Knowledge Graph Neo4j Database Settings
      'knowledge_graph.neo4j.enabled': 'Enable or disable Neo4j knowledge graph database connection.',
      'knowledge_graph.neo4j.host': 'Neo4j database server hostname or IP address.',
      'knowledge_graph.neo4j.port': 'Neo4j Bolt protocol port (default: 7687).',
      'knowledge_graph.neo4j.http_port': 'Neo4j HTTP interface port for browser access (default: 7474).',
      'knowledge_graph.neo4j.database': 'Neo4j database name to connect to.',
      'knowledge_graph.neo4j.username': 'Username for Neo4j database authentication.',
      'knowledge_graph.neo4j.password': 'Password for Neo4j database authentication.',
      'knowledge_graph.neo4j.uri': 'Complete Neo4j connection URI (bolt://host:port).',
      'knowledge_graph.neo4j.connection_pool.max_connections': 'Maximum number of concurrent connections to Neo4j.',
      'knowledge_graph.neo4j.connection_pool.connection_timeout': 'Connection timeout in seconds.',
      'knowledge_graph.neo4j.connection_pool.max_transaction_retry_time': 'Maximum time to retry failed transactions.',
      'knowledge_graph.neo4j.memory_config.heap_initial': 'Initial heap size for Neo4j (e.g., 512m).',
      'knowledge_graph.neo4j.memory_config.heap_max': 'Maximum heap size for Neo4j (e.g., 2g).',
      'knowledge_graph.neo4j.memory_config.pagecache': 'Page cache size for Neo4j (e.g., 1g).',
      'knowledge_graph.neo4j.plugins.apoc_enabled': 'Enable APOC (Awesome Procedures On Cypher) plugin for advanced graph operations.',
      'knowledge_graph.neo4j.plugins.gds_enabled': 'Enable Graph Data Science plugin for graph algorithms and analytics.',
      'knowledge_graph.neo4j.security.encrypted': 'Enable SSL/TLS encryption for Neo4j connections.',
      'knowledge_graph.neo4j.security.trust_strategy': 'SSL certificate trust strategy (TRUST_ALL_CERTIFICATES, TRUST_SYSTEM_CA_SIGNED_CERTIFICATES).',
      
      // Query Classifier Settings
      'query_classifier.model': 'LLM model used for query classification and routing decisions.',
      'query_classifier.max_tokens': 'Maximum number of tokens the query classifier can use for analysis.',
      'query_classifier.model_server': 'Server endpoint URL for the query classifier LLM model.',
      'query_classifier.system_prompt': 'System prompt that guides the query classifier in making routing decisions.',
      'query_classifier.min_confidence_threshold': 'Minimum confidence score required for classification decisions. Lower values = more permissive.',
      'query_classifier.enable_llm_classification': 'Use LLM-based classification in addition to pattern-based classification for better accuracy.',
      'query_classifier.llm_direct_threshold': 'Confidence threshold for routing queries directly to LLM without additional processing.',
      'query_classifier.multi_agent_threshold': 'Confidence threshold for routing queries to multi-agent workflows.',
      'query_classifier.direct_execution_threshold': 'Confidence threshold for direct tool execution without additional validation.',
      
      // Thinking Mode Parameters
      'thinking_mode_params.temperature': 'Controls randomness in thinking mode responses. Higher values = more creative but less predictable.',
      'thinking_mode_params.top_p': 'Controls diversity via nucleus sampling in thinking mode. Lower values = more focused responses.',
      'thinking_mode_params.top_k': 'Limits vocabulary to top K tokens in thinking mode. Lower values = more constrained responses.',
      'thinking_mode_params.min_p': 'Minimum probability threshold for token selection in thinking mode.',
      
      // Non-Thinking Mode Parameters
      'non_thinking_mode_params.temperature': 'Controls randomness in non-thinking mode responses. Higher values = more creative but less predictable.',
      'non_thinking_mode_params.top_p': 'Controls diversity via nucleus sampling in non-thinking mode. Lower values = more focused responses.',
      'non_thinking_mode_params.top_k': 'Limits vocabulary to top K tokens in non-thinking mode. Lower values = more constrained responses.',
      'non_thinking_mode_params.min_p': 'Minimum probability threshold for token selection in non-thinking mode.'
    };
    
    const lowerKey = key.toLowerCase();
    
    // Try exact match first
    if (helpTexts[lowerKey]) {
      return helpTexts[lowerKey];
    }
    
    // Try partial matches
    for (const [helpKey, helpText] of Object.entries(helpTexts)) {
      if (lowerKey.includes(helpKey.toLowerCase()) || helpKey.toLowerCase().includes(lowerKey)) {
        return helpText;
      }
    }
    
    return null;
  };

  const renderField = (key: string, value: any, depth: number = 0, customOnChange?: (field: string, value: any) => void, fieldCategory?: string, onShowSuccessCallback?: (message?: string) => void) => {
    const formatLabel = (str: string) => {
      // Remove redundant "settings." prefix if present
      let cleanStr = str;
      if (cleanStr.startsWith('settings.settings.')) {
        cleanStr = cleanStr.replace('settings.settings.', '');
      } else if (cleanStr.startsWith('settings.')) {
        cleanStr = cleanStr.replace('settings.', '');
      }
      
      // Custom labels for query classifier fields
      const customLabels = {
        // Classification thresholds
        'min_confidence_threshold': 'Minimum Confidence Threshold',
        'direct_execution_threshold': 'Direct Tool Execution Threshold',
        'llm_direct_threshold': 'LLM Direct Response Threshold',
        'multi_agent_threshold': 'Multi-Agent Task Threshold',
        
        // Pattern-based classification
        'max_classifications': 'Maximum Classifications',
        'enable_hybrid_detection': 'Enable Hybrid Detection',
        'confidence_decay_factor': 'Confidence Decay Factor',
        'pattern_combination_bonus': 'Pattern Combination Bonus',
        
        // LLM-based classification
        'enable_llm_classification': 'Enable LLM Classification',
        'llm_model': 'LLM Model',
        'context_length': 'Context Length',
        'llm_temperature': 'LLM Temperature',
        'llm_max_tokens': 'LLM Max Tokens',
        'llm_timeout_seconds': 'LLM Timeout (seconds)',
        'llm_system_prompt': 'LLM System Prompt',
        'fallback_to_patterns': 'Fallback to Patterns',
        'llm_classification_priority': 'Use LLM First (vs Patterns First)',
        
        // Nested field handling
        'query_classifier.min_confidence_threshold': 'Minimum Confidence Threshold',
        'query_classifier.direct_execution_threshold': 'Direct Tool Execution Threshold',
        'query_classifier.llm_direct_threshold': 'LLM Direct Response Threshold',
        'query_classifier.multi_agent_threshold': 'Multi-Agent Task Threshold',
        'query_classifier.max_classifications': 'Maximum Classifications',
        'query_classifier.enable_hybrid_detection': 'Enable Hybrid Detection',
        'query_classifier.confidence_decay_factor': 'Confidence Decay Factor',
        'query_classifier.pattern_combination_bonus': 'Pattern Combination Bonus',
        'query_classifier.enable_llm_classification': 'Enable LLM Classification',
        'query_classifier.llm_model': 'LLM Model',
        'query_classifier.context_length': 'Context Length',
        'query_classifier.llm_temperature': 'LLM Temperature',
        'query_classifier.llm_max_tokens': 'LLM Max Tokens',
        'query_classifier.llm_timeout_seconds': 'LLM Timeout (seconds)',
        'query_classifier.llm_system_prompt': 'LLM System Prompt',
        'query_classifier.fallback_to_patterns': 'Fallback to Patterns',
        'query_classifier.llm_classification_priority': 'Use LLM First (vs Patterns First)',
        'query_classifier.model_server': 'Model Server URL',
        
        // Search Optimization fields
        'search_optimization.enable_search_optimization': 'Enable Search Optimization',
        'search_optimization.optimization_timeout': 'Optimization Timeout (seconds)',
        'search_optimization.optimization_prompt': 'Optimization Prompt Template',
        'search_optimization.top_p': 'Top P',
        'enable_search_optimization': 'Enable Search Optimization',
        'optimization_timeout': 'Optimization Timeout (seconds)',
        'optimization_prompt': 'Optimization Prompt Template',
        'top_p': 'Top P',
        
        // Conflict Prevention fields
        'conflict_prevention.enabled': 'Conflict Prevention Enabled',
        'conflict_prevention.disclaimer_patterns': 'Disclaimer Patterns',
        'conflict_prevention.severity_thresholds.high': 'High Severity Threshold',
        'conflict_prevention.severity_thresholds.medium': 'Medium Severity Threshold',
        'conflict_prevention.severity_thresholds.critical': 'Critical Severity Threshold',
        'conflict_prevention.allow_with_disclaimers': 'Allow With Disclaimers'
      };
      
      // Check for custom label first
      if (customLabels[cleanStr]) {
        return customLabels[cleanStr];
      }
      
      // Default: replace underscores with spaces and capitalize words
      return cleanStr.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    };
    const onChangeHandler = customOnChange || onChange;
    
    // Detect field complexity for layout
    const isComplexField = (key: string, value: any): boolean => {
      const lowerKey = key.toLowerCase();
      
      // Arrays are always complex
      if (Array.isArray(value)) return true;
      
      // Long text fields (textareas) are complex
      if (typeof value === 'string' && value.length > 100) return true;
      
      // System prompts and other prompt fields are complex
      if (lowerKey.includes('prompt') || lowerKey.includes('system')) return true;
      
      // Slider parameters (identified by specific naming patterns)
      if (lowerKey.includes('temperature') || lowerKey.includes('top_p') || 
          lowerKey.includes('top_k') || lowerKey.includes('penalty')) return true;
      
      return false;
    };
    
    const isComplex = isComplexField(key, value);
    const fieldClass = `jarvis-form-group${isComplex ? ' complex' : ''}`;
    const validationError = validateRAGField(key, value, fieldCategory);
    
    // Helper function to render label with help tooltip
    const renderLabelWithHelp = (labelText: string, helpText: string | null = null, isInlineCheckbox: boolean = false) => {
      if ((fieldCategory === 'rag' || fieldCategory === 'large_generation' || fieldCategory === 'overflow' || fieldCategory === 'langfuse' || fieldCategory === 'environment' || fieldCategory === 'llm') && helpText) {
        return (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
            <span>{labelText}</span>
            <Tooltip title={helpText} placement="top-start" arrow>
              <HelpOutlineIcon 
                sx={{ 
                  fontSize: '16px', 
                  color: 'text.secondary',
                  cursor: 'help',
                  '&:hover': { color: 'primary.main' }
                }} 
              />
            </Tooltip>
          </span>
        );
      }
      return labelText;
    };

    // Helper function to render validation error
    const renderValidationError = (error: string | null) => {
      if (!error) return null;
      return (
        <div style={{ 
          color: '#f44336', 
          fontSize: '12px', 
          marginTop: '4px',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}>
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      );
    };
    
    if (typeof value === 'boolean') {
      const helpText = fieldCategory === 'rag' ? getRAGHelpText(key) : fieldCategory === 'large_generation' ? getPerformanceHelpText(key) : fieldCategory === 'langfuse' ? getLangfuseHelpText(key) : fieldCategory === 'environment' ? getEnvironmentHelpText(key) : fieldCategory === 'llm' ? getLLMHelpText(key) : fieldCategory === 'overflow' ? getOverflowHelpText(key) : null;
      return (
        <div key={key} className={fieldClass} style={{ marginLeft: `${depth * 20}px` }}>
          <label className="jarvis-form-label">
            <input
              type="checkbox"
              checked={value}
              onChange={(e) => onChangeHandler(key, e.target.checked)}
              style={{ marginRight: '8px' }}
            />
            {renderLabelWithHelp(formatLabel(key), helpText, true)}
          </label>
          {renderValidationError(validationError)}
        </div>
      );
    }

    if (typeof value === 'number') {
      const helpText = fieldCategory === 'rag' ? getRAGHelpText(key) : fieldCategory === 'large_generation' ? getPerformanceHelpText(key) : fieldCategory === 'langfuse' ? getLangfuseHelpText(key) : fieldCategory === 'environment' ? getEnvironmentHelpText(key) : fieldCategory === 'llm' ? getLLMHelpText(key) : fieldCategory === 'overflow' ? getOverflowHelpText(key) : null;
      
      // Check if this looks like a slider parameter (temperature, top_p, etc.)
      const isSliderParam = key.toLowerCase().includes('temperature') || 
                          key.toLowerCase().includes('top_p') || 
                          key.toLowerCase().includes('top_k') || 
                          key.toLowerCase().includes('penalty') ||
                          // RAG-specific slider parameters
                          (fieldCategory === 'rag' && (
                            key.toLowerCase().includes('weight') ||
                            key.toLowerCase().includes('threshold') ||
                            key.toLowerCase().includes('score') ||
                            key.toLowerCase().includes('similarity')
                          ));
      
      if (isSliderParam) {
        const lowerKey = key.toLowerCase();
        let min = 0, max = 1, step = 0.01;
        
        if (lowerKey.includes('top_k')) {
          min = 0; max = 100; step = 1;
        } else if (lowerKey.includes('temperature')) {
          min = 0; max = 2; step = 0.01;
        } else if (lowerKey.includes('top_p')) {
          min = 0; max = 1; step = 0.01;
        } else if (lowerKey.includes('penalty')) {
          min = 1; max = 2; step = 0.01;
        } else if (fieldCategory === 'rag') {
          // RAG-specific slider ranges
          if (lowerKey.includes('weight')) {
            min = 0; max = 1; step = 0.01;
          } else if (lowerKey.includes('threshold') || lowerKey.includes('similarity')) {
            min = 0; max = 1; step = 0.01;
          } else if (lowerKey.includes('score')) {
            min = 0; max = 1; step = 0.01;
          } else if (lowerKey.includes('k1')) {
            min = 1.0; max = 3.0; step = 0.1;
          } else if (lowerKey.includes('b') && lowerKey.includes('bm25')) {
            min = 0.0; max = 1.0; step = 0.05;
          }
        }
        
        return (
          <div key={key} className={fieldClass} style={{ marginLeft: `${depth * 20}px` }}>
            <label className="jarvis-form-label">{renderLabelWithHelp(formatLabel(key), helpText)}</label>
            <div className="jarvis-slider-container">
              <div className="jarvis-slider-header">
                <span></span>
                <span className="jarvis-slider-value">{value}</span>
              </div>
              <input
                type="range"
                className="jarvis-slider"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={(e) => onChangeHandler(key, parseFloat(e.target.value))}
              />
              <div className="jarvis-slider-labels">
                <span>{min}</span>
                <span>{max}</span>
              </div>
            </div>
            {renderValidationError(validationError)}
          </div>
        );
      } else {
        return (
          <div key={key} className={fieldClass} style={{ marginLeft: `${depth * 20}px` }}>
            <label className="jarvis-form-label">{renderLabelWithHelp(formatLabel(key), helpText)}</label>
            <input
              type="number"
              className="jarvis-form-input"
              value={value}
              onChange={(e) => onChangeHandler(key, parseFloat(e.target.value) || 0)}
              style={{ borderColor: validationError ? '#f44336' : undefined }}
            />
            {renderValidationError(validationError)}
          </div>
        );
      }
    }

    if (typeof value === 'string') {
      const lowerKey = key.toLowerCase();
      const helpText = fieldCategory === 'rag' ? getRAGHelpText(key) : fieldCategory === 'large_generation' ? getPerformanceHelpText(key) : fieldCategory === 'langfuse' ? getLangfuseHelpText(key) : fieldCategory === 'environment' ? getEnvironmentHelpText(key) : fieldCategory === 'llm' ? getLLMHelpText(key) : fieldCategory === 'overflow' ? getOverflowHelpText(key) : null;
      
      // Always use textarea for prompt fields to prevent height changes while typing
      const isLongText = value.length > 100 || lowerKey.includes('prompt') || lowerKey.includes('system');
      
      // Debug logging for max_tokens
      if (lowerKey.includes('max_tokens') || lowerKey.includes('maxtoken')) {
        console.log('[DEBUG] Max tokens field:', key, 'Value:', value, 'Type:', typeof value);
      }
      
      // Special handling for model field in LLM category
      console.log('[DEBUG] Checking model field:', { fieldCategory, key, value, fullKey: key, customOnChange: !!customOnChange });
      // Check for both 'model' and 'settings.model' due to potential nesting, including llm_model and second_llm.model
      if (fieldCategory === 'llm' && (key === 'model' || key === 'settings.model' || key.endsWith('.model') || key.endsWith('.llm_model') || key === 'llm_model' || key === 'main_llm.model' || key === 'second_llm.model' || key === 'knowledge_graph.model')) {
        console.log('[DEBUG] Rendering model selector for LLM, using onChangeHandler:', onChangeHandler.toString().substring(0, 100));
        return (
          <ModelSelector
            key={key}
            fieldKey={key}
            value={value}
            onChangeHandler={onChangeHandler}
            depth={depth}
            onShowSuccess={onShowSuccessCallback || onShowSuccess}
            customOnChange={customOnChange}
          />
        );
      }
      
      // Explicitly exclude max_tokens from password fields
      const isPassword = !lowerKey.includes('max_tokens') && !lowerKey.includes('maxtoken') && (
                        lowerKey.includes('password') || 
                        lowerKey.includes('secret') || 
                        lowerKey.includes('api_key') ||
                        lowerKey.includes('apikey') ||
                        lowerKey.includes('access_key') ||
                        lowerKey.includes('token') ||
                        lowerKey.includes('credentials'));
      
      const isVisible = passwordVisibility[key] || false;
      
      // Check if this should be a dropdown for RAG settings
      const shouldUseDropdown = fieldCategory === 'rag' && !isPassword && !isLongText;
      const dropdownOptions = shouldUseDropdown ? getRAGDropdownOptions(key) : null;
      
      return (
        <div key={key} className={fieldClass} style={{ marginLeft: `${depth * 20}px` }}>
          <label className="jarvis-form-label">{renderLabelWithHelp(formatLabel(key), helpText)}</label>
          {dropdownOptions ? (
            <select
              className="jarvis-form-select"
              value={value}
              onChange={(e) => onChangeHandler(key, e.target.value)}
            >
              {dropdownOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          ) : isLongText ? (
            <textarea
              className="jarvis-form-textarea"
              value={value}
              onChange={(e) => onChangeHandler(key, e.target.value)}
              placeholder={`Enter ${formatLabel(key).toLowerCase()}...`}
            />
          ) : (
            <div className="jarvis-input-container" style={{ position: 'relative' }}>
              <input
                type={isPassword && !isVisible ? 'password' : 'text'}
                className="jarvis-form-input"
                value={value}
                onChange={(e) => onChangeHandler(key, e.target.value)}
                placeholder={`Enter ${formatLabel(key).toLowerCase()}...`}
                style={isPassword ? { paddingRight: '40px' } : {}}
              />
              {isPassword && (
                <IconButton
                  onClick={() => togglePasswordVisibility(key)}
                  style={{
                    position: 'absolute',
                    right: '8px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    padding: '4px',
                    color: 'var(--jarvis-text-secondary)'
                  }}
                  size="small"
                >
                  {isVisible ? <VisibilityOffIcon fontSize="small" /> : <VisibilityIcon fontSize="small" />}
                </IconButton>
              )}
            </div>
          )}
          {key.toLowerCase().includes('server') && value && (
            <div className="jarvis-status-indicator jarvis-status-connected">
              <div className="jarvis-connection-dot"></div>
              Connected
            </div>
          )}
        </div>
      );
    }

    if (Array.isArray(value)) {
      // Special handling for prompts arrays - use PromptManagement component instead of raw JSON
      if (key === 'prompts' || (Array.isArray(value) && value.length > 0 && value[0]?.prompt_template)) {
        return (
          <div key={key} className={fieldClass} style={{ marginLeft: `${depth * 20}px`, gridColumn: '1 / -1' }}>
            <PromptManagement
              data={value}
              onChange={(field, newValue) => onChangeHandler(key, newValue)}
              onShowSuccess={onShowSuccess}
            />
          </div>
        );
      }
      
      const helpText = fieldCategory === 'rag' ? getRAGHelpText(key) : fieldCategory === 'large_generation' ? getPerformanceHelpText(key) : fieldCategory === 'langfuse' ? getLangfuseHelpText(key) : fieldCategory === 'environment' ? getEnvironmentHelpText(key) : fieldCategory === 'llm' ? getLLMHelpText(key) : fieldCategory === 'overflow' ? getOverflowHelpText(key) : null;
      return (
        <div key={key} className={fieldClass} style={{ marginLeft: `${depth * 20}px` }}>
          <label className="jarvis-form-label">{renderLabelWithHelp(formatLabel(key), helpText)}</label>
          <div className="settings-section" style={{ padding: 'var(--jarvis-spacing-md)', maxHeight: '50vh', overflow: 'auto' }}>
            {value.map((item, index) => (
              <div key={index} style={{ display: 'flex', gap: '8px', marginBottom: '8px', alignItems: 'center' }}>
                <input
                  className="jarvis-form-input"
                  value={typeof item === 'object' ? JSON.stringify(item) : item}
                  onChange={(e) => {
                    const newArray = [...value];
                    try {
                      newArray[index] = JSON.parse(e.target.value);
                    } catch {
                      newArray[index] = e.target.value;
                    }
                    onChangeHandler(key, newArray);
                  }}
                  style={{ flex: 1 }}
                />
                <button 
                  className="jarvis-btn jarvis-btn-secondary"
                  onClick={() => {
                    const newArray = value.filter((_, i) => i !== index);
                    onChangeHandler(key, newArray);
                  }}
                  style={{ padding: '6px 12px', fontSize: '12px' }}
                >
                  Remove
                </button>
              </div>
            ))}
            <button 
              className="jarvis-btn jarvis-btn-primary"
              onClick={() => onChangeHandler(key, [...value, ''])}
              style={{ padding: '6px 12px', fontSize: '12px' }}
            >
              Add Item
            </button>
          </div>
        </div>
      );
    }

    // Special handling for vector_db configuration
    if (key === 'vector_db' && typeof value === 'object' && value !== null) {
      console.log('[DEBUG] vector_db value:', value);
      
      // Get embedding values from parent data
      const parentData = data || {};
      const embeddingModel = parentData.embedding_model || '';
      const embeddingEndpoint = parentData.embedding_endpoint || '';
      
      return (
        <div key={key} style={{ marginLeft: `${depth * 20}px`, marginBottom: '16px' }}>
          <VectorDatabaseManager
            data={value}
            onChange={(updatedValue) => onChangeHandler(key, updatedValue)}
            embeddingModel={embeddingModel}
            embeddingEndpoint={embeddingEndpoint}
            onEmbeddingChange={onChange}
          />
        </div>
      );
    }
    
    
    // Special handling for Iceberg configuration
    if (key === 'iceberg' && typeof value === 'object' && value !== null) {
      return (
        <Card
          key={key}
          variant="outlined"
          className="jarvis-form-group full-width"
          sx={{
            marginLeft: `${depth * 20}px`,
            marginBottom: '16px',
            gridColumn: '1 / -1'
          }}
        >
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Apache Iceberg Configuration
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {/* Catalog Settings Section */}
              <Box>
                <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600 }}>
                  Catalog Settings
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {['uri', 'name', 'token', 'warehouse'].map(field => {
                    if (value[field] !== undefined) {
                      const isPasswordField = field === 'token';
                      const fieldKey = `iceberg.${field}`;
                      const isVisible = icebergPasswordVisibility[fieldKey] || false;
                      
                      return (
                        <TextField
                          key={field}
                          label={field.charAt(0).toUpperCase() + field.slice(1)}
                          value={value[field]}
                          onChange={(e) => {
                            const updatedValue = { ...value, [field]: e.target.value };
                            onChangeHandler(key, updatedValue);
                          }}
                          fullWidth
                          type={isPasswordField && !isVisible ? 'password' : 'text'}
                          placeholder={field === 'uri' ? 'https://catalog.example.com/...' : ''}
                          InputProps={isPasswordField ? {
                            endAdornment: (
                              <IconButton
                                onClick={() => setIcebergPasswordVisibility(prev => ({ 
                                  ...prev, 
                                  [fieldKey]: !prev[fieldKey] 
                                }))}
                                edge="end"
                                size="small"
                              >
                                {isVisible ? <VisibilityOffIcon fontSize="small" /> : <VisibilityIcon fontSize="small" />}
                              </IconButton>
                            )
                          } : undefined}
                        />
                      );
                    }
                    return null;
                  })}
                </Box>
              </Box>
              
              {/* S3 Configuration Section */}
              {Object.entries(value).some(([k]) => k.startsWith('s3.')) && (
                <Box>
                  <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600 }}>
                    S3 Configuration
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {Object.entries(value).filter(([k]) => k.startsWith('s3.')).map(([nestedKey, nestedValue]) => {
                      const fieldName = nestedKey.replace('s3.', '');
                      const isPasswordField = fieldName.toLowerCase().includes('secret') || fieldName.toLowerCase().includes('key');
                      const fieldKey = `iceberg.${nestedKey}`;
                      const isVisible = icebergPasswordVisibility[fieldKey] || false;
                      
                      return (
                        <TextField
                          key={nestedKey}
                          label={fieldName.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                          value={nestedValue as string}
                          onChange={(e) => {
                            const updatedValue = { ...value, [nestedKey]: e.target.value };
                            onChangeHandler(key, updatedValue);
                          }}
                          fullWidth
                          type={isPasswordField && !isVisible ? 'password' : 'text'}
                          InputProps={isPasswordField ? {
                            endAdornment: (
                              <IconButton
                                onClick={() => setIcebergPasswordVisibility(prev => ({ 
                                  ...prev, 
                                  [fieldKey]: !prev[fieldKey] 
                                }))}
                                edge="end"
                                size="small"
                              >
                                {isVisible ? <VisibilityOffIcon fontSize="small" /> : <VisibilityIcon fontSize="small" />}
                              </IconButton>
                            )
                          } : undefined}
                        />
                      );
                    })}
                  </Box>
                </Box>
              )}
            </Box>
          </CardContent>
        </Card>
      );
    }
    
    // Handle nested objects - but flatten query_classifier, main_llm, second_llm, knowledge_graph, and search_optimization to work like Settings
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      if (key === 'query_classifier' || key === 'main_llm' || key === 'second_llm' || key === 'knowledge_graph' || key === 'search_optimization') {
        // Flatten these fields to work like Settings fields, but skip mode fields and apply ordering
        const getFieldOrder = (fieldKey: string): number => {
          const lowerKey = fieldKey.toLowerCase();
          const orderMap: Record<string, number> = {
            'model': 100,
            'max_tokens': 200,
            'model_server': 300,
            'system_prompt': 400,
            'context_length': 500,
            'repeat_penalty': 600,
            'temperature': 700,
            'top_p': 800,
            'top_k': 900,
            'min_p': 1000,
            'stop': 1100 // Stop at bottom
          };
          return orderMap[lowerKey] || 10000;
        };
        
        const sortedEntries = Object.entries(value).sort(([keyA], [keyB]) => 
          getFieldOrder(keyA) - getFieldOrder(keyB)
        );
        
        return (
          <div key={key} style={{ marginLeft: `${depth * 20}px`, marginBottom: '16px' }}>
            {sortedEntries.map(([nestedKey, nestedValue]) => {
              // Skip mode field for nested objects
              if (nestedKey === 'mode') {
                return null;
              }
              return renderField(`${key}.${nestedKey}`, nestedValue, depth, onChangeHandler, fieldCategory);
            })}
          </div>
        );
      } else {
        // Normal nested object handling for other cases
        return (
          <div key={key} className="jarvis-nested-group" style={{ marginLeft: `${depth * 20}px`, marginBottom: '16px' }}>
            <h4 className="jarvis-nested-title" style={{ marginBottom: '12px', fontSize: '14px', fontWeight: 600 }}>
              {formatLabel(key)}
            </h4>
            <div style={{ marginLeft: '16px' }}>
              {Object.entries(value).map(([nestedKey, nestedValue]) => 
                renderField(`${key}.${nestedKey}`, nestedValue, depth + 1, (field, val) => {
                  console.log('[DEBUG] Nested field onChange:', { field, nestedKey, val, parentKey: key });
                  const updatedValue = { ...value, [nestedKey]: val };
                  onChangeHandler(key, updatedValue);
                }, fieldCategory)
              )}
            </div>
          </div>
        );
      }
    }

    return (
      <div key={key} className="jarvis-form-group" style={{ marginLeft: `${depth * 20}px` }}>
        <div className="jarvis-help-text">
          {key}: {String(value)} (type: {typeof value})
        </div>
      </div>
    );
  };

  const categories = categorizeFields(data, category);
  const categoryKeys = Object.keys(categories);
  
  // Set first available category as active if current active tab doesn't exist
  // Removed useEffect to avoid hooks in non-component functions
  if (categoryKeys.length > 0 && !categoryKeys.includes(activeTab)) {
    setActiveTab(categoryKeys[0]);
  }

  if (categoryKeys.length === 0) {
    return <div className="jarvis-help-text">No settings available to configure.</div>;
  }

  return (
    <div className="jarvis-tabs">
      {/* Tab Navigation */}
      <div className="jarvis-tab-list">
        {categoryKeys.map((categoryKey) => (
          <button
            key={categoryKey}
            className={`jarvis-tab-button ${activeTab === categoryKey ? 'active' : ''}`}
            onClick={() => setActiveTab(categoryKey)}
          >
            {categories[categoryKey].title}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {categoryKeys.map((categoryKey) => (
        <div
          key={categoryKey}
          className={`jarvis-tab-content ${activeTab === categoryKey ? 'active' : ''}`}
        >
          <div className="jarvis-tab-panel">
            {/* Special rendering for Thinking Mode tab */}
            {categoryKey === 'thinking' ? (
              <div style={{ padding: '16px', maxWidth: '1200px', margin: '0 auto' }}>
                {/* General Information */}
                <Alert severity="info" sx={{ mb: 3 }}>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    <strong>Parameter Configuration Guidelines</strong>
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    For more information, refer to{' '}
                    <a 
                      href="https://huggingface.co/Qwen/Qwen3-4B-MLX-4bit#best-practices" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      style={{ color: 'inherit', textDecoration: 'underline' }}
                    >
                      Qwen3 Best Practices
                    </a>
                  </Typography>
                </Alert>
                
                <div style={{ 
                  display: 'grid', 
                  gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', 
                  gap: '24px'
                }}>
                {/* Thinking Mode Card */}
                <Card variant="outlined" sx={{ height: 'fit-content' }}>
                  <CardHeader 
                    title="Thinking Mode Parameters"
                    subheader="Used when the model shows step-by-step reasoning"
                    sx={{ pb: 1 }}
                  />
                  <CardContent>
                    <Alert severity="info" sx={{ mb: 2 }}>
                      <Typography variant="body2">
                        <strong>Recommended:</strong> Temperature=0.6, TopP=0.95, TopK=20, MinP=0
                      </Typography>
                    </Alert>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      {(() => {
                        const thinkingModeParams = data.thinking_mode_params || {};
                        const parameterFields = [
                          { key: 'temperature', label: 'Temperature', type: 'number', step: 0.1, min: 0, max: 2 },
                          { key: 'top_p', label: 'Top P', type: 'number', step: 0.05, min: 0, max: 1 },
                          { key: 'top_k', label: 'Top K', type: 'number', step: 1, min: 0, max: 100 },
                          { key: 'min_p', label: 'Min P', type: 'number', step: 0.01, min: 0, max: 1 }
                        ];
                        
                        return parameterFields.map(({ key, label, type, step, min, max }) => (
                          <TextField
                            key={key}
                            label={label}
                            type={type}
                            value={thinkingModeParams[key] !== undefined ? thinkingModeParams[key] : ''}
                            onChange={(e) => {
                              const newValue = type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value;
                              const updatedParams = { ...thinkingModeParams, [key]: newValue };
                              onChange('thinking_mode_params', updatedParams);
                            }}
                            fullWidth
                            variant="outlined"
                            size="small"
                            inputProps={{ step, min, max }}
                          />
                        ));
                      })()}
                    </Box>
                  </CardContent>
                </Card>

                {/* Non-Thinking Mode Card */}
                <Card variant="outlined" sx={{ height: 'fit-content' }}>
                  <CardHeader 
                    title="Non-Thinking Mode Parameters"
                    subheader="Used for direct responses without explicit reasoning"
                    sx={{ pb: 1 }}
                  />
                  <CardContent>
                    <Alert severity="success" sx={{ mb: 2 }}>
                      <Typography variant="body2">
                        <strong>Recommended:</strong> Temperature=0.7, TopP=0.8, TopK=20, MinP=0
                      </Typography>
                    </Alert>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      {(() => {
                        const nonThinkingModeParams = data.non_thinking_mode_params || {};
                        const parameterFields = [
                          { key: 'temperature', label: 'Temperature', type: 'number', step: 0.1, min: 0, max: 2 },
                          { key: 'top_p', label: 'Top P', type: 'number', step: 0.05, min: 0, max: 1 },
                          { key: 'top_k', label: 'Top K', type: 'number', step: 1, min: 0, max: 100 },
                          { key: 'min_p', label: 'Min P', type: 'number', step: 0.01, min: 0, max: 1 }
                        ];
                        
                        return parameterFields.map(({ key, label, type, step, min, max }) => (
                          <TextField
                            key={key}
                            label={label}
                            type={type}
                            value={nonThinkingModeParams[key] !== undefined ? nonThinkingModeParams[key] : ''}
                            onChange={(e) => {
                              const newValue = type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value;
                              const updatedParams = { ...nonThinkingModeParams, [key]: newValue };
                              onChange('non_thinking_mode_params', updatedParams);
                            }}
                            fullWidth
                            variant="outlined"
                            size="small"
                            inputProps={{ step, min, max }}
                          />
                        ));
                      })()}
                    </Box>
                  </CardContent>
                </Card>
                </div>
              </div>
            ) : (
              /* Regular form rendering for other tabs */
              <div style={{ width: '100%', maxWidth: 'none' }}>
                {/* Mode Selection for Settings and Query Classifier tabs */}
                {category === 'llm' && (categoryKey === 'settings' || categoryKey === 'second_llm' || categoryKey === 'knowledge_graph' || categoryKey === 'classifier' || categoryKey === 'search_optimization') && (
                  <Card variant="outlined" sx={{ mb: 3 }}>
                    <CardHeader 
                      title={
                        categoryKey === 'settings' ? "LLM Mode Selection" : 
                        categoryKey === 'second_llm' ? "Second LLM Mode Selection" : 
                        categoryKey === 'knowledge_graph' ? "Knowledge Graph Mode Selection" :
                        categoryKey === 'search_optimization' ? "Search Optimization Mode Selection" :
                        "Query Classifier Mode Selection"
                      }
                      subheader={
                        categoryKey === 'settings' ? "Select between thinking and non-thinking modes" : 
                        categoryKey === 'second_llm' ? "Select mode for the second LLM" : 
                        categoryKey === 'knowledge_graph' ? "Select mode for knowledge graph extraction" :
                        categoryKey === 'search_optimization' ? "Select mode for search query optimization" :
                        "Select mode for query classification"
                      }
                    />
                    <CardContent>
                      <FormControl component="fieldset">
                        <RadioGroup
                          value={
                            categoryKey === 'settings' 
                              ? data.main_llm?.mode || 'thinking'
                              : categoryKey === 'second_llm'
                              ? data.second_llm?.mode || 'thinking'
                              : categoryKey === 'knowledge_graph'
                              ? data.knowledge_graph?.mode || 'thinking'
                              : categoryKey === 'search_optimization'
                              ? data.search_optimization?.mode || 'thinking'
                              : data.query_classifier?.mode || 'non-thinking'
                          }
                          onChange={(e) => {
                            if (categoryKey === 'settings') {
                              const updatedMainLlm = { ...data.main_llm, mode: e.target.value };
                              onChange('main_llm', updatedMainLlm);
                            } else if (categoryKey === 'second_llm') {
                              const updatedSecondLlm = { ...data.second_llm, mode: e.target.value };
                              onChange('second_llm', updatedSecondLlm);
                            } else if (categoryKey === 'knowledge_graph') {
                              const updatedKnowledgeGraph = { ...data.knowledge_graph, mode: e.target.value };
                              onChange('knowledge_graph', updatedKnowledgeGraph);
                            } else if (categoryKey === 'search_optimization') {
                              const updatedSearchOptimization = { ...data.search_optimization, mode: e.target.value };
                              onChange('search_optimization', updatedSearchOptimization);
                            } else {
                              const updatedQueryClassifier = { ...data.query_classifier, mode: e.target.value };
                              onChange('query_classifier', updatedQueryClassifier);
                            }
                          }}
                        >
                          <FormControlLabel 
                            value="thinking" 
                            control={<Radio />} 
                            label={
                              <Box>
                                <Typography variant="body2" fontWeight={600}>
                                  Thinking Mode
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {categoryKey === 'settings' 
                                    ? "Enable step-by-step reasoning with <think> tags"
                                    : categoryKey === 'second_llm'
                                    ? "Enable step-by-step reasoning with <think> tags"
                                    : categoryKey === 'knowledge_graph'
                                    ? "Enable step-by-step reasoning for entity and relationship extraction"
                                    : categoryKey === 'search_optimization'
                                    ? "Enable step-by-step reasoning for query optimization"
                                    : "Use thinking mode parameters for classification"
                                  }
                                </Typography>
                              </Box>
                            }
                          />
                          <FormControlLabel 
                            value="non-thinking" 
                            control={<Radio />} 
                            label={
                              <Box>
                                <Typography variant="body2" fontWeight={600}>
                                  Non-Thinking Mode
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {categoryKey === 'settings'
                                    ? "Direct responses without explicit reasoning steps"
                                    : categoryKey === 'second_llm'
                                    ? "Direct responses without explicit reasoning steps"
                                    : categoryKey === 'knowledge_graph'
                                    ? "Direct extraction without explicit reasoning steps"
                                    : categoryKey === 'search_optimization'
                                    ? "Direct optimization without explicit reasoning steps"
                                    : "Use non-thinking mode parameters for classification"
                                  }
                                </Typography>
                              </Box>
                            }
                          />
                        </RadioGroup>
                      </FormControl>
                    </CardContent>
                  </Card>
                )}

                
                <div className={category === 'rag' || category === 'large_generation' || category === 'langfuse' || category === 'environment' ? '' : `jarvis-form-grid ${(categoryKey === 'classifier' || categoryKey === 'second_llm' || categoryKey === 'knowledge_graph' || categoryKey === 'settings') ? 'single-column' : ''}`}>
                {(() => {
                  // Deduplicate fields before rendering
                  const fieldEntries = Object.entries(categories[categoryKey].fields);
                  const renderedFields = new Map<string, { key: string, value: any }>();
                  
                  fieldEntries.forEach(([fieldKey, fieldValue]) => {
                    // Skip mode fields since we handle them with radio buttons
                    if (fieldKey === 'mode' || fieldKey.endsWith('.mode') || 
                        fieldKey === 'main_llm.mode' || fieldKey === 'second_llm.mode' || fieldKey === 'knowledge_graph.mode' || fieldKey === 'query_classifier.mode') {
                      return;
                    }
                    
                    // Get the base field name (last part after dots)
                    const baseKey = fieldKey.split('.').pop() || fieldKey;
                    
                    // Keep track of what we're rendering to avoid duplicates
                    const existing = renderedFields.get(baseKey);
                    
                    // Prefer non-dotted keys over dotted ones
                    if (!existing || (!fieldKey.includes('.') && existing.key.includes('.'))) {
                      renderedFields.set(baseKey, { key: fieldKey, value: fieldValue });
                    }
                  });
                  
                  // Define preferred field order for LLM settings
                  const getFieldOrder = (fieldKey: string): number => {
                    const lowerKey = fieldKey.toLowerCase();
                    const orderMap: Record<string, number> = {
                      'model': 100,
                      'max_tokens': 200,
                      'maxtoken': 200,
                      'model_server': 300,
                      'system_prompt': 400,
                      'systemprompt': 400,
                      'context_length': 500,
                      'contextlength': 500,
                      'repeat_penalty': 600,
                      'repeatpenalty': 600,
                      'temperature': 700,
                      'top_p': 800,
                      'top_k': 900,
                      'min_p': 1000,
                      'stop': 1100 // Stop parameter positioned at the bottom
                    };
                    
                    // Check for exact matches first
                    for (const [pattern, order] of Object.entries(orderMap)) {
                      if (lowerKey === pattern || lowerKey.endsWith('.' + pattern)) {
                        return order;
                      }
                    }
                    
                    // Check for partial matches
                    for (const [pattern, order] of Object.entries(orderMap)) {
                      if (lowerKey.includes(pattern)) {
                        return order;
                      }
                    }
                    
                    return 10000; // Default for unmatched fields
                  };

                  // Sort fields by priority for LLM settings
                  const sortedFields = category === 'llm' && categoryKey === 'settings' 
                    ? Array.from(renderedFields.values()).sort((a, b) => 
                        getFieldOrder(a.key) - getFieldOrder(b.key)
                      )
                    : Array.from(renderedFields.values());

                  // Special rendering for RAG settings with card grouping
                  if (category === 'rag') {
                    return renderRAGFieldsWithCards(sortedFields, categoryKey, onChange, onShowSuccess, renderField);
                  }
                  
                  // Special rendering for Performance Optimization settings with card grouping
                  if (category === 'large_generation') {
                    return renderPerformanceFieldsWithCards(sortedFields, categoryKey, onChange, onShowSuccess, renderField);
                  }
                  
                  // Special rendering for Langfuse/Monitoring settings with card grouping
                  if (category === 'langfuse') {
                    return renderLangfuseFieldsWithCards(sortedFields, categoryKey, onChange, onShowSuccess, renderField);
                  }
                  
                  // Special rendering for Environment & Runtime settings with card grouping
                  if (category === 'environment') {
                    return renderEnvironmentFieldsWithCards(sortedFields, categoryKey, onChange, onShowSuccess, renderField);
                  }
                  
                  // Special handling for Knowledge Graph category OR Knowledge Graph tab in LLM category
                  if (category === 'knowledge_graph' || (category === 'llm' && categoryKey === 'knowledge_graph')) {
                    return (
                      <KnowledgeGraphSettings
                        data={data}
                        onChange={onChange}
                        onShowSuccess={onShowSuccess}
                      />
                    );
                  }
                  
                  // Special 2-column layout for Query Classifier and Search Optimization tabs
                  if (category === 'llm' && (categoryKey === 'classifier' || categoryKey === 'search_optimization')) {
                    // Debug: Log all field keys to see what we're working with
                    console.log(`[DEBUG] ${categoryKey} fields:`, sortedFields.map(f => f.key));
                    console.log(`[DEBUG] ${categoryKey} raw category data:`, categories[categoryKey].fields);
                    
                    // Extract the model field which will be rendered as full width
                    const modelField = sortedFields.find(field => {
                      const key = field.key.toLowerCase();
                      return key.includes('.model') || key === 'model';
                    });
                    
                    // Search Optimization specific field organization
                    if (categoryKey === 'search_optimization') {
                      // Filter out model field for separate rendering
                      const otherFields = sortedFields.filter(field => {
                        const key = field.key.toLowerCase();
                        return !(key.includes('.model') || key === 'model');
                      });
                      
                      console.log('[DEBUG] Search Optimization fields (excluding model):', otherFields.map(f => f.key));
                      
                      // Organize fields into left and right columns for better visual balance
                      const leftColumnFields = otherFields.filter(field => {
                        const key = field.key.toLowerCase();
                        // Left column: Enable, Max Tokens, Context Length, Top P, System Prompt (5 fields)
                        return key.includes('enable_search_optimization') ||
                               key.includes('max_tokens') ||
                               key.includes('context_length') ||
                               key === 'top_p' ||
                               key.includes('.top_p') ||
                               key.includes('system_prompt') ||
                               key.includes('.system_prompt');
                      });
                      
                      const rightColumnFields = otherFields.filter(field => {
                        const key = field.key.toLowerCase();
                        // Right column: Temperature, Repeat Penalty, Timeout, Optimization Prompt (4 fields)
                        return key.includes('temperature') ||
                               key.includes('repeat_penalty') ||
                               key.includes('timeout') ||
                               (key.includes('optimization_prompt') || (key.includes('prompt') && !key.includes('system')));
                      });
                      
                      console.log('[DEBUG] Left column fields:', leftColumnFields.map(f => f.key));
                      console.log('[DEBUG] Right column fields:', rightColumnFields.map(f => f.key));
                      
                      return (
                        <div style={{ width: '100%' }}>
                          {/* Render the LLM Model Configuration card full width FIRST */}
                          {modelField && (
                            <div style={{ width: '100%', marginBottom: '24px' }}>
                              {renderField(modelField.key, modelField.value, 0, (fieldKey, fieldValue) => {
                                onChange(fieldKey, fieldValue);
                              }, category, onShowSuccess)}
                            </div>
                          )}
                          
                          {/* Use exact same 2-column structure as query classifier */}
                          <div style={{ 
                            display: 'flex',
                            gap: '24px',
                            width: '100%'
                          }}>
                            {/* Left Column */}
                            <div style={{ 
                              flex: '1 1 0%',
                              display: 'flex', 
                              flexDirection: 'column', 
                              gap: '16px' 
                            }}>
                              {leftColumnFields.map(({ key, value }) => 
                                renderField(key, value, 0, (fieldKey, fieldValue) => {
                                  onChange(fieldKey, fieldValue);
                                }, category, onShowSuccess)
                              )}
                            </div>
                            
                            {/* Right Column */}
                            <div style={{ 
                              flex: '1 1 0%',
                              display: 'flex', 
                              flexDirection: 'column', 
                              gap: '16px' 
                            }}>
                              {rightColumnFields.map(({ key, value }) => 
                                renderField(key, value, 0, (fieldKey, fieldValue) => {
                                  onChange(fieldKey, fieldValue);
                                }, category, onShowSuccess)
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    }
                    
                    // Query Classifier layout (existing logic)
                    else {
                      // More precise matching for second column fields
                      // Handles both 'search_optimization.field' and standalone 'field' patterns
                      const secondColumnFields = sortedFields.filter(field => {
                        const key = field.key.toLowerCase();
                        
                        // More precise matching for second column fields
                        return key.endsWith('context_length') ||
                               key.endsWith('max_tokens') ||
                               key.includes('timeout') ||
                               key.includes('threshold') ||
                               (key.includes('context') && key.includes('length')) ||
                               (key.includes('max') && key.includes('tokens'));
                      });
                      
                      console.log('[DEBUG] Found second column fields:', secondColumnFields.map(f => f.key));
                      
                      // Get all other fields for first column - exclude the second column fields AND model field
                      const firstColumnFields = sortedFields.filter(field => {
                        const key = field.key.toLowerCase();
                        
                        // Exclude model field (will be rendered separately)
                        if (key.includes('.model') || key === 'model') {
                          return false;
                        }
                        
                        // Exclude fields that go in second column (use same precise matching)
                        return !(key.endsWith('context_length') ||
                               key.endsWith('max_tokens') ||
                               key.includes('timeout') ||
                               key.includes('threshold') ||
                               (key.includes('context') && key.includes('length')) ||
                               (key.includes('max') && key.includes('tokens')));
                      });
                      
                      console.log('[DEBUG] Found first column fields:', firstColumnFields.map(f => f.key));
                      
                      return (
                        <div style={{ width: '100%' }}>
                          {/* Render the LLM Model Configuration card full width FIRST */}
                          {modelField && (
                            <div style={{ width: '100%', marginBottom: '24px' }}>
                              {renderField(modelField.key, modelField.value, 0, (fieldKey, fieldValue) => {
                                onChange(fieldKey, fieldValue);
                              }, category, onShowSuccess)}
                            </div>
                          )}
                          
                          {/* Then render the 2-column layout for other fields */}
                          <div style={{ 
                            display: 'flex',
                            gap: '24px',
                            width: '100%'
                          }}>
                            {/* First Column - Most fields */}
                            <div style={{ 
                              flex: 1,
                              display: 'flex', 
                              flexDirection: 'column', 
                              gap: '16px' 
                            }}>
                              {firstColumnFields.map(({ key, value }) => 
                                renderField(key, value, 0, (fieldKey, fieldValue) => {
                                  onChange(fieldKey, fieldValue);
                                }, category, onShowSuccess)
                              )}
                            </div>
                            
                            {/* Second Column - Context Length and Timeout fields */}
                            <div style={{ 
                              flex: 1,
                              display: 'flex', 
                              flexDirection: 'column', 
                              gap: '16px' 
                            }}>
                              {secondColumnFields.map(({ key, value }) => 
                                renderField(key, value, 0, (fieldKey, fieldValue) => {
                                  onChange(fieldKey, fieldValue);
                                }, category, onShowSuccess)
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    }
                  }
                  
                  // Render sorted fields normally for other categories
                  return sortedFields.map(({ key, value }) => 
                    renderField(key, value, 0, (fieldKey, fieldValue) => {
                      // Handle nested field updates properly
                      onChange(fieldKey, fieldValue);
                    }, category, onShowSuccess)
                  );
                })()}
                </div>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

const TimeoutConfiguration: React.FC<{
  data: any,
  onChange: (field: string, value: any) => void,
  onShowSuccess?: (message?: string) => void
}> = ({ data, onChange, onShowSuccess }) => {
  const [activeTimeoutTab, setActiveTimeoutTab] = React.useState('api_network');

  const timeoutCategories = {
    api_network: { 
      title: 'API & Network', 
      description: 'HTTP requests, database connections, and network operations',
      fields: {} 
    },
    llm_ai: { 
      title: 'LLM & AI Processing', 
      description: 'Language model inference, classification, and AI operations',
      fields: {} 
    },
    document_processing: { 
      title: 'Document & RAG', 
      description: 'Document retrieval, vector search, and RAG processing',
      fields: {} 
    },
    mcp_tools: { 
      title: 'MCP Tools', 
      description: 'Tool execution, server communication, and integrations',
      fields: {} 
    },
    workflow_automation: { 
      title: 'Workflow & Automation', 
      description: 'Agent workflows, task execution, and large generation',
      fields: {} 
    },
    session_cache: { 
      title: 'Session & Cache', 
      description: 'Redis TTL, conversation memory, and cache expiration',
      fields: {} 
    },
    redis_cache_ttl: {
      title: 'Redis Cache TTL',
      description: 'Time-to-live settings for various Redis caches (controls how long data is kept in cache)',
      fields: {}
    }
  };

  // Organize timeout data into categories
  Object.entries(data || {}).forEach(([categoryKey, categoryData]) => {
    if (timeoutCategories[categoryKey] && typeof categoryData === 'object') {
      timeoutCategories[categoryKey].fields = categoryData;
    }
  });

  const renderTimeoutField = (key: string, value: any, categoryKey: string) => {
    const formatLabel = (str: string) => {
      return str
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
    };

    const getHelpText = (fieldKey: string) => {
      const helpTexts = {
        // API & Network
        'http_request_timeout': 'Timeout for HTTP API requests (seconds)',
        'http_streaming_timeout': 'Timeout for streaming HTTP responses (seconds)',
        'http_upload_timeout': 'Timeout for file upload operations (seconds)',
        'database_connection_timeout': 'Database connection establishment timeout (seconds)',
        'database_query_timeout': 'Database query execution timeout (seconds)',
        'redis_operation_timeout': 'Redis operations timeout (seconds)',
        'redis_connection_timeout': 'Redis connection timeout (seconds)',
        
        // LLM & AI
        'llm_inference_timeout': 'LLM model inference timeout (seconds)',
        'llm_streaming_timeout': 'LLM streaming response timeout (seconds)',
        'query_classification_timeout': 'Query classification timeout (seconds)',
        'multi_agent_timeout': 'Multi-agent coordination timeout (seconds)',
        'agent_processing_timeout': 'Individual agent processing timeout (seconds)',
        'agent_coordination_timeout': 'Agent-to-agent coordination timeout (seconds)',
        'thinking_mode_timeout': 'Extended thinking mode timeout (seconds)',
        
        // Document Processing
        'rag_retrieval_timeout': 'RAG document retrieval timeout (seconds)',
        'rag_processing_timeout': 'RAG processing pipeline timeout (seconds)',
        'vector_search_timeout': 'Vector database search timeout (seconds)',
        'embedding_generation_timeout': 'Text embedding generation timeout (seconds)',
        'document_processing_timeout': 'Document processing timeout (seconds)',
        'collection_search_timeout': 'Collection search timeout (seconds)',
        'bm25_processing_timeout': 'BM25 text processing timeout (seconds)',
        
        // MCP Tools
        'tool_execution_timeout': 'Tool execution timeout (seconds)',
        'tool_initialization_timeout': 'Tool initialization timeout (seconds)',
        'manifest_fetch_timeout': 'MCP manifest fetch timeout (seconds)',
        'server_communication_timeout': 'MCP server communication timeout (seconds)',
        'server_startup_timeout': 'MCP server startup timeout (seconds)',
        'stdio_bridge_timeout': 'STDIO bridge communication timeout (seconds)',
        
        // Workflow & Automation
        'workflow_execution_timeout': 'Workflow execution timeout (seconds)',
        'workflow_step_timeout': 'Individual workflow step timeout (seconds)',
        'task_timeout': 'General task execution timeout (seconds)',
        'agent_task_timeout': 'Agent task execution timeout (seconds)',
        'large_generation_timeout': 'Large content generation timeout (seconds)',
        'chunk_generation_timeout': 'Content chunk generation timeout (seconds)',
        
        // Session & Cache
        'redis_ttl_seconds': 'Redis cache TTL (seconds)',
        'conversation_cache_ttl': 'Conversation cache TTL (seconds)',
        'result_cache_ttl': 'Result cache TTL (seconds)',
        'temp_data_ttl': 'Temporary data TTL (seconds)',
        'session_cleanup_interval': 'Session cleanup interval (seconds)',
        'cache_cleanup_interval': 'Cache cleanup interval (seconds)',
        
        // Redis Cache TTL Settings
        'settings_cache_ttl': 'How long to cache system settings (default: 3600s / 1 hour)',
        'pipeline_cache_ttl': 'How long to cache pipeline states (default: 3600s / 1 hour)',
        'list_cache_ttl': 'How long to cache list data like pipeline lists (default: 300s / 5 minutes)',
        'agent_response_cache_ttl': 'How long to cache agent responses (default: 600s / 10 minutes)',
        'collection_registry_ttl': 'How long to cache collection registry data (default: 300s / 5 minutes)',
        'conversation_cache_ttl': 'How long to keep conversation history in cache (default: 86400s / 24 hours)',
        'temporary_data_ttl': 'How long to keep temporary/intermediate data (default: 1800s / 30 minutes)',
        'idc_cache_ttl': 'How long to cache IDC extraction results (default: 3600s / 1 hour)',
        'validation_cache_ttl': 'How long to cache validation results (default: 7200s / 2 hours)',
        'knowledge_graph_cache_ttl': 'How long to cache knowledge graph data (default: 1800s / 30 minutes)',
        'rag_cache_ttl': 'How long to cache RAG retrieval results (default: 3600s / 1 hour)',
        'embedding_cache_ttl': 'How long to cache generated embeddings (default: 7200s / 2 hours)',
        'vector_search_cache_ttl': 'How long to cache vector search results (default: 1800s / 30 minutes)',
        'workflow_state_ttl': 'How long to keep workflow execution states (default: 3600s / 1 hour)',
        'mcp_tool_cache_ttl': 'How long to cache MCP tool configurations (default: 600s / 10 minutes)'
      };
      
      return helpTexts[fieldKey] || null;
    };

    const helpText = getHelpText(key);
    const numValue = typeof value === 'number' ? value : parseInt(value) || 0;

    // Determine reasonable min/max values based on category
    let minValue = 1;
    let maxValue = 600; // 10 minutes default
    
    if (categoryKey === 'session_cache' || categoryKey === 'redis_cache_ttl') {
      maxValue = 604800; // 7 days for cache TTL
    } else if (categoryKey === 'workflow_automation') {
      maxValue = 1800; // 30 minutes for workflows
    }

    return (
      <Box key={key} sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <Typography variant="body2" fontWeight="medium">
            {formatLabel(key)}
          </Typography>
          {helpText && (
            <Tooltip title={helpText} arrow>
              <HelpOutlineIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
            </Tooltip>
          )}
        </Box>
        <TextField
          type="number"
          value={numValue}
          onChange={(e) => {
            const newValue = parseInt(e.target.value) || 0;
            const clampedValue = Math.max(minValue, Math.min(maxValue, newValue));
            onChange(`${categoryKey}.${key}`, clampedValue);
          }}
          inputProps={{ 
            min: minValue, 
            max: maxValue,
            step: 1
          }}
          size="small"
          fullWidth
          helperText={numValue !== Math.max(minValue, Math.min(maxValue, numValue)) 
            ? `Value will be clamped to range ${minValue}-${maxValue}` 
            : `Range: ${minValue}-${maxValue} seconds`}
          error={numValue < minValue || numValue > maxValue}
        />
      </Box>
    );
  };

  return (
    <Box>
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs 
          value={activeTimeoutTab} 
          onChange={(_, newValue) => setActiveTimeoutTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
        >
          {Object.entries(timeoutCategories).map(([key, category]) => (
            <Tab 
              key={key}
              label={category.title} 
              value={key} 
            />
          ))}
        </Tabs>
      </Box>

      {Object.entries(timeoutCategories).map(([categoryKey, categoryData]) => (
        activeTimeoutTab === categoryKey && (
          <Box key={categoryKey}>
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                {categoryData.title} Timeouts
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                {categoryData.description}
              </Typography>
            </Box>

            <Card variant="outlined">
              <CardContent>
                {Object.keys(categoryData.fields || {}).length === 0 ? (
                  <Alert severity="info">
                    No timeout settings available for this category.
                  </Alert>
                ) : (
                  <Grid container spacing={3}>
                    {Object.entries(categoryData.fields || {}).map(([fieldKey, fieldValue]) => (
                      <Grid item xs={12} sm={6} md={4} key={fieldKey}>
                        {renderTimeoutField(fieldKey, fieldValue, categoryKey)}
                      </Grid>
                    ))}
                  </Grid>
                )}
              </CardContent>
            </Card>

            <Box sx={{ mt: 2 }}>
              <Alert severity="warning" sx={{ mb: 2 }}>
                <Typography variant="body2">
                  <strong>Important:</strong> Timeout changes require a system restart to take full effect. 
                  Very low values may cause system instability, while very high values may cause poor user experience.
                </Typography>
              </Alert>
            </Box>
          </Box>
        )
      ))}
    </Box>
  );
};

export default SettingsFormRenderer;