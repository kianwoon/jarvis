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
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material';
import YamlEditor from './YamlEditor';
import DatabaseTableManager from './DatabaseTableManager';
import MCPServerManager from './MCPServerManager';
import MCPToolManager from './MCPToolManager';
import VectorDatabaseManager from './VectorDatabaseManager';

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
  // Move all hooks to the top level - they must always be called in the same order
  const [activeTab, setActiveTab] = React.useState(() => {
    if (category === 'rag') return 'retrieval';
    if (category === 'storage') return 'vector';
    return 'settings';
  });
  const [passwordVisibility, setPasswordVisibility] = React.useState<Record<string, boolean>>({});
  const [icebergPasswordVisibility, setIcebergPasswordVisibility] = React.useState<Record<string, boolean>>({});
  const [mcpTab, setMcpTab] = React.useState(0);
  
  // Update activeTab when category changes
  React.useEffect(() => {
    if (category === 'rag') setActiveTab('retrieval');
    else if (category === 'storage') setActiveTab('vector');
    else setActiveTab('settings');
  }, [category]);

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

  // Special handling for environment variables
  if (category === 'environment') {
    return renderEnvironmentEditor(data, onChange);
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

  // Default form rendering for regular settings
  return renderStandardForm(data, onChange, category, activeTab, setActiveTab, passwordVisibility, setPasswordVisibility, icebergPasswordVisibility, setIcebergPasswordVisibility, onShowSuccess);
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
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
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
        setModels(data.models || []);
      } else {
        // Fallback to hardcoded models if API fails
        setModels([
          { name: 'qwen3:0.6b', id: '7df6b6e09427', size: '522 MB', modified: '2 minutes ago', context_length: 'Unknown' },
          { name: 'deepseek-r1:8b', id: '6995872bfe4c', size: '5.2 GB', modified: '5 weeks ago', context_length: 'Unknown' },
          { name: 'qwen3:30b-a3b', id: '2ee832bc15b5', size: '18 GB', modified: '7 weeks ago', context_length: 'Unknown' }
        ]);
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
      // Use hardcoded models as fallback
      setModels([
        { name: 'qwen3:0.6b', id: '7df6b6e09427', size: '522 MB', modified: '2 minutes ago', context_length: 'Unknown' },
        { name: 'deepseek-r1:8b', id: '6995872bfe4c', size: '5.2 GB', modified: '5 weeks ago', context_length: 'Unknown' },
        { name: 'qwen3:30b-a3b', id: '2ee832bc15b5', size: '18 GB', modified: '7 weeks ago', context_length: 'Unknown' }
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
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button 
                size="small" 
                onClick={fetchAvailableModels}
                startIcon={<RefreshIcon />}
                disabled={loading}
              >
                Refresh
              </Button>
              <Tooltip title="Reload LLM cache with current settings">
                <Button 
                  size="small" 
                  onClick={async () => {
                    try {
                      const response = await fetch('/api/v1/settings/llm/cache/reload', { method: 'POST' });
                      if (response.ok) {
                        const result = await response.json();
                        console.log('LLM cache reloaded:', result);
                        if (onShowSuccess) {
                          onShowSuccess('Cache reloaded successfully!');
                        }
                      } else {
                        console.error('Failed to reload cache');
                        console.error('Failed to reload cache');
                      }
                    } catch (error) {
                      console.error('Error reloading cache:', error);
                      console.error('Error reloading cache:', error);
                    }
                  }}
                  startIcon={<CacheIcon />}
                  variant="outlined"
                >
                  Reload Cache
                </Button>
              </Tooltip>
            </Box>
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
                    
                    if (fieldKey === 'model') {
                      // Settings tab
                      onChangeHandler('context_length', contextLength);
                      const suggestedMaxTokens = Math.floor(contextLength * 0.75);
                      onChangeHandler('max_tokens', suggestedMaxTokens);
                    } else if (fieldKey === 'query_classifier.llm_model') {
                      // Query Classifier tab - exactly same as Settings tab
                      onChangeHandler('query_classifier.context_length', contextLength);
                      const suggestedLlmMaxTokens = Math.floor(contextLength * 0.75);
                      onChangeHandler('query_classifier.llm_max_tokens', suggestedLlmMaxTokens);
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
  onShowSuccess?: (message?: string) => void
) => {

  // Flatten nested objects and categorize fields into domain-intelligent tabs
  // Known nested structures that should be preserved (not flattened)
  const preserveNested = [
    'query_classifier',
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
    } else {
      // Default LLM category structure - consolidated context into settings
      categories = {
        settings: { title: 'Settings', fields: {} },
        classifier: { title: 'Query Classifier', fields: {} },
        thinking: { title: 'Thinking Mode', fields: {} }
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
      
      // Skip keys that are already part of a preserved nested object
      const isPartOfNestedObject = key.includes('.') && 
        preserveNested.some(nested => key.startsWith(nested + '.'));
      
      if (isPartOfNestedObject) {
        return; // Skip this field as it's part of a preserved nested structure
      }
      
      if (category === 'rag') {
        // RAG-specific field categorization
        if (lowerKey.includes('embedding') || lowerKey.includes('vector') || lowerKey.includes('similarity') || 
            lowerKey.includes('chunk') || lowerKey.includes('top_k') || lowerKey.includes('retrieval')) {
          categories.retrieval.fields[key] = value;
        }
        // Reranking & Scoring
        else if (lowerKey.includes('rerank') || lowerKey.includes('score') || lowerKey.includes('bm25') || 
                 lowerKey.includes('weight') || lowerKey.includes('threshold')) {
          categories.reranking.fields[key] = value;
        }
        // Search Strategy
        else if (lowerKey.includes('search') || lowerKey.includes('query') || lowerKey.includes('strategy') || 
                 lowerKey.includes('hybrid') || lowerKey.includes('filter')) {
          categories.search.fields[key] = value;
        }
        // Document Processing
        else if (lowerKey.includes('document') || lowerKey.includes('processing') || lowerKey.includes('indexing') || 
                 lowerKey.includes('text') || lowerKey.includes('content')) {
          categories.processing.fields[key] = value;
        }
        // Default to retrieval if we can't categorize
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
      } else {
        // LLM-specific field categorization
        // Query Classifier Tab - All classifier-related settings
        if (lowerKey.includes('query_classifier') || lowerKey.includes('classifier')) {
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
        'query_classifier.llm_classification_priority': 'Use LLM First (vs Patterns First)'
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
    
    if (typeof value === 'boolean') {
      return (
        <div key={key} className={fieldClass} style={{ marginLeft: `${depth * 20}px` }}>
          <label className="jarvis-form-label">
            <input
              type="checkbox"
              checked={value}
              onChange={(e) => onChangeHandler(key, e.target.checked)}
              style={{ marginRight: '8px' }}
            />
            {formatLabel(key)}
          </label>
        </div>
      );
    }

    if (typeof value === 'number') {
      // Check if this looks like a slider parameter (temperature, top_p, etc.)
      const isSliderParam = key.toLowerCase().includes('temperature') || 
                          key.toLowerCase().includes('top_p') || 
                          key.toLowerCase().includes('top_k') || 
                          key.toLowerCase().includes('penalty');
      
      if (isSliderParam) {
        const min = key.toLowerCase().includes('top_k') ? 0 : 
                   key.toLowerCase().includes('temperature') ? 0 : 
                   key.toLowerCase().includes('top_p') ? 0 : 1;
        const max = key.toLowerCase().includes('top_k') ? 100 : 
                   key.toLowerCase().includes('temperature') ? 2 : 
                   key.toLowerCase().includes('top_p') ? 1 : 2;
        const step = key.toLowerCase().includes('top_k') ? 1 : 0.01;
        
        return (
          <div key={key} className={fieldClass} style={{ marginLeft: `${depth * 20}px` }}>
            <label className="jarvis-form-label">{formatLabel(key)}</label>
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
          </div>
        );
      } else {
        return (
          <div key={key} className={fieldClass} style={{ marginLeft: `${depth * 20}px` }}>
            <label className="jarvis-form-label">{formatLabel(key)}</label>
            <input
              type="number"
              className="jarvis-form-input"
              value={value}
              onChange={(e) => onChangeHandler(key, parseFloat(e.target.value) || 0)}
            />
          </div>
        );
      }
    }

    if (typeof value === 'string') {
      const lowerKey = key.toLowerCase();
      // Always use textarea for prompt fields to prevent height changes while typing
      const isLongText = value.length > 100 || lowerKey.includes('prompt') || lowerKey.includes('system');
      
      // Debug logging for max_tokens
      if (lowerKey.includes('max_tokens') || lowerKey.includes('maxtoken')) {
        console.log('[DEBUG] Max tokens field:', key, 'Value:', value, 'Type:', typeof value);
      }
      
      // Special handling for model field in LLM category
      console.log('[DEBUG] Checking model field:', { fieldCategory, key, value, fullKey: key, customOnChange: !!customOnChange });
      // Check for both 'model' and 'settings.model' due to potential nesting, including llm_model
      if (fieldCategory === 'llm' && (key === 'model' || key === 'settings.model' || key.endsWith('.model') || key.endsWith('.llm_model') || key === 'llm_model')) {
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
      
      return (
        <div key={key} className={fieldClass} style={{ marginLeft: `${depth * 20}px` }}>
          <label className="jarvis-form-label">{formatLabel(key)}</label>
          {isLongText ? (
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
      return (
        <div key={key} className={fieldClass} style={{ marginLeft: `${depth * 20}px` }}>
          <label className="jarvis-form-label">{formatLabel(key)}</label>
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
    
    // Handle nested objects - but flatten query_classifier to work like Settings
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      if (key === 'query_classifier') {
        // Flatten query_classifier fields to work like Settings fields
        return (
          <div key={key} style={{ marginLeft: `${depth * 20}px`, marginBottom: '16px' }}>
            {Object.entries(value).map(([nestedKey, nestedValue]) => 
              renderField(`${key}.${nestedKey}`, nestedValue, depth, onChangeHandler, fieldCategory)
            )}
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
              <div style={{ padding: '16px' }}>
                {/* Mode Selection for Settings and Query Classifier tabs */}
                {(categoryKey === 'settings' || categoryKey === 'classifier') && (
                  <Card variant="outlined" sx={{ mb: 3 }}>
                    <CardHeader 
                      title={categoryKey === 'settings' ? "LLM Mode Selection" : "Query Classifier Mode Selection"}
                      subheader={categoryKey === 'settings' ? "Select between thinking and non-thinking modes" : "Select mode for query classification"}
                    />
                    <CardContent>
                      <FormControl component="fieldset">
                        <RadioGroup
                          value={
                            categoryKey === 'settings' 
                              ? data.main_llm?.mode || 'thinking'
                              : data.query_classifier?.mode || 'non-thinking'
                          }
                          onChange={(e) => {
                            if (categoryKey === 'settings') {
                              const updatedMainLlm = { ...data.main_llm, mode: e.target.value };
                              onChange('main_llm', updatedMainLlm);
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
                
                <div className={`jarvis-form-grid ${categoryKey === 'classifier' ? 'single-column' : ''}`}>
                {(() => {
                  // Deduplicate fields before rendering
                  const fieldEntries = Object.entries(categories[categoryKey].fields);
                  const renderedFields = new Map<string, { key: string, value: any }>();
                  
                  fieldEntries.forEach(([fieldKey, fieldValue]) => {
                    // Skip mode fields since we handle them with radio buttons
                    if (fieldKey === 'mode' || fieldKey.endsWith('.mode') || 
                        (categoryKey === 'settings' && fieldKey === 'main_llm.mode') ||
                        (categoryKey === 'classifier' && fieldKey === 'query_classifier.mode')) {
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
                      'stop': 700, // Stop parameter positioned after basic settings
                      'temperature': 800,
                      'top_p': 900,
                      'top_k': 1000,
                      'min_p': 1100
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

                  // Render sorted fields
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

export default SettingsFormRenderer;