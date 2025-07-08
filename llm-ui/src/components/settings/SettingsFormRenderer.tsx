import React, { useState } from 'react';
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
  Tab
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import YamlEditor from './YamlEditor';
import DatabaseTableManager from './DatabaseTableManager';
import MCPServerManager from './MCPServerManager';
import MCPToolManager from './MCPToolManager';

interface SettingsFormRendererProps {
  category: string;
  data: any;
  onChange: (field: string, value: any) => void;
  onRefresh?: () => void;
  isYamlBased?: boolean;
}

const SettingsFormRenderer: React.FC<SettingsFormRendererProps> = ({
  category,
  data,
  onChange,
  onRefresh,
  isYamlBased = false
}) => {

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
    const mcpData = data || { servers: [], tools: [] };
    return renderMCPConfiguration(mcpData, onChange, onRefresh);
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
  return renderStandardForm(data, onChange);
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

const renderMCPConfiguration = (data: any, onChange: (field: string, value: any) => void, onRefresh?: () => void) => {
  const [mcpTab, setMcpTab] = useState(0);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setMcpTab(newValue);
  };

  const loadMCPData = async (type: 'servers' | 'tools') => {
    try {
      const endpoint = type === 'servers' ? '/api/v1/mcp/servers/' : '/api/v1/mcp/tools/';
      const response = await fetch(endpoint);
      if (response.ok) {
        const result = await response.json();
        console.log(`Loaded MCP ${type}:`, result.length, 'items');
        onChange(type, Array.isArray(result) ? result : result.data || []);
      } else {
        console.error(`Failed to load MCP ${type}: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      console.error(`Error loading MCP ${type}:`, error);
    }
  };

  return (
    <Box>
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
        <Tabs value={mcpTab} onChange={handleTabChange} aria-label="mcp tabs">
          <Tab label="MCP Servers" />
          <Tab label="MCP Tools" />
          <Tab label="Cache Management" />
        </Tabs>
      </Box>

      {mcpTab === 0 && (
        <MCPServerManager
          data={data.servers || []}
          onChange={(servers) => onChange('servers', servers)}
          onRefresh={() => {
            loadMCPData('servers');
            if (onRefresh) onRefresh();
          }}
        />
      )}

      {mcpTab === 1 && (
        <MCPToolManager
          data={data.tools || []}
          onChange={(tools) => onChange('tools', tools)}
          onRefresh={() => {
            loadMCPData('tools');
            if (onRefresh) onRefresh();
          }}
        />
      )}

      {mcpTab === 2 && (
        <Box>
          <Typography variant="h6" gutterBottom>Cache Management</Typography>
          <Paper sx={{ p: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  onClick={async () => {
                    try {
                      await fetch('/api/v1/mcp/tools/cache/reload', { method: 'POST' });
                      alert('MCP tools cache reloaded');
                    } catch (error) {
                      alert('Failed to reload cache');
                    }
                  }}
                >
                  Reload Tools Cache
                </Button>
              </Grid>
              <Grid item xs={12} md={4}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  onClick={async () => {
                    try {
                      await fetch('/api/v1/mcp/servers/cache/reload', { method: 'POST' });
                      alert('MCP servers cache reloaded');
                    } catch (error) {
                      alert('Failed to reload cache');
                    }
                  }}
                >
                  Reload Servers Cache
                </Button>
              </Grid>
              <Grid item xs={12} md={4}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  onClick={() => {
                    loadMCPData('servers');
                    loadMCPData('tools');
                    alert('MCP data refreshed');
                  }}
                >
                  Refresh All Data
                </Button>
              </Grid>
            </Grid>
          </Paper>
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

const renderStandardForm = (data: any, onChange: (field: string, value: any) => void) => {
  const renderField = (key: string, value: any, depth: number = 0) => {
    const indent = depth * 20;
    
    if (typeof value === 'boolean') {
      return (
        <Box key={key} sx={{ ml: `${indent}px`, mb: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={value}
                onChange={(e) => onChange(key, e.target.checked)}
              />
            }
            label={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
          />
        </Box>
      );
    }

    if (typeof value === 'number') {
      return (
        <Box key={key} sx={{ ml: `${indent}px`, mb: 2 }}>
          <TextField
            fullWidth
            type="number"
            label={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            value={value}
            onChange={(e) => onChange(key, parseFloat(e.target.value) || 0)}
            variant="outlined"
            size="small"
          />
        </Box>
      );
    }

    if (typeof value === 'string') {
      return (
        <Box key={key} sx={{ ml: `${indent}px`, mb: 2 }}>
          <TextField
            fullWidth
            label={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            value={value}
            onChange={(e) => onChange(key, e.target.value)}
            variant="outlined"
            size="small"
            multiline={value.length > 50}
            rows={value.length > 50 ? 3 : 1}
            type={key.toLowerCase().includes('password') || key.toLowerCase().includes('secret') || key.toLowerCase().includes('key') ? 'password' : 'text'}
          />
        </Box>
      );
    }

    if (Array.isArray(value)) {
      return (
        <Box key={key} sx={{ ml: `${indent}px`, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
          </Typography>
          <Paper variant="outlined" sx={{ p: 2, maxHeight: '50vh', overflow: 'auto' }}>
            {value.map((item, index) => (
              <Box key={index} sx={{ display: 'flex', gap: 1, mb: 1, alignItems: 'center' }}>
                <TextField
                  size="small"
                  value={typeof item === 'object' ? JSON.stringify(item) : item}
                  onChange={(e) => {
                    const newArray = [...value];
                    try {
                      newArray[index] = JSON.parse(e.target.value);
                    } catch {
                      newArray[index] = e.target.value;
                    }
                    onChange(key, newArray);
                  }}
                  fullWidth
                />
                <IconButton 
                  size="small" 
                  onClick={() => {
                    const newArray = value.filter((_, i) => i !== index);
                    onChange(key, newArray);
                  }}
                >
                  <DeleteIcon />
                </IconButton>
              </Box>
            ))}
            <Button 
              size="small" 
              startIcon={<AddIcon />}
              onClick={() => onChange(key, [...value, ''])}
            >
              Add Item
            </Button>
          </Paper>
        </Box>
      );
    }

    if (typeof value === 'object' && value !== null) {
      return (
        <Box key={key} sx={{ ml: `${indent}px`, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
          </Typography>
          <Paper variant="outlined" sx={{ p: 2, maxHeight: '60vh', overflow: 'auto' }}>
            {Object.entries(value).map(([subKey, subValue]) => 
              renderField(`${key}.${subKey}`, subValue, depth + 1)
            )}
          </Paper>
        </Box>
      );
    }

    return (
      <Box key={key} sx={{ ml: `${indent}px`, mb: 2 }}>
        <Typography variant="body2" color="text.secondary">
          {key}: {String(value)} (type: {typeof value})
        </Typography>
      </Box>
    );
  };

  return (
    <Box>
      {Object.entries(data).map(([key, value]) => renderField(key, value))}
    </Box>
  );
};

export default SettingsFormRenderer;