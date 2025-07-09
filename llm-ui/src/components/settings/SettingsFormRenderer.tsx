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
  Refresh as RefreshIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon
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
  return renderStandardForm(data, onChange, category);
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

const renderStandardForm = (data: any, onChange: (field: string, value: any) => void, category?: string) => {
  const getDefaultTab = () => {
    if (category === 'rag') return 'retrieval';
    if (category === 'storage') return 'vector';
    return 'settings';
  };
  
  const [activeTab, setActiveTab] = React.useState(getDefaultTab());
  const [passwordVisibility, setPasswordVisibility] = React.useState<Record<string, boolean>>({});

  // Flatten nested objects and categorize fields into domain-intelligent tabs
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
      // Default LLM category structure
      categories = {
        settings: { title: 'Settings', fields: {} },
        context: { title: 'Context Length', fields: {} },
        classifier: { title: 'Query Classifier', fields: {} },
        thinking: { title: 'Thinking Mode', fields: {} }
      };
    }

    // Flatten nested objects recursively
    const flattenObject = (obj: any, prefix: string = ''): Record<string, any> => {
      const flattened: Record<string, any> = {};
      
      Object.entries(obj).forEach(([key, value]) => {
        const fullKey = prefix ? `${prefix}.${key}` : key;
        
        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
          // Recursively flatten nested objects
          Object.assign(flattened, flattenObject(value, fullKey));
        } else {
          flattened[fullKey] = value;
        }
      });
      
      return flattened;
    };

    const flattenedData = flattenObject(data);

    Object.entries(flattenedData).forEach(([key, value]) => {
      const lowerKey = key.toLowerCase();
      
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
        if (lowerKey.includes('milvus') || lowerKey.includes('qdrant') || lowerKey.includes('vector') || 
            lowerKey.includes('embedding') || lowerKey.includes('pinecone') || lowerKey.includes('weaviate') ||
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
        if (lowerKey.includes('max_tokens') || lowerKey.includes('context_length') || lowerKey.includes('context')) {
          categories.context.fields[key] = value;
        }
        // Query Classifier Tab - All classifier-related settings
        else if (lowerKey.includes('query_classifier') || lowerKey.includes('classifier')) {
          categories.classifier.fields[key] = value;
        }
        // Thinking Mode Tab - Thinking-specific parameters (including non-thinking mode)
        else if (lowerKey.includes('thinking_mode') || lowerKey.includes('thinking') || lowerKey.includes('non_thinking')) {
          categories.thinking.fields[key] = value;
        }
        // Settings Tab - Core model configuration (everything else)
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

  const renderField = (key: string, value: any, depth: number = 0, customOnChange?: (field: string, value: any) => void) => {
    const formatLabel = (str: string) => str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
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
      const isLongText = value.length > 100;
      const isPassword = key.toLowerCase().includes('password') || 
                        key.toLowerCase().includes('secret') || 
                        key.toLowerCase().includes('key') ||
                        key.toLowerCase().includes('token') ||
                        key.toLowerCase().includes('access');
      
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

    // Skip nested objects since we've already flattened them
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      return null;
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
  React.useEffect(() => {
    if (categoryKeys.length > 0 && !categoryKeys.includes(activeTab)) {
      setActiveTab(categoryKeys[0]);
    }
  }, [categoryKeys, activeTab]);

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
            <div className="jarvis-form-grid">
              {Object.entries(categories[categoryKey].fields).map(([key, value]) => 
                renderField(key, value, 0, (fieldKey, fieldValue) => {
                  // Handle nested field updates properly
                  onChange(fieldKey, fieldValue);
                })
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default SettingsFormRenderer;