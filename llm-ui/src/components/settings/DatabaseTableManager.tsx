import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Chip,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  OutlinedInput,
  Checkbox,
  ListItemText
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as RunIcon,
  Pause as PauseIcon,
  Refresh as RefreshIcon,
  Visibility as ViewIcon
} from '@mui/icons-material';

interface DatabaseTableManagerProps {
  category: 'automation' | 'collection_registry' | 'langgraph_agents';
  data: any[];
  onChange: (data: any[]) => void;
  onRefresh: () => void;
}

interface WorkflowRecord {
  id?: string;
  name: string;
  description?: string;
  status: 'active' | 'inactive' | 'draft';
  workflow_type: 'langflow' | 'custom';
  config: any;
  created_at?: string;
  updated_at?: string;
}

interface CollectionRecord {
  id?: string;
  collection_name: string;
  description?: string;
  collection_type: string;
  access_config?: {
    restricted: boolean;
    allowed_users: string[];
  };
  metadata: any;
  created_at?: string;
  // Statistics fields
  statistics?: {
    document_count?: number;
    total_chunks?: number;
    storage_size_mb?: number;
    last_updated?: string;
  };
}

interface AgentRecord {
  id?: string;
  name: string;
  role: string;
  system_prompt: string;
  tools: string[];
  description?: string;
  is_active: boolean;
  config: any;
}

const DatabaseTableManager: React.FC<DatabaseTableManagerProps> = ({
  category,
  data,
  onChange,
  onRefresh
}) => {
  const [editDialog, setEditDialog] = useState(false);
  const [editingRecord, setEditingRecord] = useState<any>(null);
  const [refreshingStats, setRefreshingStats] = useState(false);
  const [viewDialog, setViewDialog] = useState(false);
  const [viewingRecord, setViewingRecord] = useState<any>(null);
  const [localData, setLocalData] = useState(data);
  const [availableTools, setAvailableTools] = useState<string[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);

  // Update local data when props change
  React.useEffect(() => {
    setLocalData(data);
  }, [data]);

  // Fetch available tools for LangGraph agents
  useEffect(() => {
    if (category === 'langgraph_agents') {
      fetchAvailableTools();
      fetchAvailableModels();
    }
  }, [category]);

  const fetchAvailableTools = async () => {
    try {
      const response = await fetch('/api/v1/mcp/tools/');
      if (response.ok) {
        const tools = await response.json();
        const toolNames = Array.isArray(tools) 
          ? tools.map((tool: any) => tool.name).filter(Boolean)
          : (tools.data || []).map((tool: any) => tool.name).filter(Boolean);
        setAvailableTools(toolNames);
      } else {
        console.error('Failed to fetch tools');
        // Fallback to common tools if API fails
        setAvailableTools(['search', 'calculator', 'weather', 'email', 'file_manager', 'database']);
      }
    } catch (error) {
      console.error('Error fetching tools:', error);
      // Fallback to common tools if API fails
      setAvailableTools(['search', 'calculator', 'weather', 'email', 'file_manager', 'database']);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch('/api/v1/ollama/models');
      if (response.ok) {
        const data = await response.json();
        const modelNames = (data.models || []).map((model: any) => model.name).filter(Boolean);
        setAvailableModels(modelNames);
      }
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const getTableHeaders = () => {
    switch (category) {
      case 'automation':
        return ['Name', 'Type', 'Status', 'Description', 'Last Updated', 'Actions'];
      case 'collection_registry':
        return ['Name', 'Type', 'Access Level', 'Documents', 'Chunks', 'Size (MB)', 'Description', 'Actions'];
      case 'langgraph_agents':
        return ['Name', 'Role', 'Active', 'Tools', 'Description', 'Actions'];
      default:
        return ['ID', 'Data', 'Actions'];
    }
  };

  const getNewRecordTemplate = () => {
    switch (category) {
      case 'automation':
        return {
          name: '',
          description: '',
          status: 'draft',
          workflow_type: 'langflow',
          config: {}
        } as WorkflowRecord;
      case 'collection_registry':
        return {
          collection_name: '',
          description: '',
          collection_type: 'vector',
          access_config: {
            restricted: false,
            allowed_users: []
          },
          metadata: {}
        } as CollectionRecord;
      case 'langgraph_agents':
        return {
          name: '',
          role: '',
          system_prompt: '',
          tools: [],
          description: '',
          is_active: true,
          config: {}
        } as AgentRecord;
      default:
        return {};
    }
  };

  const handleAdd = () => {
    setEditingRecord(getNewRecordTemplate());
    setEditDialog(true);
  };

  const handleEdit = (record: any) => {
    setEditingRecord({ ...record });
    setEditDialog(true);
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this record?')) return;
    
    try {
      const endpoint = getDeleteEndpoint(id);
      const response = await fetch(endpoint, { method: 'DELETE' });
      
      if (response.ok) {
        onRefresh();
      } else {
        alert('Failed to delete record');
      }
    } catch (error) {
      alert('Error deleting record: ' + error);
    }
  };

  const handleSave = async () => {
    try {
      const endpoint = getSaveEndpoint();
      const method = editingRecord.id ? 'PUT' : 'POST';
      const url = editingRecord.id ? `${endpoint}/${editingRecord.id}` : endpoint;
      
      const response = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editingRecord)
      });
      
      if (response.ok) {
        setEditDialog(false);
        setEditingRecord(null);
        
        // For LangGraph agents, explicitly reload the cache
        if (category === 'langgraph_agents') {
          try {
            const cacheResponse = await fetch('/api/v1/langgraph/agents/cache/reload', {
              method: 'POST'
            });
            if (!cacheResponse.ok) {
              console.error('Failed to reload agent cache');
            }
          } catch (error) {
            console.error('Error reloading agent cache:', error);
          }
        }
        
        onRefresh();
      } else {
        alert('Failed to save record');
      }
    } catch (error) {
      alert('Error saving record: ' + error);
    }
  };

  const handleRun = async (record: any) => {
    if (category === 'automation') {
      try {
        const response = await fetch(`/api/v1/automation/workflows/${record.id}/execute`, {
          method: 'POST'
        });
        
        if (response.ok) {
          alert('Workflow execution started');
        } else {
          alert('Failed to start workflow');
        }
      } catch (error) {
        alert('Error running workflow: ' + error);
      }
    } else {
      alert('Run functionality not available for this category');
    }
  };

  const getSaveEndpoint = () => {
    switch (category) {
      case 'automation':
        return '/api/v1/automation/workflows';
      case 'collection_registry':
        return '/api/v1/collections/';
      case 'langgraph_agents':
        return '/api/v1/langgraph/agents';
      default:
        return '/api/v1/data';
    }
  };

  const getDeleteEndpoint = (id: string) => {
    return `${getSaveEndpoint()}/${id}`;
  };

  const handleRefreshStatistics = async () => {
    if (category !== 'collection_registry') return;
    
    setRefreshingStats(true);
    try {
      const response = await fetch('/api/v1/collections/refresh-statistics', {
        method: 'POST'
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Update statistics in local data immediately
        const updatedData = localData.map((record: any) => {
          if (result.statistics && result.statistics[record.collection_name]) {
            const stats = result.statistics[record.collection_name];
            return {
              ...record,
              statistics: {
                document_count: stats.document_count || 0,
                total_chunks: stats.total_chunks || 0,
                storage_size_mb: stats.storage_size_mb || 0,
                last_updated: stats.last_updated
              }
            };
          }
          return record;
        });
        setLocalData(updatedData);
      } else {
        const error = await response.text();
        console.error(`Failed to refresh statistics: ${error}`);
      }
    } catch (error) {
      console.error('Error refreshing statistics:', error);
    } finally {
      setRefreshingStats(false);
    }
  };

  const renderTableRow = (record: any, index: number) => {
    switch (category) {
      case 'automation':
        return (
          <TableRow key={record.id || index}>
            <TableCell>{record.name}</TableCell>
            <TableCell>
              <Chip label={record.workflow_type} size="small" />
            </TableCell>
            <TableCell>
              <Chip 
                label={record.status} 
                color={record.status === 'active' ? 'success' : record.status === 'inactive' ? 'error' : 'default'}
                size="small" 
              />
            </TableCell>
            <TableCell>{record.description || 'No description'}</TableCell>
            <TableCell>{record.updated_at ? new Date(record.updated_at).toLocaleDateString() : 'N/A'}</TableCell>
            <TableCell>
              <IconButton size="small" onClick={() => { setViewingRecord(record); setViewDialog(true); }}>
                <ViewIcon />
              </IconButton>
              <IconButton size="small" onClick={() => handleEdit(record)}>
                <EditIcon />
              </IconButton>
              <IconButton size="small" onClick={() => handleRun(record)} color="primary">
                <RunIcon />
              </IconButton>
              <IconButton size="small" onClick={() => handleDelete(record.id)} color="error">
                <DeleteIcon />
              </IconButton>
            </TableCell>
          </TableRow>
        );
      case 'collection_registry':
        return (
          <TableRow key={record.id || index}>
            <TableCell>{record.collection_name}</TableCell>
            <TableCell>
              <Chip label={record.collection_type} size="small" />
            </TableCell>
            <TableCell>
              <Chip 
                label={record.access_config?.restricted ? 'Restricted' : 'Public'} 
                color={record.access_config?.restricted ? 'warning' : 'success'}
                size="small" 
              />
              {record.access_config?.restricted && record.access_config?.allowed_users?.length > 0 && (
                <Typography variant="caption" display="block" color="text.secondary">
                  {record.access_config.allowed_users.slice(0, 3).join(', ')}
                  {record.access_config.allowed_users.length > 3 && ` +${record.access_config.allowed_users.length - 3} more`}
                </Typography>
              )}
            </TableCell>
            <TableCell align="center">{record.statistics?.document_count || 0}</TableCell>
            <TableCell align="center">{record.statistics?.total_chunks || 0}</TableCell>
            <TableCell align="center">{record.statistics?.storage_size_mb || 0}</TableCell>
            <TableCell>{record.description || 'No description'}</TableCell>
            <TableCell>
              <IconButton size="small" onClick={() => handleEdit(record)}>
                <EditIcon />
              </IconButton>
              <IconButton size="small" onClick={() => handleDelete(record.id)} color="error">
                <DeleteIcon />
              </IconButton>
            </TableCell>
          </TableRow>
        );
      case 'langgraph_agents':
        return (
          <TableRow key={record.id || index}>
            <TableCell>{record.name}</TableCell>
            <TableCell>{record.role}</TableCell>
            <TableCell>
              <Chip 
                label={record.is_active ? 'Active' : 'Inactive'} 
                color={record.is_active ? 'success' : 'error'}
                size="small" 
              />
            </TableCell>
            <TableCell>
              {record.tools?.slice(0, 3).map((tool: string) => (
                <Chip key={tool} label={tool} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
              ))}
              {record.tools?.length > 3 && <Chip label={`+${record.tools.length - 3} more`} size="small" />}
            </TableCell>
            <TableCell>{record.description || 'No description'}</TableCell>
            <TableCell>
              <IconButton size="small" onClick={() => handleEdit(record)}>
                <EditIcon />
              </IconButton>
              <IconButton size="small" onClick={() => handleDelete(record.id)} color="error">
                <DeleteIcon />
              </IconButton>
            </TableCell>
          </TableRow>
        );
      default:
        return (
          <TableRow key={index}>
            <TableCell>{record.id || index}</TableCell>
            <TableCell>{JSON.stringify(record)}</TableCell>
            <TableCell>
              <IconButton size="small" onClick={() => handleEdit(record)}>
                <EditIcon />
              </IconButton>
            </TableCell>
          </TableRow>
        );
    }
  };

  const renderEditForm = () => {
    if (!editingRecord) return null;

    switch (category) {
      case 'automation':
        return (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              label="Workflow Name"
              value={editingRecord.name}
              onChange={(e) => setEditingRecord({...editingRecord, name: e.target.value})}
              fullWidth
              required
            />
            <TextField
              label="Description"
              value={editingRecord.description}
              onChange={(e) => setEditingRecord({...editingRecord, description: e.target.value})}
              fullWidth
              multiline
              rows={3}
            />
            <FormControl fullWidth>
              <InputLabel>Workflow Type</InputLabel>
              <Select
                value={editingRecord.workflow_type}
                onChange={(e) => setEditingRecord({...editingRecord, workflow_type: e.target.value})}
              >
                <MenuItem value="langflow">Langflow</MenuItem>
                <MenuItem value="custom">Custom</MenuItem>
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>Status</InputLabel>
              <Select
                value={editingRecord.status}
                onChange={(e) => setEditingRecord({...editingRecord, status: e.target.value})}
              >
                <MenuItem value="draft">Draft</MenuItem>
                <MenuItem value="active">Active</MenuItem>
                <MenuItem value="inactive">Inactive</MenuItem>
              </Select>
            </FormControl>
            <TextField
              label="Configuration (JSON)"
              value={JSON.stringify(editingRecord.config, null, 2)}
              onChange={(e) => {
                try {
                  const config = JSON.parse(e.target.value);
                  setEditingRecord({...editingRecord, config});
                } catch {}
              }}
              fullWidth
              multiline
              rows={6}
              sx={{ fontFamily: 'monospace' }}
            />
          </Box>
        );
      case 'langgraph_agents':
        return (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              label="Agent Name"
              value={editingRecord.name}
              onChange={(e) => setEditingRecord({...editingRecord, name: e.target.value})}
              fullWidth
              required
            />
            <TextField
              label="Role"
              value={editingRecord.role}
              onChange={(e) => setEditingRecord({...editingRecord, role: e.target.value})}
              fullWidth
              required
            />
            <TextField
              label="Description"
              value={editingRecord.description}
              onChange={(e) => setEditingRecord({...editingRecord, description: e.target.value})}
              fullWidth
              multiline
              rows={2}
            />
            <TextField
              label="System Prompt"
              value={editingRecord.system_prompt}
              onChange={(e) => setEditingRecord({...editingRecord, system_prompt: e.target.value})}
              fullWidth
              multiline
              rows={8}
              required
              sx={{ fontFamily: 'monospace' }}
            />
            <FormControl fullWidth>
              <InputLabel id="tools-select-label">Tools</InputLabel>
              <Select
                labelId="tools-select-label"
                id="tools-select"
                multiple
                value={editingRecord.tools || []}
                onChange={(event) => {
                  const value = event.target.value;
                  setEditingRecord({
                    ...editingRecord,
                    tools: typeof value === 'string' ? value.split(',') : value
                  });
                }}
                input={<OutlinedInput label="Tools" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {(selected as string[]).map((value) => (
                      <Chip 
                        key={value} 
                        label={value} 
                        size="small"
                        onDelete={(event) => {
                          event.stopPropagation();
                          const newTools = (editingRecord.tools || []).filter((tool: string) => tool !== value);
                          setEditingRecord({
                            ...editingRecord,
                            tools: newTools
                          });
                        }}
                        onMouseDown={(event) => {
                          event.stopPropagation();
                        }}
                      />
                    ))}
                  </Box>
                )}
              >
                {availableTools.map((tool) => (
                  <MenuItem key={tool} value={tool}>
                    <Checkbox checked={(editingRecord.tools || []).indexOf(tool) > -1} />
                    <ListItemText primary={tool} />
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            {/* Configuration Section */}
            <Box sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Agent Configuration
              </Typography>
              
              {/* Quick Presets */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="caption" color="text.secondary" gutterBottom>
                  Quick Presets
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 1 }}>
                  <Chip
                    label="Use Main LLM"
                    onClick={() => setEditingRecord({
                      ...editingRecord,
                      config: { use_main_llm: true }
                    })}
                    variant="outlined"
                    size="small"
                    color={editingRecord.config?.use_main_llm ? "primary" : "default"}
                  />
                  <Chip
                    label="Use Second LLM"
                    onClick={() => setEditingRecord({
                      ...editingRecord,
                      config: { use_second_llm: true }
                    })}
                    variant="outlined"
                    size="small"
                    color={editingRecord.config?.use_second_llm ? "primary" : "default"}
                  />
                  <Chip
                    label="Lightweight"
                    onClick={() => setEditingRecord({
                      ...editingRecord,
                      config: { max_tokens: 1500, temperature: 0.7, timeout: 30 }
                    })}
                    variant="outlined"
                    size="small"
                  />
                  <Chip
                    label="Standard"
                    onClick={() => setEditingRecord({
                      ...editingRecord,
                      config: { max_tokens: 2000, temperature: 0.7, timeout: 45 }
                    })}
                    variant="outlined"
                    size="small"
                  />
                  <Chip
                    label="Heavy Processing"
                    onClick={() => setEditingRecord({
                      ...editingRecord,
                      config: { max_tokens: 3000, temperature: 0.6, timeout: 90 }
                    })}
                    variant="outlined"
                    size="small"
                  />
                </Box>
              </Box>
              
              {/* Configuration Info */}
              {editingRecord.config?.use_main_llm && (
                <Alert severity="info" sx={{ mb: 2 }}>
                  This agent will use main_llm configuration as base. You can still override specific settings below.
                </Alert>
              )}
              {(editingRecord.config?.use_second_llm || (!editingRecord.config?.model && !editingRecord.config?.use_main_llm)) && (
                <Alert severity="info" sx={{ mb: 2 }}>
                  This agent will use second_llm configuration as base. You can override specific settings below.
                </Alert>
              )}
              
              {/* Model Selection */}
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Model</InputLabel>
                <Select
                  value={
                    editingRecord.config?.use_main_llm ? '__use_main_llm__' : 
                    (editingRecord.config?.use_second_llm || !editingRecord.config?.model) ? '__use_second_llm__' : 
                    editingRecord.config.model
                  }
                  onChange={(e) => {
                    if (e.target.value === '__use_main_llm__') {
                      setEditingRecord({
                        ...editingRecord,
                        config: { ...editingRecord.config, use_main_llm: true, use_second_llm: undefined, model: undefined }
                      });
                    } else if (e.target.value === '__use_second_llm__') {
                      setEditingRecord({
                        ...editingRecord,
                        config: { ...editingRecord.config, use_second_llm: true, use_main_llm: undefined, model: undefined }
                      });
                    } else {
                      setEditingRecord({
                        ...editingRecord,
                        config: { ...editingRecord.config, model: e.target.value, use_main_llm: undefined, use_second_llm: undefined }
                      });
                    }
                  }}
                  label="Model"
                >
                  <MenuItem value="__use_second_llm__">
                    <em>Use system default (second_llm)</em>
                  </MenuItem>
                  <MenuItem value="__use_main_llm__">
                    <em>Use main_llm configuration</em>
                  </MenuItem>
                  {availableModels.map(model => (
                    <MenuItem key={model} value={model}>
                      {model}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Max Tokens */}
              <TextField
                label="Max Tokens"
                type="number"
                value={editingRecord.config?.max_tokens || ''}
                onChange={(e) => setEditingRecord({
                  ...editingRecord,
                  config: { 
                    ...editingRecord.config, 
                    max_tokens: e.target.value ? parseInt(e.target.value) : undefined 
                  }
                })}
                fullWidth
                sx={{ mb: 2 }}
                helperText="Maximum tokens (500-3000). Leave empty for default"
                InputProps={{
                  inputProps: { min: 100, max: 10000, step: 100 }
                }}
              />

              {/* Temperature */}
              <TextField
                label="Temperature"
                type="number"
                value={editingRecord.config?.temperature !== undefined ? editingRecord.config.temperature : ''}
                onChange={(e) => setEditingRecord({
                  ...editingRecord,
                  config: { 
                    ...editingRecord.config, 
                    temperature: e.target.value ? parseFloat(e.target.value) : undefined 
                  }
                })}
                fullWidth
                sx={{ mb: 2 }}
                helperText="Creativity (0.1-1.0). Default: 0.7"
                InputProps={{
                  inputProps: { min: 0.1, max: 1.0, step: 0.1 }
                }}
              />

              {/* Timeout */}
              <TextField
                label="Timeout (seconds)"
                type="number"
                value={editingRecord.config?.timeout || ''}
                onChange={(e) => setEditingRecord({
                  ...editingRecord,
                  config: { 
                    ...editingRecord.config, 
                    timeout: e.target.value ? parseInt(e.target.value) : undefined 
                  }
                })}
                fullWidth
                sx={{ mb: 2 }}
                helperText="Execution timeout (15-90s). Default: 30s"
                InputProps={{
                  inputProps: { min: 15, max: 90, step: 5 }
                }}
              />

            </Box>
            <FormControlLabel
              control={
                <Switch
                  checked={editingRecord.is_active}
                  onChange={(e) => setEditingRecord({...editingRecord, is_active: e.target.checked})}
                />
              }
              label="Active"
            />
          </Box>
        );
      case 'collection_registry':
        return (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              label="Collection Name"
              value={editingRecord.collection_name || ''}
              onChange={(e) => setEditingRecord({...editingRecord, collection_name: e.target.value})}
              fullWidth
              required
              helperText="Unique identifier for the collection"
            />
            <TextField
              label="Description"
              value={editingRecord.description || ''}
              onChange={(e) => setEditingRecord({...editingRecord, description: e.target.value})}
              fullWidth
              multiline
              rows={3}
              helperText="Brief description of the collection purpose"
            />
            <FormControl fullWidth>
              <InputLabel>Collection Type</InputLabel>
              <Select
                value={editingRecord.collection_type || 'general'}
                onChange={(e) => setEditingRecord({...editingRecord, collection_type: e.target.value})}
              >
                <MenuItem value="general">General</MenuItem>
                <MenuItem value="technical_docs">Technical Documentation</MenuItem>
                <MenuItem value="regulatory_compliance">Regulatory Compliance</MenuItem>
                <MenuItem value="product_documentation">Product Documentation</MenuItem>
                <MenuItem value="risk_management">Risk Management</MenuItem>
                <MenuItem value="customer_support">Customer Support</MenuItem>
                <MenuItem value="audit_reports">Audit Reports</MenuItem>
                <MenuItem value="training_materials">Training Materials</MenuItem>
                <MenuItem value="partnership">Partnership</MenuItem>
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>Access Level</InputLabel>
              <Select
                value={editingRecord.access_config?.restricted === false ? 'public' : 
                       editingRecord.access_config?.restricted === true ? 'restricted' : 'public'}
                onChange={(e) => {
                  const isRestricted = e.target.value === 'restricted';
                  setEditingRecord({
                    ...editingRecord, 
                    access_config: {
                      ...editingRecord.access_config,
                      restricted: isRestricted,
                      allowed_users: isRestricted ? (editingRecord.access_config?.allowed_users || []) : []
                    }
                  });
                }}
              >
                <MenuItem value="public">Public</MenuItem>
                <MenuItem value="restricted">Restricted</MenuItem>
              </Select>
            </FormControl>
            {editingRecord.access_config?.restricted && (
              <TextField
                label="Allowed Users (comma-separated)"
                value={editingRecord.access_config?.allowed_users?.join(', ') || ''}
                onChange={(e) => {
                  const users = e.target.value.split(',').map(u => u.trim()).filter(u => u);
                  setEditingRecord({
                    ...editingRecord,
                    access_config: {
                      ...editingRecord.access_config,
                      allowed_users: users
                    }
                  });
                }}
                fullWidth
                helperText="Enter usernames or roles separated by commas"
              />
            )}
            <TextField
              label="Chunk Size"
              type="number"
              value={editingRecord.metadata_schema?.chunk_size || 1500}
              onChange={(e) => setEditingRecord({
                ...editingRecord,
                metadata_schema: {
                  ...editingRecord.metadata_schema,
                  chunk_size: parseInt(e.target.value) || 1500
                }
              })}
              fullWidth
              helperText="Size of text chunks for processing (default: 1500)"
            />
            <TextField
              label="Chunk Overlap"
              type="number"
              value={editingRecord.metadata_schema?.chunk_overlap || 200}
              onChange={(e) => setEditingRecord({
                ...editingRecord,
                metadata_schema: {
                  ...editingRecord.metadata_schema,
                  chunk_overlap: parseInt(e.target.value) || 200
                }
              })}
              fullWidth
              helperText="Overlap between chunks (default: 200)"
            />
            <FormControl fullWidth>
              <InputLabel>Search Strategy</InputLabel>
              <Select
                value={editingRecord.search_config?.strategy || 'balanced'}
                onChange={(e) => setEditingRecord({
                  ...editingRecord,
                  search_config: {
                    ...editingRecord.search_config,
                    strategy: e.target.value
                  }
                })}
              >
                <MenuItem value="balanced">Balanced</MenuItem>
                <MenuItem value="precise">Precise</MenuItem>
                <MenuItem value="comprehensive">Comprehensive</MenuItem>
                <MenuItem value="temporal">Temporal</MenuItem>
                <MenuItem value="exact">Exact</MenuItem>
              </Select>
            </FormControl>
            <TextField
              label="Similarity Threshold"
              type="number"
              inputProps={{ min: 0, max: 1, step: 0.01 }}
              value={editingRecord.search_config?.similarity_threshold || 0.7}
              onChange={(e) => setEditingRecord({
                ...editingRecord,
                search_config: {
                  ...editingRecord.search_config,
                  similarity_threshold: parseFloat(e.target.value) || 0.7
                }
              })}
              fullWidth
              helperText="Minimum similarity score (0.0 - 1.0, default: 0.7)"
            />
          </Box>
        );
      // Add similar forms for other categories
      default:
        return (
          <TextField
            label="JSON Data"
            value={JSON.stringify(editingRecord, null, 2)}
            onChange={(e) => {
              try {
                const parsed = JSON.parse(e.target.value);
                setEditingRecord(parsed);
              } catch {}
            }}
            fullWidth
            multiline
            rows={10}
            sx={{ fontFamily: 'monospace' }}
          />
        );
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          {category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} Management
        </Typography>
        <Box>
          {category === 'collection_registry' && (
            <Button 
              startIcon={<RefreshIcon />} 
              onClick={handleRefreshStatistics}
              disabled={refreshingStats}
              sx={{ mr: 1 }}
              variant="contained"
              color="secondary"
            >
              {refreshingStats ? 'Updating...' : 'Update Statistics'}
            </Button>
          )}
          <Button startIcon={<AddIcon />} variant="contained" onClick={handleAdd}>
            Add New
          </Button>
        </Box>
      </Box>

      {localData.length === 0 ? (
        <Alert severity="info">No records found</Alert>
      ) : (
        <TableContainer component={Paper} sx={{ maxHeight: '60vh', overflow: 'auto' }}>
          <Table stickyHeader>
            <TableHead>
              <TableRow>
                {getTableHeaders().map((header) => (
                  <TableCell key={header}>{header}</TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {localData.sort((a, b) => {
                const nameA = category === 'collection_registry' ? (a.collection_name || '') : (a.name || '');
                const nameB = category === 'collection_registry' ? (b.collection_name || '') : (b.name || '');
                return nameA.localeCompare(nameB);
              }).map((record, index) => renderTableRow(record, index))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Edit Dialog */}
      <Dialog 
        open={editDialog} 
        onClose={() => setEditDialog(false)} 
        maxWidth="md" 
        fullWidth
        PaperProps={{
          sx: { 
            maxHeight: '85vh',
            margin: '48px 32px',
            overflow: 'hidden'
          }
        }}
      >
        <DialogTitle sx={{ paddingBottom: '16px' }}>
          {editingRecord?.id ? 'Edit Record' : 'Add New Record'}
        </DialogTitle>
        <DialogContent sx={{ overflow: 'auto', paddingTop: '24px !important', paddingBottom: 2 }}>
          {renderEditForm()}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialog(false)}>Cancel</Button>
          <Button onClick={handleSave} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>

      {/* View Dialog */}
      <Dialog 
        open={viewDialog} 
        onClose={() => setViewDialog(false)} 
        maxWidth="lg" 
        fullWidth
      >
        <DialogTitle>
          View Record Details
        </DialogTitle>
        <DialogContent>
          {viewingRecord && (
            <Box sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(viewingRecord, null, 2)}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>

    </Box>
  );
};

export default DatabaseTableManager;