import React, { useState } from 'react';
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
  FormControlLabel
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as RunIcon,
  Pause as PauseIcon
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
  access_level: 'public' | 'private' | 'restricted';
  metadata: any;
  created_at?: string;
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

  const getTableHeaders = () => {
    switch (category) {
      case 'automation':
        return ['Name', 'Type', 'Status', 'Description', 'Last Updated', 'Actions'];
      case 'collection_registry':
        return ['Name', 'Type', 'Access Level', 'Description', 'Created', 'Actions'];
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
          access_level: 'public',
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
              <IconButton size="small" onClick={() => setViewingRecord(record) || setViewDialog(true)}>
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
                label={record.access_level} 
                color={record.access_level === 'public' ? 'success' : record.access_level === 'private' ? 'warning' : 'error'}
                size="small" 
              />
            </TableCell>
            <TableCell>{record.description || 'No description'}</TableCell>
            <TableCell>{record.created_at ? new Date(record.created_at).toLocaleDateString() : 'N/A'}</TableCell>
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
            <TextField
              label="Tools (JSON Array)"
              value={JSON.stringify(editingRecord.tools || [])}
              onChange={(e) => {
                try {
                  const tools = JSON.parse(e.target.value);
                  setEditingRecord({...editingRecord, tools});
                } catch {}
              }}
              fullWidth
              multiline
              rows={3}
              sx={{ fontFamily: 'monospace' }}
            />
            <TextField
              label="Configuration (JSON)"
              value={JSON.stringify(editingRecord.config || {}, null, 2)}
              onChange={(e) => {
                try {
                  const config = JSON.parse(e.target.value);
                  setEditingRecord({...editingRecord, config});
                } catch {}
              }}
              fullWidth
              multiline
              rows={4}
              sx={{ fontFamily: 'monospace' }}
            />
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
          <Button onClick={onRefresh} sx={{ mr: 1 }}>
            Refresh
          </Button>
          <Button startIcon={<AddIcon />} variant="contained" onClick={handleAdd}>
            Add New
          </Button>
        </Box>
      </Box>

      {data.length === 0 ? (
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
              {data.sort((a, b) => {
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

    </Box>
  );
};

export default DatabaseTableManager;