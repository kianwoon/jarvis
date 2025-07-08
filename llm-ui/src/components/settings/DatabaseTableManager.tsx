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
  MenuItem
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as RunIcon,
  Pause as PauseIcon,
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
  name: string;
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
  const [viewDialog, setViewDialog] = useState(false);
  const [viewingRecord, setViewingRecord] = useState<any>(null);

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
          name: '',
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
        return '/api/v1/collections';
      case 'langgraph_agents':
        return '/api/v1/agents';
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
            <TableCell>{record.name}</TableCell>
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
              <IconButton size="small" onClick={() => setViewingRecord(record) || setViewDialog(true)}>
                <ViewIcon />
              </IconButton>
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
              <IconButton size="small" onClick={() => setViewingRecord(record) || setViewDialog(true)}>
                <ViewIcon />
              </IconButton>
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
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                {getTableHeaders().map((header) => (
                  <TableCell key={header}>{header}</TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {data.map((record, index) => renderTableRow(record, index))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Edit Dialog */}
      <Dialog open={editDialog} onClose={() => setEditDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingRecord?.id ? 'Edit Record' : 'Add New Record'}
        </DialogTitle>
        <DialogContent>
          {renderEditForm()}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialog(false)}>Cancel</Button>
          <Button onClick={handleSave} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>

      {/* View Dialog */}
      <Dialog open={viewDialog} onClose={() => setViewDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>View Record</DialogTitle>
        <DialogContent>
          <pre style={{ fontSize: '12px', overflow: 'auto' }}>
            {JSON.stringify(viewingRecord, null, 2)}
          </pre>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DatabaseTableManager;