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
  FormControlLabel,
  Grid,
  Card,
  CardContent,
  CardHeader,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Checkbox
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Visibility as ViewIcon,
  Build as ToolIcon,
  Computer as ServerIcon,
  Settings as SettingsIcon,
  Cached as CacheIcon,
  FileUpload as ImportIcon,
  FileDownload as ExportIcon
} from '@mui/icons-material';

interface MCPTool {
  id?: number;
  name: string;
  description?: string;
  endpoint: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  parameters?: Record<string, any>;
  headers?: Record<string, string>;
  is_active: boolean;
  is_manual: boolean;
  server_id?: number;
  server_name?: string;
  manifest_id?: number;
}

interface MCPToolManagerProps {
  data: MCPTool[];
  onChange: (data: MCPTool[]) => void;
  onRefresh: () => void;
}

const MCPToolManager: React.FC<MCPToolManagerProps> = ({
  data,
  onChange,
  onRefresh
}) => {
  const [editDialog, setEditDialog] = useState(false);
  const [editingTool, setEditingTool] = useState<MCPTool | null>(null);
  const [viewDialog, setViewDialog] = useState(false);
  const [viewingTool, setViewingTool] = useState<MCPTool | null>(null);
  const [bulkDialog, setBulkDialog] = useState(false);
  const [selectedTools, setSelectedTools] = useState<number[]>([]);

  const getNewToolTemplate = (): MCPTool => ({
    name: '',
    description: '',
    endpoint: '',
    method: 'POST',
    parameters: {},
    headers: {},
    is_active: true,
    is_manual: true
  });

  const handleAdd = () => {
    setEditingTool(getNewToolTemplate());
    setEditDialog(true);
  };

  const handleEdit = (tool: MCPTool) => {
    setEditingTool({ ...tool });
    setEditDialog(true);
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this MCP tool?')) return;
    
    try {
      const response = await fetch(`/api/v1/mcp/tools/${id}/`, { method: 'DELETE' });
      if (response.ok) {
        onRefresh();
      } else {
        alert('Failed to delete tool');
      }
    } catch (error) {
      alert('Error deleting tool: ' + error);
    }
  };

  const handleSave = async () => {
    if (!editingTool) return;
    
    try {
      const endpoint = '/api/v1/mcp/tools/';
      const method = editingTool.id ? 'PUT' : 'POST';
      const url = editingTool.id ? `${endpoint}${editingTool.id}/` : endpoint;
      
      const response = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editingTool)
      });
      
      if (response.ok) {
        setEditDialog(false);
        setEditingTool(null);
        onRefresh();
      } else {
        alert('Failed to save tool');
      }
    } catch (error) {
      alert('Error saving tool: ' + error);
    }
  };

  const handleBulkToggle = async () => {
    try {
      const enabledTools = data.filter(tool => 
        selectedTools.includes(tool.id!) ? !tool.is_active : tool.is_active
      ).map(tool => tool.id!);

      const response = await fetch('/api/v1/mcp/tools/enabled/', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled_tool_ids: enabledTools })
      });
      
      if (response.ok) {
        setBulkDialog(false);
        setSelectedTools([]);
        onRefresh();
      } else {
        alert('Failed to update tools');
      }
    } catch (error) {
      alert('Error updating tools: ' + error);
    }
  };

  const handleCacheReload = async () => {
    try {
      const response = await fetch('/api/v1/mcp/tools/cache/reload/', {
        method: 'POST'
      });
      
      if (response.ok) {
        alert('MCP tools cache reloaded successfully');
        onRefresh();
      } else {
        alert('Failed to reload cache');
      }
    } catch (error) {
      alert('Error reloading cache: ' + error);
    }
  };

  const handleExportTools = () => {
    const exportData = data.map(tool => ({
      name: tool.name,
      description: tool.description,
      endpoint: tool.endpoint,
      method: tool.method,
      parameters: tool.parameters,
      headers: tool.headers,
      is_active: tool.is_active
    }));
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'mcp-tools-export.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const renderToolTable = () => (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell padding="checkbox">
              <Checkbox
                indeterminate={selectedTools.length > 0 && selectedTools.length < data.length}
                checked={data.length > 0 && selectedTools.length === data.length}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSelectedTools(data.map(tool => tool.id!));
                  } else {
                    setSelectedTools([]);
                  }
                }}
              />
            </TableCell>
            <TableCell>Name</TableCell>
            <TableCell>Method</TableCell>
            <TableCell>Endpoint</TableCell>
            <TableCell>Source</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((tool, index) => (
            <TableRow key={tool.id || index}>
              <TableCell padding="checkbox">
                <Checkbox
                  checked={selectedTools.includes(tool.id!)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedTools([...selectedTools, tool.id!]);
                    } else {
                      setSelectedTools(selectedTools.filter(id => id !== tool.id));
                    }
                  }}
                />
              </TableCell>
              <TableCell>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ToolIcon fontSize="small" />
                  <Box>
                    <Typography variant="body2" fontWeight="bold">
                      {tool.name}
                    </Typography>
                    {tool.description && (
                      <Typography variant="caption" color="text.secondary">
                        {tool.description}
                      </Typography>
                    )}
                  </Box>
                </Box>
              </TableCell>
              <TableCell>
                <Chip 
                  label={tool.method} 
                  size="small"
                  color={tool.method === 'GET' ? 'primary' : tool.method === 'POST' ? 'secondary' : 'default'}
                />
              </TableCell>
              <TableCell>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                  {tool.endpoint.length > 40 ? `${tool.endpoint.substring(0, 40)}...` : tool.endpoint}
                </Typography>
              </TableCell>
              <TableCell>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {tool.is_manual ? (
                    <Chip label="Manual" size="small" color="warning" />
                  ) : (
                    <Chip 
                      label={tool.server_name || `Server ${tool.server_id}`} 
                      size="small" 
                      color="info"
                    />
                  )}
                </Box>
              </TableCell>
              <TableCell>
                <Chip 
                  label={tool.is_active ? 'Active' : 'Inactive'} 
                  color={tool.is_active ? 'success' : 'default'}
                  size="small" 
                />
              </TableCell>
              <TableCell>
                <IconButton size="small" onClick={() => setViewingTool(tool) || setViewDialog(true)}>
                  <ViewIcon />
                </IconButton>
                <IconButton size="small" onClick={() => handleEdit(tool)}>
                  <EditIcon />
                </IconButton>
                <IconButton 
                  size="small" 
                  onClick={() => handleDelete(tool.id!)}
                  color="error"
                >
                  <DeleteIcon />
                </IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );

  const renderEditForm = () => {
    if (!editingTool) return null;

    const updateTool = (updates: Partial<MCPTool>) => {
      setEditingTool({ ...editingTool, ...updates });
    };

    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, maxHeight: '70vh', overflow: 'auto' }}>
        <Card>
          <CardHeader title="Basic Configuration" />
          <CardContent>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Tool Name"
                  value={editingTool.name}
                  onChange={(e) => updateTool({ name: e.target.value })}
                  fullWidth
                  required
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>HTTP Method</InputLabel>
                  <Select
                    value={editingTool.method}
                    onChange={(e) => updateTool({ method: e.target.value as any })}
                  >
                    <MenuItem value="GET">GET</MenuItem>
                    <MenuItem value="POST">POST</MenuItem>
                    <MenuItem value="PUT">PUT</MenuItem>
                    <MenuItem value="DELETE">DELETE</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Description"
                  value={editingTool.description || ''}
                  onChange={(e) => updateTool({ description: e.target.value })}
                  fullWidth
                  multiline
                  rows={2}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Endpoint URL"
                  value={editingTool.endpoint}
                  onChange={(e) => updateTool({ endpoint: e.target.value })}
                  fullWidth
                  required
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={editingTool.is_active}
                      onChange={(e) => updateTool({ is_active: e.target.checked })}
                    />
                  }
                  label="Active"
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        <Card>
          <CardHeader title="Parameters & Headers" />
          <CardContent>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  label="Parameters (JSON)"
                  value={JSON.stringify(editingTool.parameters || {}, null, 2)}
                  onChange={(e) => {
                    try {
                      const parameters = JSON.parse(e.target.value);
                      updateTool({ parameters });
                    } catch {}
                  }}
                  fullWidth
                  multiline
                  rows={4}
                  sx={{ fontFamily: 'monospace' }}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Headers (JSON)"
                  value={JSON.stringify(editingTool.headers || {}, null, 2)}
                  onChange={(e) => {
                    try {
                      const headers = JSON.parse(e.target.value);
                      updateTool({ headers });
                    } catch {}
                  }}
                  fullWidth
                  multiline
                  rows={3}
                  sx={{ fontFamily: 'monospace' }}
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Box>
    );
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">MCP Tool Management</Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button 
            onClick={handleCacheReload} 
            startIcon={<CacheIcon />}
            variant="outlined"
          >
            Reload Cache
          </Button>
          <Button 
            onClick={handleExportTools} 
            startIcon={<ExportIcon />}
            variant="outlined"
          >
            Export Tools
          </Button>
          <Button onClick={onRefresh} startIcon={<RefreshIcon />}>
            Refresh
          </Button>
          <Button 
            onClick={handleAdd} 
            variant="contained" 
            startIcon={<AddIcon />}
          >
            Add Manual Tool
          </Button>
        </Box>
      </Box>

      {selectedTools.length > 0 && (
        <Alert 
          severity="info" 
          sx={{ mb: 2 }}
          action={
            <Button 
              color="inherit" 
              size="small" 
              onClick={() => setBulkDialog(true)}
            >
              Bulk Actions
            </Button>
          }
        >
          {selectedTools.length} tool(s) selected
        </Alert>
      )}

      {data.length === 0 ? (
        <Alert severity="info">No MCP tools found. Add servers to discover tools automatically.</Alert>
      ) : (
        renderToolTable()
      )}

      {/* Edit Dialog */}
      <Dialog open={editDialog} onClose={() => setEditDialog(false)} maxWidth="lg" fullWidth>
        <DialogTitle>
          {editingTool?.id ? 'Edit MCP Tool' : 'Add MCP Tool'}
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
        <DialogTitle>MCP Tool Details</DialogTitle>
        <DialogContent>
          <pre style={{ fontSize: '12px', overflow: 'auto' }}>
            {JSON.stringify(viewingTool, null, 2)}
          </pre>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Bulk Actions Dialog */}
      <Dialog open={bulkDialog} onClose={() => setBulkDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Bulk Actions</DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            Selected {selectedTools.length} tool(s). Choose an action:
          </Typography>
          <List>
            <ListItem button onClick={handleBulkToggle}>
              <ListItemText 
                primary="Toggle Active Status" 
                secondary="Enable inactive tools, disable active tools"
              />
            </ListItem>
          </List>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBulkDialog(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default MCPToolManager;