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
  Menu,
  MenuItem as MuiMenuItem,
  Snackbar,
  FormControlLabel,
  Grid,
  Card,
  CardContent,
  CardHeader,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Checkbox,
  Divider
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
  FileDownload as ExportIcon,
  MoreVert as MoreVertIcon
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

interface MCPServer {
  id: number;
  name: string;
}

interface MCPToolManagerProps {
  data: MCPTool[];
  servers?: MCPServer[];
  onChange: (data: MCPTool[]) => void;
  onRefresh: () => void;
}

const MCPToolManager: React.FC<MCPToolManagerProps> = ({
  data,
  servers,
  onChange,
  onRefresh
}) => {
  const [editDialog, setEditDialog] = useState(false);
  const [editingTool, setEditingTool] = useState<MCPTool | null>(null);
  const [viewDialog, setViewDialog] = useState(false);
  const [viewingTool, setViewingTool] = useState<MCPTool | null>(null);
  const [bulkDialog, setBulkDialog] = useState(false);
  const [selectedTools, setSelectedTools] = useState<number[]>([]);
  const [statusFilter, setStatusFilter] = useState<'all' | 'active' | 'inactive'>('all');
  const [serverFilter, setServerFilter] = useState<'all' | number>('all');
  const [actionMenuAnchor, setActionMenuAnchor] = useState<null | HTMLElement>(null);
  const [actionMenuTool, setActionMenuTool] = useState<MCPTool | null>(null);
  const [snackbar, setSnackbar] = useState<{open: boolean, message: string, severity: 'success' | 'error' | 'info'}>({
    open: false,
    message: '',
    severity: 'info'
  });

  const showNotification = (message: string, severity: 'success' | 'error' | 'info' = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

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
        showNotification('Tool deleted successfully', 'success');
        onRefresh();
      } else {
        showNotification('Failed to delete tool', 'error');
      }
    } catch (error) {
      showNotification('Error deleting tool: ' + error, 'error');
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
        showNotification('Tool saved successfully', 'success');
        onRefresh();
      } else {
        showNotification('Failed to save tool', 'error');
      }
    } catch (error) {
      showNotification('Error saving tool: ' + error, 'error');
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
        showNotification('Tools updated successfully', 'success');
        onRefresh();
      } else {
        showNotification('Failed to update tools', 'error');
      }
    } catch (error) {
      showNotification('Error updating tools: ' + error, 'error');
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

  // Filter tools based on status and server
  const filteredTools = data.filter(tool => {
    // Apply status filter
    if (statusFilter !== 'all') {
      if (statusFilter === 'active' && !tool.is_active) return false;
      if (statusFilter === 'inactive' && tool.is_active) return false;
    }
    
    // Apply server filter
    if (serverFilter !== 'all') {
      if (serverFilter === 0) {
        // For manual tools, check if is_manual is true AND not internal AND server_id is null or undefined
        if (!tool.is_manual || tool.endpoint.startsWith('internal://') || (tool.server_id !== null && tool.server_id !== undefined)) return false;
      } else if (serverFilter === -1) {
        // For internal tools, check if endpoint starts with internal://
        if (!tool.endpoint.startsWith('internal://')) return false;
      } else {
        // For specific server, check server_id matches
        if (tool.server_id !== serverFilter) return false;
      }
    }
    
    return true;
  });
  
  // Get unique servers from tools data
  const uniqueServers = React.useMemo(() => {
    const serverMap = new Map<number, string>();
    data.forEach(tool => {
      if (tool.server_id && tool.server_name) {
        serverMap.set(tool.server_id, tool.server_name);
      }
    });
    return Array.from(serverMap.entries()).map(([id, name]) => ({ id, name }));
  }, [data]);
  
  // Check if there are internal tools
  const hasInternalTools = React.useMemo(() => {
    return data.some(tool => tool.endpoint.startsWith('internal://'));
  }, [data]);
  
  // Check if there are manual tools (non-internal)
  const hasManualTools = React.useMemo(() => {
    return data.some(tool => tool.is_manual && !tool.endpoint.startsWith('internal://') && (tool.server_id === null || tool.server_id === undefined));
  }, [data]);

  const handleCloseActionMenu = () => {
    setActionMenuAnchor(null);
    setActionMenuTool(null);
  };

  const handleMenuAction = async (action: string) => {
    if (!actionMenuTool) return;
    
    if (action === 'view') {
      setViewingTool(actionMenuTool);
      setViewDialog(true);
    } else if (action === 'delete') {
      await handleDelete(actionMenuTool.id!);
    } else if (action === 'toggle') {
      // Toggle the active status directly via API
      try {
        const response = await fetch(`/api/v1/mcp/tools/${actionMenuTool.id}/`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...actionMenuTool, is_active: !actionMenuTool.is_active })
        });
        
        if (response.ok) {
          showNotification(`Tool ${actionMenuTool.is_active ? 'disabled' : 'enabled'} successfully`, 'success');
          onRefresh();
        } else {
          showNotification('Failed to toggle tool status', 'error');
        }
      } catch (error) {
        showNotification('Error toggling tool: ' + error, 'error');
      }
    }
    
    handleCloseActionMenu();
  };

  const renderToolTable = () => (
    <TableContainer component={Paper}>
      <Table sx={{ tableLayout: 'fixed' }}>
        <TableHead>
          <TableRow>
            <TableCell padding="checkbox" sx={{ width: '50px' }}>
              <Checkbox
                indeterminate={selectedTools.length > 0 && selectedTools.length < filteredTools.length}
                checked={filteredTools.length > 0 && selectedTools.length === filteredTools.length}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSelectedTools(filteredTools.map(tool => tool.id!));
                  } else {
                    setSelectedTools([]);
                  }
                }}
              />
            </TableCell>
            <TableCell sx={{ width: '20%', maxWidth: '200px' }}>Name</TableCell>
            <TableCell sx={{ width: '80px' }}>Method</TableCell>
            <TableCell sx={{ width: '35%' }}>Endpoint</TableCell>
            <TableCell sx={{ width: '120px' }}>Source</TableCell>
            <TableCell sx={{ width: '90px' }}>Status</TableCell>
            <TableCell sx={{ width: '100px' }}>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {filteredTools.map((tool, index) => (
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
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, overflow: 'hidden' }}>
                  <ToolIcon fontSize="small" sx={{ flexShrink: 0 }} />
                  <Box sx={{ overflow: 'hidden' }}>
                    <Typography 
                      variant="body2" 
                      fontWeight="bold"
                      title={tool.name}
                      sx={{ 
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}
                    >
                      {tool.name}
                    </Typography>
                    {tool.description && (
                      <Typography 
                        variant="caption" 
                        color="text.secondary"
                        title={tool.description}
                        sx={{ 
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          display: 'block'
                        }}
                      >
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
                  {tool.endpoint.startsWith('internal://') ? (
                    <Chip label="Internal" size="small" color="primary" />
                  ) : tool.is_manual ? (
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
                <IconButton size="small" onClick={() => handleEdit(tool)}>
                  <EditIcon />
                </IconButton>
                <IconButton 
                  size="small" 
                  onClick={(e) => {
                    setActionMenuAnchor(e.currentTarget);
                    setActionMenuTool(tool);
                  }}
                >
                  <MoreVertIcon />
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
            onClick={handleExportTools} 
            startIcon={<ExportIcon />}
            variant="outlined"
          >
            Export Tools
          </Button>
          <Button onClick={onRefresh} startIcon={<RefreshIcon />}>
            Refresh Data
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

      {/* Filters */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Status Filter</InputLabel>
            <Select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as 'all' | 'active' | 'inactive')}
              label="Status Filter"
            >
              <MenuItem value="all">All Status</MenuItem>
              <MenuItem value="active">Active ({data.filter(t => t.is_active).length})</MenuItem>
              <MenuItem value="inactive">Inactive ({data.filter(t => !t.is_active).length})</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Server Filter</InputLabel>
            <Select
              value={serverFilter}
              onChange={(e) => setServerFilter(e.target.value === 'all' ? 'all' : Number(e.target.value))}
              label="Server Filter"
            >
              <MenuItem value="all">All Servers</MenuItem>
              {hasInternalTools && (
                <MenuItem value={-1} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip label="Internal" size="small" color="primary" sx={{ height: 20 }} />
                  <span>({data.filter(t => t.endpoint.startsWith('internal://')).length})</span>
                </MenuItem>
              )}
              {hasManualTools && (
                <MenuItem value={0} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip label="Manual" size="small" color="warning" sx={{ height: 20 }} />
                  <span>({data.filter(t => t.is_manual && !t.endpoint.startsWith('internal://') && (t.server_id === null || t.server_id === undefined)).length})</span>
                </MenuItem>
              )}
              {uniqueServers.map(server => (
                <MenuItem key={server.id} value={server.id}>
                  {server.name} ({data.filter(t => t.server_id === server.id).length})
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          {(statusFilter !== 'all' || serverFilter !== 'all') && (
            <Typography variant="body2" color="text.secondary">
              Showing {filteredTools.length} of {data.length} tools
            </Typography>
          )}
        </Box>

        {selectedTools.length > 0 && (
          <Button 
            color="primary" 
            size="small" 
            onClick={() => setBulkDialog(true)}
            variant="outlined"
          >
            Bulk Actions ({selectedTools.length})
          </Button>
        )}
      </Box>

      {data.length === 0 ? (
        <Alert severity="info">No MCP tools found. Add servers to discover tools automatically.</Alert>
      ) : filteredTools.length === 0 ? (
        <Alert severity="info">No tools match the selected filter. Try changing the status filter.</Alert>
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
            <ListItem onClick={handleBulkToggle} sx={{ cursor: 'pointer' }}>
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

      {/* Action Menu */}
      <Menu
        anchorEl={actionMenuAnchor}
        open={Boolean(actionMenuAnchor)}
        onClose={handleCloseActionMenu}
      >
        <MuiMenuItem onClick={() => handleMenuAction('view')}>
          <ViewIcon sx={{ mr: 1 }} fontSize="small" />
          View Details
        </MuiMenuItem>
        <MuiMenuItem onClick={() => handleMenuAction('toggle')}>
          <SettingsIcon sx={{ mr: 1 }} fontSize="small" />
          {actionMenuTool?.is_active ? 'Disable Tool' : 'Enable Tool'}
        </MuiMenuItem>
        <Divider />
        <MuiMenuItem onClick={() => handleMenuAction('delete')}>
          <DeleteIcon sx={{ mr: 1 }} fontSize="small" color="error" />
          <Typography color="error">Delete Tool</Typography>
        </MuiMenuItem>
      </Menu>

      {/* Notification Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Alert 
          onClose={() => setSnackbar({ ...snackbar, open: false })} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default MCPToolManager;