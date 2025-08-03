import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Dialog, 
  DialogTitle, 
  DialogContent, 
  DialogActions,
  TextField,
  IconButton,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Tooltip,
  FormControlLabel,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Snackbar,
  Card,
  CardContent,
  Divider
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Code as CodeIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Build as BuildIcon
} from '@mui/icons-material';

interface SystemTool {
  id: number;
  name: string;
  description?: string;
  endpoint: string;
  method: string;
  parameters?: any;
  headers?: any;
  is_active: boolean;
  is_internal: boolean;
  tool_type: 'system' | 'internal';
  version?: string;
}

interface SystemToolsManagerProps {
  onRefresh?: () => void;
}

interface ToolFormData {
  name: string;
  description: string;
  parameters: string;
  headers: string;
  method: string;
  endpoint: string;
  is_active: boolean;
}

const SystemToolsManager: React.FC<SystemToolsManagerProps> = ({ onRefresh }) => {
  const [systemTools, setSystemTools] = useState<SystemTool[]>([]);
  const [loading, setLoading] = useState(false);
  const [snackbar, setSnackbar] = useState<{open: boolean, message: string, severity: 'success' | 'error' | 'info'}>({
    open: false, message: '', severity: 'info'
  });
  const [detailsDialog, setDetailsDialog] = useState<{open: boolean, tool: SystemTool | null}>({
    open: false, tool: null
  });
  const [editDialog, setEditDialog] = useState(false);
  const [editingTool, setEditingTool] = useState<SystemTool | null>(null);
  const [formData, setFormData] = useState<ToolFormData>({
    name: '',
    description: '',
    parameters: '{}',
    headers: '{}',
    method: 'POST',
    endpoint: '',
    is_active: true
  });

  const showSnackbar = (message: string, severity: 'success' | 'error' | 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const fetchSystemTools = async () => {
    setLoading(true);
    try {
      // Get all MCP tools and filter for internal/system tools
      const response = await fetch('/api/v1/mcp/tools');
      const data = await response.json();
      
      // Filter for system tools (internal endpoint or is_manual=true with system characteristics)
      const systemToolsData = data.filter((tool: any) => 
        tool.endpoint?.startsWith('internal://') || 
        (tool.is_manual && tool.endpoint?.includes('system'))
      ).map((tool: any) => ({
        ...tool,
        is_internal: tool.endpoint?.startsWith('internal://') || false,
        tool_type: tool.endpoint?.startsWith('internal://') ? 'internal' : 'system'
      }));
      
      setSystemTools(systemToolsData);
    } catch (error) {
      //console.error('Failed to fetch system tools:', error);
      showSnackbar('Failed to load system tools', 'error');
    } finally {
      setLoading(false);
    }
  };

  const refreshSystemToolsCache = async () => {
    setLoading(true);
    try {
      // Reload Redis cache for MCP tools
      const response = await fetch('/api/v1/mcp/system-tools/cache/reload', {
        method: 'POST'
      });
      
      if (response.ok) {
        await fetchSystemTools();
        showSnackbar('MCP tools cache refreshed successfully', 'success');
        onRefresh?.();
      } else {
        showSnackbar('Failed to refresh MCP tools cache', 'error');
      }
    } catch (error) {
      //console.error('Failed to refresh MCP tools cache:', error);
      showSnackbar('Failed to refresh MCP tools cache', 'error');
    } finally {
      setLoading(false);
    }
  };

  const toggleToolStatus = async (tool: SystemTool) => {
    try {
      const response = await fetch(`/api/v1/mcp/tools/${tool.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...tool,
          is_active: !tool.is_active
        })
      });

      if (response.ok) {
        await fetchSystemTools();
        showSnackbar(`Tool ${tool.is_active ? 'disabled' : 'enabled'} successfully`, 'success');
        onRefresh?.();
      } else {
        showSnackbar('Failed to update tool status', 'error');
      }
    } catch (error) {
      //console.error('Failed to toggle tool status:', error);
      showSnackbar('Failed to update tool status', 'error');
    }
  };

  const reinstallSystemTools = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/mcp/system-tools/reinstall', {
        method: 'POST'
      });
      
      if (response.ok) {
        await fetchSystemTools();
        showSnackbar('System tools reinstalled successfully', 'success');
        onRefresh?.();
      } else {
        showSnackbar('Failed to reinstall system tools', 'error');
      }
    } catch (error) {
      //console.error('Failed to reinstall system tools:', error);
      showSnackbar('Failed to reinstall system tools', 'error');
    } finally {
      setLoading(false);
    }
  };

  const showToolDetails = (tool: SystemTool) => {
    setDetailsDialog({ open: true, tool });
  };

  const handleEditTool = (tool: SystemTool) => {
    setEditingTool(tool);
    setFormData({
      name: tool.name,
      description: tool.description || '',
      parameters: JSON.stringify(tool.parameters || {}, null, 2),
      headers: JSON.stringify(tool.headers || {}, null, 2),
      method: tool.method,
      endpoint: tool.endpoint,
      is_active: tool.is_active
    });
    setEditDialog(true);
  };

  const handleDeleteTool = async (tool: SystemTool) => {
    if (!window.confirm(`Delete tool "${tool.name}"?`)) return;

    try {
      const response = await fetch(`/api/v1/mcp/tools/${tool.id}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        showSnackbar('Tool deleted successfully', 'success');
        fetchSystemTools();
        onRefresh?.();
      } else {
        throw new Error('Failed to delete tool');
      }
    } catch (error) {
      showSnackbar('Failed to delete tool', 'error');
    }
  };

  const handleSaveTool = async () => {
    try {
      // Validate JSON
      let parameters, headers;
      try {
        parameters = JSON.parse(formData.parameters);
        headers = JSON.parse(formData.headers);
      } catch (e) {
        showSnackbar('Invalid JSON in parameters or headers', 'error');
        return;
      }

      const toolData = {
        name: formData.name,
        description: formData.description || undefined,
        parameters,
        headers,
        method: formData.method,
        endpoint: formData.endpoint || undefined,
        is_active: formData.is_active
      };

      const response = await fetch(`/api/v1/mcp/tools/${editingTool?.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(toolData)
      });

      if (response.ok) {
        showSnackbar('Tool updated successfully', 'success');
        setEditDialog(false);
        fetchSystemTools();
        onRefresh?.();
      } else {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to save tool');
      }
    } catch (error: any) {
      showSnackbar(error.message || 'Failed to save tool', 'error');
    }
  };

  useEffect(() => {
    fetchSystemTools();
  }, []);

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" component="div" display="flex" alignItems="center">
            <BuildIcon sx={{ mr: 1 }} />
            System Tools
          </Typography>
          <Box>
            <Tooltip title="Refresh Redis Cache for MCP Tools">
              <span>
                <IconButton onClick={refreshSystemToolsCache} disabled={loading}>
                  <RefreshIcon />
                </IconButton>
              </span>
            </Tooltip>
            <Tooltip title="Reinstall System Tools">
              <span>
                <Button 
                  variant="outlined" 
                  size="small" 
                  onClick={reinstallSystemTools}
                  disabled={loading}
                  sx={{ ml: 1 }}
                >
                  Reinstall
                </Button>
              </span>
            </Tooltip>
          </Box>
        </Box>

        <Alert severity="info" sx={{ mb: 2 }}>
          System tools are built-in services that provide core functionality like RAG search, 
          document processing, and internal APIs. These tools can now be edited, deleted, or have their configuration modified.
        </Alert>

        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Tool Name</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Description</TableCell>
                <TableCell align="center">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {systemTools.length === 0 && !loading ? (
                <TableRow>
                  <TableCell colSpan={5} align="center">
                    <Typography color="textSecondary">
                      No system tools found. Click "Reinstall" to register system tools.
                    </Typography>
                  </TableCell>
                </TableRow>
              ) : (
                systemTools.map((tool) => (
                  <TableRow key={tool.id} hover>
                    <TableCell>
                      <Typography variant="body2" fontWeight="medium">
                        {tool.name}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={tool.tool_type} 
                        size="small"
                        color={tool.tool_type === 'internal' ? 'primary' : 'secondary'}
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={tool.is_active}
                            onChange={() => toggleToolStatus(tool)}
                            size="small"
                          />
                        }
                        label={tool.is_active ? 'Active' : 'Inactive'}
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="textSecondary">
                        {tool.description || 'No description available'}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Tooltip title="View Details">
                        <IconButton size="small" onClick={() => showToolDetails(tool)}>
                          <CodeIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Edit Tool">
                        <IconButton size="small" onClick={() => handleEditTool(tool)}>
                          <EditIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete Tool">
                        <IconButton 
                          size="small" 
                          onClick={() => handleDeleteTool(tool)}
                          color="error"
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>

        {/* Tool Details Dialog */}
        <Dialog 
          open={detailsDialog.open} 
          onClose={() => setDetailsDialog({ open: false, tool: null })}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>
            System Tool Details: {detailsDialog.tool?.name}
          </DialogTitle>
          <DialogContent>
            {detailsDialog.tool && (
              <Box>
                <Typography variant="subtitle2" gutterBottom>Description:</Typography>
                <Typography variant="body2" paragraph>
                  {detailsDialog.tool.description || 'No description available'}
                </Typography>

                <Typography variant="subtitle2" gutterBottom>Endpoint:</Typography>
                <Typography variant="body2" paragraph fontFamily="monospace">
                  {detailsDialog.tool.endpoint}
                </Typography>

                <Typography variant="subtitle2" gutterBottom>Method:</Typography>
                <Typography variant="body2" paragraph>
                  {detailsDialog.tool.method}
                </Typography>

                <Typography variant="subtitle2" gutterBottom>Parameters Schema:</Typography>
                <Box sx={{ 
                  p: 1, 
                  bgcolor: (theme) => theme.palette.mode === 'dark' ? theme.palette.grey[800] : theme.palette.grey[50],
                  borderRadius: 1, 
                  mt: 1,
                  border: (theme) => theme.palette.mode === 'dark' ? '1px solid' : 'none',
                  borderColor: (theme) => theme.palette.grey[700]
                }}>
                  <pre style={{ 
                    margin: 0, 
                    fontSize: '0.75rem', 
                    overflow: 'auto',
                    color: 'inherit',
                    fontFamily: 'monospace'
                  }}>
                    {JSON.stringify(detailsDialog.tool.parameters || {}, null, 2)}
                  </pre>
                </Box>
              </Box>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDetailsDialog({ open: false, tool: null })}>
              Close
            </Button>
          </DialogActions>
        </Dialog>

        {/* Edit Tool Dialog */}
        <Dialog open={editDialog} onClose={() => setEditDialog(false)} maxWidth="md" fullWidth>
          <DialogTitle>Edit System Tool</DialogTitle>
          <DialogContent>
            <Box sx={{ pt: 2 }}>
              <TextField
                fullWidth
                label="Tool Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                margin="normal"
                required
              />
              <TextField
                fullWidth
                label="Description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                margin="normal"
                multiline
                rows={2}
              />
              <TextField
                fullWidth
                label="Endpoint"
                value={formData.endpoint}
                onChange={(e) => setFormData({ ...formData, endpoint: e.target.value })}
                margin="normal"
              />
              <TextField
                fullWidth
                label="Method"
                value={formData.method}
                onChange={(e) => setFormData({ ...formData, method: e.target.value })}
                margin="normal"
                select
                SelectProps={{ native: true }}
              >
                <option value="POST">POST</option>
                <option value="GET">GET</option>
                <option value="PUT">PUT</option>
                <option value="DELETE">DELETE</option>
              </TextField>
              
              <Accordion sx={{ mt: 2 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Parameters Schema</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <TextField
                    fullWidth
                    multiline
                    rows={6}
                    value={formData.parameters}
                    onChange={(e) => setFormData({ ...formData, parameters: e.target.value })}
                    placeholder='{"type": "object", "properties": {...}}'
                    sx={{ fontFamily: 'monospace' }}
                  />
                </AccordionDetails>
              </Accordion>

              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Headers</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <TextField
                    fullWidth
                    multiline
                    rows={4}
                    value={formData.headers}
                    onChange={(e) => setFormData({ ...formData, headers: e.target.value })}
                    placeholder='{"Content-Type": "application/json"}'
                    sx={{ fontFamily: 'monospace' }}
                  />
                </AccordionDetails>
              </Accordion>

              <FormControlLabel
                control={
                  <Switch
                    checked={formData.is_active}
                    onChange={(e) => setFormData({ ...formData, is_active: e.target.checked })}
                  />
                }
                label="Active"
                sx={{ mt: 2 }}
              />
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setEditDialog(false)}>Cancel</Button>
            <Button onClick={handleSaveTool} variant="contained" disabled={!formData.name}>
              Update
            </Button>
          </DialogActions>
        </Dialog>

        {/* Snackbar for notifications */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={() => setSnackbar({ ...snackbar, open: false })}
        >
          <Alert 
            onClose={() => setSnackbar({ ...snackbar, open: false })} 
            severity={snackbar.severity}
          >
            {snackbar.message}
          </Alert>
        </Snackbar>
      </CardContent>
    </Card>
  );
};

export default SystemToolsManager;