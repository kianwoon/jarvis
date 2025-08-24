import React, { useState, useEffect, useCallback, useMemo, memo } from 'react';
import {
  Box,
  Button,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  IconButton,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  Alert,
  Skeleton,
  InputAdornment,
  Menu,
  MenuItem,
  CircularProgress,
  Tooltip,
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Search as SearchIcon,
  Description as DocumentIcon,
  ViewList as ListIcon,
  ViewModule as GridIcon,
  MoreVert as MoreVertIcon,
  Folder as FolderIcon,
  AccessTime as TimeIcon,
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Security as SecurityIcon
} from '@mui/icons-material';
import NavigationBar from '../shared/NavigationBar';
import { notebookAPI, Notebook, CreateNotebookRequest, UpdateNotebookRequest, getErrorMessage, formatRelativeTime } from './NotebookAPI';

interface NotebookManagerProps {
  onNotebookSelect?: (notebook: Notebook) => void;
}

const NotebookManager: React.FC<NotebookManagerProps> = ({ onNotebookSelect }) => {
  const [notebooks, setNotebooks] = useState<Notebook[]>([]);
  const [loading, setLoading] = useState(true);
  const [createLoading, setCreateLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [retryCount, setRetryCount] = useState(0);
  
  // Theme management
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('jarvis-dark-mode');
    if (saved) {
      const darkMode = JSON.parse(saved);
      document.body.setAttribute('data-theme', darkMode ? 'dark' : 'light');
      return darkMode;
    }
    document.body.setAttribute('data-theme', 'light');
    return false;
  });
  
  // Create/Edit Dialog
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingNotebook, setEditingNotebook] = useState<Notebook | null>(null);
  const [dialogForm, setDialogForm] = useState<CreateNotebookRequest>({ name: '', description: '' });
  const [dialogLoading, setDialogLoading] = useState(false);
  
  // Context Menu
  const [contextMenu, setContextMenu] = useState<{
    mouseX: number;
    mouseY: number;
    notebook: Notebook;
  } | null>(null);

  // Load notebooks on component mount
  useEffect(() => {
    loadNotebooks();
  }, []);

  // Listen for theme changes from localStorage
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'jarvis-dark-mode' && e.newValue) {
        const newDarkMode = JSON.parse(e.newValue);
        setIsDarkMode(newDarkMode);
        document.body.setAttribute('data-theme', newDarkMode ? 'dark' : 'light');
      }
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  const loadNotebooks = useCallback(async (showLoadingIndicator = true) => {
    try {
      if (showLoadingIndicator) {
        setLoading(true);
      }
      setError('');
      const data = await notebookAPI.getNotebooks();
      setNotebooks(Array.isArray(data) ? data : []);
      setRetryCount(0);
    } catch (err) {
      setError(getErrorMessage(err));
      console.error('Failed to load notebooks:', err);
    } finally {
      if (showLoadingIndicator) {
        setLoading(false);
      }
    }
  }, []);

  const handleCreateNotebook = useCallback(() => {
    setEditingNotebook(null);
    setDialogForm({ name: '', description: '' });
    setDialogOpen(true);
  }, []);

  const handleEditNotebook = useCallback((notebook: Notebook) => {
    setEditingNotebook(notebook);
    setDialogForm({ name: notebook.name, description: notebook.description || '' });
    setDialogOpen(true);
    setContextMenu(null);
  }, []);

  const handleSaveNotebook = useCallback(async () => {
    if (!dialogForm.name.trim()) {
      setError('Notebook name is required');
      return;
    }

    try {
      setDialogLoading(true);
      setError('');
      
      if (editingNotebook) {
        // Update existing notebook
        const updated = await notebookAPI.updateNotebook(editingNotebook.id, {
          name: dialogForm.name,
          description: dialogForm.description || undefined
        });
        setNotebooks(prev => (prev || []).map(nb => nb.id === updated.id ? updated : nb));
      } else {
        // Create new notebook
        setCreateLoading(true);
        const created = await notebookAPI.createNotebook(dialogForm);
        console.log('Created notebook response:', created);
        
        // Validate the response has required properties
        if (!created || typeof created !== 'object') {
          throw new Error('Invalid response from server: Expected notebook object');
        }
        if (!created.name) {
          throw new Error('Invalid response from server: Missing notebook name');
        }
        
        setNotebooks(prev => [created, ...(prev || [])]);
      }
      
      setDialogOpen(false);
      setDialogForm({ name: '', description: '' });
      setEditingNotebook(null);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setDialogLoading(false);
      setCreateLoading(false);
    }
  }, [dialogForm, editingNotebook]);

  const handleDeleteNotebook = useCallback(async (notebook: Notebook) => {
    if (!window.confirm(`Are you sure you want to delete "${notebook.name}"? This action cannot be undone.`)) {
      return;
    }

    try {
      setError('');
      await notebookAPI.deleteNotebook(notebook.id);
      setNotebooks(prev => (prev || []).filter(nb => nb.id !== notebook.id));
      setContextMenu(null);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }, []);

  const handleContextMenu = useCallback((event: React.MouseEvent, notebook: Notebook) => {
    event.preventDefault();
    setContextMenu(
      contextMenu === null
        ? {
            mouseX: event.clientX + 2,
            mouseY: event.clientY - 6,
            notebook
          }
        : null
    );
  }, [contextMenu]);

  const handleContextMenuClose = useCallback(() => {
    setContextMenu(null);
  }, []);

  const toggleDarkMode = useCallback(() => {
    const newDarkMode = !isDarkMode;
    setIsDarkMode(newDarkMode);
    localStorage.setItem('jarvis-dark-mode', JSON.stringify(newDarkMode));
    document.body.setAttribute('data-theme', newDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  const filteredNotebooks = useMemo(() => 
    (notebooks || []).filter(notebook =>
      notebook.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (notebook.description && notebook.description.toLowerCase().includes(searchTerm.toLowerCase()))
    ), [notebooks, searchTerm]
  );

  const NotebookCard = memo<{ notebook: Notebook }>(({ notebook }) => (
    <Card 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        cursor: 'pointer',
        '&:hover': { 
          boxShadow: 4,
          transform: 'translateY(-2px)',
          transition: 'all 0.2s ease-in-out'
        }
      }}
      onClick={() => onNotebookSelect?.(notebook)}
      onContextMenu={(e) => handleContextMenu(e, notebook)}
    >
      <CardContent sx={{ flex: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <FolderIcon color="primary" sx={{ mr: 1 }} />
          <Typography variant="h6" noWrap sx={{ flex: 1 }}>
            {notebook.name}
          </Typography>
          <IconButton 
            size="small" 
            onClick={(e) => {
              e.stopPropagation();
              handleContextMenu(e, notebook);
            }}
          >
            <MoreVertIcon />
          </IconButton>
        </Box>
        
        {notebook.description && (
          <Typography 
            variant="body2" 
            color="text.secondary" 
            sx={{ 
              mb: 2,
              display: '-webkit-box',
              WebkitLineClamp: 3,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
              textOverflow: 'ellipsis'
            }}
          >
            {notebook.description}
          </Typography>
        )}
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Chip
            icon={<DocumentIcon />}
            label={`${notebook.document_count} doc${notebook.document_count !== 1 ? 's' : ''}`}
            size="small"
            variant="outlined"
            color={notebook.document_count > 0 ? 'primary' : 'default'}
          />
        </Box>
      </CardContent>
      
      <CardActions sx={{ justifyContent: 'space-between', p: 2, pt: 0 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', color: 'text.secondary' }}>
          <TimeIcon fontSize="small" sx={{ mr: 0.5 }} />
          <Typography variant="caption">
            {formatRelativeTime(notebook.updated_at)}
          </Typography>
        </Box>
      </CardActions>
    </Card>
  ));

  const NotebookListItem = memo<{ notebook: Notebook }>(({ notebook }) => (
    <Paper 
      sx={{ 
        p: 2, 
        mb: 1,
        cursor: 'pointer',
        '&:hover': { backgroundColor: 'action.hover' }
      }}
      onClick={() => onNotebookSelect?.(notebook)}
      onContextMenu={(e) => handleContextMenu(e, notebook)}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', flex: 1, minWidth: 0 }}>
          <FolderIcon color="primary" sx={{ mr: 2 }} />
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography variant="h6" noWrap>
              {notebook.name}
            </Typography>
            {notebook.description && (
              <Typography variant="body2" color="text.secondary" noWrap>
                {notebook.description}
              </Typography>
            )}
          </Box>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip
            icon={<DocumentIcon />}
            label={`${notebook.document_count} doc${notebook.document_count !== 1 ? 's' : ''}`}
            size="small"
            variant="outlined"
            color={notebook.document_count > 0 ? 'primary' : 'default'}
          />
          
          <Typography variant="caption" color="text.secondary" sx={{ minWidth: 80 }}>
            {formatRelativeTime(notebook.updated_at)}
          </Typography>
          
          <IconButton 
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              handleContextMenu(e, notebook);
            }}
          >
            <MoreVertIcon />
          </IconButton>
        </Box>
      </Box>
    </Paper>
  ));

  // Create Material-UI theme with memoization
  const theme = useMemo(() => createTheme({
    palette: {
      mode: isDarkMode ? 'dark' : 'light',
      primary: {
        main: '#2196f3',
      },
      background: {
        default: isDarkMode ? '#121212' : '#f5f5f5',
        paper: isDarkMode ? '#1e1e1e' : '#ffffff',
      },
    },
  }), [isDarkMode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Jarvis AI Assistant
            </Typography>
            <IconButton onClick={toggleDarkMode} color="inherit">
              {isDarkMode ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Toolbar>
        </AppBar>
        
        <NavigationBar currentTab={7} />
      
      <Box sx={{ p: 3, flex: 1, overflow: 'auto' }}>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
          <Typography variant="h4">Notebooks</Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Box
              onClick={() => window.dispatchEvent(new CustomEvent('toggleAdminMode'))}
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                px: 2,
                py: 1,
                border: '2px solid',
                borderColor: '#666',
                borderRadius: 1,
                backgroundColor: '#333',
                cursor: 'pointer',
                '&:hover': {
                  backgroundColor: '#555',
                  borderColor: '#888',
                }
              }}
            >
              <SecurityIcon sx={{ color: '#fff' }} />
              <Typography sx={{ color: '#fff', fontWeight: 'bold' }}>
                Admin Mode
              </Typography>
            </Box>
            <Button
              variant="contained"
              startIcon={createLoading ? <CircularProgress size={16} /> : <AddIcon />}
              onClick={handleCreateNotebook}
              disabled={createLoading}
            >
              {createLoading ? 'Creating...' : 'New Notebook'}
            </Button>
          </Box>
        </Box>

        {/* Search and View Controls */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
          <TextField
            placeholder="Search notebooks..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
            sx={{ flex: 1, maxWidth: 400 }}
          />
          
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Tooltip title="Grid view">
              <IconButton 
                onClick={() => setViewMode('grid')}
                color={viewMode === 'grid' ? 'primary' : 'default'}
              >
                <GridIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="List view">
              <IconButton 
                onClick={() => setViewMode('list')}
                color={viewMode === 'list' ? 'primary' : 'default'}
              >
                <ListIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Error Display */}
        {error && (
          <Alert 
            severity="error" 
            sx={{ mb: 3 }} 
            onClose={() => setError('')}
            action={
              <Button
                color="inherit"
                size="small"
                onClick={() => {
                  setError('');
                  loadNotebooks(false);
                  setRetryCount(prev => prev + 1);
                }}
              >
                Retry
              </Button>
            }
          >
            {error}
          </Alert>
        )}

        {/* Loading State */}
        {loading && (
          <Box>
            {viewMode === 'grid' ? (
              <Grid container spacing={3}>
                {[...Array(6)].map((_, index) => (
                  <Grid item xs={12} sm={6} md={4} key={index}>
                    <Skeleton variant="rectangular" height={200} sx={{ borderRadius: 1 }} />
                  </Grid>
                ))}
              </Grid>
            ) : (
              [...Array(5)].map((_, index) => (
                <Skeleton key={index} variant="rectangular" height={80} sx={{ mb: 1, borderRadius: 1 }} />
              ))
            )}
          </Box>
        )}

        {/* Notebooks Display */}
        {!loading && (
          <>
            {filteredNotebooks.length === 0 ? (
              <Paper sx={{ p: 6, textAlign: 'center' }}>
                <FolderIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  {searchTerm ? 'No notebooks found' : 'No notebooks yet'}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  {searchTerm 
                    ? 'Try adjusting your search terms' 
                    : 'Create your first notebook to organize and chat with your documents'
                  }
                </Typography>
                {!searchTerm && (
                  <Button
                    variant="contained"
                    startIcon={createLoading ? <CircularProgress size={16} /> : <AddIcon />}
                    onClick={handleCreateNotebook}
                    disabled={createLoading}
                  >
                    {createLoading ? 'Creating...' : 'Create Notebook'}
                  </Button>
                )}
              </Paper>
            ) : (
              <>
                {viewMode === 'grid' ? (
                  <Grid container spacing={3}>
                    {filteredNotebooks.map((notebook) => (
                      <Grid item xs={12} sm={6} md={4} key={`grid-${notebook.id}-${notebook.updated_at}`}>
                        <NotebookCard notebook={notebook} />
                      </Grid>
                    ))}
                  </Grid>
                ) : (
                  <Box>
                    {filteredNotebooks.map((notebook) => (
                      <NotebookListItem key={`list-${notebook.id}-${notebook.updated_at}`} notebook={notebook} />
                    ))}
                  </Box>
                )}
              </>
            )}
          </>
        )}
      </Box>

      {/* Create/Edit Dialog */}
      <Dialog 
        open={dialogOpen} 
        onClose={() => !dialogLoading && setDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {editingNotebook ? 'Edit Notebook' : 'Create New Notebook'}
        </DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Name"
            fullWidth
            variant="outlined"
            value={dialogForm.name}
            onChange={(e) => setDialogForm(prev => ({ ...prev, name: e.target.value }))}
            disabled={dialogLoading}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Description (optional)"
            fullWidth
            multiline
            rows={3}
            variant="outlined"
            value={dialogForm.description}
            onChange={(e) => setDialogForm(prev => ({ ...prev, description: e.target.value }))}
            disabled={dialogLoading}
          />
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => setDialogOpen(false)}
            disabled={dialogLoading}
          >
            Cancel
          </Button>
          <Button 
            onClick={handleSaveNotebook} 
            variant="contained"
            disabled={dialogLoading || !dialogForm.name.trim()}
            startIcon={dialogLoading ? <CircularProgress size={16} /> : null}
          >
            {editingNotebook ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Context Menu */}
      <Menu
        open={contextMenu !== null}
        onClose={handleContextMenuClose}
        anchorReference="anchorPosition"
        anchorPosition={
          contextMenu !== null
            ? { top: contextMenu.mouseY, left: contextMenu.mouseX }
            : undefined
        }
      >
        <MenuItem 
          onClick={() => {
            if (contextMenu) {
              onNotebookSelect?.(contextMenu.notebook);
              handleContextMenuClose();
            }
          }}
        >
          <DocumentIcon sx={{ mr: 1 }} />
          Open Notebook
        </MenuItem>
        <MenuItem 
          onClick={() => {
            if (contextMenu) {
              handleEditNotebook(contextMenu.notebook);
            }
          }}
        >
          <EditIcon sx={{ mr: 1 }} />
          Edit
        </MenuItem>
        <MenuItem 
          onClick={() => {
            if (contextMenu) {
              handleDeleteNotebook(contextMenu.notebook);
            }
          }}
          sx={{ color: 'error.main' }}
        >
          <DeleteIcon sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>
      </Box>
    </ThemeProvider>
  );
};

export default NotebookManager;