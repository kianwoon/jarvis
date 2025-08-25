import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  Grid,
  Fab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Chip,
  IconButton,
  TextField,
  InputAdornment,
  Skeleton,
  Tooltip,
  Menu,
  MenuItem,
  ToggleButton,
  ToggleButtonGroup,
  Divider
} from '@mui/material';
import {
  Add as AddIcon,
  Memory as MemoryIcon,
  Search as SearchIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  MoreVert as MoreVertIcon,
  ViewList as ListViewIcon,
  ViewModule as GridViewIcon,
  AccessTime as TimeIcon,
  TextSnippet as ContentIcon,
  Psychology as BrainIcon,
  Clear as ClearIcon
} from '@mui/icons-material';
import { 
  notebookAPI, 
  NotebookWithDocuments,
  Memory,
  getErrorMessage, 
  formatRelativeTime
} from './NotebookAPI';

interface NotebookMemoryProps {
  notebook: NotebookWithDocuments;
  onMemoryChange: () => void;
}

interface ContextMenuState {
  mouseX: number;
  mouseY: number;
  memory: Memory;
}

const NotebookMemory: React.FC<NotebookMemoryProps> = ({ 
  notebook, 
  onMemoryChange 
}) => {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  
  // Create memory dialog
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [memoryName, setMemoryName] = useState('');
  const [memoryDescription, setMemoryDescription] = useState('');
  const [memoryContent, setMemoryContent] = useState('');
  const [creating, setCreating] = useState(false);
  
  // Edit memory dialog
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingMemory, setEditingMemory] = useState<Memory | null>(null);
  const [editMemoryName, setEditMemoryName] = useState('');
  const [editMemoryDescription, setEditMemoryDescription] = useState('');
  const [editMemoryContent, setEditMemoryContent] = useState('');
  const [updating, setUpdating] = useState(false);
  
  // Delete confirmation dialog
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [memoryToDelete, setMemoryToDelete] = useState<Memory | null>(null);
  const [deleting, setDeleting] = useState(false);
  
  // Context menu
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  
  // Pagination
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const pageSize = 12;

  useEffect(() => {
    loadMemories();
  }, [notebook.id, page]);

  const loadMemories = async () => {
    try {
      setLoading(true);
      setError('');
      
      const response = await notebookAPI.getMemories(notebook.id, page, pageSize);
      
      if (page === 1) {
        setMemories(response.memories);
      } else {
        setMemories(prev => [...prev, ...response.memories]);
      }
      
      setHasMore(response.memories.length === pageSize);
      
    } catch (err) {
      setError(getErrorMessage(err));
      console.error('Failed to load memories:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateMemory = async () => {
    if (!memoryName.trim() || !memoryContent.trim()) {
      setError('Memory name and content are required');
      return;
    }

    try {
      setCreating(true);
      setError('');
      
      await notebookAPI.createMemory(notebook.id, {
        name: memoryName.trim(),
        description: memoryDescription.trim() || undefined,
        content: memoryContent.trim()
      });
      
      // Reset form
      setMemoryName('');
      setMemoryDescription('');
      setMemoryContent('');
      setCreateDialogOpen(false);
      
      // Refresh memories
      setPage(1);
      await loadMemories();
      onMemoryChange();
      
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setCreating(false);
    }
  };

  const handleEditMemory = async () => {
    if (!editingMemory || !editMemoryName.trim() || !editMemoryContent.trim()) {
      setError('Memory name and content are required');
      return;
    }

    try {
      setUpdating(true);
      setError('');
      
      await notebookAPI.updateMemory(notebook.id, editingMemory.memory_id, {
        name: editMemoryName.trim(),
        description: editMemoryDescription.trim() || undefined,
        content: editMemoryContent.trim()
      });
      
      setEditDialogOpen(false);
      setEditingMemory(null);
      
      // Refresh memories
      await loadMemories();
      onMemoryChange();
      
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setUpdating(false);
    }
  };

  const handleDeleteMemory = async () => {
    if (!memoryToDelete) return;

    try {
      setDeleting(true);
      setError('');
      
      await notebookAPI.deleteMemory(notebook.id, memoryToDelete.memory_id);
      
      setDeleteDialogOpen(false);
      setMemoryToDelete(null);
      
      // Refresh memories
      setPage(1);
      await loadMemories();
      onMemoryChange();
      
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setDeleting(false);
    }
  };

  const openEditDialog = (memory: Memory) => {
    setEditingMemory(memory);
    setEditMemoryName(memory.name);
    setEditMemoryDescription(memory.description || '');
    setEditMemoryContent(memory.content);
    setEditDialogOpen(true);
    setContextMenu(null);
  };

  const openDeleteDialog = (memory: Memory) => {
    setMemoryToDelete(memory);
    setDeleteDialogOpen(true);
    setContextMenu(null);
  };

  const handleContextMenu = (event: React.MouseEvent, memory: Memory) => {
    event.preventDefault();
    setContextMenu({
      mouseX: event.clientX - 2,
      mouseY: event.clientY - 4,
      memory
    });
  };

  const closeContextMenu = () => {
    setContextMenu(null);
  };

  const filteredMemories = memories.filter(memory =>
    memory.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    memory.description?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    memory.content.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const renderMemoryCard = (memory: Memory) => (
    <Card 
      key={memory.id}
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
      onContextMenu={(e) => handleContextMenu(e, memory)}
    >
      <CardContent sx={{ flexGrow: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <BrainIcon sx={{ color: 'primary.main', mr: 1, fontSize: 20 }} />
          <Typography variant="h6" component="div" sx={{ 
            flexGrow: 1,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap'
          }}>
            {memory.name}
          </Typography>
          <IconButton 
            size="small"
            onClick={(e) => handleContextMenu(e, memory)}
          >
            <MoreVertIcon fontSize="small" />
          </IconButton>
        </Box>
        
        {memory.description && (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {memory.description}
          </Typography>
        )}
        
        <Typography 
          variant="body2" 
          sx={{ 
            mb: 2,
            display: '-webkit-box',
            WebkitLineClamp: 3,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
            textOverflow: 'ellipsis'
          }}
        >
          {memory.content}
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
          <Chip
            icon={<ContentIcon />}
            label={`${memory.chunk_count} chunks`}
            size="small"
            color="primary"
            variant="outlined"
          />
          <Chip
            icon={<TimeIcon />}
            label={formatRelativeTime(memory.created_at)}
            size="small"
            variant="outlined"
          />
        </Box>
      </CardContent>
      
      <CardActions>
        <Button 
          size="small" 
          startIcon={<EditIcon />}
          onClick={() => openEditDialog(memory)}
        >
          Edit
        </Button>
        <Button 
          size="small" 
          color="error"
          startIcon={<DeleteIcon />}
          onClick={() => openDeleteDialog(memory)}
        >
          Delete
        </Button>
      </CardActions>
    </Card>
  );

  const renderMemoryList = (memory: Memory) => (
    <Paper 
      key={memory.id}
      sx={{ 
        p: 2, 
        mb: 1,
        cursor: 'pointer',
        '&:hover': { bgcolor: 'action.hover' }
      }}
      onContextMenu={(e) => handleContextMenu(e, memory)}
    >
      <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
        <BrainIcon sx={{ color: 'primary.main', mr: 2, mt: 0.5 }} />
        <Box sx={{ flexGrow: 1, minWidth: 0 }}>
          <Typography variant="h6" sx={{ mb: 0.5 }}>
            {memory.name}
          </Typography>
          {memory.description && (
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              {memory.description}
            </Typography>
          )}
          <Typography 
            variant="body2" 
            sx={{ 
              mb: 1,
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden'
            }}
          >
            {memory.content}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              icon={<ContentIcon />}
              label={`${memory.chunk_count} chunks`}
              size="small"
              color="primary"
              variant="outlined"
            />
            <Chip
              icon={<TimeIcon />}
              label={formatRelativeTime(memory.created_at)}
              size="small"
              variant="outlined"
            />
          </Box>
        </Box>
        <Box sx={{ display: 'flex', gap: 1, ml: 2 }}>
          <IconButton 
            size="small"
            onClick={() => openEditDialog(memory)}
          >
            <EditIcon />
          </IconButton>
          <IconButton 
            size="small"
            color="error"
            onClick={() => openDeleteDialog(memory)}
          >
            <DeleteIcon />
          </IconButton>
        </Box>
      </Box>
    </Paper>
  );

  if (loading && memories.length === 0) {
    return (
      <Box sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h5">Memories</Typography>
          <Skeleton variant="rectangular" width={120} height={36} />
        </Box>
        <Grid container spacing={2}>
          {[...Array(6)].map((_, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Skeleton variant="rectangular" height={200} />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  return (
    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <MemoryIcon sx={{ fontSize: 32, color: 'primary.main', mr: 2 }} />
          <Typography variant="h5">
            Memories ({filteredMemories.length})
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(_, newMode) => newMode && setViewMode(newMode)}
            size="small"
          >
            <ToggleButton value="grid">
              <GridViewIcon />
            </ToggleButton>
            <ToggleButton value="list">
              <ListViewIcon />
            </ToggleButton>
          </ToggleButtonGroup>
          
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setCreateDialogOpen(true)}
          >
            Add Memory
          </Button>
        </Box>
      </Box>

      {/* Search */}
      <TextField
        placeholder="Search memories..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        sx={{ mb: 3 }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon />
            </InputAdornment>
          ),
          endAdornment: searchTerm && (
            <InputAdornment position="end">
              <IconButton onClick={() => setSearchTerm('')}>
                <ClearIcon />
              </IconButton>
            </InputAdornment>
          )
        }}
      />

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* Content */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {filteredMemories.length === 0 && !loading ? (
          <Paper sx={{ p: 6, textAlign: 'center' }}>
            <BrainIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              {memories.length === 0 ? 'No Memories Yet' : 'No Matching Memories'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              {memories.length === 0 
                ? 'Create your first memory to store important information and insights.'
                : 'Try adjusting your search terms to find the memories you\'re looking for.'
              }
            </Typography>
            {memories.length === 0 && (
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={() => setCreateDialogOpen(true)}
              >
                Create First Memory
              </Button>
            )}
          </Paper>
        ) : (
          <>
            {viewMode === 'grid' ? (
              <Grid container spacing={2}>
                {filteredMemories.map(memory => (
                  <Grid item xs={12} sm={6} md={4} key={memory.id}>
                    {renderMemoryCard(memory)}
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Box>
                {filteredMemories.map(renderMemoryList)}
              </Box>
            )}
            
            {hasMore && (
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                <Button 
                  onClick={() => setPage(p => p + 1)}
                  disabled={loading}
                >
                  Load More
                </Button>
              </Box>
            )}
          </>
        )}
      </Box>

      {/* Create Memory Dialog */}
      <Dialog 
        open={createDialogOpen} 
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Create New Memory</DialogTitle>
        <DialogContent>
          <TextField
            label="Memory Name"
            value={memoryName}
            onChange={(e) => setMemoryName(e.target.value)}
            fullWidth
            margin="normal"
            required
          />
          <TextField
            label="Description (optional)"
            value={memoryDescription}
            onChange={(e) => setMemoryDescription(e.target.value)}
            fullWidth
            margin="normal"
          />
          <TextField
            label="Content"
            value={memoryContent}
            onChange={(e) => setMemoryContent(e.target.value)}
            fullWidth
            margin="normal"
            multiline
            rows={6}
            required
            helperText={`${memoryContent.length} characters`}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleCreateMemory}
            variant="contained"
            disabled={creating || !memoryName.trim() || !memoryContent.trim()}
          >
            {creating ? 'Creating...' : 'Create Memory'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Edit Memory Dialog */}
      <Dialog 
        open={editDialogOpen} 
        onClose={() => setEditDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Edit Memory</DialogTitle>
        <DialogContent>
          <TextField
            label="Memory Name"
            value={editMemoryName}
            onChange={(e) => setEditMemoryName(e.target.value)}
            fullWidth
            margin="normal"
            required
          />
          <TextField
            label="Description (optional)"
            value={editMemoryDescription}
            onChange={(e) => setEditMemoryDescription(e.target.value)}
            fullWidth
            margin="normal"
          />
          <TextField
            label="Content"
            value={editMemoryContent}
            onChange={(e) => setEditMemoryContent(e.target.value)}
            fullWidth
            margin="normal"
            multiline
            rows={6}
            required
            helperText={`${editMemoryContent.length} characters`}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleEditMemory}
            variant="contained"
            disabled={updating || !editMemoryName.trim() || !editMemoryContent.trim()}
          >
            {updating ? 'Updating...' : 'Update Memory'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Delete Memory</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the memory "{memoryToDelete?.name}"?
            This action cannot be undone and will remove all associated chunks from the vector database.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleDeleteMemory}
            color="error"
            variant="contained"
            disabled={deleting}
          >
            {deleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Context Menu */}
      <Menu
        open={contextMenu !== null}
        onClose={closeContextMenu}
        anchorReference="anchorPosition"
        anchorPosition={
          contextMenu !== null
            ? { top: contextMenu.mouseY, left: contextMenu.mouseX }
            : undefined
        }
      >
        <MenuItem onClick={() => contextMenu && openEditDialog(contextMenu.memory)}>
          <EditIcon sx={{ mr: 1 }} />
          Edit Memory
        </MenuItem>
        <Divider />
        <MenuItem 
          onClick={() => contextMenu && openDeleteDialog(contextMenu.memory)}
          sx={{ color: 'error.main' }}
        >
          <DeleteIcon sx={{ mr: 1 }} />
          Delete Memory
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default NotebookMemory;