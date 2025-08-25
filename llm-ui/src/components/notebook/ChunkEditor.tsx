import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Chip,
  IconButton,
  Checkbox,
  FormControlLabel,
  LinearProgress,
  Divider,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  CircularProgress
} from '@mui/material';
import {
  Edit as EditIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Refresh as RefreshIcon,
  History as HistoryIcon,
  Psychology as EmbeddingIcon,
  RestoreFromTrash as RestoreIcon,
  SelectAll as SelectAllIcon,
  Deselect as DeselectIcon,
  ExpandMore as ExpandMoreIcon,
  Compare as CompareIcon,
  Difference as DiffIcon,
  MoreVert as MoreVertIcon,
  AutoFixHigh as AutoFixIcon
} from '@mui/icons-material';
import { notebookAPI, Chunk, ChunkEditHistory, getErrorMessage } from './NotebookAPI';

interface ChunkEditorProps {
  collectionName: string;
  documentId: string;
  contentType: 'document' | 'memory';
  documentName: string;
  onChunkChange?: () => void;
  onClose?: () => void;
}

interface EditingChunk {
  chunkId: string;
  originalContent: string;
  editedContent: string;
  hasChanges: boolean;
}

const ChunkEditor: React.FC<ChunkEditorProps> = ({
  collectionName,
  documentId,
  contentType,
  documentName,
  onChunkChange,
  onClose
}) => {
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [totalCount, setTotalCount] = useState(0);
  
  // Editing state
  const [editingChunks, setEditingChunks] = useState<Map<string, EditingChunk>>(new Map());
  const [selectedChunks, setSelectedChunks] = useState<Set<string>>(new Set());
  const [bulkOperationLoading, setBulkOperationLoading] = useState(false);
  
  // History dialog
  const [historyDialogOpen, setHistoryDialogOpen] = useState(false);
  const [selectedChunk, setSelectedChunk] = useState<Chunk | null>(null);
  
  // Context menu
  const [contextMenu, setContextMenu] = useState<{
    mouseX: number;
    mouseY: number;
    chunk: Chunk;
  } | null>(null);

  useEffect(() => {
    loadChunks();
  }, [collectionName, documentId, page, rowsPerPage]);

  const loadChunks = async () => {
    try {
      setLoading(true);
      setError('');
      
      const response = await notebookAPI.getChunksForDocument(
        collectionName,
        documentId,
        page + 1, // API uses 1-based pagination
        rowsPerPage
      );
      
      setChunks(response.chunks);
      setTotalCount(response.total_count);
      
    } catch (err) {
      setError(getErrorMessage(err));
      console.error('Failed to load chunks:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleEditChunk = (chunk: Chunk) => {
    const editingChunk: EditingChunk = {
      chunkId: chunk.chunk_id,
      originalContent: chunk.content,
      editedContent: chunk.content,
      hasChanges: false
    };
    
    setEditingChunks(prev => new Map(prev.set(chunk.chunk_id, editingChunk)));
  };

  const handleContentChange = (chunkId: string, newContent: string) => {
    setEditingChunks(prev => {
      const updated = new Map(prev);
      const editing = updated.get(chunkId);
      if (editing) {
        updated.set(chunkId, {
          ...editing,
          editedContent: newContent,
          hasChanges: newContent !== editing.originalContent
        });
      }
      return updated;
    });
  };

  const handleSaveChunk = async (chunkId: string, reEmbed: boolean = true) => {
    const editingChunk = editingChunks.get(chunkId);
    if (!editingChunk || !editingChunk.hasChanges) return;

    try {
      const result = await notebookAPI.updateChunk(collectionName, chunkId, {
        content: editingChunk.editedContent,
        re_embed: reEmbed
      });

      if (result.success) {
        // Remove from editing state
        setEditingChunks(prev => {
          const updated = new Map(prev);
          updated.delete(chunkId);
          return updated;
        });
        
        // Refresh chunks
        await loadChunks();
        onChunkChange?.();
      } else {
        setError(result.message);
      }
      
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const handleCancelEdit = (chunkId: string) => {
    setEditingChunks(prev => {
      const updated = new Map(prev);
      updated.delete(chunkId);
      return updated;
    });
  };

  const handleBulkReEmbed = async () => {
    if (selectedChunks.size === 0) return;

    try {
      setBulkOperationLoading(true);
      setError('');
      
      const result = await notebookAPI.bulkReEmbedChunks(
        collectionName,
        Array.from(selectedChunks)
      );

      if (result.success) {
        setSelectedChunks(new Set());
        await loadChunks();
        onChunkChange?.();
      } else {
        setError(result.message);
      }
      
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setBulkOperationLoading(false);
    }
  };

  const handleSelectAll = () => {
    if (selectedChunks.size === chunks.length) {
      setSelectedChunks(new Set());
    } else {
      setSelectedChunks(new Set(chunks.map(chunk => chunk.chunk_id)));
    }
  };

  const handleChunkSelect = (chunkId: string) => {
    setSelectedChunks(prev => {
      const updated = new Set(prev);
      if (updated.has(chunkId)) {
        updated.delete(chunkId);
      } else {
        updated.add(chunkId);
      }
      return updated;
    });
  };

  const handleShowHistory = (chunk: Chunk) => {
    setSelectedChunk(chunk);
    setHistoryDialogOpen(true);
    setContextMenu(null);
  };

  const handleContextMenu = (event: React.MouseEvent, chunk: Chunk) => {
    event.preventDefault();
    setContextMenu({
      mouseX: event.clientX - 2,
      mouseY: event.clientY - 4,
      chunk
    });
  };

  const closeContextMenu = () => {
    setContextMenu(null);
  };

  const renderEditHistory = (history: ChunkEditHistory[]) => {
    if (history.length === 0) {
      return (
        <Typography variant="body2" color="text.secondary">
          No edit history available
        </Typography>
      );
    }

    return (
      <Box>
        {history.map((edit, index) => (
          <Accordion key={edit.id}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                <Typography variant="subtitle2">
                  Edit {history.length - index}
                </Typography>
                <Chip
                  size="small"
                  label={edit.re_embedded ? 'Re-embedded' : 'Content only'}
                  color={edit.re_embedded ? 'primary' : 'default'}
                  sx={{ ml: 2 }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
                  {new Date(edit.edited_at).toLocaleString()}
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Original Content:
                </Typography>
                <Paper sx={{ p: 2, bgcolor: 'error.50', maxHeight: 200, overflow: 'auto' }}>
                  <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                    {edit.original_content}
                  </Typography>
                </Paper>
              </Box>
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Edited Content:
                </Typography>
                <Paper sx={{ p: 2, bgcolor: 'success.50', maxHeight: 200, overflow: 'auto' }}>
                  <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                    {edit.edited_content}
                  </Typography>
                </Paper>
              </Box>
              {edit.edited_by && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  Edited by: {edit.edited_by}
                </Typography>
              )}
            </AccordionDetails>
          </Accordion>
        ))}
      </Box>
    );
  };

  if (loading && chunks.length === 0) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading chunks...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'between', mb: 2 }}>
          <Box>
            <Typography variant="h6">
              Chunk Editor: {documentName}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {contentType === 'memory' ? 'Memory' : 'Document'} • {totalCount} chunks
            </Typography>
          </Box>
          {onClose && (
            <Button onClick={onClose} sx={{ ml: 'auto' }}>
              Close
            </Button>
          )}
        </Box>

        {/* Bulk Operations */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
          <FormControlLabel
            control={
              <Checkbox
                checked={selectedChunks.size === chunks.length && chunks.length > 0}
                indeterminate={selectedChunks.size > 0 && selectedChunks.size < chunks.length}
                onChange={handleSelectAll}
              />
            }
            label={`Select all (${selectedChunks.size} selected)`}
          />
          
          <Button
            startIcon={<EmbeddingIcon />}
            onClick={handleBulkReEmbed}
            disabled={selectedChunks.size === 0 || bulkOperationLoading}
            variant="outlined"
          >
            {bulkOperationLoading ? 'Re-embedding...' : `Re-embed Selected (${selectedChunks.size})`}
          </Button>
          
          <Button
            startIcon={<RefreshIcon />}
            onClick={loadChunks}
            disabled={loading}
          >
            Refresh
          </Button>
          
          {editingChunks.size > 0 && (
            <Chip 
              label={`${editingChunks.size} chunks being edited`} 
              color="warning"
            />
          )}
        </Box>
      </Paper>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* Progress Bar */}
      {bulkOperationLoading && (
        <LinearProgress sx={{ mb: 2 }} />
      )}

      {/* Chunks Table */}
      <Paper sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <TableContainer sx={{ flex: 1 }}>
          <Table stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox">
                  <Checkbox
                    checked={selectedChunks.size === chunks.length && chunks.length > 0}
                    indeterminate={selectedChunks.size > 0 && selectedChunks.size < chunks.length}
                    onChange={handleSelectAll}
                  />
                </TableCell>
                <TableCell>Content</TableCell>
                <TableCell width="150">Metadata</TableCell>
                <TableCell width="120">History</TableCell>
                <TableCell width="120">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {chunks.map((chunk) => {
                const editing = editingChunks.get(chunk.chunk_id);
                const hasEditHistory = chunk.edit_history && chunk.edit_history.length > 0;
                
                return (
                  <TableRow
                    key={chunk.chunk_id}
                    hover
                    onContextMenu={(e) => handleContextMenu(e, chunk)}
                    sx={{ 
                      '& td': { verticalAlign: 'top' },
                      ...(editing?.hasChanges && { 
                        bgcolor: 'warning.50',
                        '&:hover': { bgcolor: 'warning.100' }
                      })
                    }}
                  >
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={selectedChunks.has(chunk.chunk_id)}
                        onChange={() => handleChunkSelect(chunk.chunk_id)}
                      />
                    </TableCell>
                    
                    <TableCell>
                      {editing ? (
                        <TextField
                          multiline
                          fullWidth
                          rows={4}
                          value={editing.editedContent}
                          onChange={(e) => handleContentChange(chunk.chunk_id, e.target.value)}
                          variant="outlined"
                          size="small"
                        />
                      ) : (
                        <Typography 
                          variant="body2" 
                          component="pre"
                          sx={{ 
                            whiteSpace: 'pre-wrap',
                            maxHeight: 200,
                            overflow: 'auto',
                            fontFamily: 'monospace'
                          }}
                        >
                          {chunk.content}
                        </Typography>
                      )}
                    </TableCell>
                    
                    <TableCell>
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                        {chunk.metadata.source && (
                          <Chip size="small" label={`Source: ${chunk.metadata.source}`} />
                        )}
                        {chunk.metadata.page > 0 && (
                          <Chip size="small" label={`Page: ${chunk.metadata.page}`} />
                        )}
                        {chunk.last_edited && (
                          <Chip 
                            size="small" 
                            color="warning"
                            label={`Edited: ${new Date(chunk.last_edited).toLocaleDateString()}`}
                          />
                        )}
                      </Box>
                    </TableCell>
                    
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {hasEditHistory && (
                          <Chip
                            size="small"
                            color="info"
                            label={`${chunk.edit_history.length} edits`}
                            onClick={() => handleShowHistory(chunk)}
                            clickable
                          />
                        )}
                        {chunk.last_edited && (
                          <Tooltip title="This chunk has been edited">
                            <HistoryIcon color="warning" fontSize="small" />
                          </Tooltip>
                        )}
                      </Box>
                    </TableCell>
                    
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        {editing ? (
                          <>
                            <Tooltip title="Save changes">
                              <IconButton
                                size="small"
                                color="primary"
                                onClick={() => handleSaveChunk(chunk.chunk_id)}
                                disabled={!editing.hasChanges}
                              >
                                <SaveIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Cancel editing">
                              <IconButton
                                size="small"
                                onClick={() => handleCancelEdit(chunk.chunk_id)}
                              >
                                <CancelIcon />
                              </IconButton>
                            </Tooltip>
                          </>
                        ) : (
                          <>
                            <Tooltip title="Edit chunk">
                              <IconButton
                                size="small"
                                onClick={() => handleEditChunk(chunk)}
                              >
                                <EditIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="More options">
                              <IconButton
                                size="small"
                                onClick={(e) => handleContextMenu(e, chunk)}
                              >
                                <MoreVertIcon />
                              </IconButton>
                            </Tooltip>
                          </>
                        )}
                      </Box>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
        
        <TablePagination
          rowsPerPageOptions={[5, 10, 25, 50]}
          component="div"
          count={totalCount}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={(_, newPage) => setPage(newPage)}
          onRowsPerPageChange={(e) => {
            setRowsPerPage(parseInt(e.target.value, 10));
            setPage(0);
          }}
        />
      </Paper>

      {/* History Dialog */}
      <Dialog 
        open={historyDialogOpen} 
        onClose={() => setHistoryDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Edit History for Chunk
          {selectedChunk && (
            <Typography variant="body2" color="text.secondary">
              ID: {selectedChunk.chunk_id}
            </Typography>
          )}
        </DialogTitle>
        <DialogContent>
          {selectedChunk && renderEditHistory(selectedChunk.edit_history)}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHistoryDialogOpen(false)}>
            Close
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
        <MenuItem onClick={() => contextMenu && handleEditChunk(contextMenu.chunk)}>
          <ListItemIcon>
            <EditIcon />
          </ListItemIcon>
          <ListItemText>Edit Chunk</ListItemText>
        </MenuItem>
        <MenuItem 
          onClick={() => contextMenu && handleShowHistory(contextMenu.chunk)}
          disabled={!contextMenu?.chunk.edit_history?.length}
        >
          <ListItemIcon>
            <HistoryIcon />
          </ListItemIcon>
          <ListItemText>View Edit History</ListItemText>
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => contextMenu && handleSaveChunk(contextMenu.chunk.chunk_id, true)}>
          <ListItemIcon>
            <EmbeddingIcon />
          </ListItemIcon>
          <ListItemText>Re-embed Chunk</ListItemText>
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default ChunkEditor;