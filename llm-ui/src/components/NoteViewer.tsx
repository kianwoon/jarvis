import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Paper
} from '@mui/material';

interface NoteViewerProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  content?: string;
  notes?: Array<{
    id: string;
    title: string;
    content: string;
    timestamp?: string;
  }>;
}

const NoteViewer: React.FC<NoteViewerProps> = ({
  open,
  onClose,
  title = "Notes",
  content = "",
  notes = []
}) => {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>{title}</DialogTitle>
      <DialogContent>
        {content && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body1" style={{ whiteSpace: 'pre-wrap' }}>
              {content}
            </Typography>
          </Box>
        )}
        
        {notes.length > 0 && (
          <Box>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Notes ({notes.length})
            </Typography>
            {notes.map((note) => (
              <Paper key={note.id} sx={{ p: 2, mb: 2 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                  {note.title}
                </Typography>
                <Typography variant="body2" style={{ whiteSpace: 'pre-wrap' }}>
                  {note.content}
                </Typography>
                {note.timestamp && (
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    {new Date(note.timestamp).toLocaleString()}
                  </Typography>
                )}
              </Paper>
            ))}
          </Box>
        )}
        
        {!content && notes.length === 0 && (
          <Typography variant="body2" color="text.secondary">
            No notes available.
          </Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default NoteViewer;