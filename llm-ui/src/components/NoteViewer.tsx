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
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

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
  // Helper function to clean output text
  const cleanOutputText = (text: string) => {
    // Remove <think> tags and their content
    const withoutThinkTags = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
    
    // Clean up extra whitespace
    return withoutThinkTags.replace(/\n\s*\n\s*\n/g, '\n\n').trim();
  };
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
          <Box sx={{ 
            mb: 2,
            '& pre': { 
              bgcolor: 'action.hover', 
              p: 1, 
              borderRadius: 1,
              overflow: 'auto'
            },
            '& code': {
              bgcolor: 'action.hover',
              px: 0.5,
              borderRadius: 0.5
            }
          }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {cleanOutputText(content)}
            </ReactMarkdown>
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