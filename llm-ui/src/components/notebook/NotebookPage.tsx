import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Switch,
  FormControlLabel,
  Paper,
  Typography,
  Divider,
  useTheme,
  ThemeProvider,
  createTheme,
  CssBaseline
} from '@mui/material';
import {
  Security as SecurityIcon,
  Warning as WarningIcon
} from '@mui/icons-material';
import NotebookManager from './NotebookManager';
import NotebookViewer from './NotebookViewer';
import DocumentAdminPage from '../admin/DocumentAdminPage';
import ErrorBoundary from '../shared/ErrorBoundary';
import { Notebook } from './NotebookAPI';

/**
 * Main notebook page component that manages the state between 
 * NotebookManager (list view) and NotebookViewer (individual notebook view)
 */
const NotebookPage: React.FC = () => {
  const [selectedNotebook, setSelectedNotebook] = useState<Notebook | null>(null);
  const [editingNotebook, setEditingNotebook] = useState<Notebook | null>(null);
  const [adminMode, setAdminMode] = useState(false);
  const [adminConfirmOpen, setAdminConfirmOpen] = useState(false);
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

  // Listen for theme changes from localStorage
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'jarvis-dark-mode' && e.newValue) {
        const newDarkMode = JSON.parse(e.newValue);
        setIsDarkMode(newDarkMode);
        document.body.setAttribute('data-theme', newDarkMode ? 'dark' : 'light');
        document.documentElement.setAttribute('data-theme', newDarkMode ? 'dark' : 'light');
      }
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  // Create Material-UI theme with memoization
  const theme = useMemo(() => createTheme({
    palette: {
      mode: isDarkMode ? 'dark' : 'light',
      primary: {
        main: '#2196f3',
      },
      secondary: {
        main: '#f50057',
      },
      background: {
        default: isDarkMode ? '#121212' : '#f5f5f5',
        paper: isDarkMode ? '#1e1e1e' : '#ffffff',
      },
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            backgroundColor: isDarkMode ? '#121212' : '#f5f5f5',
          },
        },
      },
    },
  }), [isDarkMode]);

  const handleNotebookSelect = (notebook: Notebook) => {
    console.log('Selected notebook:', notebook);
    setSelectedNotebook(notebook);
  };

  const handleBack = () => {
    setSelectedNotebook(null);
    setEditingNotebook(null);
  };

  const handleEdit = (notebook: Notebook) => {
    setEditingNotebook(notebook);
    // You could implement an edit dialog here or navigate to an edit view
    console.log('Edit notebook:', notebook);
  };

  const handleAdminModeToggle = () => {
    if (!adminMode) {
      // Entering admin mode - show confirmation
      setAdminConfirmOpen(true);
    } else {
      // Exiting admin mode - direct toggle
      setAdminMode(false);
      // Reset any selected states when exiting admin mode
      setSelectedNotebook(null);
      setEditingNotebook(null);
    }
  };

  const handleAdminConfirm = () => {
    setAdminMode(true);
    setAdminConfirmOpen(false);
    // Reset notebook states when entering admin mode
    setSelectedNotebook(null);
    setEditingNotebook(null);
  };

  const handleAdminCancel = () => {
    setAdminConfirmOpen(false);
  };

  // Listen for admin mode toggle from NotebookManager
  useEffect(() => {
    const handleToggleAdminMode = () => {
      handleAdminModeToggle();
    };
    
    window.addEventListener('toggleAdminMode', handleToggleAdminMode);
    return () => window.removeEventListener('toggleAdminMode', handleToggleAdminMode);
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ErrorBoundary>
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {adminMode && (
          <Alert severity="warning" variant="filled" sx={{ m: 2, mb: 0 }}>
            <Typography variant="body2">
              <strong>Admin Mode Active:</strong> You can now view and permanently delete documents across all notebooks and users. 
              Use with caution - deletions cannot be undone.
            </Typography>
          </Alert>
        )}

        {/* Content Area */}
        <Box sx={{ flex: 1, overflow: 'hidden' }}>
          {adminMode ? (
            /* Admin Mode: Show Document Administration */
            <DocumentAdminPage />
          ) : (
            /* Normal Mode: Show Notebook Interface */
            selectedNotebook ? (
              <NotebookViewer
                notebookId={selectedNotebook.id}
                onBack={handleBack}
                onEdit={handleEdit}
              />
            ) : (
              <NotebookManager onNotebookSelect={handleNotebookSelect} />
            )
          )}
        </Box>
      </Box>

      {/* Admin Mode Confirmation Dialog */}
      <Dialog
        open={adminConfirmOpen}
        onClose={handleAdminCancel}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle 
          sx={{ 
            bgcolor: 'warning.light', 
            color: 'warning.contrastText',
            display: 'flex',
            alignItems: 'center',
            gap: 1
          }}
        >
          <SecurityIcon />
          Enable Admin Mode?
        </DialogTitle>
        <DialogContent sx={{ pt: 3 }}>
          <Alert severity="warning" sx={{ mb: 2 }}>
            <Typography variant="body1" fontWeight="bold" gutterBottom>
              System Administrator Access
            </Typography>
            <Typography variant="body2">
              Admin mode provides system-wide document management capabilities that affect all notebooks and users:
            </Typography>
          </Alert>
          
          <Box component="ul" sx={{ pl: 2, mt: 2 }}>
            <Typography component="li" variant="body2" sx={{ mb: 1 }}>
              <strong>View all system documents</strong> across all notebooks
            </Typography>
            <Typography component="li" variant="body2" sx={{ mb: 1 }}>
              <strong>Permanently delete documents</strong> from all systems (database, vectors, cache)
            </Typography>
            <Typography component="li" variant="body2" sx={{ mb: 1 }}>
              <strong>Manage orphaned documents</strong> not associated with any notebooks
            </Typography>
            <Typography component="li" variant="body2" sx={{ mb: 1 }}>
              <strong>Bulk operations</strong> that can impact system performance
            </Typography>
          </Box>
          
          <Alert severity="error" sx={{ mt: 2 }}>
            <Typography variant="body2">
              <strong>Warning:</strong> Actions performed in admin mode cannot be undone and may affect other users.
            </Typography>
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleAdminCancel}>
            Cancel
          </Button>
          <Button
            onClick={handleAdminConfirm}
            variant="contained"
            color="error"
            startIcon={<SecurityIcon />}
          >
            Enable Admin Mode
          </Button>
        </DialogActions>
      </Dialog>
      </ErrorBoundary>
    </ThemeProvider>
  );
};

export default NotebookPage;