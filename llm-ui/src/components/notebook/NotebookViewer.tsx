import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Tabs,
  Tab,
  Paper,
  Button,
  Chip,
  Alert,
  CircularProgress,
  Breadcrumbs,
  Link,
  Divider,
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar
} from '@mui/material';
import {
  ArrowBack as BackIcon,
  Edit as EditIcon,
  Description as DocumentIcon,
  Chat as ChatIcon,
  Memory as MemoryIcon,
  Folder as FolderIcon,
  AccessTime as TimeIcon,
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon
} from '@mui/icons-material';
import NavigationBar from '../shared/NavigationBar';
import NotebookDocumentList from './NotebookDocumentList';
import NotebookChat from './NotebookChat';
import NotebookMemory from './NotebookMemory';
import { 
  notebookAPI, 
  Notebook, 
  NotebookWithDocuments, 
  getErrorMessage, 
  formatDate,
  formatRelativeTime
} from './NotebookAPI';

interface NotebookViewerProps {
  notebookId: string;
  onBack: () => void;
  onEdit?: (notebook: Notebook) => void;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index }: TabPanelProps) {
  return (
    <Box
      role="tabpanel"
      id={`notebook-tabpanel-${index}`}
      aria-labelledby={`notebook-tab-${index}`}
      sx={{
        flex: 1,
        display: value === index ? 'flex' : 'none',
        flexDirection: 'column',
        height: '100%',
        minHeight: 0
      }}
    >
      {children}
    </Box>
  );
}

const NotebookViewer: React.FC<NotebookViewerProps> = ({ 
  notebookId, 
  onBack, 
  onEdit 
}) => {
  const [notebook, setNotebook] = useState<NotebookWithDocuments | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [currentTab, setCurrentTab] = useState(0);
  
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

  // Load notebook details on component mount or when notebookId changes
  useEffect(() => {
    loadNotebook();
  }, [notebookId]);

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

  const loadNotebook = async () => {
    try {
      setLoading(true);
      setError('');
      console.log('Loading notebook with ID:', notebookId);
      const data = await notebookAPI.getNotebook(notebookId);
      console.log('Loaded notebook data:', data);
      setNotebook(data);
    } catch (err) {
      console.error('Error loading notebook:', err);
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleDocumentChange = () => {
    // Refresh notebook data when documents are added/removed
    loadNotebook();
  };

  const toggleDarkMode = () => {
    const newDarkMode = !isDarkMode;
    setIsDarkMode(newDarkMode);
    localStorage.setItem('jarvis-dark-mode', JSON.stringify(newDarkMode));
    document.body.setAttribute('data-theme', newDarkMode ? 'dark' : 'light');
  };

  // Create Material-UI theme
  const theme = createTheme({
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
  });

  if (loading) {
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
          <Box sx={{ 
            flex: 1, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center' 
          }}>
            <CircularProgress size={48} />
          </Box>
        </Box>
      </ThemeProvider>
    );
  }

  if (error || !notebook) {
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
          <Box sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <IconButton onClick={onBack} sx={{ mr: 1 }}>
                <BackIcon />
              </IconButton>
              <Typography variant="h5">Error Loading Notebook</Typography>
            </Box>
            
            <Alert severity="error" sx={{ mb: 2 }}>
              {error || 'Notebook not found'}
            </Alert>
            
            <Button variant="contained" onClick={onBack}>
              Back to Notebooks
            </Button>
          </Box>
        </Box>
      </ThemeProvider>
    );
  }

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
      
      <Box sx={{ p: 3, flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        {/* Header */}
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'between', mb: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', flex: 1, minWidth: 0 }}>
              <IconButton onClick={onBack} sx={{ mr: 1 }}>
                <BackIcon />
              </IconButton>
              
              <Breadcrumbs separator="›" sx={{ mr: 2 }}>
                <Link 
                  component="button" 
                  variant="body1" 
                  onClick={onBack}
                  sx={{ 
                    textDecoration: 'none',
                    '&:hover': { textDecoration: 'underline' }
                  }}
                >
                  Notebooks
                </Link>
                <Typography variant="body1" color="text.primary">
                  {notebook.name}
                </Typography>
              </Breadcrumbs>
            </Box>
            
            {onEdit && (
              <IconButton 
                onClick={() => onEdit(notebook)}
                color="primary"
              >
                <EditIcon />
              </IconButton>
            )}
          </Box>

          {/* Notebook Info */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <FolderIcon sx={{ fontSize: 32, color: 'primary.main', mr: 2 }} />
              <Box sx={{ flex: 1 }}>
                <Typography variant="h4" sx={{ mb: 0.5 }}>
                  {notebook.name}
                </Typography>
                {notebook.description && (
                  <Typography variant="body1" color="text.secondary" sx={{ mb: 1 }}>
                    {notebook.description}
                  </Typography>
                )}
              </Box>
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
              <Chip
                icon={<DocumentIcon />}
                label={`${notebook.document_count} document${notebook.document_count !== 1 ? 's' : ''}`}
                color={notebook.document_count > 0 ? 'primary' : 'default'}
                variant="outlined"
              />
              
              <Chip
                icon={<TimeIcon />}
                label={`Created ${formatRelativeTime(notebook.created_at)}`}
                variant="outlined"
              />
              
              {notebook.updated_at !== notebook.created_at && (
                <Chip
                  label={`Updated ${formatRelativeTime(notebook.updated_at)}`}
                  variant="outlined"
                />
              )}
            </Box>
          </Paper>
        </Box>

        {/* Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
          <Tabs 
            value={currentTab} 
            onChange={handleTabChange}
            sx={{
              '& .MuiTab-root': {
                fontSize: '1rem',
                fontWeight: 600,
                textTransform: 'none',
                minWidth: 120,
                padding: '12px 24px',
                '&.Mui-selected': {
                  color: 'primary.main',
                  fontWeight: 700
                }
              }
            }}
          >
            <Tab 
              icon={<DocumentIcon />} 
              label="Documents" 
              iconPosition="start"
              id="notebook-tab-0"
              aria-controls="notebook-tabpanel-0"
            />
            <Tab 
              icon={<MemoryIcon />} 
              label="Memory" 
              iconPosition="start"
              id="notebook-tab-1"
              aria-controls="notebook-tabpanel-1"
            />
            <Tab 
              icon={<ChatIcon />} 
              label="Chat" 
              iconPosition="start"
              id="notebook-tab-2"
              aria-controls="notebook-tabpanel-2"
              disabled={notebook.document_count === 0}
            />
          </Tabs>
        </Box>

        {/* Tab Panels */}
        <Box sx={{ 
          flex: 1, 
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column'
        }}>
          <TabPanel value={currentTab} index={0}>
            <NotebookDocumentList
              notebook={notebook}
              onDocumentChange={handleDocumentChange}
            />
          </TabPanel>
          
          <TabPanel value={currentTab} index={1}>
            <NotebookMemory
              notebook={notebook}
              onMemoryChange={handleDocumentChange}
            />
          </TabPanel>
          
          <TabPanel value={currentTab} index={2}>
            {notebook.document_count > 0 ? (
              <NotebookChat
                notebook={notebook}
                onDocumentChange={handleDocumentChange}
              />
            ) : (
              <Paper sx={{ p: 6, textAlign: 'center' }}>
                <ChatIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No Documents Available
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Add some documents to this notebook to start chatting with them.
                </Typography>
              </Paper>
            )}
          </TabPanel>
        </Box>
      </Box>
      </Box>
    </ThemeProvider>
  );
};

export default NotebookViewer;