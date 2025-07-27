import React, { useState, useEffect } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Container,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Tabs,
  Tab,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Chat as ChatIcon,
  Group as GroupIcon,
  AccountTree as WorkflowIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import WorkflowLanding from './components/workflow/WorkflowLanding';
import CustomWorkflowEditor from './components/CustomWorkflowEditor';

interface WorkflowData {
  id?: string;
  name: string;
  description: string;
  langflow_config: any;
  is_active: boolean;
  created_at?: string;
  updated_at?: string;
}

function WorkflowApp() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('jarvis-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });

  const [currentView, setCurrentView] = useState<'landing' | 'editor'>('landing');
  const [selectedWorkflow, setSelectedWorkflow] = useState<WorkflowData | null>(null);
  const [isNewWorkflow, setIsNewWorkflow] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Create theme
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#2196f3',
      },
      secondary: {
        main: '#ff9800',
      },
      background: {
        default: darkMode ? '#121212' : '#f5f5f5',
        paper: darkMode ? '#1e1e1e' : '#ffffff',
      },
    },
  });

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('jarvis-dark-mode', JSON.stringify(newDarkMode));
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    switch (newValue) {
      case 0:
        window.location.href = '/';
        break;
      case 1:
        window.location.href = '/multi-agent.html';
        break;
      case 2:
        // Already on workflow page
        break;
      case 3:
        window.location.href = '/settings.html';
        break;
      case 4:
        window.location.href = '/knowledge-graph.html';
        break;
    }
  };

  const handleCreateNewWorkflow = () => {
    setSelectedWorkflow(null);
    setIsNewWorkflow(true);
    setCurrentView('editor');
  };

  const handleEditWorkflow = (workflow: WorkflowData) => {
    setSelectedWorkflow(workflow);
    setIsNewWorkflow(false);
    setCurrentView('editor');
  };

  const handleBackToLanding = () => {
    setCurrentView('landing');
    setSelectedWorkflow(null);
    setIsNewWorkflow(false);
  };

  const handleSaveWorkflow = async (workflowData: any) => {
    setLoading(true);
    setError(null);

    try {
      const url = selectedWorkflow?.id
        ? `http://127.0.0.1:8000/api/v1/automation/workflows/${selectedWorkflow.id}`
        : 'http://127.0.0.1:8000/api/v1/automation/workflows';
      
      const method = selectedWorkflow?.id ? 'PUT' : 'POST';
      
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: workflowData.name || 'Untitled Workflow',
          description: workflowData.description || '',
          langflow_config: workflowData.config,
          is_active: true,
          created_by: 'user'
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to save workflow: ${response.statusText}`);
      }

      const savedWorkflow = await response.json();
      setSelectedWorkflow(savedWorkflow);
      setIsNewWorkflow(false);
      
      // Trigger refresh of workflow list
      setRefreshTrigger(prev => prev + 1);
      
      // Don't automatically go back to landing after save
      // This allows users to continue working on the workflow
      // handleBackToLanding();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save workflow');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Jarvis Workflow Designer
            </Typography>

            <IconButton onClick={toggleDarkMode} color="inherit">
              {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Toolbar>
        </AppBar>

        {/* Navigation Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={2}
            onChange={handleTabChange} 
            aria-label="jarvis modes"
            centered
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
              label="Standard Chat" 
              id="tab-0"
              aria-controls="tabpanel-0"
            />
            <Tab 
              label="Multi-Agent" 
              id="tab-1"
              aria-controls="tabpanel-1"
            />
            <Tab 
              label="Workflow" 
              id="tab-2"
              aria-controls="tabpanel-2"
            />
            <Tab 
              label="Settings" 
              id="tab-3"
              aria-controls="tabpanel-3"
            />
            <Tab 
              label="Knowledge Graph" 
              id="tab-4"
              aria-controls="tabpanel-4"
            />
          </Tabs>
        </Box>

        {/* Main Content */}
        <Box sx={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
          {loading && (
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'rgba(0, 0, 0, 0.5)',
                zIndex: 9999,
              }}
            >
              <CircularProgress />
            </Box>
          )}

          {error && (
            <Alert severity="error" onClose={() => setError(null)} sx={{ m: 2 }}>
              {error}
            </Alert>
          )}

          {currentView === 'landing' ? (
            <WorkflowLanding
              onCreateNew={handleCreateNewWorkflow}
              onEditWorkflow={handleEditWorkflow}
              refreshTrigger={refreshTrigger}
            />
          ) : (
            <CustomWorkflowEditor
              initialWorkflow={selectedWorkflow?.langflow_config}
              workflowId={selectedWorkflow?.id}
              onSave={handleSaveWorkflow}
              onBack={handleBackToLanding}
              isNewWorkflow={isNewWorkflow}
            />
          )}
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default WorkflowApp;