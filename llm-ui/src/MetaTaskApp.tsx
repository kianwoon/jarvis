import React, { useState, useEffect } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Paper,
  Container,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Tabs,
  Tab,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Chip,
  LinearProgress,
  Alert
} from '@mui/material';
import {
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Add as AddIcon,
  PlayArrow as PlayIcon,
  Settings as SettingsIcon,
  Description as DocumentIcon,
  AccountTree as WorkflowIcon,
  Timeline as TimelineIcon
} from '@mui/icons-material';

interface Template {
  id: string;
  name: string;
  description: string;
  template_type: string;
  template_config: any;
  is_active: boolean;
}

interface Workflow {
  id: string;
  name: string;
  status: string;
  progress: any;
  created_at: string;
}

function MetaTaskApp() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('jarvis-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });

  const [templates, setTemplates] = useState<Template[]>([]);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [workflowName, setWorkflowName] = useState<string>('');
  const [inputData, setInputData] = useState<any>({});
  const [createWorkflowOpen, setCreateWorkflowOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

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
        window.location.href = '/workflow.html';
        break;
      case 3:
        // Already on meta-tasks page
        break;
      case 4:
        window.location.href = '/settings.html';
        break;
      case 5:
        window.location.href = '/knowledge-graph.html';
        break;
      case 6:
        window.location.href = '/idc.html';
        break;
    }
  };

  useEffect(() => {
    loadTemplates();
    loadWorkflows();
  }, []);

  const loadTemplates = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/meta-task/templates');
      if (response.ok) {
        const data = await response.json();
        setTemplates(data.templates || []);
      } else {
        setError('Failed to load templates');
      }
    } catch (err) {
      setError('Error loading templates');
    } finally {
      setLoading(false);
    }
  };

  const loadWorkflows = async () => {
    try {
      // This endpoint would need to be implemented
      const response = await fetch('/api/v1/meta-task/workflows');
      if (response.ok) {
        const data = await response.json();
        setWorkflows(data.workflows || []);
      }
    } catch (err) {
      console.log('Workflows endpoint not implemented yet');
    }
  };

  const createWorkflow = async () => {
    try {
      setLoading(true);
      setError('');
      
      const response = await fetch('/api/v1/meta-task/workflows', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          template_id: selectedTemplate,
          name: workflowName,
          input_data: inputData
        })
      });

      if (response.ok) {
        setCreateWorkflowOpen(false);
        setWorkflowName('');
        setInputData({});
        setSelectedTemplate('');
        loadWorkflows();
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to create workflow');
      }
    } catch (err) {
      setError('Error creating workflow');
    } finally {
      setLoading(false);
    }
  };

  const executeWorkflow = async (workflowId: string) => {
    try {
      // This would open a streaming connection to monitor execution
      window.open(`/api/v1/meta-task/workflows/${workflowId}/execute`, '_blank');
    } catch (err) {
      setError('Error executing workflow');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'running': return 'primary';
      case 'failed': return 'error';
      case 'pending': return 'default';
      default: return 'default';
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Box sx={{ 
          position: 'sticky', 
          top: 0, 
          zIndex: 1100, 
          bgcolor: 'background.default',
          boxShadow: 1
        }}>
          <AppBar position="static">
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                Jarvis AI Assistant - Meta-Tasks
              </Typography>
              <IconButton onClick={toggleDarkMode} color="inherit">
                {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
              </IconButton>
            </Toolbar>
          </AppBar>

          {/* Navigation Tabs */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper' }}>
            <Tabs 
              value={3}
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
              <Tab label="Standard Chat" />
              <Tab label="Multi-Agent" />
              <Tab label="Workflow" />
              <Tab label="Meta-Tasks" />
              <Tab label="Settings" />
              <Tab label="Knowledge Graph" />
              <Tab label="IDC" />
            </Tabs>
          </Box>
        </Box>

        {/* Main Content */}
        <Container maxWidth={false} sx={{ flex: 1, py: 3, overflow: 'hidden' }}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {loading && <LinearProgress sx={{ mb: 2 }} />}

          <Grid container spacing={3}>
            {/* Templates Section */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, height: 'fit-content' }}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="h5" component="h2">
                    Templates
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    onClick={() => setCreateWorkflowOpen(true)}
                    disabled={templates.length === 0}
                  >
                    Create Workflow
                  </Button>
                </Box>

                {templates.length === 0 ? (
                  <Typography variant="body1" color="text.secondary">
                    No templates available. Templates are created through database migrations.
                  </Typography>
                ) : (
                  <Grid container spacing={2}>
                    {templates.map((template) => (
                      <Grid item xs={12} key={template.id}>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="h6" component="h3">
                              {template.name}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" paragraph>
                              {template.description}
                            </Typography>
                            <Chip 
                              label={template.template_type} 
                              size="small" 
                              color="primary" 
                              variant="outlined" 
                            />
                            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                              Phases: {template.template_config?.phases?.length || 0}
                            </Typography>
                          </CardContent>
                          <CardActions>
                            <Button 
                              size="small" 
                              startIcon={<PlayIcon />}
                              onClick={() => {
                                setSelectedTemplate(template.id);
                                setCreateWorkflowOpen(true);
                              }}
                            >
                              Use Template
                            </Button>
                          </CardActions>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                )}
              </Paper>
            </Grid>

            {/* Workflows Section */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, height: 'fit-content' }}>
                <Typography variant="h5" component="h2" mb={2}>
                  Active Workflows
                </Typography>

                {workflows.length === 0 ? (
                  <Typography variant="body1" color="text.secondary">
                    No active workflows. Create a workflow from a template to get started.
                  </Typography>
                ) : (
                  <List>
                    {workflows.map((workflow) => (
                      <ListItem key={workflow.id} divider>
                        <ListItemText
                          primary={workflow.name}
                          secondary={`Created: ${new Date(workflow.created_at).toLocaleDateString()}`}
                        />
                        <Chip 
                          label={workflow.status}
                          color={getStatusColor(workflow.status) as any}
                          size="small"
                        />
                        <IconButton 
                          onClick={() => executeWorkflow(workflow.id)}
                          disabled={workflow.status === 'running'}
                        >
                          <PlayIcon />
                        </IconButton>
                      </ListItem>
                    ))}
                  </List>
                )}
              </Paper>
            </Grid>
          </Grid>
        </Container>

        {/* Create Workflow Dialog */}
        <Dialog 
          open={createWorkflowOpen} 
          onClose={() => setCreateWorkflowOpen(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>Create New Workflow</DialogTitle>
          <DialogContent>
            <Box sx={{ pt: 1 }}>
              <FormControl fullWidth margin="normal">
                <InputLabel>Template</InputLabel>
                <Select
                  value={selectedTemplate}
                  onChange={(e) => setSelectedTemplate(e.target.value)}
                  label="Template"
                >
                  {templates.map((template) => (
                    <MenuItem key={template.id} value={template.id}>
                      {template.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <TextField
                fullWidth
                label="Workflow Name"
                value={workflowName}
                onChange={(e) => setWorkflowName(e.target.value)}
                margin="normal"
                required
              />

              <TextField
                fullWidth
                label="Input Data (JSON)"
                value={JSON.stringify(inputData, null, 2)}
                onChange={(e) => {
                  try {
                    setInputData(JSON.parse(e.target.value));
                  } catch {
                    // Invalid JSON, keep as string
                  }
                }}
                multiline
                rows={6}
                margin="normal"
                placeholder='{"topic": "Your topic here", "target_length": 10}'
                helperText="Enter the input data as JSON. Required fields depend on the selected template."
              />
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setCreateWorkflowOpen(false)}>
              Cancel
            </Button>
            <Button 
              onClick={createWorkflow}
              variant="contained"
              disabled={!selectedTemplate || !workflowName || loading}
            >
              Create Workflow
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </ThemeProvider>
  );
}

export default MetaTaskApp;