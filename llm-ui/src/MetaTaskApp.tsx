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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Chip,
  Alert,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextareaAutosize
} from '@mui/material';
import {
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as PlayIcon,
  Description as TemplateIcon,
  AccountTree as WorkflowIcon,
  ContentCopy as DuplicateIcon
} from '@mui/icons-material';


interface Template {
  id: string;
  name: string;
  description: string;
  category: string;
  variables: string[];
  content: string;
  created_at: string;
  updated_at: string;
}

interface WorkflowStep {
  id: string;
  name: string;
  template_id: string;
  order: number;
  variables: Record<string, any>;
}

interface Workflow {
  id: string;
  name: string;
  description: string;
  steps: WorkflowStep[];
  created_at: string;
  updated_at: string;
  last_run?: string;
  status?: 'idle' | 'running' | 'completed' | 'failed';
}


function MetaTaskApp() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('jarvis-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  
  // Templates and Workflows management
  const [templates, setTemplates] = useState<Template[]>([]);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null);
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null);
  
  // Dialog states
  const [createWorkflowOpen, setCreateWorkflowOpen] = useState(false);
  const [editTemplateOpen, setEditTemplateOpen] = useState(false);
  const [newWorkflow, setNewWorkflow] = useState<Partial<Workflow>>({
    name: '',
    description: '',
    steps: []
  });
  
  // Active tab for Templates/Workflows
  const [activeTab, setActiveTab] = useState(0);

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

  // Load templates
  const loadTemplates = async () => {
    try {
      setLoading(true);
      // Mock data for now - replace with actual API call
      setTemplates([
        {
          id: '1',
          name: 'Technical Documentation',
          description: 'Template for creating technical documentation',
          category: 'Documentation',
          variables: ['project_name', 'version', 'author'],
          content: '# {{project_name}} Documentation\nVersion: {{version}}\nAuthor: {{author}}',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        },
        {
          id: '2',
          name: 'API Specification',
          description: 'Template for API documentation',
          category: 'API',
          variables: ['api_name', 'base_url', 'version'],
          content: '# {{api_name}} API\nBase URL: {{base_url}}\nVersion: {{version}}',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        },
        {
          id: '3',
          name: 'Project Proposal',
          description: 'Template for project proposals',
          category: 'Business',
          variables: ['project_title', 'client', 'budget'],
          content: '# Project Proposal: {{project_title}}\nClient: {{client}}\nBudget: {{budget}}',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        }
      ]);
    } catch (err) {
      setError('Failed to load templates');
    } finally {
      setLoading(false);
    }
  };

  // Load workflows
  const loadWorkflows = async () => {
    try {
      // Mock data for now - replace with actual API call
      setWorkflows([
        {
          id: '1',
          name: 'Complete Documentation Pipeline',
          description: 'Full documentation generation workflow',
          steps: [
            {
              id: '1',
              name: 'Generate Overview',
              template_id: '1',
              order: 1,
              variables: { project_name: 'My Project', version: '1.0', author: 'Team' }
            },
            {
              id: '2',
              name: 'Generate API Docs',
              template_id: '2',
              order: 2,
              variables: { api_name: 'REST API', base_url: 'https://api.example.com', version: 'v1' }
            }
          ],
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          last_run: new Date().toISOString(),
          status: 'completed'
        }
      ]);
    } catch (err) {
      console.error('Failed to load workflows:', err);
    }
  };

  // Create new workflow
  const createWorkflow = async () => {
    try {
      // API call to create workflow
      console.log('Creating workflow:', newWorkflow);
      setSuccess('Workflow created successfully!');
      setCreateWorkflowOpen(false);
      await loadWorkflows();
    } catch (err) {
      setError('Failed to create workflow');
    }
  };

  // Run workflow
  const runWorkflow = async (workflowId: string) => {
    try {
      // API call to run workflow
      console.log('Running workflow:', workflowId);
      setSuccess('Workflow started successfully!');
    } catch (err) {
      setError('Failed to run workflow');
    }
  };

  // Delete template
  const deleteTemplate = async (templateId: string) => {
    try {
      // API call to delete template
      console.log('Deleting template:', templateId);
      setSuccess('Template deleted successfully!');
      await loadTemplates();
    } catch (err) {
      setError('Failed to delete template');
    }
  };

  // Delete workflow
  const deleteWorkflow = async (workflowId: string) => {
    try {
      // API call to delete workflow
      console.log('Deleting workflow:', workflowId);
      setSuccess('Workflow deleted successfully!');
      await loadWorkflows();
    } catch (err) {
      setError('Failed to delete workflow');
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
                Jarvis AI Assistant - Meta-Task Templates & Workflows
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
        <Container maxWidth={false} sx={{ flex: 1, py: 2, overflow: 'hidden', width: '100%' }}>
          <Paper sx={{ height: '100%', overflow: 'auto', p: 3 }}>
            <Box>
              {/* Header */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="h4" gutterBottom>
                  Meta-Task Templates & Workflows
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Manage document templates and create automated workflows for complex document generation
                </Typography>
              </Box>

              {error && (
                <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
                  {error}
                </Alert>
              )}

              {success && (
                <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess('')}>
                  {success}
                </Alert>
              )}

              {/* Content Tabs */}
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs
                  value={activeTab}
                  onChange={(_, newValue) => setActiveTab(newValue)}
                  centered
                >
                  <Tab label="Templates" icon={<TemplateIcon />} iconPosition="start" />
                  <Tab label="Workflows" icon={<WorkflowIcon />} iconPosition="start" />
                </Tabs>
              </Box>

              {/* Templates Tab */}
              <Box hidden={activeTab !== 0} sx={{ pt: 3 }}>
                <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="h5">Document Templates</Typography>
                  <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    onClick={() => setEditTemplateOpen(true)}
                  >
                    Create Template
                  </Button>
                </Box>

                <Grid container spacing={2}>
                  {templates.map((template) => (
                    <Grid item xs={12} md={6} lg={4} key={template.id}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            {template.name}
                          </Typography>
                          <Chip label={template.category} size="small" color="primary" sx={{ mb: 1 }} />
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            {template.description}
                          </Typography>
                          <Typography variant="caption" display="block">
                            Variables: {template.variables.join(', ')}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Updated: {new Date(template.updated_at).toLocaleDateString()}
                          </Typography>
                        </CardContent>
                        <CardActions>
                          <Button size="small" startIcon={<EditIcon />}>
                            Edit
                          </Button>
                          <Button size="small" startIcon={<DuplicateIcon />}>
                            Duplicate
                          </Button>
                          <Button 
                            size="small" 
                            color="error"
                            startIcon={<DeleteIcon />}
                            onClick={() => deleteTemplate(template.id)}
                          >
                            Delete
                          </Button>
                        </CardActions>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>

              {/* Workflows Tab */}
              <Box hidden={activeTab !== 1} sx={{ pt: 3 }}>
                <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="h5">Document Workflows</Typography>
                  <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    onClick={() => setCreateWorkflowOpen(true)}
                  >
                    Create Workflow
                  </Button>
                </Box>

                <Grid container spacing={2}>
                  {workflows.map((workflow) => (
                    <Grid item xs={12} key={workflow.id}>
                      <Card>
                        <CardContent>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
                            <Box>
                              <Typography variant="h6">
                                {workflow.name}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {workflow.description}
                              </Typography>
                            </Box>
                            {workflow.status && (
                              <Chip 
                                label={workflow.status} 
                                color={workflow.status === 'completed' ? 'success' : workflow.status === 'failed' ? 'error' : 'default'}
                                size="small"
                              />
                            )}
                          </Box>

                          <Divider sx={{ my: 2 }} />

                          <Typography variant="subtitle2" gutterBottom>
                            Workflow Steps ({workflow.steps.length}):
                          </Typography>
                          <List dense>
                            {workflow.steps.map((step, index) => (
                              <ListItem key={step.id}>
                                <ListItemText
                                  primary={`${index + 1}. ${step.name}`}
                                  secondary={`Template: ${templates.find(t => t.id === step.template_id)?.name || 'Unknown'}`}
                                />
                              </ListItem>
                            ))}
                          </List>

                          <Typography variant="caption" color="text.secondary">
                            Last run: {workflow.last_run ? new Date(workflow.last_run).toLocaleString() : 'Never'}
                          </Typography>
                        </CardContent>
                        <CardActions>
                          <Button 
                            size="small" 
                            variant="contained"
                            startIcon={<PlayIcon />}
                            onClick={() => runWorkflow(workflow.id)}
                          >
                            Run
                          </Button>
                          <Button size="small" startIcon={<EditIcon />}>
                            Edit
                          </Button>
                          <Button size="small" startIcon={<DuplicateIcon />}>
                            Duplicate
                          </Button>
                          <Button 
                            size="small" 
                            color="error"
                            startIcon={<DeleteIcon />}
                            onClick={() => deleteWorkflow(workflow.id)}
                          >
                            Delete
                          </Button>
                        </CardActions>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>

              {/* Create Workflow Dialog */}
              <Dialog 
                open={createWorkflowOpen} 
                onClose={() => setCreateWorkflowOpen(false)}
                maxWidth="md"
                fullWidth
              >
                <DialogTitle>Create New Workflow</DialogTitle>
                <DialogContent>
                  <TextField
                    fullWidth
                    label="Workflow Name"
                    value={newWorkflow.name}
                    onChange={(e) => setNewWorkflow({ ...newWorkflow, name: e.target.value })}
                    margin="normal"
                  />
                  <TextField
                    fullWidth
                    label="Description"
                    value={newWorkflow.description}
                    onChange={(e) => setNewWorkflow({ ...newWorkflow, description: e.target.value })}
                    margin="normal"
                    multiline
                    rows={3}
                  />
                  
                  <Typography variant="subtitle1" sx={{ mt: 2, mb: 1 }}>
                    Add Steps
                  </Typography>
                  <Button 
                    variant="outlined" 
                    startIcon={<AddIcon />}
                    fullWidth
                  >
                    Add Step from Template
                  </Button>
                </DialogContent>
                <DialogActions>
                  <Button onClick={() => setCreateWorkflowOpen(false)}>Cancel</Button>
                  <Button onClick={createWorkflow} variant="contained">Create</Button>
                </DialogActions>
              </Dialog>

              {/* Edit Template Dialog */}
              <Dialog 
                open={editTemplateOpen} 
                onClose={() => setEditTemplateOpen(false)}
                maxWidth="md"
                fullWidth
              >
                <DialogTitle>Create New Template</DialogTitle>
                <DialogContent>
                  <TextField
                    fullWidth
                    label="Template Name"
                    margin="normal"
                  />
                  <TextField
                    fullWidth
                    label="Description"
                    margin="normal"
                    multiline
                    rows={2}
                  />
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Category</InputLabel>
                    <Select label="Category">
                      <MenuItem value="Documentation">Documentation</MenuItem>
                      <MenuItem value="API">API</MenuItem>
                      <MenuItem value="Business">Business</MenuItem>
                      <MenuItem value="Technical">Technical</MenuItem>
                    </Select>
                  </FormControl>
                  <TextField
                    fullWidth
                    label="Variables (comma-separated)"
                    margin="normal"
                    helperText="e.g., project_name, version, author"
                  />
                  <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                    Template Content
                  </Typography>
                  <TextareaAutosize
                    minRows={10}
                    style={{ 
                      width: '100%', 
                      padding: '8px',
                      fontFamily: 'monospace',
                      fontSize: '14px',
                      border: '1px solid #ccc',
                      borderRadius: '4px'
                    }}
                    placeholder="Enter template content with variables in {{variable_name}} format"
                  />
                </DialogContent>
                <DialogActions>
                  <Button onClick={() => setEditTemplateOpen(false)}>Cancel</Button>
                  <Button variant="contained">Create Template</Button>
                </DialogActions>
              </Dialog>

            </Box>
          </Paper>
        </Container>

      </Box>
    </ThemeProvider>
  );
}

export default MetaTaskApp;