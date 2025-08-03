import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
  Tooltip,
  CardHeader,
  CardContent,
  Card,
  Grid,
  Divider,
  Autocomplete
} from '@mui/material';
import {
  Edit as EditIcon,
  Refresh as RefreshIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  ExpandMore as ExpandMoreIcon,
  Code as CodeIcon,
  Preview as PreviewIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';

interface Prompt {
  id: string;
  name: string;
  description?: string;
  prompt_type: string;
  version: number;
  is_active: boolean;
  parameters?: Record<string, any>;
  prompt_template?: string;
}

interface PromptVariable {
  name: string;
  description: string;
  required: boolean;
}

interface PromptManagementProps {
  data: Prompt[];
  onChange: (field: string, value: any) => void;
  onShowSuccess?: (message?: string) => void;
}

const PromptManagement: React.FC<PromptManagementProps> = ({ 
  data, 
  onChange, 
  onShowSuccess 
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedPrompt, setSelectedPrompt] = useState<Prompt | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false);
  const [editedTemplate, setEditedTemplate] = useState('');
  const [previewVariables, setPreviewVariables] = useState<Record<string, string>>({});
  const [previewResult, setPreviewResult] = useState('');

  const promptTypes = [
    'entity_discovery',
    'relationship_discovery',
    'knowledge_extraction',
    'custom'
  ];

  // Use prompts from unified settings data
  const prompts = Array.isArray(data) ? data : [];

  // Auto-load prompts when component mounts
  useEffect(() => {
    if (prompts.length === 0) {
      // If no prompts exist, initialize with default prompts
      const defaultPrompts: Prompt[] = [
        {
          id: '1',
          name: 'entity_discovery',
          description: 'Extract entities from text for knowledge graph construction',
          prompt_type: 'entity_discovery',
          version: 1,
          is_active: true,
          prompt_template: 'Analyze the following text and extract all entities. Return entities in JSON format: {text}',
          parameters: { format: 'json', confidence_threshold: 0.7 }
        },
        {
          id: '2',
          name: 'relationship_discovery',
          description: 'Discover relationships between entities',
          prompt_type: 'relationship_discovery',
          version: 1,
          is_active: true,
          prompt_template: 'Identify relationships between entities in the text. Return relationships as triples: {text}',
          parameters: { format: 'triples', confidence_threshold: 0.8 }
        },
        {
          id: '3',
          name: 'knowledge_extraction',
          description: 'Extract comprehensive knowledge from documents',
          prompt_type: 'knowledge_extraction',
          version: 1,
          is_active: true,
          prompt_template: 'Extract key knowledge from the document: {text}',
          parameters: { depth: 'comprehensive', include_metadata: true }
        }
      ];
      onChange('prompts', defaultPrompts);
    }
  }, [prompts.length, onChange]);

  const handleEditPrompt = (prompt: Prompt) => {
    setSelectedPrompt(prompt);
    setEditedTemplate(prompt.prompt_template || '');
    setEditDialogOpen(true);
  };

  const handleSavePrompt = async () => {
    if (!selectedPrompt) return;

    try {
      const updatedPrompt = {
        ...selectedPrompt,
        prompt_template: editedTemplate,
        version: selectedPrompt.version + 1
      };

      const updatedPrompts = prompts.map(p => 
        p.id === selectedPrompt.id ? updatedPrompt : p
      );
      
      onChange('prompts', updatedPrompts);
      
      if (onShowSuccess) {
        onShowSuccess('Prompt saved successfully');
      }
      
      setEditDialogOpen(false);
      setSelectedPrompt(null);
    } catch (err) {
      setError('Failed to save prompt');
      console.error('Error saving prompt:', err);
    }
  };

  const handlePreviewPrompt = (prompt: Prompt) => {
    setSelectedPrompt(prompt);
    setPreviewDialogOpen(true);
    setPreviewVariables({});
    setPreviewResult('');
  };

  const handleGeneratePreview = () => {
    if (!selectedPrompt) return;

    try {
      // Simple preview generation by replacing variables in template
      let result = selectedPrompt.prompt_template || '';
      
      Object.entries(previewVariables).forEach(([key, value]) => {
        const regex = new RegExp(`\\{${key}\\}`, 'g');
        result = result.replace(regex, value);
      });
      
      setPreviewResult(result);
    } catch (err) {
      setError('Failed to generate preview');
      console.error('Error generating preview:', err);
    }
  };

  const extractVariables = (template: string): PromptVariable[] => {
    const regex = /\{(\w+)\}/g;
    const matches = template.match(regex) || [];
    const uniqueVars = [...new Set(matches.map(m => m.slice(1, -1)))].sort();
    
    return uniqueVars.map(name => ({
      name: name,
      description: `Variable: ${name}`,
      required: true
    }));
  };

  const getTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      entity_discovery: 'primary',
      relationship_discovery: 'secondary',
      knowledge_extraction: 'success',
      custom: 'default'
    };
    return colors[type] || 'default';
  };

  const getTypeLabel = (type: string) => {
    return type.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          LLM Prompt Management
        </Typography>
        <Box>
          <Button
            startIcon={<RefreshIcon />}
            onClick={() => {
              // Reset to default prompts
              const defaultPrompts: Prompt[] = [
                {
                  id: '1',
                  name: 'entity_discovery',
                  description: 'Extract entities from text for knowledge graph construction',
                  prompt_type: 'entity_discovery',
                  version: 1,
                  is_active: true,
                  prompt_template: 'Analyze the following text and extract all entities. Return entities in JSON format: {text}',
                  parameters: { format: 'json', confidence_threshold: 0.7 }
                },
                {
                  id: '2',
                  name: 'relationship_discovery',
                  description: 'Discover relationships between entities',
                  prompt_type: 'relationship_discovery',
                  version: 1,
                  is_active: true,
                  prompt_template: 'Identify relationships between entities in the text. Return relationships as triples: {text}',
                  parameters: { format: 'triples', confidence_threshold: 0.8 }
                },
                {
                  id: '3',
                  name: 'knowledge_extraction',
                  description: 'Extract comprehensive knowledge from documents',
                  prompt_type: 'knowledge_extraction',
                  version: 1,
                  is_active: true,
                  prompt_template: 'Extract key knowledge from the document: {text}',
                  parameters: { depth: 'comprehensive', include_metadata: true }
                }
              ];
              onChange('prompts', defaultPrompts);
              if (onShowSuccess) {
                onShowSuccess('Prompts reset to defaults');
              }
            }}
            variant="outlined"
            sx={{ mr: 1 }}
          >
            Reset
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Paper elevation={2} sx={{ mb: 3 }}>
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            System Prompts
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Manage the prompts used for knowledge graph extraction. These prompts control how the LLM identifies entities and relationships.
          </Typography>
        </Box>
      </Paper>

      <List>
        {prompts.map((prompt) => (
          <Accordion key={prompt.id} sx={{ mb: 1 }}>
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              aria-controls={`panel-${prompt.id}-content`}
              id={`panel-${prompt.id}-header`}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
                <Box sx={{ flexGrow: 1 }}>
                  <Typography variant="h6" component="div">
                    {prompt.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {prompt.description}
                  </Typography>
                  <Box sx={{ mt: 1 }}>
                    <Chip
                      label={getTypeLabel(prompt.prompt_type)}
                      color={getTypeColor(prompt.prompt_type) as any}
                      size="small"
                    />
                    <Chip
                      label={`v${prompt.version}`}
                      variant="outlined"
                      size="small"
                      sx={{ ml: 1 }}
                    />
                    {!prompt.is_active && (
                      <Chip
                        label="Inactive"
                        color="error"
                        size="small"
                        sx={{ ml: 1 }}
                      />
                    )}
                  </Box>
                </Box>
                <ListItemSecondaryAction>
                  <IconButton
                    edge="end"
                    aria-label="preview"
                    onClick={(e) => {
                      e.stopPropagation();
                      handlePreviewPrompt(prompt);
                    }}
                  >
                    <PreviewIcon />
                  </IconButton>
                  <IconButton
                    edge="end"
                    aria-label="edit"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleEditPrompt(prompt);
                    }}
                  >
                    <EditIcon />
                  </IconButton>
                </ListItemSecondaryAction>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              {/* User-friendly parameters display */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                  Parameters:
                </Typography>
                <Grid container spacing={1}>
                  {prompt.parameters && Object.entries(prompt.parameters).map(([key, value]) => (
                    <Grid item xs={12} sm={6} md={4} key={key}>
                      <Chip
                        label={`${key}: ${Array.isArray(value) ? value.join(', ') : String(value)}`}
                        variant="outlined"
                        size="small"
                        sx={{ 
                          width: '100%',
                          justifyContent: 'flex-start',
                          '& .MuiChip-label': {
                            display: 'block',
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            maxWidth: '100%'
                          }
                        }}
                      />
                    </Grid>
                  ))}
                  {(!prompt.parameters || Object.keys(prompt.parameters).length === 0) && (
                    <Grid item xs={12}>
                      <Typography variant="body2" color="text.secondary" fontStyle="italic">
                        No parameters configured
                      </Typography>
                    </Grid>
                  )}
                </Grid>
              </Box>
              {prompt.prompt_template && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Template Preview:
                  </Typography>
                  <Box
                    component="pre"
                    sx={{
                      backgroundColor: (theme) => theme.palette.mode === 'dark' ? theme.palette.grey[900] : theme.palette.grey[100],
                      color: (theme) => theme.palette.mode === 'dark' ? theme.palette.grey[100] : theme.palette.grey[900],
                      p: 2,
                      borderRadius: 1,
                      fontSize: '0.8rem',
                      maxHeight: 200,
                      overflow: 'auto',
                      whiteSpace: 'pre-wrap'
                    }}
                  >
                    {prompt.prompt_template.substring(0, 500)}
                    {prompt.prompt_template.length > 500 && '...'}
                  </Box>
                </Box>
              )}
            </AccordionDetails>
          </Accordion>
        ))}
      </List>

      {/* Edit Dialog */}
      <Dialog
        open={editDialogOpen}
        onClose={() => setEditDialogOpen(false)}
        maxWidth="xl"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="h6">Edit Prompt</Typography>
            <Chip label={selectedPrompt?.name} color="primary" />
            <Box sx={{ flexGrow: 1 }} />
            <FormControlLabel
              control={
                <Switch
                  checked={selectedPrompt?.is_active || false}
                  onChange={(e) => {
                    if (selectedPrompt) {
                      const updatedPrompts = prompts.map(p => 
                        p.id === selectedPrompt.id 
                          ? { ...p, is_active: e.target.checked }
                          : p
                      );
                      onChange('prompts', updatedPrompts);
                      setSelectedPrompt({ ...selectedPrompt, is_active: e.target.checked });
                    }
                  }}
                />
              }
              label="Active"
            />
          </Box>
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            {/* Left Column - Basic Info */}
            <Grid item xs={12} md={4}>
              <Card variant="outlined">
                <CardHeader title="Prompt Information" />
                <CardContent>
                  <TextField
                    fullWidth
                    label="Name"
                    value={selectedPrompt?.name || ''}
                    onChange={(e) => {
                      if (selectedPrompt) {
                        const updatedPrompts = prompts.map(p => 
                          p.id === selectedPrompt.id 
                            ? { ...p, name: e.target.value }
                            : p
                        );
                        onChange('prompts', updatedPrompts);
                        setSelectedPrompt({ ...selectedPrompt, name: e.target.value });
                      }
                    }}
                    margin="normal"
                    variant="outlined"
                  />
                  <TextField
                    fullWidth
                    label="Description"
                    multiline
                    rows={3}
                    value={selectedPrompt?.description || ''}
                    onChange={(e) => {
                      if (selectedPrompt) {
                        const updatedPrompts = prompts.map(p => 
                          p.id === selectedPrompt.id 
                            ? { ...p, description: e.target.value }
                            : p
                        );
                        onChange('prompts', updatedPrompts);
                        setSelectedPrompt({ ...selectedPrompt, description: e.target.value });
                      }
                    }}
                    margin="normal"
                    variant="outlined"
                    helperText="Brief description of what this prompt does"
                  />
                  <Autocomplete
                    fullWidth
                    options={promptTypes}
                    value={selectedPrompt?.prompt_type || ''}
                    onChange={(_, newValue) => {
                      if (selectedPrompt && newValue) {
                        const updatedPrompts = prompts.map(p => 
                          p.id === selectedPrompt.id 
                            ? { ...p, prompt_type: newValue }
                            : p
                        );
                        onChange('prompts', updatedPrompts);
                        setSelectedPrompt({ ...selectedPrompt, prompt_type: newValue });
                      }
                    }}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Prompt Type"
                        margin="normal"
                        variant="outlined"
                        helperText="Category of this prompt"
                      />
                    )}
                  />
                </CardContent>
              </Card>

              {/* Parameters Card */}
              <Card variant="outlined" sx={{ mt: 2 }}>
                <CardHeader 
                  title="Parameters" 
                  subheader="Configure prompt-specific parameters"
                />
                <CardContent>
                  {selectedPrompt?.parameters && Object.entries(selectedPrompt.parameters).map(([key, value]) => (
                    <Box key={key} sx={{ mb: 2 }}>
                      <TextField
                        fullWidth
                        label={key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ')}
                        value={Array.isArray(value) ? value.join(', ') : String(value)}
                        onChange={(e) => {
                          if (selectedPrompt) {
                            const newParams = { ...selectedPrompt.parameters };
                            // Handle arrays vs single values
                            if (Array.isArray(value)) {
                              newParams[key] = e.target.value.split(',').map(v => v.trim()).filter(v => v);
                            } else if (typeof value === 'number') {
                              newParams[key] = parseFloat(e.target.value) || 0;
                            } else if (typeof value === 'boolean') {
                              newParams[key] = e.target.value.toLowerCase() === 'true';
                            } else {
                              newParams[key] = e.target.value;
                            }
                            
                            const updatedPrompts = prompts.map(p => 
                              p.id === selectedPrompt.id 
                                ? { ...p, parameters: newParams }
                                : p
                            );
                            onChange('prompts', updatedPrompts);
                            setSelectedPrompt({ ...selectedPrompt, parameters: newParams });
                          }
                        }}
                        variant="outlined"
                        size="small"
                        helperText={
                          Array.isArray(value) 
                            ? "Comma-separated list" 
                            : typeof value === 'number' 
                            ? "Numeric value" 
                            : typeof value === 'boolean'
                            ? "true or false"
                            : "Text value"
                        }
                      />
                    </Box>
                  ))}
                  {(!selectedPrompt?.parameters || Object.keys(selectedPrompt.parameters).length === 0) && (
                    <Typography variant="body2" color="text.secondary" fontStyle="italic">
                      No parameters configured for this prompt
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Right Column - Template Editor */}
            <Grid item xs={12} md={8}>
              <Card variant="outlined" sx={{ height: 'fit-content' }}>
                <CardHeader 
                  title="Prompt Template" 
                  subheader="Edit the template using {variable_name} for placeholders"
                />
                <CardContent>
                  <TextField
                    fullWidth
                    multiline
                    rows={16}
                    value={editedTemplate}
                    onChange={(e) => setEditedTemplate(e.target.value)}
                    variant="outlined"
                    label="Template"
                    placeholder="Enter your prompt template here..."
                    sx={{ 
                      '& .MuiInputBase-input': {
                        fontFamily: 'monospace',
                        fontSize: '14px'
                      }
                    }}
                  />
                  {selectedPrompt && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                        Template Variables:
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {extractVariables(editedTemplate).map((variable) => (
                          <Tooltip key={variable.name} title={variable.description}>
                            <Chip
                              label={`{${variable.name}}`}
                              size="small"
                              variant="outlined"
                              color={variable.required ? "primary" : "default"}
                              icon={variable.required ? <span style={{fontSize: '12px'}}>*</span> : undefined}
                            />
                          </Tooltip>
                        ))}
                        {extractVariables(editedTemplate).length === 0 && (
                          <Typography variant="body2" color="text.secondary" fontStyle="italic">
                            No variables detected. Use {"{variable_name}"} syntax to add variables.
                          </Typography>
                        )}
                      </Box>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)} startIcon={<CancelIcon />}>
            Cancel
          </Button>
          <Button
            onClick={handleSavePrompt}
            variant="contained"
            startIcon={<SaveIcon />}
            color="primary"
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* Preview Dialog */}
      <Dialog
        open={previewDialogOpen}
        onClose={() => setPreviewDialogOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          Preview Prompt: {selectedPrompt?.name}
        </DialogTitle>
        <DialogContent>
          {selectedPrompt && extractVariables(selectedPrompt.prompt_template || '').map((variable) => (
            <TextField
              key={variable.name}
              fullWidth
              label={variable.name}
              value={previewVariables[variable.name] || ''}
              onChange={(e) => setPreviewVariables(prev => ({
                ...prev,
                [variable.name]: e.target.value
              }))}
              variant="outlined"
              margin="normal"
            />
          ))}
          
          <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
            <Button
              onClick={handleGeneratePreview}
              variant="contained"
              startIcon={<CodeIcon />}
            >
              Generate Preview
            </Button>
          </Box>

          {previewResult && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Generated Prompt:
              </Typography>
              <Box
                component="pre"
                sx={{
                  backgroundColor: (theme) => theme.palette.mode === 'dark' ? theme.palette.grey[900] : theme.palette.grey[100],
                  color: (theme) => theme.palette.mode === 'dark' ? theme.palette.grey[100] : theme.palette.grey[900],
                  p: 2,
                  borderRadius: 1,
                  fontSize: '0.8rem',
                  maxHeight: 400,
                  overflow: 'auto',
                  whiteSpace: 'pre-wrap'
                }}
              >
                {previewResult}
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewDialogOpen(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default PromptManagement;