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
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid
} from '@mui/material';
import {
  Edit as EditIcon,
  Refresh as RefreshIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  ExpandMore as ExpandMoreIcon,
  Code as CodeIcon
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

const SimplePromptManagement: React.FC = () => {
  const { enqueueSnackbar } = useSnackbar();
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedPrompt, setSelectedPrompt] = useState<Prompt | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editedTemplate, setEditedTemplate] = useState('');

  // Hardcoded prompts data (fallback when API is down)
  const fallbackPrompts: Prompt[] = [
    {
      id: 'entity_discovery_default',
      name: 'Entity Discovery Prompt',
      description: 'Prompt for discovering new entity types from text',
      prompt_type: 'entity_discovery',
      version: 1,
      is_active: true,
      parameters: {
        variables: ['text', 'existing_types'],
        required: ['text'],
        confidence_threshold: 0.75
      },
      prompt_template: `You are an expert knowledge graph architect. Analyze the provided text and discover unique entity types beyond standard categories.

Text: {text}

Current accepted entity types: {existing_types}

Instructions:
1. Identify any unique entity types that don't fit standard categories
2. Provide a clear, singular label for each entity type
3. Include a brief description and 1-2 examples from the text
4. Rate confidence (0-1) based on clarity and frequency
5. Group similar entities under unified types

Return JSON format with discovered entities.`
    },
    {
      id: 'relationship_discovery_default',
      name: 'Relationship Discovery Prompt',
      description: 'Prompt for discovering new relationship types between entities',
      prompt_type: 'relationship_discovery',
      version: 1,
      is_active: true,
      parameters: {
        variables: ['entities', 'existing_types'],
        required: ['entities'],
        confidence_threshold: 0.7
      },
      prompt_template: `You are an expert knowledge graph relationship designer. Discover unique relationship types between entities.

Entities to analyze: {entities}

Current accepted relationship types: {existing_types}

Instructions:
1. Identify relationship types beyond standard ones
2. Use clear verb phrases
3. Provide inverse relationships where applicable
4. Rate confidence (0-1) based on semantic clarity

Return JSON format with discovered relationships.`
    },
    {
      id: 'knowledge_extraction_default',
      name: 'Knowledge Extraction Prompt',
      description: 'Main prompt for extracting entities and relationships from text',
      prompt_type: 'knowledge_extraction',
      version: 1,
      is_active: true,
      parameters: {
        variables: ['text', 'context_info', 'domain_guidance', 'entity_types', 'relationship_types'],
        required: ['text'],
        quality_guidelines: 'strict_entity_validation'
      },
      prompt_template: `You are an expert knowledge graph extraction system with dynamic schema discovery capabilities. Your task is to extract high-quality entities and relationships from the provided text.

{context_info}
{domain_guidance}

DYNAMIC SCHEMA:
Entity Types: {entity_types}
Relationship Types: {relationship_types}

Instructions:
1. Extract only proper nouns, names, specific concepts
2. Avoid generic terms, actions, or phrases
3. Use high confidence scores for clear entities
4. Create relationships only between valid entities

Return JSON format with entities, relationships, and discoveries.`
    }
  ];

  useEffect(() => {
    loadPrompts();
  }, []);

  const loadPrompts = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/v1/prompts/');
      if (response.ok) {
        const data = await response.json();
        setPrompts(data.prompts);
      } else {
        console.warn('API not available, using fallback prompts');
        setPrompts(fallbackPrompts);
      }
    } catch (error) {
      console.error('Error loading prompts:', error);
      setError('API unavailable, showing default prompts');
      setPrompts(fallbackPrompts);
    } finally {
      setLoading(false);
    }
  };

  const handleEditPrompt = (prompt: Prompt) => {
    setSelectedPrompt(prompt);
    setEditedTemplate(prompt.prompt_template || '');
    setEditDialogOpen(true);
  };

  const handleSavePrompt = async () => {
    if (!selectedPrompt) return;

    try {
      const response = await fetch(`/api/v1/prompts/${selectedPrompt.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt_template: editedTemplate }),
      });

      if (response.ok) {
        enqueueSnackbar('Prompt updated successfully', { variant: 'success' });
        setEditDialogOpen(false);
        setSelectedPrompt(null);
        loadPrompts(); // Reload to reflect changes
      } else {
        const errorData = await response.json();
        enqueueSnackbar(`Failed to update prompt: ${errorData.detail || 'Unknown error'}`, { variant: 'error' });
      }
    } catch (error) {
      console.error('Error updating prompt:', error);
      enqueueSnackbar('Failed to update prompt', { variant: 'error' });
    }
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
        <Button
          startIcon={<RefreshIcon />}
          onClick={loadPrompts}
          variant="outlined"
        >
          Refresh
        </Button>
      </Box>

      {error && (
        <Alert severity="warning" sx={{ mb: 2 }}>
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
                  </Box>
                </Box>
                <ListItemSecondaryAction>
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
              <Typography variant="body2" color="text.secondary" paragraph>
                Parameters: {JSON.stringify(prompt.parameters, null, 2)}
              </Typography>
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
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          Edit Prompt: {selectedPrompt?.name}
        </DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            multiline
            rows={20}
            value={editedTemplate}
            onChange={(e) => setEditedTemplate(e.target.value)}
            variant="outlined"
            label="Prompt Template"
            helperText="Use {variable_name} for template variables"
            sx={{ mt: 2 }}
          />
          {selectedPrompt && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Available Variables:
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {selectedPrompt.parameters?.variables?.join(', ') || 'text'}
              </Typography>
            </Box>
          )}
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
    </Box>
  );
};

export default SimplePromptManagement;