import React, { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Typography,
  Alert,
  Button,
  Chip,
  Paper,
  Grid,
  FormControlLabel,
  Switch,
  Divider,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Visibility as PreviewIcon,
  Code as VariablesIcon,
  CheckCircle as ValidIcon,
  Error as ErrorIcon,
  Help as HelpIcon
} from '@mui/icons-material';

interface Template {
  name: string;
  content: string;
  description: string;
  variables: string[];
  active: boolean;
  metadata?: {
    version: string;
    author: string;
    created: string;
    modified: string;
  };
}

interface SynthesisPromptEditorProps {
  template: Template;
  onChange: (template: Template) => void;
  onSave?: () => void;
  onReset?: () => void;
  onPreview?: (template: Template) => void;
  disabled?: boolean;
  availableVariables?: string[];
  category: 'synthesis_prompts' | 'formatting_templates' | 'system_behaviors';
}

const SynthesisPromptEditor: React.FC<SynthesisPromptEditorProps> = ({
  template,
  onChange,
  onSave,
  onReset,
  onPreview,
  disabled = false,
  availableVariables = [],
  category
}) => {
  const [validationError, setValidationError] = useState<string>('');
  const [isValid, setIsValid] = useState(true);
  const [wordCount, setWordCount] = useState(0);
  const [variableCount, setVariableCount] = useState(0);

  // Validate template content and extract variables
  const validateTemplate = (content: string) => {
    try {
      // Extract variables from template content using {variable_name} pattern
      const variableMatches = content.match(/\{([^}]+)\}/g) || [];
      const extractedVariables = variableMatches.map(match => match.slice(1, -1));
      
      // Check for unmatched braces
      const openBraces = (content.match(/\{/g) || []).length;
      const closeBraces = (content.match(/\}/g) || []).length;
      
      if (openBraces !== closeBraces) {
        throw new Error('Unmatched braces in template - ensure all {variables} are properly closed');
      }
      
      // Update word and variable counts
      const words = content.trim().split(/\s+/).filter(word => word.length > 0);
      setWordCount(words.length);
      setVariableCount(extractedVariables.length);
      
      // Update template variables
      const updatedTemplate = {
        ...template,
        content,
        variables: [...new Set(extractedVariables)] // Remove duplicates
      };
      onChange(updatedTemplate);
      
      setValidationError('');
      setIsValid(true);
      return true;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Invalid template syntax';
      setValidationError(errorMsg);
      setIsValid(false);
      return false;
    }
  };

  // Handle content changes with debounced validation
  const handleContentChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newContent = event.target.value;
    
    // Update template immediately for UI responsiveness
    const updatedTemplate = { ...template, content: newContent };
    onChange(updatedTemplate);
    
    // Debounced validation
    setTimeout(() => {
      validateTemplate(newContent);
    }, 300);
  };

  const handleMetadataChange = (field: keyof Template, value: any) => {
    const updatedTemplate = { ...template, [field]: value };
    onChange(updatedTemplate);
  };

  const insertVariable = (variableName: string) => {
    const textarea = document.getElementById('template-content') as HTMLTextAreaElement;
    if (textarea) {
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const currentContent = template.content;
      const newContent = currentContent.substring(0, start) + `{${variableName}}` + currentContent.substring(end);
      
      const updatedTemplate = { ...template, content: newContent };
      onChange(updatedTemplate);
      
      // Restore cursor position
      setTimeout(() => {
        textarea.focus();
        textarea.setSelectionRange(start + variableName.length + 2, start + variableName.length + 2);
      }, 0);
    }
  };

  const getTemplatePlaceholder = () => {
    switch (category) {
      case 'synthesis_prompts':
        return `# Synthesis Prompt Template

You are synthesizing information for: {topic}

Context from documents:
{document_content}

Key themes to address:
{key_themes}

Please provide a comprehensive synthesis that:
1. Integrates the main concepts from {source_count} sources
2. Addresses the user query: {user_query}
3. Maintains academic rigor and clarity

Target audience: {audience_level}
Desired length: {output_length}`;

      case 'formatting_templates':
        return `# Formatting Template

## {section_title}

**Summary:** {summary}

### Key Points:
{formatted_points}

**Sources:** {source_list}
**Generated on:** {timestamp}
**Confidence:** {confidence_score}`;

      case 'system_behaviors':
        return `# System Behavior Template

When processing requests of type: {request_type}

System should:
- Apply reasoning mode: {reasoning_mode}
- Use confidence threshold: {confidence_threshold}
- Include verification steps: {verification_required}

Quality controls:
- Fact-checking level: {fact_check_level}
- Citation requirements: {citation_style}

Fallback behavior: {fallback_strategy}`;

      default:
        return '# Enter your template content here\n\nUse {variable_name} for dynamic content substitution.';
    }
  };

  // Initialize validation on mount
  useEffect(() => {
    if (template.content) {
      validateTemplate(template.content);
    }
  }, []);

  return (
    <Box sx={{ width: '100%' }}>
      {/* Header with metadata */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Template Name"
              value={template.name || ''}
              onChange={(e) => handleMetadataChange('name', e.target.value)}
              disabled={disabled}
              size="small"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={template.active || false}
                  onChange={(e) => handleMetadataChange('active', e.target.checked)}
                  disabled={disabled}
                />
              }
              label="Active Template"
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Description"
              value={template.description || ''}
              onChange={(e) => handleMetadataChange('description', e.target.value)}
              disabled={disabled}
              size="small"
              multiline
              rows={2}
            />
          </Grid>
        </Grid>
      </Paper>

      {/* Template Editor */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <Typography variant="h6">Template Content</Typography>
          <Chip 
            icon={isValid ? <ValidIcon /> : <ErrorIcon />}
            label={isValid ? 'Valid Template' : 'Invalid Template'}
            color={isValid ? 'success' : 'error'}
            size="small"
          />
          <Chip 
            label={`${wordCount} words`}
            size="small"
            variant="outlined"
          />
          <Chip 
            label={`${variableCount} variables`}
            size="small"
            variant="outlined"
            icon={<VariablesIcon />}
          />
          <Box sx={{ ml: 'auto', display: 'flex', gap: 1 }}>
            <Tooltip title="Insert variables by clicking them in the panel below">
              <IconButton size="small">
                <HelpIcon />
              </IconButton>
            </Tooltip>
            {onPreview && (
              <Button
                size="small"
                startIcon={<PreviewIcon />}
                onClick={() => onPreview(template)}
                disabled={disabled || !isValid}
              >
                Preview
              </Button>
            )}
          </Box>
        </Box>

        {validationError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            <Typography variant="body2">{validationError}</Typography>
          </Alert>
        )}

        <TextField
          id="template-content"
          fullWidth
          multiline
          minRows={12}
          maxRows={25}
          value={template.content || ''}
          onChange={handleContentChange}
          disabled={disabled}
          placeholder={getTemplatePlaceholder()}
          sx={{
            '& .MuiInputBase-input': {
              fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
              fontSize: '14px',
              lineHeight: 1.6,
            },
            '& .MuiOutlinedInput-root': {
              backgroundColor: 'background.default',
            }
          }}
        />

        <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="caption" color="text.secondary">
            Use {'{variable_name}'} syntax for dynamic content. Variables will be extracted automatically.
          </Typography>
          {!isValid && (
            <Typography variant="caption" color="error">
              Please fix template syntax errors before saving
            </Typography>
          )}
        </Box>
      </Paper>

      {/* Variable References */}
      {availableVariables.length > 0 && (
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Available Variables
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
            Click to insert into template at cursor position
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {availableVariables.map((variable) => (
              <Chip
                key={variable}
                label={variable}
                size="small"
                clickable
                onClick={() => insertVariable(variable)}
                disabled={disabled}
                sx={{ 
                  cursor: 'pointer',
                  '&:hover': {
                    backgroundColor: 'primary.light',
                    color: 'primary.contrastText'
                  }
                }}
              />
            ))}
          </Box>
        </Paper>
      )}

      {/* Used Variables */}
      {template.variables && template.variables.length > 0 && (
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Variables Used in Template
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {template.variables.map((variable) => (
              <Chip
                key={variable}
                label={variable}
                size="small"
                color="primary"
                variant="outlined"
              />
            ))}
          </Box>
        </Paper>
      )}

      {/* Action Buttons */}
      <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
        {onReset && (
          <Button
            startIcon={<RefreshIcon />}
            onClick={onReset}
            disabled={disabled}
          >
            Reset
          </Button>
        )}
        {onSave && (
          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            onClick={onSave}
            disabled={disabled || !isValid || !template.name?.trim()}
          >
            Save Template
          </Button>
        )}
      </Box>
    </Box>
  );
};

export default SynthesisPromptEditor;