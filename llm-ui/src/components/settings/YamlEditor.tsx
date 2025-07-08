import React, { useState } from 'react';
import {
  Box,
  TextField,
  Typography,
  Alert,
  Button,
  Chip,
  Paper
} from '@mui/material';
import {
  Description as YamlIcon,
  CheckCircle as ValidIcon,
  Error as ErrorIcon
} from '@mui/icons-material';

interface YamlEditorProps {
  value: string;
  onChange: (value: string) => void;
  label?: string;
  disabled?: boolean;
  minRows?: number;
  maxRows?: number;
}

const YamlEditor: React.FC<YamlEditorProps> = ({
  value,
  onChange,
  label = 'YAML Configuration',
  disabled = false,
  minRows = 10,
  maxRows = 30
}) => {
  const [yamlError, setYamlError] = useState<string>('');
  const [isValid, setIsValid] = useState(true);

  const validateYaml = (yamlText: string) => {
    try {
      // Basic YAML validation - check for common syntax errors
      const lines = yamlText.split('\n');
      let indentStack: number[] = [];
      
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();
        
        // Skip empty lines and comments
        if (!trimmed || trimmed.startsWith('#')) continue;
        
        // Check indentation consistency
        const indent = line.search(/\S/);
        if (indent === -1) continue;
        
        // Basic bracket/quote matching
        const openBrackets = (line.match(/\[/g) || []).length;
        const closeBrackets = (line.match(/\]/g) || []).length;
        const openQuotes = (line.match(/"/g) || []).length;
        
        if (openBrackets !== closeBrackets) {
          throw new Error(`Line ${i + 1}: Unmatched brackets`);
        }
        
        if (openQuotes % 2 !== 0) {
          throw new Error(`Line ${i + 1}: Unmatched quotes`);
        }
      }
      
      setYamlError('');
      setIsValid(true);
      return true;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Invalid YAML syntax';
      setYamlError(errorMsg);
      setIsValid(false);
      return false;
    }
  };

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = event.target.value;
    onChange(newValue);
    
    // Debounced validation
    setTimeout(() => {
      validateYaml(newValue);
    }, 500);
  };

  const formatYaml = () => {
    try {
      // Basic YAML formatting - normalize indentation
      const lines = value.split('\n');
      const formatted = lines.map(line => {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) return line;
        
        // Count the logical indentation level
        const colonIndex = line.indexOf(':');
        const dashIndex = line.indexOf('-');
        
        if (colonIndex > -1 && (dashIndex === -1 || colonIndex < dashIndex)) {
          // This is a key-value pair
          const key = line.substring(0, colonIndex + 1).trim();
          const value = line.substring(colonIndex + 1).trim();
          return `  ${key}${value ? ` ${value}` : ''}`;
        } else if (dashIndex > -1) {
          // This is a list item
          const item = line.substring(dashIndex).trim();
          return `  ${item}`;
        }
        
        return line;
      }).join('\n');
      
      onChange(formatted);
      validateYaml(formatted);
    } catch (error) {
      setYamlError('Could not format YAML: ' + (error instanceof Error ? error.message : 'Unknown error'));
    }
  };

  const getLineCount = () => {
    return value.split('\n').length;
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <YamlIcon color="primary" />
        <Typography variant="subtitle1">{label}</Typography>
        <Chip 
          icon={isValid ? <ValidIcon /> : <ErrorIcon />}
          label={isValid ? 'Valid YAML' : 'Invalid YAML'}
          color={isValid ? 'success' : 'error'}
          size="small"
        />
        <Chip 
          label={`${getLineCount()} lines`}
          size="small"
          variant="outlined"
        />
        <Button 
          size="small"
          onClick={formatYaml}
          disabled={disabled}
        >
          Format
        </Button>
      </Box>

      {yamlError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          <Typography variant="body2">{yamlError}</Typography>
        </Alert>
      )}

      <Paper variant="outlined" sx={{ p: 1, maxHeight: '70vh', overflow: 'auto' }}>
        <TextField
          fullWidth
          multiline
          minRows={minRows}
          maxRows={maxRows}
          value={value}
          onChange={handleChange}
          disabled={disabled}
          variant="outlined"
          placeholder="# Enter YAML configuration here&#10;example_key: example_value&#10;nested:&#10;  key: value&#10;  list:&#10;    - item1&#10;    - item2"
          sx={{
            '& .MuiInputBase-input': {
              fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
              fontSize: '14px',
              lineHeight: 1.5,
            },
            '& .MuiOutlinedInput-root': {
              backgroundColor: 'background.default',
              maxHeight: '60vh',
              overflow: 'auto',
            }
          }}
        />
      </Paper>

      <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="caption" color="text.secondary">
          YAML format - Use proper indentation (2 spaces recommended)
        </Typography>
        {!isValid && (
          <Typography variant="caption" color="error">
            Please fix YAML syntax errors before saving
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default YamlEditor;