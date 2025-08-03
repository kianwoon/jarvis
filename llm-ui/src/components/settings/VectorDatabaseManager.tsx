import React, { useState } from 'react';
import {
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Card,
  CardContent,
  CardActions,
  Typography,
  Chip,
  Switch,
  FormControlLabel,
  Alert,
  Tooltip,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  ToggleButton,
  ToggleButtonGroup
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  ContentCopy as CopyIcon,
  Check as CheckIcon,
  ViewModule as CardViewIcon,
  TableChart as TableViewIcon
} from '@mui/icons-material';

interface VectorDatabase {
  id: string;
  name: string;
  enabled: boolean;
  config: Record<string, any>;
}

interface VectorDatabaseManagerProps {
  data: {
    active: string;
    databases?: VectorDatabase[];
    // Legacy format support
    [key: string]: any;
  };
  onChange: (data: any) => void;
  embeddingModel?: string;
  embeddingEndpoint?: string;
  onEmbeddingChange?: (field: string, value: string) => void;
}

// Database templates
const DATABASE_TEMPLATES: Record<string, { name: string; config: Record<string, any> }> = {
  milvus: {
    name: 'Milvus',
    config: {
      MILVUS_URI: 'http://localhost:19530',
      MILVUS_TOKEN: '',
      collection: 'default_knowledge',
      dimension: 1536
    }
  },
  qdrant: {
    name: 'Qdrant',
    config: {
      QDRANT_HOST: 'localhost',
      QDRANT_PORT: 6333,
      api_key: '',
      collection: 'default_knowledge',
      dimension: 1536
    }
  },
  pinecone: {
    name: 'Pinecone',
    config: {
      api_key: '',
      environment: 'us-east-1',
      index_name: 'default-index',
      dimension: 1536
    }
  },
  weaviate: {
    name: 'Weaviate',
    config: {
      url: 'http://localhost:8080',
      api_key: '',
      class_name: 'Document',
      dimension: 1536
    }
  },
  chroma: {
    name: 'Chroma',
    config: {
      host: 'localhost',
      port: 8000,
      collection: 'default_collection',
      dimension: 1536
    }
  },
  custom: {
    name: 'Custom Database',
    config: {
      endpoint: '',
      auth_type: 'api_key',
      auth_value: '',
      collection: 'default'
    }
  }
};

const VectorDatabaseManager: React.FC<VectorDatabaseManagerProps> = ({ data, onChange, embeddingModel, embeddingEndpoint, onEmbeddingChange }) => {
  const [editDialog, setEditDialog] = useState<{ open: boolean; database?: VectorDatabase; index?: number }>({ open: false });
  const [addDialog, setAddDialog] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [jsonError, setJsonError] = useState('');
  const [copied, setCopied] = useState<string | null>(null);

  // Convert legacy format to new format if needed
  const getDatabases = (): VectorDatabase[] => {
    if (data.databases) {
      return data.databases;
    }

    // Convert legacy format (milvus: {}, qdrant: {}, etc.) to new format
    const databases: VectorDatabase[] = [];
    const knownDbs = ['milvus', 'qdrant', 'pinecone', 'weaviate', 'chroma'];
    
    knownDbs.forEach(dbId => {
      if (data[dbId] && typeof data[dbId] === 'object') {
        const dbConfig = { ...data[dbId] };
        // Remove status from config if it exists
        const status = dbConfig.status;
        delete dbConfig.status;
        
        databases.push({
          id: dbId,
          name: DATABASE_TEMPLATES[dbId]?.name || dbId.charAt(0).toUpperCase() + dbId.slice(1),
          // If this is the active database, it MUST be enabled
          // Otherwise, check for explicit status field
          enabled: data.active === dbId ? true : (status === true),
          config: dbConfig
        });
      }
    });

    return databases;
  };

  const databases = getDatabases();

  const handleActiveChange = (newActive: string) => {
    onChange({
      ...data,
      active: newActive,
      databases
    });
  };

  const handleDatabaseToggle = (index: number) => {
    const updated = [...databases];
    const database = updated[index];
    database.enabled = !database.enabled;
    
    // If enabling a database, make it active
    // If disabling the active database, switch to another enabled one
    let newActive = data.active;
    if (database.enabled) {
      newActive = database.id;
    } else if (database.id === data.active) {
      // Find another enabled database
      const otherEnabled = updated.find(db => db.id !== database.id && db.enabled);
      newActive = otherEnabled ? otherEnabled.id : '';
    }
    
    onChange({
      ...data,
      active: newActive,
      databases: updated
    });
  };

  const handleEdit = (database: VectorDatabase, index: number) => {
    setEditDialog({ open: true, database: { ...database }, index });
    setJsonError('');
  };

  const handleAdd = () => {
    setAddDialog(true);
    setSelectedTemplate('');
    setJsonError('');
  };

  const handleDelete = (index: number) => {
    if (confirm('Are you sure you want to delete this database configuration?')) {
      const updated = databases.filter((_, i) => i !== index);
      onChange({
        ...data,
        databases: updated,
        // If we deleted the active database, switch to the first enabled one
        active: data.active === databases[index].id && updated.length > 0 
          ? updated.find(db => db.enabled)?.id || updated[0].id 
          : data.active
      });
    }
  };

  const handleSaveEdit = () => {
    if (!editDialog.database) return;

    try {
      // Validate JSON
      const configStr = JSON.stringify(editDialog.database.config);
      JSON.parse(configStr);

      const updated = [...databases];
      updated[editDialog.index!] = editDialog.database;
      
      onChange({
        ...data,
        databases: updated
      });
      
      setEditDialog({ open: false });
    } catch (error) {
      setJsonError('Invalid configuration');
    }
  };

  const handleSaveAdd = (newDb: VectorDatabase) => {
    const updated = [...databases, newDb];
    onChange({
      ...data,
      databases: updated,
      // If this is the first database, make it active
      active: databases.length === 0 ? newDb.id : data.active
    });
    setAddDialog(false);
  };

  const handleCopyConfig = (config: Record<string, any>) => {
    navigator.clipboard.writeText(JSON.stringify(config, null, 2));
    setCopied(JSON.stringify(config));
    setTimeout(() => setCopied(null), 2000);
  };

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Vector Database Configuration
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleAdd}
            size="small"
          >
            Add Database
          </Button>
        </Box>
      </Box>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {databases.map((database, index) => (
            <Card 
            key={database.id} 
            variant="outlined"
            sx={{ 
              borderColor: data.active === database.id ? 'primary.main' : 'divider',
              borderWidth: data.active === database.id ? 2 : 1,
              opacity: database.enabled ? 1 : 0.6,
              backgroundColor: (theme) => data.active === database.id 
                ? theme.palette.mode === 'dark' 
                  ? 'rgba(33, 150, 243, 0.08)' 
                  : 'rgba(33, 150, 243, 0.04)'
                : 'background.paper'
            }}
          >
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="h6">
                    {database.name}
                  </Typography>
                  {data.active === database.id && (
                    <Chip 
                      label="Active" 
                      color="primary" 
                      size="small"
                      icon={<CheckIcon />}
                    />
                  )}
                </Box>
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={database.enabled}
                      onChange={() => handleDatabaseToggle(index)}
                    />
                  }
                  label={database.enabled ? 'Enabled' : 'Disabled'}
                />
              </Box>

              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                Configuration:
              </Typography>
              
              <Box 
                sx={{ 
                  backgroundColor: (theme) => theme.palette.mode === 'dark' ? theme.palette.grey[900] : theme.palette.grey[100],
                  color: (theme) => theme.palette.mode === 'dark' ? theme.palette.grey[100] : theme.palette.grey[900],
                  p: 1,
                  borderRadius: 1,
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                  overflow: 'auto',
                  maxHeight: 150,
                  border: 1,
                  borderColor: 'divider'
                }}
              >
                <pre style={{ margin: 0 }}>
                  {JSON.stringify(database.config, null, 2)}
                </pre>
              </Box>
            </CardContent>
            
            <CardActions>
              <Button
                size="small"
                startIcon={<EditIcon />}
                onClick={() => handleEdit(database, index)}
              >
                Edit
              </Button>
              <Button
                size="small"
                startIcon={<CopyIcon />}
                onClick={() => handleCopyConfig(database.config)}
              >
                {copied === JSON.stringify(database.config) ? 'Copied!' : 'Copy'}
              </Button>
              <Button
                size="small"
                color="error"
                startIcon={<DeleteIcon />}
                onClick={() => handleDelete(index)}
              >
                Delete
              </Button>
            </CardActions>
          </Card>
        ))}
        
        {/* Embedding Configuration Card */}
        {(embeddingModel !== undefined || embeddingEndpoint !== undefined) && onEmbeddingChange && (
          <Card 
            variant="outlined"
            sx={{ 
              borderColor: 'divider',
              borderWidth: 1
            }}
          >
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Embedding Configuration
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  label="Embedding Model"
                  value={embeddingModel || ''}
                  onChange={(e) => onEmbeddingChange('embedding_model', e.target.value)}
                  fullWidth
                  placeholder="e.g., BAAI/bge-base-en-v1.5"
                />
                <TextField
                  label="Embedding Endpoint"
                  value={embeddingEndpoint || ''}
                  onChange={(e) => onEmbeddingChange('embedding_endpoint', e.target.value)}
                  fullWidth
                  placeholder="e.g., http://localhost:8080/embed"
                />
              </Box>
            </CardContent>
          </Card>
        )}
      </Box>

      {/* Edit Dialog */}
      <Dialog 
        open={editDialog.open} 
        onClose={() => setEditDialog({ open: false })}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Edit {editDialog.database?.name} Configuration
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="Database Name"
              value={editDialog.database?.name || ''}
              onChange={(e) => setEditDialog({
                ...editDialog,
                database: { ...editDialog.database!, name: e.target.value }
              })}
              fullWidth
            />
            
            <Box>
              <Typography variant="body2" sx={{ mb: 1 }}>
                Configuration (JSON):
              </Typography>
              <TextField
                multiline
                rows={10}
                value={JSON.stringify(editDialog.database?.config || {}, null, 2)}
                onChange={(e) => {
                  try {
                    const config = JSON.parse(e.target.value);
                    setEditDialog({
                      ...editDialog,
                      database: { ...editDialog.database!, config }
                    });
                    setJsonError('');
                  } catch (error) {
                    setJsonError('Invalid JSON format');
                  }
                }}
                fullWidth
                error={!!jsonError}
                helperText={jsonError}
                sx={{ fontFamily: 'monospace' }}
              />
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialog({ open: false })}>
            Cancel
          </Button>
          <Button 
            onClick={handleSaveEdit} 
            variant="contained"
            disabled={!!jsonError || !editDialog.database?.name}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* Add Dialog */}
      <AddDatabaseDialog
        open={addDialog}
        onClose={() => setAddDialog(false)}
        onSave={handleSaveAdd}
        existingIds={databases.map(db => db.id)}
      />
    </Box>
  );
};

// Add Database Dialog Component
const AddDatabaseDialog: React.FC<{
  open: boolean;
  onClose: () => void;
  onSave: (database: VectorDatabase) => void;
  existingIds: string[];
}> = ({ open, onClose, onSave, existingIds }) => {
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [database, setDatabase] = useState<VectorDatabase>({
    id: '',
    name: '',
    enabled: true,
    config: {}
  });
  const [jsonError, setJsonError] = useState('');

  const handleTemplateChange = (template: string) => {
    setSelectedTemplate(template);
    const tmpl = DATABASE_TEMPLATES[template];
    if (tmpl) {
      setDatabase({
        id: template,
        name: tmpl.name,
        enabled: true,
        config: { ...tmpl.config }
      });
    }
  };

  const handleSave = () => {
    if (!database.id || !database.name) {
      setJsonError('ID and Name are required');
      return;
    }

    if (existingIds.includes(database.id)) {
      setJsonError('A database with this ID already exists');
      return;
    }

    onSave(database);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Add Vector Database</DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
          <FormControl fullWidth>
            <InputLabel>Database Template</InputLabel>
            <Select
              value={selectedTemplate}
              onChange={(e) => handleTemplateChange(e.target.value)}
              label="Database Template"
            >
              <MenuItem value="">Custom</MenuItem>
              <Divider />
              {Object.entries(DATABASE_TEMPLATES).map(([key, template]) => (
                <MenuItem key={key} value={key}>
                  {template.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <TextField
            label="Database ID"
            value={database.id}
            onChange={(e) => setDatabase({ ...database, id: e.target.value.toLowerCase().replace(/\s+/g, '_') })}
            fullWidth
            required
            helperText="Unique identifier (lowercase, no spaces)"
          />

          <TextField
            label="Database Name"
            value={database.name}
            onChange={(e) => setDatabase({ ...database, name: e.target.value })}
            fullWidth
            required
          />

          <Box>
            <Typography variant="body2" sx={{ mb: 1 }}>
              Configuration (JSON):
            </Typography>
            <TextField
              multiline
              rows={10}
              value={JSON.stringify(database.config, null, 2)}
              onChange={(e) => {
                try {
                  const config = JSON.parse(e.target.value);
                  setDatabase({ ...database, config });
                  setJsonError('');
                } catch (error) {
                  setJsonError('Invalid JSON format');
                }
              }}
              fullWidth
              error={!!jsonError}
              helperText={jsonError}
              sx={{ fontFamily: 'monospace' }}
            />
          </Box>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={!!jsonError || !database.id || !database.name}
        >
          Add Database
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default VectorDatabaseManager;