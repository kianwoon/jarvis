import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Button,
  ToggleButton,
  ToggleButtonGroup,
  TextField,
  InputAdornment,
  Grid,
  Alert,
  CircularProgress
} from '@mui/material';
import {
  Add as AddIcon,
  ViewModule as CardViewIcon,
  ViewList as TableViewIcon,
  Search as SearchIcon
} from '@mui/icons-material';
import WorkflowCard from './WorkflowCard';
import WorkflowTable from './WorkflowTable';

interface WorkflowData {
  id: string;
  name: string;
  description: string;
  langflow_config: any;
  is_active: boolean;
  created_by: string;
  created_at: string;
  updated_at: string;
}

interface WorkflowLandingProps {
  onCreateNew: () => void;
  onEditWorkflow: (workflow: WorkflowData) => void;
  refreshTrigger?: number;
}

const WorkflowLanding: React.FC<WorkflowLandingProps> = ({
  onCreateNew,
  onEditWorkflow,
  refreshTrigger
}) => {
  const [workflows, setWorkflows] = useState<WorkflowData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'card' | 'table'>('card');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    loadWorkflows();
  }, [refreshTrigger]);

  const loadWorkflows = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/automation/workflows');
      if (!response.ok) {
        throw new Error(`Failed to load workflows: ${response.statusText}`);
      }

      const data = await response.json();
      const workflowList = Array.isArray(data) ? data : data.workflows || [];
      
      // Sort workflows by updated_at in descending order (newest first)
      const sortedWorkflows = workflowList.sort((a: WorkflowData, b: WorkflowData) => {
        const dateA = new Date(a.updated_at).getTime();
        const dateB = new Date(b.updated_at).getTime();
        return dateB - dateA; // Descending order (newest first)
      });
      
      setWorkflows(sortedWorkflows);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load workflows');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteWorkflow = async (workflowId: string) => {
    if (!confirm('Are you sure you want to delete this workflow?')) {
      return;
    }

    try {
      const response = await fetch(`http://127.0.0.1:8000/api/v1/automation/workflows/${workflowId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error(`Failed to delete workflow: ${response.statusText}`);
      }

      // Refresh the list
      await loadWorkflows();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete workflow');
    }
  };

  const handleDuplicateWorkflow = async (workflow: WorkflowData) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/automation/workflows', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: `${workflow.name} (Copy)`,
          description: workflow.description,
          langflow_config: workflow.langflow_config,
          is_active: workflow.is_active,
          created_by: 'user'
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to duplicate workflow: ${response.statusText}`);
      }

      // Refresh the list
      await loadWorkflows();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to duplicate workflow');
    }
  };

  const handleToggleActive = async (workflow: WorkflowData) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/v1/automation/workflows/${workflow.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...workflow,
          is_active: !workflow.is_active
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to update workflow: ${response.statusText}`);
      }

      // Refresh the list
      await loadWorkflows();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update workflow');
    }
  };

  const handleExecuteWorkflow = async (workflowId: string) => {
    // This could open a dialog or navigate to an execution view
    alert(`Execute workflow ${workflowId} - Implementation pending`);
  };

  const filteredWorkflows = workflows.filter(workflow =>
    workflow.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    workflow.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getWorkflowStats = (config: any) => {
    try {
      const nodeCount = config?.nodes?.length || 0;
      const edgeCount = config?.edges?.length || 0;
      const nodeTypes = config?.metadata?.node_types || [];
      const hasCache = config?.metadata?.has_cache_nodes || false;
      
      return { nodeCount, edgeCount, nodeTypes, hasCache };
    } catch (error) {
      //console.error('Error in getWorkflowStats:', error);
      return { nodeCount: 0, edgeCount: 0, nodeTypes: [], hasCache: false };
    }
  };

  return (
    <Container maxWidth={false} sx={{ py: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Workflow Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={onCreateNew}
          size="large"
        >
          New Workflow
        </Button>
      </Box>

      {/* Controls */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center' }}>
        <TextField
          placeholder="Search workflows..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          sx={{ flex: 1, maxWidth: 400 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
        />
        
        <Button
          variant="outlined"
          onClick={loadWorkflows}
          disabled={loading}
          sx={{ minWidth: 120 }}
        >
          {loading ? <CircularProgress size={20} /> : 'Refresh'}
        </Button>
        
        <Button
          variant="outlined" 
          color="secondary"
          onClick={() => window.location.reload()}
          sx={{ minWidth: 140 }}
        >
          Force Reload
        </Button>
        
        <ToggleButtonGroup
          value={viewMode}
          exclusive
          onChange={(_, value) => value && setViewMode(value)}
          aria-label="view mode"
        >
          <ToggleButton value="card" aria-label="card view">
            <CardViewIcon />
          </ToggleButton>
          <ToggleButton value="table" aria-label="table view">
            <TableViewIcon />
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Content */}
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress />
        </Box>
      ) : filteredWorkflows.length === 0 ? (
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            {searchQuery ? 'No workflows found matching your search' : 'No workflows created yet'}
          </Typography>
          {!searchQuery && (
            <Button
              variant="outlined"
              startIcon={<AddIcon />}
              onClick={onCreateNew}
              sx={{ mt: 2 }}
            >
              Create Your First Workflow
            </Button>
          )}
        </Box>
      ) : viewMode === 'card' ? (
        <Grid container spacing={3}>
          {filteredWorkflows.map((workflow) => {
            const stats = getWorkflowStats(workflow.langflow_config);
            return (
              <Grid item xs={12} sm={6} md={4} key={workflow.id}>
                <WorkflowCard
                  workflow={workflow}
                  stats={stats}
                  onEdit={() => onEditWorkflow(workflow)}
                  onDelete={() => handleDeleteWorkflow(workflow.id)}
                  onDuplicate={() => handleDuplicateWorkflow(workflow)}
                  onToggleActive={() => handleToggleActive(workflow)}
                  onExecute={() => handleExecuteWorkflow(workflow.id)}
                />
              </Grid>
            );
          })}
        </Grid>
      ) : (
        <WorkflowTable
          workflows={filteredWorkflows}
          onEdit={onEditWorkflow}
          onDelete={handleDeleteWorkflow}
          onDuplicate={handleDuplicateWorkflow}
          onToggleActive={handleToggleActive}
          onExecute={handleExecuteWorkflow}
        />
      )}
    </Container>
  );
};

export default WorkflowLanding;