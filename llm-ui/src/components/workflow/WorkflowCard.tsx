import React from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  IconButton,
  Box,
  Chip,
  Tooltip,
  Menu,
  MenuItem
} from '@mui/material';
import {
  Edit as EditIcon,
  Delete as DeleteIcon,
  MoreVert as MoreIcon,
  ContentCopy as CopyIcon,
  PowerSettingsNew as PowerIcon,
  AccountTree as NodeIcon,
  Link as LinkIcon,
  Storage as CacheIcon
} from '@mui/icons-material';

interface WorkflowCardProps {
  workflow: {
    id: string;
    name: string;
    description: string;
    langflow_config: any;
    is_active: boolean;
    created_at: string;
    updated_at: string;
  };
  stats: {
    nodeCount: number;
    edgeCount: number;
    nodeTypes: string[];
    hasCache: boolean;
  };
  onEdit: () => void;
  onDelete: () => void;
  onDuplicate: () => void;
  onToggleActive: () => void;
}

const WorkflowCard: React.FC<WorkflowCardProps> = ({
  workflow,
  stats,
  onEdit,
  onDelete,
  onDuplicate,
  onToggleActive
}) => {
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getRelativeTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
      if (diffHours === 0) {
        const diffMinutes = Math.floor(diffMs / (1000 * 60));
        return diffMinutes === 0 ? 'Just now' : `${diffMinutes}m ago`;
      }
      return `${diffHours}h ago`;
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return formatDate(dateString);
    }
  };

  return (
    <Card 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        transition: 'all 0.3s ease',
        '&:hover': {
          boxShadow: 6,
          transform: 'translateY(-2px)'
        }
      }}
    >
      <CardContent sx={{ flex: 1 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" component="h2" gutterBottom>
              {workflow.name}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
              <Chip
                label={workflow.is_active ? 'Active' : 'Inactive'}
                color={workflow.is_active ? 'success' : 'default'}
                size="small"
                icon={<PowerIcon />}
              />
              {stats.hasCache && (
                <Chip
                  label="Cached"
                  color="info"
                  size="small"
                  icon={<CacheIcon />}
                />
              )}
            </Box>
          </Box>
          <IconButton size="small" onClick={handleMenuOpen}>
            <MoreIcon />
          </IconButton>
        </Box>

        {/* Description */}
        <Typography 
          variant="body2" 
          color="text.secondary" 
          sx={{ 
            mb: 2,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            minHeight: '2.5em'
          }}
        >
          {workflow.description || 'No description provided'}
        </Typography>

        {/* Stats */}
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <NodeIcon fontSize="small" color="action" />
            <Typography variant="body2" color="text.secondary">
              {stats.nodeCount} nodes
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <LinkIcon fontSize="small" color="action" />
            <Typography variant="body2" color="text.secondary">
              {stats.edgeCount} connections
            </Typography>
          </Box>
        </Box>

        {/* Timestamps */}
        <Typography variant="caption" color="text.secondary" display="block">
          Updated: {getRelativeTime(workflow.updated_at)}
        </Typography>
      </CardContent>

      <CardActions sx={{ px: 2, pb: 2 }}>
        <Button
          startIcon={<EditIcon />}
          onClick={onEdit}
          size="small"
          fullWidth
          variant="outlined"
        >
          Edit Workflow
        </Button>
      </CardActions>

      {/* More Options Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem 
          onClick={() => {
            onToggleActive();
            handleMenuClose();
          }}
        >
          <PowerIcon sx={{ mr: 1 }} fontSize="small" />
          {workflow.is_active ? 'Deactivate' : 'Activate'}
        </MenuItem>
        <MenuItem 
          onClick={() => {
            onDuplicate();
            handleMenuClose();
          }}
        >
          <CopyIcon sx={{ mr: 1 }} fontSize="small" />
          Duplicate
        </MenuItem>
        <MenuItem 
          onClick={() => {
            onDelete();
            handleMenuClose();
          }}
          sx={{ color: 'error.main' }}
        >
          <DeleteIcon sx={{ mr: 1 }} fontSize="small" />
          Delete
        </MenuItem>
      </Menu>
    </Card>
  );
};

export default WorkflowCard;