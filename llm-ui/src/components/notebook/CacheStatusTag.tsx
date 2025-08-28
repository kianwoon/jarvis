import React, { useState } from 'react';
import {
  Chip,
  Tooltip,
  CircularProgress,
  Box,
  Typography,
  Fade
} from '@mui/material';
import {
  AccessTime as AccessTimeIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import { CacheStatus } from './NotebookAPI';

interface CacheStatusTagProps {
  cacheStatus: CacheStatus & {
    original_query?: string;
    cache_age_human?: string;
  };
  onClearCache: (notebookId: string, conversationId: string) => Promise<void>;
  loading?: boolean;
  notebookId: string;
  conversationId: string;
}

const CacheStatusTag: React.FC<CacheStatusTagProps> = ({
  cacheStatus,
  onClearCache,
  loading = false,
  notebookId,
  conversationId
}) => {
  const [isDeleting, setIsDeleting] = useState(false);

  // Don't render if cache doesn't exist
  if (!cacheStatus?.cache_exists) {
    return null;
  }

  const handleClearCache = async (event: React.MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    
    if (isDeleting) return;
    
    setIsDeleting(true);
    try {
      await onClearCache(notebookId, conversationId);
    } finally {
      setIsDeleting(false);
    }
  };

  // Calculate cache age display
  const getCacheAgeDisplay = () => {
    if (cacheStatus.cache_age_human) {
      return cacheStatus.cache_age_human;
    }
    
    if (cacheStatus.created_at) {
      const createdAt = new Date(cacheStatus.created_at);
      const now = new Date();
      const diffMinutes = Math.floor((now.getTime() - createdAt.getTime()) / (1000 * 60));
      
      if (diffMinutes < 1) return 'just now';
      if (diffMinutes < 60) return `${diffMinutes} min ago`;
      
      const diffHours = Math.floor(diffMinutes / 60);
      if (diffHours < 24) return `${diffHours}h ago`;
      
      const diffDays = Math.floor(diffHours / 24);
      return `${diffDays}d ago`;
    }
    
    return 'cached';
  };

  // Truncate query text for display
  const getTruncatedQuery = () => {
    const query = cacheStatus.original_query || '';
    return query.length > 50 ? `${query.substring(0, 47)}...` : query;
  };

  // Create tooltip content
  const tooltipContent = (
    <Box sx={{ p: 1 }}>
      <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>
        Cached Response Available
      </Typography>
      {cacheStatus.original_query && (
        <Typography variant="body2" sx={{ mb: 1 }}>
          <strong>Query:</strong> {cacheStatus.original_query}
        </Typography>
      )}
      {cacheStatus.created_at && (
        <Typography variant="body2" sx={{ mb: 1 }}>
          <strong>Cached:</strong> {new Date(cacheStatus.created_at).toLocaleString()}
        </Typography>
      )}
      {cacheStatus.last_accessed && (
        <Typography variant="body2" sx={{ mb: 1 }}>
          <strong>Last accessed:</strong> {new Date(cacheStatus.last_accessed).toLocaleString()}
        </Typography>
      )}
      {cacheStatus.ttl_remaining && (
        <Typography variant="body2">
          <strong>Expires in:</strong> {Math.floor(cacheStatus.ttl_remaining / 60)} minutes
        </Typography>
      )}
      <Typography variant="caption" sx={{ mt: 1, display: 'block', fontStyle: 'italic' }}>
        Click × to clear cache and get fresh results
      </Typography>
    </Box>
  );

  return (
    <Fade in={true} timeout={300}>
      <Box
        sx={{
          display: 'inline-flex',
          mb: 1,
          maxWidth: '100%'
        }}
      >
        <Tooltip 
          title={tooltipContent} 
          arrow 
          placement="top"
          componentsProps={{
            tooltip: {
              sx: {
                maxWidth: 300,
                backgroundColor: 'background.paper',
                color: 'text.primary',
                border: 1,
                borderColor: 'divider',
                boxShadow: 3
              }
            }
          }}
        >
          <Chip
            icon={<AccessTimeIcon />}
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Typography variant="caption" sx={{ fontWeight: 500 }}>
                  {getTruncatedQuery() && `"${getTruncatedQuery()}" • `}
                  {getCacheAgeDisplay()}
                </Typography>
              </Box>
            }
            deleteIcon={
              isDeleting || loading ? (
                <CircularProgress 
                  size={16} 
                  sx={{ color: 'warning.contrastText' }}
                />
              ) : (
                <CloseIcon />
              )
            }
            onDelete={handleClearCache}
            variant="filled"
            color="warning"
            size="small"
            sx={{
              maxWidth: '100%',
              '& .MuiChip-label': {
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                maxWidth: 'calc(100% - 60px)' // Account for icon and delete button
              },
              '& .MuiChip-deleteIcon': {
                color: 'warning.contrastText',
                '&:hover': {
                  color: 'warning.contrastText',
                  backgroundColor: 'rgba(255, 255, 255, 0.1)'
                }
              },
              '&:hover': {
                backgroundColor: 'warning.dark'
              },
              // Focus visible for keyboard navigation
              '&:focus-visible': {
                outline: 2,
                outlineStyle: 'solid',
                outlineColor: 'warning.main',
                outlineOffset: 2
              },
              // Responsive design
              '@media (max-width: 600px)': {
                maxWidth: '280px',
                '& .MuiChip-label': {
                  maxWidth: 'calc(100% - 50px)'
                }
              }
            }}
            // Accessibility props
            role="button"
            tabIndex={0}
            aria-label={`Cached response available from ${getCacheAgeDisplay()}. ${cacheStatus.original_query ? `Original query: ${cacheStatus.original_query}. ` : ''}Press delete to clear cache.`}
            aria-describedby="cache-status-tooltip"
          />
        </Tooltip>
      </Box>
    </Fade>
  );
};

export default CacheStatusTag;