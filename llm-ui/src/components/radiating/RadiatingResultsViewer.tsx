import React, { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  Button,
  TextField,
  InputAdornment,
  Grid,
  Divider,
  Tooltip,
  Badge,
  Alert,
  Menu,
  MenuItem,
  Fade
} from '@mui/material';
import {
  ExpandMore as ExpandIcon,
  Hub as EntityIcon,
  Link as RelationshipIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  Sort as SortIcon,
  GetApp as ExportIcon,
  Visibility as ViewIcon,
  Star as RelevanceIcon,
  Timeline as PathIcon,
  AccountTree as TreeIcon,
  Circle as NodeIcon,
  ArrowForward as ArrowIcon,
  MoreVert as MoreIcon,
  ContentCopy as CopyIcon,
  OpenInNew as OpenIcon
} from '@mui/icons-material';
import { 
  RadiatingResults,
  RadiatingEntity,
  RadiatingRelationship,
  RadiatingExportRequest 
} from '../../types/radiating';

interface RadiatingResultsViewerProps {
  results: RadiatingResults;
  onEntityClick?: (entity: RadiatingEntity) => void;
  onRelationshipClick?: (relationship: RadiatingRelationship) => void;
  onExploreEntity?: (entity: RadiatingEntity) => void;
  compact?: boolean;
  maxHeight?: number | string;
}

type SortOption = 'relevance' | 'name' | 'type' | 'depth';
type FilterOption = 'all' | 'high-relevance' | 'entities' | 'relationships';

const RadiatingResultsViewer: React.FC<RadiatingResultsViewerProps> = ({
  results,
  onEntityClick,
  onRelationshipClick,
  onExploreEntity,
  compact = false,
  maxHeight = 600
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedDepths, setExpandedDepths] = useState<Set<number>>(new Set([0]));
  const [selectedEntity, setSelectedEntity] = useState<RadiatingEntity | null>(null);
  const [sortBy, setSortBy] = useState<SortOption>('relevance');
  const [filterBy, setFilterBy] = useState<FilterOption>('all');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [exportMenuAnchor, setExportMenuAnchor] = useState<null | HTMLElement>(null);

  // Group entities by depth
  const entitiesByDepth = useMemo(() => {
    const grouped = new Map<number, RadiatingEntity[]>();
    
    results.entities.forEach(entity => {
      const depth = entity.depth;
      if (!grouped.has(depth)) {
        grouped.set(depth, []);
      }
      grouped.get(depth)!.push(entity);
    });
    
    // Sort each depth group
    grouped.forEach((entities, depth) => {
      entities.sort((a, b) => {
        switch (sortBy) {
          case 'relevance':
            return b.relevanceScore - a.relevanceScore;
          case 'name':
            return a.name.localeCompare(b.name);
          case 'type':
            return a.type.localeCompare(b.type);
          case 'depth':
            return a.depth - b.depth;
          default:
            return 0;
        }
      });
    });
    
    return grouped;
  }, [results.entities, sortBy]);

  // Filter entities
  const filteredEntities = useMemo(() => {
    let filtered = results.entities;
    
    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(entity =>
        entity.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        entity.type.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    // Apply relevance filter
    if (filterBy === 'high-relevance') {
      filtered = filtered.filter(entity => entity.relevanceScore >= 0.7);
    }
    
    return filtered;
  }, [results.entities, searchTerm, filterBy]);

  // Get relationships for an entity
  const getEntityRelationships = (entityId: string) => {
    return results.relationships.filter(
      rel => rel.sourceId === entityId || rel.targetId === entityId
    );
  };

  const handleDepthToggle = (depth: number) => {
    const newExpanded = new Set(expandedDepths);
    if (newExpanded.has(depth)) {
      newExpanded.delete(depth);
    } else {
      newExpanded.add(depth);
    }
    setExpandedDepths(newExpanded);
  };

  const handleEntitySelect = (entity: RadiatingEntity) => {
    setSelectedEntity(entity);
    if (onEntityClick) {
      onEntityClick(entity);
    }
  };

  const handleExport = async (format: 'json' | 'csv' | 'graphml') => {
    try {
      const request: RadiatingExportRequest = {
        results,
        format
      };
      
      const response = await fetch('/api/v1/radiating/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.download_url) {
          window.open(data.download_url, '_blank');
        }
      }
    } catch (error) {
      console.error('Export failed:', error);
    }
    
    setExportMenuAnchor(null);
  };

  const getRelevanceColor = (score: number) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'primary';
    if (score >= 0.4) return 'warning';
    return 'default';
  };

  const renderEntityItem = (entity: RadiatingEntity) => {
    const relationships = getEntityRelationships(entity.id);
    const isSelected = selectedEntity?.id === entity.id;
    
    return (
      <ListItem
        key={entity.id}
        secondaryAction={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              icon={<RelevanceIcon />}
              label={`${(entity.relevanceScore * 100).toFixed(0)}%`}
              size="small"
              color={getRelevanceColor(entity.relevanceScore)}
              variant="filled"
            />
            
            {relationships.length > 0 && (
              <Badge badgeContent={relationships.length} color="primary">
                <RelationshipIcon fontSize="small" />
              </Badge>
            )}
            
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                setAnchorEl(e.currentTarget);
                setSelectedEntity(entity);
              }}
            >
              <MoreIcon fontSize="small" />
            </IconButton>
          </Box>
        }
        sx={{
          backgroundColor: isSelected ? 'action.selected' : 'transparent',
          borderRadius: 1,
          mb: 0.5,
          '&:hover': {
            backgroundColor: 'action.hover'
          }
        }}
      >
        <ListItemButton onClick={() => handleEntitySelect(entity)}>
          <ListItemIcon>
            <NodeIcon 
              sx={{ 
                color: `hsl(${entity.depth * 60}, 70%, 50%)`,
                fontSize: 20 - entity.depth * 2
              }} 
            />
          </ListItemIcon>
          <ListItemText
            primary={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: isSelected ? 600 : 400 }}>
                  {entity.name}
                </Typography>
                <Chip
                  label={entity.type}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.7rem', height: 20 }}
                />
              </Box>
            }
            secondary={
              <Typography variant="caption" color="text.secondary">
                Depth: {entity.depth} • {relationships.length} connections
              </Typography>
            }
          />
        </ListItemButton>
      </ListItem>
    );
  };

  // Compact view for embedding
  if (compact) {
    return (
      <Paper
        elevation={1}
        sx={{
          p: 2,
          maxHeight,
          overflow: 'auto',
          backgroundColor: 'background.paper'
        }}
      >
        <Typography variant="subtitle2" gutterBottom>
          Radiating Results Summary
        </Typography>
        
        <Grid container spacing={1} sx={{ mb: 2 }}>
          <Grid item xs={4}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6">{results.totalEntities}</Typography>
              <Typography variant="caption">Entities</Typography>
            </Box>
          </Grid>
          <Grid item xs={4}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6">{results.totalRelationships}</Typography>
              <Typography variant="caption">Relations</Typography>
            </Box>
          </Grid>
          <Grid item xs={4}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6">{results.maxDepthReached}</Typography>
              <Typography variant="caption">Max Depth</Typography>
            </Box>
          </Grid>
        </Grid>
        
        <List dense>
          {filteredEntities.slice(0, 5).map(entity => renderEntityItem(entity))}
        </List>
        
        {filteredEntities.length > 5 && (
          <Button size="small" fullWidth>
            View All ({filteredEntities.length - 5} more)
          </Button>
        )}
      </Paper>
    );
  }

  // Full view
  return (
    <Paper
      elevation={2}
      sx={{
        backgroundColor: 'background.paper',
        borderRadius: 2,
        overflow: 'hidden'
      }}
    >
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TreeIcon color="primary" />
            Radiating Coverage Results
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              size="small"
              startIcon={<ExportIcon />}
              onClick={(e) => setExportMenuAnchor(e.currentTarget)}
            >
              Export
            </Button>
          </Box>
        </Box>

        {/* Summary Stats */}
        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={6} sm={3}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <EntityIcon color="primary" />
              <Box>
                <Typography variant="h5">{results.totalEntities}</Typography>
                <Typography variant="caption" color="text.secondary">
                  Total Entities
                </Typography>
              </Box>
            </Box>
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <RelationshipIcon color="primary" />
              <Box>
                <Typography variant="h5">{results.totalRelationships}</Typography>
                <Typography variant="caption" color="text.secondary">
                  Relationships
                </Typography>
              </Box>
            </Box>
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <PathIcon color="primary" />
              <Box>
                <Typography variant="h5">{results.maxDepthReached}</Typography>
                <Typography variant="caption" color="text.secondary">
                  Max Depth
                </Typography>
              </Box>
            </Box>
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <RelevanceIcon color="primary" />
              <Box>
                <Typography variant="h5">
                  {(results.relevanceThreshold * 100).toFixed(0)}%
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Min Relevance
                </Typography>
              </Box>
            </Box>
          </Grid>
        </Grid>

        {/* Search and Filters */}
        <Box sx={{ display: 'flex', gap: 2 }}>
          <TextField
            size="small"
            placeholder="Search entities..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              )
            }}
            sx={{ flex: 1 }}
          />
          
          <Button
            size="small"
            startIcon={<FilterIcon />}
            onClick={(e) => setAnchorEl(e.currentTarget)}
          >
            Filter: {filterBy}
          </Button>
          
          <Button
            size="small"
            startIcon={<SortIcon />}
            onClick={(e) => setAnchorEl(e.currentTarget)}
          >
            Sort: {sortBy}
          </Button>
        </Box>
      </Box>

      {/* Results by Depth */}
      <Box sx={{ maxHeight, overflow: 'auto' }}>
        {Array.from(entitiesByDepth.entries())
          .sort(([a], [b]) => a - b)
          .map(([depth, entities]) => (
            <Accordion
              key={depth}
              expanded={expandedDepths.has(depth)}
              onChange={() => handleDepthToggle(depth)}
            >
              <AccordionSummary expandIcon={<ExpandIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                  <Chip
                    label={`Depth ${depth}`}
                    color={depth === 0 ? 'primary' : 'default'}
                    variant={depth === 0 ? 'filled' : 'outlined'}
                  />
                  <Typography variant="body2">
                    {entities.length} entities
                  </Typography>
                  
                  {/* Average relevance for this depth */}
                  <Box sx={{ ml: 'auto', mr: 2 }}>
                    <Chip
                      icon={<RelevanceIcon />}
                      label={`Avg: ${(
                        entities.reduce((sum, e) => sum + e.relevanceScore, 0) / 
                        entities.length * 100
                      ).toFixed(0)}%`}
                      size="small"
                      variant="outlined"
                    />
                  </Box>
                </Box>
              </AccordionSummary>
              
              <AccordionDetails>
                <List>
                  {entities
                    .filter(entity => 
                      !searchTerm || 
                      entity.name.toLowerCase().includes(searchTerm.toLowerCase())
                    )
                    .map(entity => renderEntityItem(entity))}
                </List>
              </AccordionDetails>
            </Accordion>
          ))}
      </Box>

      {/* Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
      >
        {selectedEntity && (
          <>
            <MenuItem onClick={() => {
              if (onExploreEntity) onExploreEntity(selectedEntity);
              setAnchorEl(null);
            }}>
              <ListItemIcon>
                <OpenIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Explore Further</ListItemText>
            </MenuItem>
            
            <MenuItem onClick={() => {
              navigator.clipboard.writeText(JSON.stringify(selectedEntity, null, 2));
              setAnchorEl(null);
            }}>
              <ListItemIcon>
                <CopyIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Copy Entity Data</ListItemText>
            </MenuItem>
            
            <MenuItem onClick={() => {
              console.log('Entity details:', selectedEntity);
              setAnchorEl(null);
            }}>
              <ListItemIcon>
                <ViewIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>View Details</ListItemText>
            </MenuItem>
          </>
        )}
      </Menu>

      {/* Export Menu */}
      <Menu
        anchorEl={exportMenuAnchor}
        open={Boolean(exportMenuAnchor)}
        onClose={() => setExportMenuAnchor(null)}
      >
        <MenuItem onClick={() => handleExport('json')}>Export as JSON</MenuItem>
        <MenuItem onClick={() => handleExport('csv')}>Export as CSV</MenuItem>
        <MenuItem onClick={() => handleExport('graphml')}>Export as GraphML</MenuItem>
      </Menu>
    </Paper>
  );
};

export default RadiatingResultsViewer;