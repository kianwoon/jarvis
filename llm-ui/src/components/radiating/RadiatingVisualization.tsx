import React, { useRef, useEffect, useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Slider,
  FormControl,
  Select,
  MenuItem,
  Tooltip,
  Button,
  ButtonGroup,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Divider,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  CenterFocusStrong as CenterIcon,
  Fullscreen as FullscreenIcon,
  FullscreenExit as ExitFullscreenIcon,
  Palette as ColorIcon,
  Timeline as LayoutIcon,
  Visibility as ShowIcon,
  VisibilityOff as HideIcon,
  PhotoCamera as SnapshotIcon,
  Info as InfoIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import * as d3 from 'd3';
import { 
  RadiatingVisualizationData,
  RadiatingVisualizationNode,
  RadiatingVisualizationLink,
  RadiatingEntity,
  RadiatingRelationship 
} from '../../types/radiating';

interface RadiatingVisualizationProps {
  data: RadiatingVisualizationData;
  onNodeClick?: (node: RadiatingVisualizationNode) => void;
  onLinkClick?: (link: RadiatingVisualizationLink) => void;
  width?: number;
  height?: number;
  fullscreen?: boolean;
}

type ColorScheme = 'depth' | 'type' | 'relevance';
type LayoutType = 'force' | 'radial' | 'hierarchical';

const RadiatingVisualization: React.FC<RadiatingVisualizationProps> = ({
  data,
  onNodeClick,
  onLinkClick,
  width = 800,
  height = 600,
  fullscreen = false
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [simulation, setSimulation] = useState<d3.Simulation<RadiatingVisualizationNode, RadiatingVisualizationLink> | null>(null);
  const [selectedNode, setSelectedNode] = useState<RadiatingVisualizationNode | null>(null);
  const [selectedLink, setSelectedLink] = useState<RadiatingVisualizationLink | null>(null);
  const [zoom, setZoom] = useState(1);
  const [colorScheme, setColorScheme] = useState<ColorScheme>('depth');
  const [layoutType, setLayoutType] = useState<LayoutType>('force');
  const [showLabels, setShowLabels] = useState(true);
  const [showLinks, setShowLinks] = useState(true);
  const [linkOpacity, setLinkOpacity] = useState(0.6);
  const [nodeSize, setNodeSize] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(fullscreen);
  const [detailsOpen, setDetailsOpen] = useState(false);

  // Color scales
  const colorScales = useMemo(() => ({
    depth: d3.scaleOrdinal(d3.schemeCategory10),
    type: d3.scaleOrdinal(d3.schemePastel1),
    relevance: d3.scaleSequential(d3.interpolatePlasma).domain([0, 1])
  }), []);

  // Initialize D3 visualization
  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create container groups
    const g = svg.append('g').attr('class', 'main-group');
    const linksGroup = g.append('g').attr('class', 'links');
    const nodesGroup = g.append('g').attr('class', 'nodes');
    const labelsGroup = g.append('g').attr('class', 'labels');

    // Setup zoom behavior
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        setZoom(event.transform.k);
      });

    svg.call(zoomBehavior);

    // Create force simulation
    const sim = d3.forceSimulation<RadiatingVisualizationNode>(data.nodes)
      .force('link', d3.forceLink<RadiatingVisualizationNode, RadiatingVisualizationLink>(data.links)
        .id(d => d.id)
        .distance(d => 50 + (1 - d.value) * 50))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => getNodeRadius(d) + 5));

    setSimulation(sim);

    // Draw links
    const links = linksGroup.selectAll('line')
      .data(data.links)
      .enter().append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', linkOpacity)
      .attr('stroke-width', d => Math.sqrt(d.value * 3))
      .on('click', (event, d) => {
        setSelectedLink(d);
        if (onLinkClick) onLinkClick(d);
      })
      .on('mouseover', function(event, d) {
        d3.select(this)
          .attr('stroke', '#ff6b6b')
          .attr('stroke-width', Math.sqrt(d.value * 5));
        
        // Show tooltip
        const tooltip = d3.select('body').append('div')
          .attr('class', 'graph-tooltip')
          .style('position', 'absolute')
          .style('padding', '8px')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('left', `${event.pageX + 10}px`)
          .style('top', `${event.pageY - 10}px`)
          .html(`
            <strong>${d.type}</strong><br/>
            Weight: ${d.value.toFixed(2)}
          `);
      })
      .on('mouseout', function(event, d) {
        d3.select(this)
          .attr('stroke', '#999')
          .attr('stroke-width', Math.sqrt(d.value * 3));
        
        d3.selectAll('.graph-tooltip').remove();
      });

    // Draw nodes
    const nodes = nodesGroup.selectAll('circle')
      .data(data.nodes)
      .enter().append('circle')
      .attr('r', d => getNodeRadius(d))
      .attr('fill', d => getNodeColor(d))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .call(drag(sim))
      .on('click', (event, d) => {
        setSelectedNode(d);
        setDetailsOpen(true);
        if (onNodeClick) onNodeClick(d);
      })
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', getNodeRadius(d) * 1.2);
        
        // Highlight connected links
        links
          .attr('stroke-opacity', l => 
            (l.source === d || l.target === d) ? 1 : linkOpacity * 0.3
          );
      })
      .on('mouseout', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', getNodeRadius(d));
        
        links.attr('stroke-opacity', linkOpacity);
      });

    // Draw labels
    if (showLabels) {
      const labels = labelsGroup.selectAll('text')
        .data(data.nodes)
        .enter().append('text')
        .text(d => d.name)
        .attr('font-size', '10px')
        .attr('dx', d => getNodeRadius(d) + 3)
        .attr('dy', '.35em')
        .style('pointer-events', 'none');
    }

    // Update positions on simulation tick
    sim.on('tick', () => {
      links
        .attr('x1', d => (d.source as any).x)
        .attr('y1', d => (d.source as any).y)
        .attr('x2', d => (d.target as any).x)
        .attr('y2', d => (d.target as any).y);

      nodes
        .attr('cx', d => d.x!)
        .attr('cy', d => d.y!);

      if (showLabels) {
        labelsGroup.selectAll('text')
          .attr('x', d => d.x!)
          .attr('y', d => d.y!);
      }
    });

    // Apply layout
    applyLayout(sim, layoutType);

    return () => {
      sim.stop();
    };
  }, [data, width, height, colorScheme, showLabels, showLinks, linkOpacity, nodeSize, layoutType]);

  // Helper functions
  const getNodeRadius = (node: RadiatingVisualizationNode) => {
    const baseSize = 5;
    return (baseSize + node.radius * 10) * nodeSize;
  };

  const getNodeColor = (node: RadiatingVisualizationNode) => {
    switch (colorScheme) {
      case 'depth':
        return colorScales.depth(node.group.toString());
      case 'type':
        return colorScales.type(node.type);
      case 'relevance':
        return colorScales.relevance(node.radius);
      default:
        return '#69b3a2';
    }
  };

  const drag = (simulation: d3.Simulation<RadiatingVisualizationNode, RadiatingVisualizationLink>) => {
    function dragstarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: any) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    return d3.drag<SVGCircleElement, RadiatingVisualizationNode>()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended);
  };

  const applyLayout = (sim: d3.Simulation<RadiatingVisualizationNode, RadiatingVisualizationLink>, type: LayoutType) => {
    switch (type) {
      case 'radial':
        // Arrange nodes in concentric circles by depth
        const depthGroups = d3.group(data.nodes, d => d.group);
        depthGroups.forEach((nodes, depth) => {
          const angleStep = (2 * Math.PI) / nodes.length;
          const radius = 100 + depth * 80;
          nodes.forEach((node, i) => {
            node.fx = width / 2 + radius * Math.cos(i * angleStep);
            node.fy = height / 2 + radius * Math.sin(i * angleStep);
          });
        });
        sim.alpha(0.5).restart();
        setTimeout(() => {
          data.nodes.forEach(node => {
            node.fx = null;
            node.fy = null;
          });
        }, 1000);
        break;
        
      case 'hierarchical':
        // Arrange nodes in a tree structure
        const tree = d3.tree<RadiatingVisualizationNode>()
          .size([width - 100, height - 100]);
        // Note: This would need proper tree data structure
        break;
        
      case 'force':
      default:
        // Default force-directed layout
        break;
    }
  };

  const handleZoom = (direction: 'in' | 'out' | 'reset') => {
    if (!svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>();
    
    if (direction === 'reset') {
      svg.transition()
        .duration(750)
        .call(zoomBehavior.transform, d3.zoomIdentity);
    } else {
      const scale = direction === 'in' ? 1.3 : 0.7;
      svg.transition()
        .duration(350)
        .call(zoomBehavior.scaleBy, scale);
    }
  };

  const handleSnapshot = () => {
    if (!svgRef.current) return;
    
    // Convert SVG to image and download
    const svgData = new XMLSerializer().serializeToString(svgRef.current);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      canvas.width = width;
      canvas.height = height;
      ctx?.drawImage(img, 0, 0);
      
      canvas.toBlob((blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `radiating-graph-${Date.now()}.png`;
          a.click();
          URL.revokeObjectURL(url);
        }
      });
    };
    
    img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
  };

  return (
    <Paper
      elevation={3}
      sx={{
        position: isFullscreen ? 'fixed' : 'relative',
        top: isFullscreen ? 0 : 'auto',
        left: isFullscreen ? 0 : 'auto',
        right: isFullscreen ? 0 : 'auto',
        bottom: isFullscreen ? 0 : 'auto',
        zIndex: isFullscreen ? 9999 : 'auto',
        width: isFullscreen ? '100vw' : width,
        height: isFullscreen ? '100vh' : height,
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'background.paper'
      }}
    >
      {/* Toolbar */}
      <Box sx={{ 
        p: 1, 
        borderBottom: 1, 
        borderColor: 'divider',
        display: 'flex',
        alignItems: 'center',
        gap: 2
      }}>
        <Typography variant="h6" sx={{ flex: 1 }}>
          Radiating Network Visualization
        </Typography>
        
        {/* Controls */}
        <ButtonGroup size="small">
          <Tooltip title="Zoom In">
            <IconButton onClick={() => handleZoom('in')}>
              <ZoomInIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Zoom Out">
            <IconButton onClick={() => handleZoom('out')}>
              <ZoomOutIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Reset View">
            <IconButton onClick={() => handleZoom('reset')}>
              <CenterIcon />
            </IconButton>
          </Tooltip>
        </ButtonGroup>

        <Divider orientation="vertical" flexItem />

        {/* Layout selector */}
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <Select
            value={layoutType}
            onChange={(e) => setLayoutType(e.target.value as LayoutType)}
          >
            <MenuItem value="force">Force Layout</MenuItem>
            <MenuItem value="radial">Radial Layout</MenuItem>
            <MenuItem value="hierarchical">Hierarchical</MenuItem>
          </Select>
        </FormControl>

        {/* Color scheme selector */}
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <Select
            value={colorScheme}
            onChange={(e) => setColorScheme(e.target.value as ColorScheme)}
            startAdornment={<ColorIcon fontSize="small" sx={{ mr: 0.5 }} />}
          >
            <MenuItem value="depth">Color by Depth</MenuItem>
            <MenuItem value="type">Color by Type</MenuItem>
            <MenuItem value="relevance">Color by Relevance</MenuItem>
          </Select>
        </FormControl>

        <Divider orientation="vertical" flexItem />

        {/* Toggles */}
        <FormControlLabel
          control={
            <Switch
              checked={showLabels}
              onChange={(e) => setShowLabels(e.target.checked)}
              size="small"
            />
          }
          label="Labels"
        />
        
        <FormControlLabel
          control={
            <Switch
              checked={showLinks}
              onChange={(e) => setShowLinks(e.target.checked)}
              size="small"
            />
          }
          label="Links"
        />

        <Divider orientation="vertical" flexItem />

        {/* Actions */}
        <Tooltip title="Take Snapshot">
          <IconButton onClick={handleSnapshot}>
            <SnapshotIcon />
          </IconButton>
        </Tooltip>
        
        <Tooltip title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}>
          <IconButton onClick={() => setIsFullscreen(!isFullscreen)}>
            {isFullscreen ? <ExitFullscreenIcon /> : <FullscreenIcon />}
          </IconButton>
        </Tooltip>

        {isFullscreen && (
          <IconButton onClick={() => setIsFullscreen(false)}>
            <CloseIcon />
          </IconButton>
        )}
      </Box>

      {/* Settings Panel */}
      <Box sx={{ p: 1, borderBottom: 1, borderColor: 'divider', display: 'flex', gap: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="body2">Node Size:</Typography>
          <Slider
            value={nodeSize}
            onChange={(_, value) => setNodeSize(value as number)}
            min={0.5}
            max={2}
            step={0.1}
            sx={{ width: 100 }}
          />
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="body2">Link Opacity:</Typography>
          <Slider
            value={linkOpacity}
            onChange={(_, value) => setLinkOpacity(value as number)}
            min={0.1}
            max={1}
            step={0.1}
            sx={{ width: 100 }}
          />
        </Box>

        <Box sx={{ ml: 'auto' }}>
          <Chip
            label={`${data.nodes.length} nodes`}
            size="small"
            color="primary"
            variant="outlined"
          />
          <Chip
            label={`${data.links.length} links`}
            size="small"
            color="secondary"
            variant="outlined"
            sx={{ ml: 1 }}
          />
        </Box>
      </Box>

      {/* SVG Container */}
      <Box sx={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          style={{ cursor: 'grab', backgroundColor: '#fafafa' }}
        />
        
        {/* Zoom indicator */}
        <Chip
          label={`${(zoom * 100).toFixed(0)}%`}
          size="small"
          sx={{
            position: 'absolute',
            bottom: 8,
            right: 8,
            backgroundColor: 'rgba(255, 255, 255, 0.9)'
          }}
        />
      </Box>

      {/* Node Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Entity Details
          <IconButton
            onClick={() => setDetailsOpen(false)}
            sx={{ position: 'absolute', right: 8, top: 8 }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        
        <DialogContent>
          {selectedNode && (
            <List>
              <ListItem>
                <ListItemText
                  primary="Name"
                  secondary={selectedNode.name}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Type"
                  secondary={selectedNode.type}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Depth Level"
                  secondary={selectedNode.group}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Relevance Score"
                  secondary={`${(selectedNode.radius * 100).toFixed(1)}%`}
                />
              </ListItem>
            </List>
          )}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>Close</Button>
          {selectedNode && onNodeClick && (
            <Button
              variant="contained"
              onClick={() => {
                onNodeClick(selectedNode);
                setDetailsOpen(false);
              }}
            >
              Explore Further
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default RadiatingVisualization;