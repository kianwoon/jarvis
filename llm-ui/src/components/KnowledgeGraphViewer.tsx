import React, { useState, useEffect, useRef, ErrorInfo, Component } from 'react';
import * as d3 from 'd3';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Tabs,
  Tab
} from '@mui/material';
import {
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon
} from '@mui/icons-material';

// Error boundary for D3 visualization
class VisualizationErrorBoundary extends Component<
  { children: React.ReactNode; onError: (error: string) => void },
  { hasError: boolean }
> {
  constructor(props: { children: React.ReactNode; onError: (error: string) => void }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Visualization Error Boundary caught an error:', error, errorInfo);
    this.props.onError(`React Error: ${error.message}`);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          padding: '20px', 
          background: '#f8d7da', 
          color: '#721c24', 
          borderRadius: '4px',
          textAlign: 'center'
        }}>
          <h4>Visualization Error</h4>
          <p>Something went wrong while rendering the knowledge graph. Check the console for details.</p>
        </div>
      );
    }

    return this.props.children;
  }
}

interface Entity {
  id: string;
  name: string;
  type: string;
  confidence: number;
  properties?: Record<string, any>;
  original_text?: string;
  chunk_id?: string;
  created_at?: string;
}

interface Relationship {
  id?: string;
  source_entity: string;
  target_entity: string;
  relationship_type: string;
  confidence: number;
  context?: string;
  chunk_id?: string;
  created_at?: string;
}

interface GraphStats {
  total_entities: number;
  total_relationships: number;
  entity_types: Record<string, number>;
  relationship_types: Record<string, number>;
  documents_processed: number;
  last_updated: string;
}

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  name: string;
  type: string;
  confidence: number;
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  source: string | GraphNode;
  target: string | GraphNode;
  relationship_type: string;
  confidence: number;
}

const KnowledgeGraphViewer: React.FC = () => {
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [entities, setEntities] = useState<Entity[]>([]);
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<string>('');
  const [availableDocuments, setAvailableDocuments] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fontSize, setFontSize] = useState(() => {
    const saved = localStorage.getItem('jarvis-kg-font-size');
    return saved ? parseInt(saved) : 12;
  });
  
  const handleFontSizeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newSize = parseInt(event.target.value);
    setFontSize(newSize);
    localStorage.setItem('jarvis-kg-font-size', newSize.toString());
  };
  
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: window.innerWidth, height: window.innerHeight - 200 });
  const [isStatsCollapsed, setIsStatsCollapsed] = useState(false);
  const [zoomTransform, setZoomTransform] = useState(d3.zoomIdentity);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [notification, setNotification] = useState<{type: 'success' | 'error', message: string} | null>(null);
  
  // Theme state
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('jarvis-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });

  // Entity type colors matching backend LLM extractor types
  // Enhanced color system with accessibility and theme support
  const getEntityColors = (isDark: boolean): Record<string, string> => {
    const lightTheme = {
      // Core entity types (matching backend hierarchical_entity_types)
      'PERSON': '#e67e22',           // Darker orange for better contrast
      'ORGANIZATION': '#27ae60',     // Darker green for companies, banks, groups
      'LOCATION': '#c0392b',         // Darker red for places, cities, countries
      'EVENT': '#e74c3c',            // Solid red for meetings, conferences, incidents
      'TECHNOLOGY': '#3498db',       // Bright blue for software, databases, tools
      'CONCEPT': '#2980b9',          // Darker blue for abstract ideas, strategies
      'PRODUCT': '#8e44ad',          // Darker purple for services, applications
      'SERVICE': '#d63384',          // Bootstrap pink for APIs, platforms, offerings
      'PROJECT': '#a0522d',          // Saddle brown for initiatives, programs
      'SYSTEM': '#f39c12',           // Bright orange for infrastructure, frameworks
      
      // Legacy aliases for backward compatibility
      'ORG': '#27ae60',              // Alias for ORGANIZATION
      'TECH': '#3498db',             // Alias for TECHNOLOGY
      
      // Additional business concepts
      'INITIATIVE': '#2ecc71',       // Emerald for business initiatives
      'STRATEGY': '#9b59b6',         // Amethyst for strategic concepts
      'PROCESS': '#795548',          // Brown for business processes
      'COUNTRY': '#c0392b',          // Same as LOCATION
      'CITY': '#ff5722',             // Deep orange (distinguishable from PERSON)
      
      // Default fallback
      'default': '#6c757d'           // Bootstrap secondary grey
    };
    
    const darkTheme = {
      // Brighter colors for dark mode with better visibility
      'PERSON': '#ff9f43',           // Brighter orange
      'ORGANIZATION': '#2ed573',     // Brighter green
      'LOCATION': '#ff3838',         // Brighter red
      'EVENT': '#ff6b6b',            // Bright coral
      'TECHNOLOGY': '#54a0ff',       // Bright blue
      'CONCEPT': '#74b9ff',          // Light blue
      'PRODUCT': '#a55eea',          // Bright purple
      'SERVICE': '#fd79a8',          // Bright pink
      'PROJECT': '#d63031',          // Bright red-brown
      'SYSTEM': '#fdcb6e',           // Bright yellow-orange
      
      // Legacy aliases
      'ORG': '#2ed573',
      'TECH': '#54a0ff',
      
      // Additional concepts
      'INITIATIVE': '#00b894',       // Teal
      'STRATEGY': '#6c5ce7',         // Periwinkle
      'PROCESS': '#a29bfe',          // Light purple
      'COUNTRY': '#ff3838',
      'CITY': '#fd79a8',
      
      // Default fallback
      'default': '#b2bec3'           // Light grey for dark theme
    };
    
    return isDark ? darkTheme : lightTheme;
  };
  
  const entityColors = getEntityColors(isDarkMode);

  // Enhanced color generator with accessibility features
  const generateColor = (entityType: string): string => {
    if (entityColors[entityType]) {
      return entityColors[entityType];
    }
    
    // Generate consistent color based on entity type string
    let hash = 0;
    for (let i = 0; i < entityType.length; i++) {
      const char = entityType.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    // Enhanced HSL generation with better accessibility
    const hue = Math.abs(hash) % 360;
    
    // Avoid problematic hue ranges (yellow-green that's hard to see)
    const adjustedHue = (hue >= 60 && hue <= 120) ? (hue + 60) % 360 : hue;
    
    // Higher saturation and better contrast for accessibility
    const saturation = isDarkMode ? 85 : 75; // Higher saturation for visibility
    const lightness = isDarkMode ? 70 : 40;  // Better contrast ratios
    
    return `hsl(${adjustedHue}, ${saturation}%, ${lightness}%)`;
  };

  // Theme-aware colors
  const getThemeColors = () => ({
    background: isDarkMode ? '#121212' : '#fafafa',
    surface: isDarkMode ? '#1e1e1e' : '#ffffff',
    text: isDarkMode ? '#ffffff' : '#333333',
    secondaryText: isDarkMode ? '#b3b3b3' : '#666666',
    border: isDarkMode ? '#424242' : '#e0e0e0',
    linkColor: isDarkMode ? '#999999' : '#666666',
    linkLabelColor: isDarkMode ? '#ffffff' : '#333333',
    nodeLabelColor: isDarkMode ? '#ffffff' : '#222222'
  });

  // Listen for theme changes from localStorage
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'jarvis-dark-mode' && e.newValue) {
        const newDarkMode = JSON.parse(e.newValue);
        setIsDarkMode(newDarkMode);
        document.body.setAttribute('data-theme', newDarkMode ? 'dark' : 'light');
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  // Fetch available document IDs or entity types
  const fetchAvailableDocuments = async () => {
    try {
      // First try to get documents with document_id
      const response = await fetch('/api/v1/knowledge-graph/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'MATCH (n) WHERE n.document_id IS NOT NULL RETURN DISTINCT n.document_id as document_id ORDER BY document_id'
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        const docIds = data.results?.map((r: any) => r.document_id) || [];
        
        if (docIds.length > 0) {
          setAvailableDocuments(['ALL_ENTITIES', ...docIds]);
          // Auto-select ALL_ENTITIES as default if nothing is selected
          if (!selectedDocument) {
            console.log('🔄 Auto-selecting ALL_ENTITIES as default document');
            setSelectedDocument('ALL_ENTITIES');
          }
        } else {
          // Fallback: show "All Entities" option if no document IDs exist
          setAvailableDocuments(['ALL_ENTITIES']);
          // Auto-select ALL_ENTITIES as the only option
          console.log('🔄 Auto-selecting ALL_ENTITIES as fallback');
          setSelectedDocument('ALL_ENTITIES');
        }
      }
    } catch (err) {
      console.error('Error fetching document IDs:', err);
      // Fallback: show "All Entities" option
      setAvailableDocuments(['ALL_ENTITIES']);
      // Auto-select ALL_ENTITIES as fallback
      if (!selectedDocument) {
        console.log('🔄 Auto-selecting ALL_ENTITIES after error');
        setSelectedDocument('ALL_ENTITIES');
      }
    }
  };

  // Fetch graph statistics
  const fetchStats = async () => {
    try {
      const response = await fetch('/api/v1/knowledge-graph/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      } else {
        setError('Failed to fetch graph statistics');
      }
    } catch (err) {
      setError('Error connecting to knowledge graph service');
      console.error('Stats fetch error:', err);
    }
  };

  // Fetch entities for a specific document or all entities
  const fetchEntities = async (documentId: string) => {
    if (!documentId) return;
    
    console.log('🔍 fetchEntities called with documentId:', documentId);
    setLoading(true);
    try {
      if (documentId === 'ALL_ENTITIES') {
        console.log('✅ Using unlimited endpoint for ALL_ENTITIES');
        // Use the new unlimited visualization endpoint - fetch both entities and relationships
        const timestamp = Date.now();
        const url = `/api/v1/knowledge-graph/visualization-data?t=${timestamp}`;
        console.log('🔄 Fetching from unlimited endpoint with cache bust:', url);
        const response = await fetch(url, {
          cache: 'no-cache',
          headers: {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache'
          }
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log('🔍 RAW API RESPONSE FROM UNLIMITED ENDPOINT:', {
            url: url,
            entitiesCount: data.entities?.length || 0,
            relationshipsCount: data.relationships?.length || 0,
            firstFewRelationships: data.relationships?.slice(0, 3),
            stats: data.stats
          });
          setEntities(data.entities || []);
          setRelationships(data.relationships || []);
          console.log(`🎉 SUCCESS: Loaded ${data.entities?.length || 0} entities and ${data.relationships?.length || 0} relationships from unlimited endpoint`);
          console.log('📊 Stats from backend:', data.stats);
        } else {
          console.error('❌ Failed to fetch from unlimited endpoint, status:', response.status);
          setError('Failed to fetch visualization data');
        }
      } else {
        // Original document-specific query
        const response = await fetch(`/api/v1/knowledge-graph/entities/${documentId}`);
        if (response.ok) {
          const data = await response.json();
          setEntities(data.entities || []);
        } else {
          setError('Failed to fetch entities');
        }
      }
    } catch (err) {
      setError('Error fetching entities');
      console.error('Entities fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Fetch relationships for a specific document or all relationships
  const fetchRelationships = async (documentId: string) => {
    if (!documentId) return;
    
    try {
      if (documentId === 'ALL_ENTITIES') {
        // Skip - relationships are already loaded with entities in fetchEntities
        return;
      } else {
        // Original document-specific query
        const response = await fetch(`/api/v1/knowledge-graph/relationships/${documentId}`);
        if (response.ok) {
          const data = await response.json();
          setRelationships(data.relationships || []);
        } else {
          setError('Failed to fetch relationships');
        }
      }
    } catch (err) {
      setError('Error fetching relationships');
      console.error('Relationships fetch error:', err);
    }
  };

  // Create and update D3 visualization
  const createVisualization = () => {
    console.log('=== Starting D3 Visualization ===');
    console.log('Entities:', entities.length, entities);
    console.log('Relationships:', relationships.length, relationships);
    
    if (!svgRef.current) {
      console.error('SVG ref is null');
      setError('SVG reference not available');
      return;
    }
    
    if (entities.length === 0) {
      console.log('No entities to visualize');
      return;
    }

    try {
      // Clear any existing error
      setError(null);
      
      // Clear previous visualization
      const svg = d3.select(svgRef.current);
      svg.selectAll('*').remove();
      console.log('Cleared previous visualization');

      // Prepare nodes with better error handling
      const nodes: GraphNode[] = entities.map((entity, index) => {
        const node = {
          id: entity.id || entity.name || `entity-${index}`,
          name: entity.name || `Entity ${index + 1}`,
          type: entity.type || 'UNKNOWN',
          confidence: Math.max(0, Math.min(1, entity.confidence || 0.5))
        };
        console.log(`Node ${index}:`, node);
        return node;
      });

      // Create entity lookup maps for both ID and name matching
      const entityIds = new Set(nodes.map(n => n.id));
      const entityNames = new Set(nodes.map(n => n.name));
      const nameToIdMap = new Map(nodes.map(n => [n.name.toLowerCase(), n.id]));
      const idToNameMap = new Map(nodes.map(n => [n.id, n.name]));
      
      console.log('🔗 Entity IDs available:', Array.from(entityIds));
      console.log('🔗 Entity names available:', Array.from(entityNames));
      console.log('🔗 Raw relationships received:', relationships);
      console.log('🔍 DEBUGGING RELATIONSHIP FILTERING:');
      console.log(`   • Total relationships to process: ${relationships.length}`);
      console.log(`   • Entity IDs available: ${entityIds.size}`);
      console.log(`   • Sample relationship source/target IDs:`, relationships.slice(0, 10).map(r => ({
        source: r.source_entity, 
        target: r.target_entity,
        sourceExists: entityIds.has(r.source_entity),
        targetExists: entityIds.has(r.target_entity)
      })));
      
      // Enhanced relationship filtering with fallback logic
      let filteredOutCount = 0;
      const validRelationships = relationships.filter((rel, index) => {
        let sourceId = rel.source_entity;
        let targetId = rel.target_entity;
        
        // If source_entity is not a valid ID, try to map from name
        if (!entityIds.has(sourceId)) {
          const sourceFromName = nameToIdMap.get(sourceId.toLowerCase());
          if (sourceFromName) {
            sourceId = sourceFromName;
            console.log(`🔗 Mapped source "${rel.source_entity}" -> "${sourceId}"`);
          }
        }
        
        // If target_entity is not a valid ID, try to map from name
        if (!entityIds.has(targetId)) {
          const targetFromName = nameToIdMap.get(targetId.toLowerCase());
          if (targetFromName) {
            targetId = targetFromName;
            console.log(`🔗 Mapped target "${rel.target_entity}" -> "${targetId}"`);
          }
        }
        
        const sourceValid = entityIds.has(sourceId);
        const targetValid = entityIds.has(targetId);
        
        if (!sourceValid || !targetValid) {
          filteredOutCount++;
          console.warn(`❌ Invalid relationship #${filteredOutCount}:`, {
            original: rel,
            mappedSource: sourceId,
            mappedTarget: targetId,
            sourceValid,
            targetValid,
            index: index
          });
          return false;
        }
        
        // Update the relationship with proper IDs
        rel.source_entity = sourceId;
        rel.target_entity = targetId;
        
        console.log('✅ Valid relationship:', {
          source: sourceId,
          target: targetId,
          type: rel.relationship_type,
          sourceName: idToNameMap.get(sourceId),
          targetName: idToNameMap.get(targetId)
        });
        
        return true;
      });

      const links: GraphLink[] = validRelationships.map((rel, index) => {
        const link = {
          source: rel.source_entity,
          target: rel.target_entity,
          relationship_type: rel.relationship_type || 'RELATED',
          confidence: Math.max(0, Math.min(1, rel.confidence || 0.5))
        };
        console.log(`Link ${index}:`, link);
        return link;
      });
      
      console.log('📊 RELATIONSHIP PROCESSING SUMMARY:');
      console.log(`   • Raw relationships received: ${relationships.length}`);
      console.log(`   • Relationships filtered out: ${filteredOutCount}`);
      console.log(`   • Valid relationships after filtering: ${validRelationships.length}`);
      console.log(`   • Final D3 links created: ${links.length}`);
      console.log(`   • Entities available: ${nodes.length}`);
      
      if (filteredOutCount > 0) {
        console.error(`🚨 CRITICAL ISSUE: ${filteredOutCount} relationships were filtered out due to entity ID mismatches!`);
        console.error('This suggests the backend is returning inconsistent entity IDs between entities and relationships data.');
      }
      
      if (relationships.length > 0 && validRelationships.length === 0) {
        console.error('🚨 CRITICAL: All relationships were filtered out!');
        console.error('This usually means entity IDs don\'t match between entities and relationships');
        setError(`No valid relationships found. Expected entity IDs but got mismatched data. Check backend API consistency.`);
      } else if (validRelationships.length > 0) {
        console.log(`✅ Successfully processed ${validRelationships.length} relationships`);
      }

      // Set up SVG with safe dimensions
      const width = Math.max(400, dimensions.width);
      const height = Math.max(300, dimensions.height);
      const themeColors = getThemeColors();
      
      svg
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`)
        .style('background', themeColors.background)
        .style('cursor', 'grab');

      // Create main container group for zoom/pan transforms
      const mainGroup = svg.append('g').attr('class', 'main-group');

      // Set up zoom behavior
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
          const transform = event.transform;
          setZoomTransform(transform);
          mainGroup.attr('transform', transform.toString());
          
          // Adjust font sizes based on zoom level for better readability
          // Only update font sizes if zoom level changes significantly to improve performance
          if (Math.abs(transform.k - 1) > 0.1) {
            const scaledFontSize = Math.max(8, (fontSize - 4) / Math.sqrt(transform.k));
            const scaledLinkFontSize = Math.max(6, Math.max(fontSize - 10, 6) / Math.sqrt(transform.k));
            
            // Update node label font sizes
            mainGroup.selectAll('.node-label-group text')
              .attr('font-size', `${scaledFontSize}px`);
              
            // Update link label font sizes
            mainGroup.selectAll('.link-labels text')
              .attr('font-size', `${scaledLinkFontSize}px`);
          }
        });

      // Apply zoom behavior to SVG and store reference
      svg.call(zoom);
      zoomBehaviorRef.current = zoom;

      // Helper function to wrap text and calculate node size
      const wrapText = (text: string, maxLineLength: number = 12): { lines: string[], maxWidth: number } => {
        const words = text.split(/\s+/);
        const lines: string[] = [];
        let currentLine = '';
        let maxWidth = 0;
        
        for (const word of words) {
          const testLine = currentLine + (currentLine ? ' ' : '') + word;
          
          if (testLine.length <= maxLineLength) {
            currentLine = testLine;
          } else {
            if (currentLine) {
              lines.push(currentLine);
              maxWidth = Math.max(maxWidth, currentLine.length);
              currentLine = word;
            } else {
              // Word is longer than maxLineLength, truncate it
              lines.push(word.length > maxLineLength ? word.slice(0, maxLineLength - 1) + '…' : word);
              maxWidth = Math.max(maxWidth, Math.min(word.length, maxLineLength));
              currentLine = '';
            }
          }
        }
        
        if (currentLine) {
          lines.push(currentLine);
          maxWidth = Math.max(maxWidth, currentLine.length);
        }
        
        // Limit to maximum 3 lines for readability
        if (lines.length > 3) {
          lines.splice(2, lines.length - 2, lines.slice(2).join(' '));
          if (lines[2].length > maxLineLength) {
            lines[2] = lines[2].slice(0, maxLineLength - 1) + '…';
          }
        }
        
        return { lines, maxWidth };
      };

      // Calculate text wrapping for all nodes to determine optimal sizing
      const nodeTextInfo = nodes.map(d => {
        // Dynamic max line length based on font size - smaller fonts can fit more characters
        const maxLineLength = Math.max(8, Math.min(20, 200 / fontSize));
        const { lines, maxWidth } = wrapText(d.name, maxLineLength);
        const estimatedWidth = maxWidth * (fontSize - 4) * 0.6; // Approximate character width
        const estimatedHeight = lines.length * (fontSize - 2); // Line height
        return {
          node: d,
          lines,
          textWidth: estimatedWidth,
          textHeight: estimatedHeight,
          radius: Math.max(20 + (d.confidence * 12), Math.max(estimatedWidth, estimatedHeight) / 2 + 10)
        };
      });

      // Create simulation with better force configuration
      const simulation = d3.forceSimulation<GraphNode>(nodes)
        .force('link', d3.forceLink<GraphNode, GraphLink>(links)
          .id(d => d.id)
          .distance(180)  // Much larger distance for bigger nodes
          .strength(0.8)  // Stronger links to keep connected nodes together
        )
        .force('charge', d3.forceManyBody().strength(-800))  // Much stronger repulsion for larger nodes
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius((_, i) => nodeTextInfo[i]?.radius + 5 || 55))  // Dynamic collision based on actual node size
        .force('x', d3.forceX(width / 2).strength(0.1))    // Gentle centering force
        .force('y', d3.forceY(height / 2).strength(0.1));  // Gentle centering force

      // Create links group (child of mainGroup for zoom/pan)
      const linkGroup = mainGroup.append('g').attr('class', 'links');
      const link = linkGroup
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('stroke', themeColors.linkColor)
        .attr('stroke-opacity', 0.8)
        .attr('stroke-width', d => Math.max(0.6, d.confidence * 1.2))
        .attr('stroke-dasharray', d => d.confidence < 0.7 ? '5,5' : 'none')  // Dashed lines for lower confidence
        .style('cursor', 'pointer')
        .on('mouseenter', function(_, d) {
          d3.select(this)
            .transition()
            .duration(200)
            .attr('stroke-width', Math.max(1.2, d.confidence * 1.8))
            .attr('stroke-opacity', 1.0);
        })
        .on('mouseleave', function(_, d) {
          d3.select(this)
            .transition()
            .duration(200)
            .attr('stroke-width', Math.max(0.6, d.confidence * 1.2))
            .attr('stroke-opacity', 0.8);
        });

      // Create link labels group
      const linkLabelGroup = mainGroup.append('g').attr('class', 'link-labels');
      const linkLabels = linkLabelGroup
        .selectAll('text')
        .data(links)
        .join('text')
        .attr('text-anchor', 'middle')
        .attr('dy', -10)
        .attr('font-size', `${Math.max(fontSize - 10, 6)}px`)
        .attr('font-weight', '400')
        .attr('font-family', 'Arial, Helvetica, system-ui, -apple-system, sans-serif')
        .attr('fill', themeColors.linkLabelColor)
        .attr('fill-opacity', '0.7')
        .attr('stroke', isDarkMode ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.8)')
        .attr('stroke-width', '0.2px')
        .attr('paint-order', 'stroke fill')
        .style('pointer-events', 'none')
        .style('filter', 'drop-shadow(0 1px 2px rgba(0,0,0,0.1))')
        .text(d => {
          const type = d.relationship_type.toUpperCase();
          return type.length > 12 ? type.slice(0, 12) + '…' : type;
        });

      // Create nodes group
      const nodeGroup = mainGroup.append('g').attr('class', 'nodes');
      const node = nodeGroup
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('r', d => 20 + (d.confidence * 12))  // Much larger nodes for better visibility
        .attr('fill', d => generateColor(d.type))
        .attr('stroke', '#fff')
        .attr('stroke-width', 0.5)
        .style('cursor', 'pointer')
        .style('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))')  // Add shadow for depth
        .on('mouseenter', function(_, d) {
          const nodeIndex = nodes.findIndex(n => n.id === d.id);
          const originalRadius = nodeTextInfo[nodeIndex]?.radius || (20 + (d.confidence * 12));
          d3.select(this)
            .transition()
            .duration(200)
            .attr('r', originalRadius * 1.2)
            .attr('stroke-width', 1);
        })
        .on('mouseleave', function(_, d) {
          const nodeIndex = nodes.findIndex(n => n.id === d.id);
          const originalRadius = nodeTextInfo[nodeIndex]?.radius || (20 + (d.confidence * 12));
          d3.select(this)
            .transition()
            .duration(200)
            .attr('r', originalRadius)
            .attr('stroke-width', 0.5);
        });

      // Update node radius based on text requirements
      node.attr('r', (_, i) => nodeTextInfo[i].radius);

      // Create node labels group with proper text wrapping
      const nodeLabelGroup = mainGroup.append('g').attr('class', 'node-labels');
      
      // Create a group for each node to hold multiple text lines
      const nodeLabelGroups = nodeLabelGroup
        .selectAll('g')
        .data(nodeTextInfo)
        .join('g')
        .attr('class', 'node-label-group')
        .style('pointer-events', 'none');

      // Add text lines for each node
      nodeLabelGroups.each(function(d) {
        const group = d3.select(this);
        const lineHeight = fontSize - 2;
        const totalHeight = d.lines.length * lineHeight;
        const startY = -(totalHeight - lineHeight) / 2; // Center the text block vertically
        
        d.lines.forEach((line, lineIndex) => {
          group.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', startY + lineIndex * lineHeight)
            .attr('font-size', `${fontSize - 4}px`)
            .attr('font-weight', '600')
            .attr('font-family', 'Arial, Helvetica, system-ui, -apple-system, sans-serif')
            .attr('fill', themeColors.nodeLabelColor)
            .style('user-select', 'none')
            .text(line);
        });
      });

      // Add tooltips for nodes
      node.append('title')
        .text(d => `${d.name}\nType: ${d.type}\nConfidence: ${(d.confidence * 100).toFixed(1)}%`);

      // Add tooltips for links with better entity name display
      link.append('title')
        .text(d => {
          const sourceName = typeof d.source === 'string' ? 
            (idToNameMap.get(d.source) || d.source) : 
            (d.source as GraphNode).name;
          const targetName = typeof d.target === 'string' ? 
            (idToNameMap.get(d.target) || d.target) : 
            (d.target as GraphNode).name;
          return `${d.relationship_type}\nFrom: ${sourceName}\nTo: ${targetName}\nConfidence: ${(d.confidence * 100).toFixed(1)}%`;
        });

      // Add drag behavior with error handling
      const dragHandler = d3.drag<SVGCircleElement, GraphNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        });

      node.call(dragHandler);

      // Update positions on simulation tick
      simulation.on('tick', () => {
        try {
          link
            .attr('x1', d => (d.source as GraphNode).x || 0)
            .attr('y1', d => (d.source as GraphNode).y || 0)
            .attr('x2', d => (d.target as GraphNode).x || 0)
            .attr('y2', d => (d.target as GraphNode).y || 0);

          linkLabels
            .attr('x', d => ((d.source as GraphNode).x! + (d.target as GraphNode).x!) / 2)
            .attr('y', d => ((d.source as GraphNode).y! + (d.target as GraphNode).y!) / 2);

          node
            .attr('cx', d => d.x || 0)
            .attr('cy', d => d.y || 0);

          nodeLabelGroups
            .attr('transform', (_, i) => `translate(${nodeTextInfo[i].node.x || 0}, ${nodeTextInfo[i].node.y || 0})`);
        } catch (tickError) {
          console.error('Error in simulation tick:', tickError);
        }
      });

      // Run simulation longer for better layout
      setTimeout(() => {
        simulation.stop();
        console.log('Simulation stopped after 8 seconds');
        
        // Auto-fit to view after simulation settles
        setTimeout(() => {
          fitToView();
        }, 100);
        console.log(`Final layout: ${nodes.length} nodes, ${links.length} links`);
      }, 8000);
      
      console.log('=== Visualization created successfully ===');
      
    } catch (error) {
      console.error('=== Critical error creating visualization ===', error);
      setError(`Visualization failed: ${error instanceof Error ? error.message : String(error)}`);
      
      // Fallback: show a simple error message in the SVG
      if (svgRef.current) {
        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();
        svg.append('text')
          .attr('x', dimensions.width / 2)
          .attr('y', dimensions.height / 2)
          .attr('text-anchor', 'middle')
          .attr('fill', 'red')
          .attr('font-size', `${fontSize}px`)
          .attr('font-family', 'Arial, Helvetica, system-ui, -apple-system, sans-serif')
          .text('Visualization Error - Check Console');
      }
    }
  };

  // Dynamic sizing and window resize handler
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: rect.width,
          height: rect.height
        });
      } else {
        // Fallback to window dimensions
        const headerHeight = isStatsCollapsed ? 60 : 200;
        setDimensions({
          width: window.innerWidth,
          height: window.innerHeight - headerHeight
        });
      }
    };

    // Set initial dimensions
    updateDimensions();

    // Add resize listener
    window.addEventListener('resize', updateDimensions);
    
    return () => {
      window.removeEventListener('resize', updateDimensions);
    };
  }, [isStatsCollapsed]);

  // Initial load
  useEffect(() => {
    fetchStats();
    fetchAvailableDocuments();
  }, []);

  // Auto-fetch entities when selectedDocument changes to ALL_ENTITIES
  useEffect(() => {
    if (selectedDocument === 'ALL_ENTITIES') {
      console.log('🔄 Auto-fetching entities for ALL_ENTITIES');
      fetchEntities('ALL_ENTITIES');
      fetchRelationships('ALL_ENTITIES');
    }
  }, [selectedDocument]);

  // Update visualization when data changes
  useEffect(() => {
    if (entities.length > 0) {
      createVisualization();
      
      // Auto-fit to view after a short delay to let nodes settle
      setTimeout(() => {
        fitToView();
      }, 1000);
    }
  }, [entities, relationships, dimensions, isDarkMode, fontSize]);

  // Keyboard shortcuts for zoom controls
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (entities.length === 0) return;
      
      switch (event.key) {
        case '+':
        case '=':
          event.preventDefault();
          zoomIn();
          break;
        case '-':
        case '_':
          event.preventDefault();
          zoomOut();
          break;
        case 'r':
        case 'R':
          if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            resetZoom();
          }
          break;
        case 'f':
        case 'F':
          if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            fitToView();
          }
          break;
        case 'Delete':
        case 'Backspace':
          if ((event.ctrlKey || event.metaKey) && selectedDocument) {
            event.preventDefault();
            handleDeleteClick();
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [entities.length]);

  // Store zoom behavior reference
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  // Zoom control functions
  const zoomIn = () => {
    if (svgRef.current && zoomBehaviorRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(300).call(zoomBehaviorRef.current.scaleBy, 1.5);
    }
  };

  const zoomOut = () => {
    if (svgRef.current && zoomBehaviorRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(300).call(zoomBehaviorRef.current.scaleBy, 0.67);
    }
  };

  const resetZoom = () => {
    if (svgRef.current && zoomBehaviorRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(500).call(zoomBehaviorRef.current.transform, d3.zoomIdentity);
    }
  };

  const fitToView = () => {
    if (svgRef.current && entities.length > 0 && zoomBehaviorRef.current) {
      const svg = d3.select(svgRef.current);
      // Calculate bounds of all nodes
      const nodes = svg.selectAll('.nodes circle').nodes() as SVGCircleElement[];
      if (nodes.length === 0) return;
      
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      nodes.forEach(node => {
        const x = parseFloat(node.getAttribute('cx') || '0');
        const y = parseFloat(node.getAttribute('cy') || '0');
        const r = parseFloat(node.getAttribute('r') || '0');
        minX = Math.min(minX, x - r);
        minY = Math.min(minY, y - r);
        maxX = Math.max(maxX, x + r);
        maxY = Math.max(maxY, y + r);
      });
      
      const width = maxX - minX;
      const height = maxY - minY;
      const padding = 50;
      
      const scale = Math.min(
        (dimensions.width - padding * 2) / width,
        (dimensions.height - padding * 2) / height
      );
      
      const translateX = (dimensions.width - width * scale) / 2 - minX * scale;
      const translateY = (dimensions.height - height * scale) / 2 - minY * scale;
      
      const transform = d3.zoomIdentity.translate(translateX, translateY).scale(scale);
      svg.transition().duration(750).call(zoomBehaviorRef.current.transform, transform);
    }
  };

  // Handle document selection
  const handleDocumentSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedDocument.trim()) {
      fetchEntities(selectedDocument.trim());
      fetchRelationships(selectedDocument.trim());
    }
  };

  // Delete document from knowledge graph
  const deleteDocument = async (documentId: string) => {
    if (!documentId) return;
    
    setDeleteLoading(true);
    try {
      const response = await fetch(`/api/v1/knowledge-graph/document/${documentId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('Document deleted successfully:', result);
        
        // Clear current data if we deleted the selected document
        if (documentId === selectedDocument) {
          setEntities([]);
          setRelationships([]);
          setSelectedDocument('');
        }
        
        // Refresh available documents list
        fetchAvailableDocuments();
        fetchStats();
        
        // Show success notification
        setError(null);
        setNotification({
          type: 'success',
          message: `Document "${documentId}" has been successfully deleted from the knowledge graph.`
        });
        
        // Auto-hide notification after 5 seconds
        setTimeout(() => setNotification(null), 5000);
      } else {
        const errorData = await response.json();
        setError(`Failed to delete document: ${errorData.detail || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Delete error:', err);
      setError(`Error deleting document: ${err instanceof Error ? err.message : 'Network error'}`);
    } finally {
      setDeleteLoading(false);
      setShowDeleteConfirm(false);
    }
  };

  // Handle delete button click with confirmation
  const handleDeleteClick = () => {
    if (!selectedDocument) return;
    setShowDeleteConfirm(true);
  };

  // Navigation button styling
  const getButtonStyle = (bgColor?: string) => ({
    padding: '8px 12px',
    border: `1px solid ${themeColors.border}`,
    borderRadius: '4px',
    background: bgColor || themeColors.surface,
    color: bgColor ? 'white' : themeColors.text,
    cursor: 'pointer',
    fontSize: '14px',
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    transition: 'all 0.2s ease'
  });

  const toggleDarkMode = () => {
    const newDarkMode = !isDarkMode;
    setIsDarkMode(newDarkMode);
    localStorage.setItem('jarvis-dark-mode', JSON.stringify(newDarkMode));
    document.body.setAttribute('data-theme', newDarkMode ? 'dark' : 'light');
  };

  const themeColors = getThemeColors();

  // Create Material-UI theme
  const theme = createTheme({
    palette: {
      mode: isDarkMode ? 'dark' : 'light',
      primary: {
        main: '#2196f3',
      },
      background: {
        default: isDarkMode ? '#121212' : '#f5f5f5',
        paper: isDarkMode ? '#1e1e1e' : '#ffffff',
      },
    },
  });

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    switch (newValue) {
      case 0:
        window.location.href = '/';
        break;
      case 1:
        window.location.href = '/multi-agent.html';
        break;
      case 2:
        window.location.href = '/workflow.html';
        break;
      case 3:
        window.location.href = '/settings.html';
        break;
      case 4:
        // Already on knowledge graph page
        break;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Jarvis AI Assistant
            </Typography>

            <IconButton onClick={toggleDarkMode} color="inherit">
              {isDarkMode ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Toolbar>
        </AppBar>

        {/* Navigation Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={4}
            onChange={handleTabChange} 
            aria-label="jarvis modes"
            centered
            sx={{
              '& .MuiTab-root': {
                fontSize: '1rem',
                fontWeight: 600,
                textTransform: 'none',
                minWidth: 120,
                padding: '12px 24px',
                '&.Mui-selected': {
                  color: 'primary.main',
                  fontWeight: 700
                }
              }
            }}
          >
            <Tab 
              label="Standard Chat" 
              id="tab-0"
              aria-controls="tabpanel-0"
            />
            <Tab 
              label="Multi-Agent" 
              id="tab-1"
              aria-controls="tabpanel-1"
            />
            <Tab 
              label="Workflow" 
              id="tab-2"
              aria-controls="tabpanel-2"
            />
            <Tab 
              label="Settings" 
              id="tab-3"
              aria-controls="tabpanel-3"
            />
            <Tab 
              label="Knowledge Graph" 
              id="tab-4"
              aria-controls="tabpanel-4"
            />
          </Tabs>
        </Box>

        {/* Knowledge Graph controls */}
        <Box sx={{ 
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          p: 2,
          bgcolor: 'background.paper',
          borderBottom: 1,
          borderColor: 'divider'
        }}>
          <Typography variant="h5" component="h2">Knowledge Graph Viewer</Typography>
          <button
            onClick={() => setIsStatsCollapsed(!isStatsCollapsed)}
            style={{
              background: isDarkMode ? '#424242' : '#e0e0e0',
              border: 'none',
              color: isDarkMode ? '#ffffff' : '#333333',
              padding: '8px 12px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            {isStatsCollapsed ? '📊 Show Stats' : '📊 Hide Stats'}
          </button>
        </Box>

        {/* Collapsible Stats Section */}
        {!isStatsCollapsed && stats && (
          <Box sx={{ 
            bgcolor: 'background.paper', 
            p: 2,
            borderBottom: 1,
            borderColor: 'divider',
            flexShrink: 0
          }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', marginBottom: '15px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '20px' }}>🔗</span>
              <div>
                <Typography variant="caption" color="text.secondary">Total Entities</Typography>
                <Typography variant="h6" fontWeight="bold">{stats.total_entities}</Typography>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '20px' }}>🔄</span>
              <div>
                <Typography variant="caption" color="text.secondary">Total Relationships</Typography>
                <Typography variant="h6" fontWeight="bold">{stats.total_relationships}</Typography>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '20px' }}>📄</span>
              <div>
                <Typography variant="caption" color="text.secondary">Documents Processed</Typography>
                <Typography variant="h6" fontWeight="bold">{stats.documents_processed}</Typography>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '20px' }}>⏰</span>
              <div>
                <Typography variant="caption" color="text.secondary">Last Updated</Typography>
                <Typography variant="body2" fontWeight="bold">{new Date(stats.last_updated).toLocaleString()}</Typography>
              </div>
            </div>
          </div>
          
          {Object.keys(stats.entity_types).length > 0 && (
            <div>
              <Typography variant="subtitle2" fontWeight="600" sx={{ mb: 1 }}>Entity Types:</Typography>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {Object.entries(stats.entity_types).map(([type, count]) => (
                  <span 
                    key={type} 
                    style={{ 
                      display: 'inline-flex',
                      alignItems: 'center',
                      padding: '4px 10px',
                      background: generateColor(type),
                      color: 'white',
                      borderRadius: '16px',
                      fontSize: '12px',
                      fontWeight: '500'
                    }}
                  >
                    {type}: {count}
                  </span>
                ))}
              </div>
            </div>
          )}
          </Box>
        )}

      {/* Control Toolbar */}
      <div style={{ 
        background: themeColors.surface, 
        padding: '12px 20px',
        borderBottom: `1px solid ${themeColors.border}`,
        display: 'flex',
        alignItems: 'center',
        gap: '15px',
        flexWrap: 'wrap',
        flexShrink: 0
      }}>
        {/* Document Selection */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label htmlFor="documentId" style={{ fontSize: '14px', fontWeight: '500', color: themeColors.text }}>Document:</label>
          {availableDocuments.length > 0 ? (
            <select
              id="documentId"
              value={selectedDocument}
              onChange={(e) => {
                setSelectedDocument(e.target.value);
                if (e.target.value.trim()) {
                  fetchEntities(e.target.value.trim());
                  fetchRelationships(e.target.value.trim());
                }
              }}
              style={{ 
                padding: '6px 10px', 
                border: `1px solid ${themeColors.border}`, 
                borderRadius: '4px',
                minWidth: '200px',
                fontSize: '14px',
                background: themeColors.surface,
                color: themeColors.text
              }}
            >
              <option value="">Select a document...</option>
              {availableDocuments.map(docId => (
                <option key={docId} value={docId}>
                  {docId === 'ALL_ENTITIES' ? 'View All Entities' : docId}
                </option>
              ))}
            </select>
          ) : (
            <input
              id="documentId"
              type="text"
              value={selectedDocument}
              onChange={(e) => setSelectedDocument(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && selectedDocument.trim()) {
                  fetchEntities(selectedDocument.trim());
                  fetchRelationships(selectedDocument.trim());
                }
              }}
              placeholder="Enter document ID"
              style={{ 
                padding: '6px 10px', 
                border: `1px solid ${themeColors.border}`, 
                borderRadius: '4px',
                minWidth: '200px',
                fontSize: '14px',
                background: themeColors.surface,
                color: themeColors.text
              }}
            />
          )}
        </div>

        {/* Font Size Control */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label htmlFor="fontSize" style={{ fontSize: '14px', fontWeight: '500', color: themeColors.text }}>Font Size:</label>
          <select
            id="fontSize"
            value={fontSize}
            onChange={handleFontSizeChange}
            style={{ 
              padding: '6px 10px', 
              border: `1px solid ${themeColors.border}`, 
              borderRadius: '4px',
              fontSize: '14px',
              background: themeColors.surface,
              color: themeColors.text,
              minWidth: '70px'
            }}
          >
            <option value={8}>8px</option>
            <option value={10}>10px</option>
            <option value={12}>12px</option>
            <option value={14}>14px</option>
            <option value={16}>16px</option>
            <option value={18}>18px</option>
            <option value={20}>20px</option>
          </select>
        </div>

        {/* Navigation Controls */}
        {entities.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginLeft: 'auto' }}>
            <div style={{ fontSize: '14px', color: themeColors.secondaryText, marginRight: '10px' }}>
              Zoom: {Math.round(zoomTransform.k * 100)}%
            </div>
            <button onClick={zoomIn} style={getButtonStyle()} title="Zoom In (+ key)">🔍+</button>
            <button onClick={zoomOut} style={getButtonStyle()} title="Zoom Out (- key)">🔍-</button>
            <button onClick={resetZoom} style={getButtonStyle()} title="Reset Zoom (Ctrl+R)">🔄</button>
            <button onClick={fitToView} style={getButtonStyle()} title="Fit to View (Ctrl+F)">📐</button>
            <button 
              onClick={() => {
                fetchAvailableDocuments();
                fetchStats();
              }}
              style={getButtonStyle('#28a745')}
              title="Refresh Data"
            >
              ♻️
            </button>
            <button 
              onClick={() => {
                if (selectedDocument) {
                  fetch(`/api/v1/knowledge-graph/debug/${selectedDocument}`)
                    .then(res => res.json())
                    .then(data => {
                      console.log('🔍 NEO4J DEBUG DATA:', data);
                      alert(`Debug data logged to console. Summary: ${data.summary.total_entities_in_neo4j} entities, ${data.summary.total_relationships_in_neo4j} relationships in Neo4j`);
                    })
                    .catch(err => {
                      console.error('Debug fetch error:', err);
                      alert('Failed to fetch debug data - check console');
                    });
                }
              }}
              style={getButtonStyle('#17a2b8')}
              title="Debug Neo4j Data"
            >
              🔍
            </button>
            <button 
              onClick={handleDeleteClick}
              disabled={!selectedDocument || deleteLoading}
              style={{
                ...getButtonStyle(selectedDocument && !deleteLoading ? '#dc3545' : '#6c757d'),
                opacity: selectedDocument && !deleteLoading ? 1 : 0.6,
                cursor: selectedDocument && !deleteLoading ? 'pointer' : 'not-allowed'
              }}
              title={selectedDocument ? `Delete document "${selectedDocument}" from knowledge graph (Ctrl+Delete)` : "Select a document to delete"}
            >
              {deleteLoading ? '⏳' : '🗑️'}
            </button>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div style={{ 
          background: '#f8d7da', 
          color: '#721c24', 
          padding: '10px 15px', 
          borderRadius: '4px',
          margin: '0 20px 20px 20px',
          border: '1px solid #f5c6cb'
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Success/Error Notification */}
      {notification && (
        <div style={{ 
          background: notification.type === 'success' ? '#d4edda' : '#f8d7da',
          color: notification.type === 'success' ? '#155724' : '#721c24',
          border: notification.type === 'success' ? '1px solid #c3e6cb' : '1px solid #f5c6cb',
          padding: '12px 15px', 
          borderRadius: '4px',
          margin: '0 20px 20px 20px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>{notification.type === 'success' ? '✅' : '❌'}</span>
            <span>{notification.message}</span>
          </div>
          <button
            onClick={() => setNotification(null)}
            style={{
              background: 'none',
              border: 'none',
              color: notification.type === 'success' ? '#155724' : '#721c24',
              cursor: 'pointer',
              fontSize: '16px',
              padding: '0',
              lineHeight: '1'
            }}
            title="Dismiss"
          >
            ×
          </button>
        </div>
      )}

      {/* Graph Visualization */}
      {entities.length > 0 ? (
        <VisualizationErrorBoundary onError={setError}>
          <div 
            ref={containerRef}
            style={{ 
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              background: themeColors.surface,
              border: `1px solid ${themeColors.border}`,
              borderRadius: '8px',
              overflow: 'hidden'
            }}
          >
            <div style={{ 
              padding: '10px 15px', 
              background: themeColors.surface, 
              borderBottom: `1px solid ${themeColors.border}`,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <h3 style={{ margin: '0', fontSize: '16px', fontWeight: '600' }}>
                Knowledge Graph: {selectedDocument === 'ALL_ENTITIES' ? 'All Entities' : `Document ${selectedDocument}`}
              </h3>
              <div style={{ fontSize: '12px', color: themeColors.secondaryText }}>
                {entities.length} entities • {relationships.length} relationships
                {relationships.length > 0 && (
                  <span style={{ 
                    marginLeft: '10px', 
                    padding: '2px 6px', 
                    borderRadius: '3px',
                    background: relationships.length > 0 ? '#d4edda' : '#f8d7da',
                    color: relationships.length > 0 ? '#155724' : '#721c24',
                    fontSize: '11px'
                  }}>
                    {relationships.length === 0 ? 'No connections' : `${relationships.length} connections found`}
                  </span>
                )}
              </div>
            </div>
            
            <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
              <svg 
                ref={svgRef} 
                style={{ 
                  width: '100%', 
                  height: '100%', 
                  display: 'block',
                  background: themeColors.background
                }}
              />
            </div>
          
            {/* Legend */}
            <div style={{ 
              padding: '10px 15px', 
              background: themeColors.surface, 
              borderTop: `1px solid ${themeColors.border}`,
              display: 'flex',
              alignItems: 'center',
              gap: '15px',
              flexWrap: 'wrap'
            }}>
              <strong style={{ fontSize: '12px', color: themeColors.text }}>Legend:</strong>
              {Array.from(new Set(entities.map(entity => entity.type))).sort().map(type => (
                <span 
                  key={type}
                  style={{ 
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    fontSize: '11px',
                    color: themeColors.text
                  }}
                >
                  <span 
                    style={{ 
                      width: '10px',
                      height: '10px',
                      background: generateColor(type),
                      borderRadius: '50%',
                      border: '1px solid white',
                      boxShadow: '0 1px 2px rgba(0,0,0,0.1)'
                    }}
                  />
                  {type}
                </span>
              ))}
            </div>
            
            {/* Special message for entities without relationships */}
            {entities.length > 0 && relationships.length === 0 && (
              <div style={{ 
                padding: '15px', 
                background: '#fff3cd', 
                border: '1px solid #ffeaa7',
                borderTop: 'none',
                borderBottomLeftRadius: '8px',
                borderBottomRightRadius: '8px',
                fontSize: '14px',
                color: '#856404'
              }}>
                <strong>⚠️ No relationships found</strong> - Entities are displayed but no connections between them were extracted or stored. 
                This could mean the document contains isolated entities or relationship extraction needs to be enabled during upload.
              </div>
            )}
          </div>
        </VisualizationErrorBoundary>
      ) : selectedDocument && !loading && (
        <div style={{ 
          textAlign: 'center', 
          padding: '40px', 
          color: themeColors.secondaryText,
          border: `2px dashed ${themeColors.border}`,
          borderRadius: '8px',
          margin: '20px',
          background: themeColors.surface
        }}>
          <h3 style={{ color: themeColors.text, marginBottom: '15px' }}>No Knowledge Graph Data Found</h3>
          <p>No entities found for document: <strong>{selectedDocument}</strong></p>
          <div style={{ fontSize: '14px', lineHeight: '1.6', maxWidth: '500px', margin: '0 auto' }}>
            <p><strong>Possible causes:</strong></p>
            <ul style={{ textAlign: 'left', display: 'inline-block' }}>
              <li>Document hasn't been processed for knowledge graph extraction</li>
              <li>Document ID doesn't exist in Neo4j database</li>
              <li>Entity extraction failed during document processing</li>
              <li>Neo4j service is not connected or configured properly</li>
            </ul>
            <p style={{ marginTop: '15px' }}>
              <strong>Next steps:</strong> Upload the document through the main interface with knowledge graph processing enabled.
            </p>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: themeColors.surface,
            borderRadius: '8px',
            padding: '24px',
            maxWidth: '500px',
            width: '90%',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
            color: themeColors.text
          }}>
            <h3 style={{
              margin: '0 0 16px 0',
              color: '#dc3545',
              fontSize: '18px',
              fontWeight: '600'
            }}>
              ⚠️ Delete Document from Knowledge Graph
            </h3>
            
            <p style={{
              margin: '0 0 20px 0',
              lineHeight: '1.5',
              color: themeColors.text
            }}>
              Are you sure you want to delete document <strong>"{selectedDocument}"</strong> from the knowledge graph?
            </p>
            
            <div style={{
              background: '#fff3cd',
              border: '1px solid #ffeaa7',
              borderRadius: '4px',
              padding: '12px',
              marginBottom: '20px',
              fontSize: '14px',
              color: '#856404'
            }}>
              <strong>⚠️ Warning:</strong> This action will permanently remove:
              <ul style={{ margin: '8px 0 0 0', paddingLeft: '20px' }}>
                <li>All entities extracted from this document</li>
                <li>All relationships between those entities</li>
                <li>All knowledge graph data for this document</li>
              </ul>
              <div style={{ marginTop: '8px', fontWeight: '600' }}>
                This action cannot be undone.
              </div>
            </div>

            <div style={{
              display: 'flex',
              gap: '12px',
              justifyContent: 'flex-end'
            }}>
              <button
                onClick={() => setShowDeleteConfirm(false)}
                disabled={deleteLoading}
                style={{
                  padding: '8px 16px',
                  border: `1px solid ${themeColors.border}`,
                  borderRadius: '4px',
                  background: themeColors.surface,
                  color: themeColors.text,
                  cursor: deleteLoading ? 'not-allowed' : 'pointer',
                  fontSize: '14px',
                  fontWeight: '500'
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => deleteDocument(selectedDocument)}
                disabled={deleteLoading}
                style={{
                  padding: '8px 16px',
                  border: 'none',
                  borderRadius: '4px',
                  background: deleteLoading ? '#6c757d' : '#dc3545',
                  color: 'white',
                  cursor: deleteLoading ? 'not-allowed' : 'pointer',
                  fontSize: '14px',
                  fontWeight: '500'
                }}
              >
                {deleteLoading ? '⏳ Deleting...' : '🗑️ Delete Document'}
              </button>
            </div>
          </div>
        </div>
      )}

      {!selectedDocument && !loading && (
        <div style={{ 
          textAlign: 'center', 
          padding: '40px', 
          color: themeColors.secondaryText,
          border: `2px dashed ${themeColors.border}`,
          borderRadius: '8px',
          background: themeColors.surface
        }}>
          {availableDocuments.length === 0 ? (
            <div>
              <h3>No documents found in the knowledge graph</h3>
              <p>To create knowledge graph documents, you need to:</p>
              <ol style={{ textAlign: 'left', display: 'inline-block' }}>
                <li>Upload documents through the main interface</li>
                <li>Use the Knowledge Graph ingestion endpoint:
                  <code style={{ display: 'block', background: '#f5f5f5', padding: '10px', margin: '10px 0', borderRadius: '4px' }}>
                    curl -X POST "http://localhost:8000/api/v1/knowledge-graph/ingest" \<br/>
                    &nbsp;&nbsp;-F "file=@your-document.pdf"
                  </code>
                </li>
                <li>Or fix the spaCy configuration issue in the backend</li>
              </ol>
              <p>After ingestion, document IDs will appear here automatically.</p>
            </div>
          ) : (
            'Select a document from the dropdown above to view its knowledge graph visualization.'
          )}
        </div>
      )}
      </Box>
    </ThemeProvider>
  );
};

export default KnowledgeGraphViewer;