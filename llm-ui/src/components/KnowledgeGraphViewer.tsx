import React, { useState, useEffect, useRef, ErrorInfo, Component, useMemo, useCallback, memo } from 'react';
import * as d3 from 'd3';

// OPTIMIZATION: Enhanced throttling utility with RAF for smooth animations
const throttleRAF = (func: Function) => {
  let rafId: number | null = null;
  return function(this: any, ...args: any[]) {
    if (rafId === null) {
      rafId = requestAnimationFrame(() => {
        func.apply(this, args);
        rafId = null;
      });
    }
  };
};

// OPTIMIZATION: Throttling utility for better performance
const throttle = (func: Function, limit: number) => {
  let inThrottle: boolean;
  return function(this: any, ...args: any[]) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  }
};

// OPTIMIZATION: Debouncing utility for better performance
const debounce = (func: Function, wait: number) => {
  let timeout: NodeJS.Timeout;
  return function(this: any, ...args: any[]) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
};

// OPTIMIZATION: Viewport culling for performance
const isNodeInViewport = (node: GraphNode, transform: d3.ZoomTransform, width: number, height: number, margin: number = 100): boolean => {
  if (!node.x || !node.y) return true; // Always render nodes without position
  
  const screenX = node.x * transform.k + transform.x;
  const screenY = node.y * transform.k + transform.y;
  
  return screenX >= -margin && screenX <= width + margin &&
         screenY >= -margin && screenY <= height + margin;
};

// OPTIMIZATION: Performance monitoring
const PerformanceMonitor = {
  startTime: 0,
  
  start(label: string) {
    this.startTime = performance.now();
    console.time(label);
  },
  
  end(label: string) {
    const endTime = performance.now();
    console.timeEnd(label);
    const duration = endTime - this.startTime;
    if (duration > 16) { // Flag operations taking more than one frame (16ms)
      console.warn(`Performance warning: ${label} took ${duration.toFixed(2)}ms`);
    }
    return duration;
  }
};
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
  Tab,
  TextField,
  Chip,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Search as SearchIcon,
  ExpandMore as ExpandMoreIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon
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
  
  // Node diameter and layout configuration - READABLE SIZES
  const [nodeDiameter, setNodeDiameter] = useState(() => {
    const saved = localStorage.getItem('jarvis-kg-node-diameter');
    return saved ? parseInt(saved) : 50; // Increased default from 40 to 50 for better readability
  });
  
  const [layoutType, setLayoutType] = useState(() => {
    const saved = localStorage.getItem('jarvis-kg-layout-type');
    return saved || 'force-directed';
  });
  
  const [forceStrength, setForceStrength] = useState(() => {
    const saved = localStorage.getItem('jarvis-kg-force-strength');
    return saved ? parseFloat(saved) : 1.0;
  });
  
  // Performance optimization states
  const [searchQuery, setSearchQuery] = useState('');
  const [maxVisibleNodes, setMaxVisibleNodes] = useState(50);
  const [collapsedClusters, setCollapsedClusters] = useState<Set<string>>(new Set());
  const [showIsolatedNodes, setShowIsolatedNodes] = useState(false);

  // Enhanced search results tracking
  const [searchResults, setSearchResults] = useState<{
    directMatches: Set<string>;
    associatedNodes: Set<string>;
    searchRelationships: Set<string>;
  }>({
    directMatches: new Set(),
    associatedNodes: new Set(),
    searchRelationships: new Set()
  });
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);
  
  // Theme state
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('jarvis-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });
  
  // OPTIMIZATION: Memoized entity color generation
  const generateColor = useCallback((type: string): string => {
    const colors = getEntityColors(isDarkMode);
    const typeKey = type.toUpperCase();
    
    if (colors[typeKey]) {
      return colors[typeKey];
    }
    
    // Generate consistent color for unknown types
    let hash = 0;
    for (let i = 0; i < typeKey.length; i++) {
      hash = typeKey.charCodeAt(i) + ((hash << 5) - hash);
    }
    
    const hue = Math.abs(hash) % 360;
    const adjustedHue = (hue >= 60 && hue <= 120) ? (hue + 60) % 360 : hue;
    
    const saturation = isDarkMode ? 85 : 75;
    const lightness = isDarkMode ? 70 : 40;
    
    return `hsl(${adjustedHue}, ${saturation}%, ${lightness}%)`;
  }, [isDarkMode]);
  
  // OPTIMIZATION: Memoized event handlers to prevent unnecessary re-renders
  const handleFontSizeChange = useCallback((event: React.ChangeEvent<HTMLSelectElement>) => {
    const newSize = parseInt(event.target.value);
    setFontSize(newSize);
    localStorage.setItem('jarvis-kg-font-size', newSize.toString());
  }, []);
  
  const handleNodeDiameterChange = useCallback((event: React.ChangeEvent<HTMLSelectElement>) => {
    const newDiameter = parseInt(event.target.value);
    setNodeDiameter(newDiameter);
    localStorage.setItem('jarvis-kg-node-diameter', newDiameter.toString());
  }, []);
  
  const handleLayoutTypeChange = useCallback((event: React.ChangeEvent<HTMLSelectElement>) => {
    const newLayout = event.target.value;
    setLayoutType(newLayout);
    localStorage.setItem('jarvis-kg-layout-type', newLayout);
  }, []);
  
  const handleForceStrengthChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const newStrength = parseFloat(event.target.value);
    setForceStrength(newStrength);
    localStorage.setItem('jarvis-kg-force-strength', newStrength.toString());
  }, []);
  
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const nodesRef = useRef<GraphNode[]>([]);
  const simulationRef = useRef<d3.Simulation<GraphNode, undefined> | null>(null);
  const [dimensions, setDimensions] = useState({ width: window.innerWidth, height: window.innerHeight - 200 });
  const [isStatsCollapsed, setIsStatsCollapsed] = useState(false);
  const [zoomTransform, setZoomTransform] = useState(d3.zoomIdentity);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [notification, setNotification] = useState<{type: 'success' | 'error', message: string} | null>(null);

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

  // Performance optimization: calculate node degree centrality
  const calculateNodeDegrees = useMemo(() => {
    const degrees = new Map<string, number>();
    
    // Initialize all entities with degree 0
    entities.forEach(entity => {
      degrees.set(entity.id, 0);
    });
    
    // Count connections for each entity
    relationships.forEach(rel => {
      const sourceId = rel.source_entity;
      const targetId = rel.target_entity;
      degrees.set(sourceId, (degrees.get(sourceId) || 0) + 1);
      degrees.set(targetId, (degrees.get(targetId) || 0) + 1);
    });
    
    return degrees;
  }, [entities, relationships]);

  // Group entities by type for clustering
  const entityClusters = useMemo(() => {
    const clusters = new Map<string, Entity[]>();
    
    entities.forEach(entity => {
      const type = entity.type || 'UNKNOWN';
      if (!clusters.has(type)) {
        clusters.set(type, []);
      }
      clusters.get(type)!.push(entity);
    });
    
    return clusters;
  }, [entities]);

  // Filter entities based on search and visibility settings
  const visibleEntities = useMemo(() => {
    let filtered = entities;
    
    // ENHANCEMENT: Apply search filter with associated nodes
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      
      // Find directly matching entities
      const directMatches = entities.filter(entity => 
        entity.name.toLowerCase().includes(query) ||
        (entity.type && entity.type.toLowerCase().includes(query))
      );
      
      // ENHANCEMENT: Find associated nodes (connected entities)
      const directMatchIds = new Set(directMatches.map(e => e.id));
      const associatedIds = new Set<string>();
      const searchRelationshipIds = new Set<string>();
      
      // Add entities that have relationships with the direct matches
      relationships.forEach(rel => {
        const relId = `${rel.source_entity}-${rel.target_entity}`;
        
        if (directMatchIds.has(rel.source_entity)) {
          associatedIds.add(rel.target_entity);
          searchRelationshipIds.add(relId);
        }
        if (directMatchIds.has(rel.target_entity)) {
          associatedIds.add(rel.source_entity);
          searchRelationshipIds.add(relId);
        }
      });
      
      // Remove direct matches from associated nodes to avoid duplicates
      directMatchIds.forEach(id => associatedIds.delete(id));
      
      // Update search results state
      setSearchResults({
        directMatches: directMatchIds,
        associatedNodes: associatedIds,
        searchRelationships: searchRelationshipIds
      });
      
      // Include direct matches and their associated nodes
      filtered = entities.filter(entity => 
        directMatchIds.has(entity.id) || associatedIds.has(entity.id)
      );
      
      console.log(`🔍 Search enhanced: "${query}" found ${directMatches.length} direct matches and ${associatedIds.size} associated nodes`);
    } else {
      // Clear search results when no search query
      setSearchResults({
        directMatches: new Set(),
        associatedNodes: new Set(),
        searchRelationships: new Set()
      });
    }
    
    // OPTIMIZATION: Efficient clustering with performance-aware sorting
    // First apply cluster visibility filter (cheaper operation)
    filtered = filtered.filter(entity => {
      if (entity.type && collapsedClusters.has(entity.type)) {
        return false;
      }
      return true;
    });
    
    // OPTIMIZATION: Only sort and limit if not searching (expensive operations)
    if (!searchQuery.trim()) {
      // Sort by degree centrality (most connected first) only when needed
      filtered = filtered
        .map(entity => ({
          entity,
          degree: calculateNodeDegrees.get(entity.id) || 0
        }))
        .sort((a, b) => b.degree - a.degree)
        .slice(0, maxVisibleNodes) // Apply limit early to reduce downstream processing
        .map(item => item.entity);
      
      console.log(`🎯 Performance clustering: Limited to top ${Math.min(filtered.length, maxVisibleNodes)} most connected nodes`);
    }
    
    return filtered;
  }, [entities, relationships, searchQuery, calculateNodeDegrees, collapsedClusters, maxVisibleNodes]);

  // Filter relationships to only show connections between visible entities
  const visibleRelationships = useMemo(() => {
    const visibleEntityIds = new Set(visibleEntities.map(e => e.id));
    return relationships.filter(rel => 
      visibleEntityIds.has(rel.source_entity) && 
      visibleEntityIds.has(rel.target_entity)
    );
  }, [relationships, visibleEntities]);

  // Identify isolated nodes
  const isolatedEntities = useMemo(() => {
    const connectedIds = new Set<string>();
    visibleRelationships.forEach(rel => {
      connectedIds.add(rel.source_entity);
      connectedIds.add(rel.target_entity);
    });
    
    return visibleEntities.filter(entity => !connectedIds.has(entity.id));
  }, [visibleEntities, visibleRelationships]);


  // OPTIMIZATION: Memoized theme colors to prevent recalculation on every render
  const themeColors = useMemo(() => ({
    background: isDarkMode ? '#121212' : '#fafafa',
    surface: isDarkMode ? '#1e1e1e' : '#ffffff',
    text: isDarkMode ? '#ffffff' : '#333333',
    secondaryText: isDarkMode ? '#b3b3b3' : '#666666',
    border: isDarkMode ? '#424242' : '#e0e0e0',
    linkColor: isDarkMode ? '#999999' : '#666666',
    linkLabelColor: isDarkMode ? '#ffffff' : '#333333',
    nodeLabelColor: isDarkMode ? '#ffffff' : '#222222'
  }), [isDarkMode]);

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

  // OPTIMIZATION: Memoized fetch functions to prevent recreation on every render
  const fetchAvailableDocuments = useCallback(async () => {
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
  }, [selectedDocument]);

  // Fetch graph statistics
  const fetchStats = useCallback(async () => {
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
  }, []);

  // Fetch entities for a specific document or all entities
  const fetchEntities = useCallback(async (documentId: string) => {
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
  }, []);

  // Fetch relationships for a specific document or all relationships
  const fetchRelationships = useCallback(async (documentId: string) => {
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
  }, []);

  // OPTIMIZATION: Memoized visualization function
  const createVisualization = useCallback(() => {
    console.log('=== Starting D3 Visualization ===');
    console.log('Total Entities:', entities.length);
    console.log('Visible Entities:', visibleEntities.length);
    console.log('Total Relationships:', relationships.length);
    console.log('Visible Relationships:', visibleRelationships.length);
    console.log('Performance Mode: Showing top', maxVisibleNodes, 'most connected nodes');
    
    if (!svgRef.current) {
      console.error('SVG ref is null');
      setError('SVG reference not available');
      return;
    }
    
    if (visibleEntities.length === 0) {
      console.log('No visible entities to visualize');
      return;
    }

    try {
      // Clear any existing error
      setError(null);
      
      // Clear previous visualization
      const svg = d3.select(svgRef.current);
      svg.selectAll('*').remove();
      console.log('Cleared previous visualization');

      // Prepare nodes with better error handling - using filtered visible entities
      const nodes: GraphNode[] = visibleEntities.map((entity, index) => {
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
      const nodeEntityIds = new Set(nodes.map(n => n.id));
      const nodeEntityNames = new Set(nodes.map(n => n.name));
      const nodeNameToIdMap = new Map(nodes.map(n => [n.name.toLowerCase(), n.id]));
      const nodeIdToNameMap = new Map(nodes.map(n => [n.id, n.name]));
      
      console.log('🔗 Entity IDs available:', Array.from(nodeEntityIds));
      console.log('🔗 Entity names available:', Array.from(nodeEntityNames));
      console.log('🔍 DEBUGGING RELATIONSHIP FILTERING:');
      console.log(`   • Total relationships to process: ${relationships.length}`);
      console.log(`   • Entity IDs available: ${nodeEntityIds.size}`);

      // Use the already filtered visible relationships for better performance
      console.log(`🎯 PERFORMANCE OPTIMIZATION: Reduced from ${relationships.length} to ${visibleRelationships.length} relationships`);
      
      const links: GraphLink[] = visibleRelationships.map((rel, index) => {
        const link = {
          source: rel.source_entity,
          target: rel.target_entity,
          relationship_type: rel.relationship_type || 'RELATED',
          confidence: Math.max(0, Math.min(1, rel.confidence || 0.5))
        };
        console.log(`Link ${index}:`, link);
        return link;
      });
      
      console.log('📊 PERFORMANCE-OPTIMIZED RELATIONSHIP PROCESSING:');
      console.log(`   • Total relationships in dataset: ${relationships.length}`);
      console.log(`   • Visible relationships after performance filtering: ${visibleRelationships.length}`);
      console.log(`   • Final D3 links created: ${links.length}`);
      console.log(`   • Total entities in dataset: ${entities.length}`);
      console.log(`   • Visible entities after performance filtering: ${nodes.length}`);
      console.log(`   • Performance improvement: ${Math.round((1 - (links.length / relationships.length)) * 100)}% reduction in rendered relationships`);
      
      if (visibleRelationships.length > 0) {
        console.log(`✅ Successfully processed ${visibleRelationships.length} visible relationships`);
      } else if (relationships.length > 0) {
        console.log(`⚠️ No visible relationships (filtered for performance)`);
      }

      // Set up SVG with safe dimensions
      const width = Math.max(400, dimensions.width);
      const height = Math.max(300, dimensions.height);
      
      svg
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`)
        .style('background', themeColors.background)
        .style('cursor', 'grab');

      // Create main container group for zoom/pan transforms
      const mainGroup = svg.append('g').attr('class', 'main-group');

      // OPTIMIZATION: Enhanced zoom behavior with RAF throttling and viewport culling updates
      const throttledZoomUpdate = throttleRAF((transform: any) => {
        PerformanceMonitor.start('zoom-update');
        
        // Update viewport culling when zoom changes significantly
        const visibleNodes = nodes.filter(d => isNodeInViewport(d, transform, width, height));
        const visibleNodeIds = new Set(visibleNodes.map(d => d.id));
        
        // Update node visibility based on viewport
        mainGroup.selectAll('circle')
          .style('display', (d: any) => isNodeInViewport(d, transform, width, height) ? 'block' : 'none');
        
        // Update link visibility based on viewport
        mainGroup.selectAll('.links line')
          .style('display', (d: any) => {
            const sourceVisible = visibleNodeIds.has((d.source as GraphNode).id);
            const targetVisible = visibleNodeIds.has((d.target as GraphNode).id);
            return (sourceVisible || targetVisible) ? 'block' : 'none';
          });
        
        // Adjust font sizes based on zoom level for better readability (only for significant changes)
        if (Math.abs(transform.k - 1) > 0.2) { // Increased threshold to reduce updates
          const scaledFontSize = Math.max(8, (fontSize - 4) / Math.sqrt(transform.k));
          const scaledLinkFontSize = Math.max(6, Math.max(fontSize - 10, 6) / Math.sqrt(transform.k));
          
          // Update node label font sizes
          mainGroup.selectAll('.node-label-group text')
            .attr('font-size', `${scaledFontSize}px`);
            
          // Update link label font sizes
          mainGroup.selectAll('.link-labels text')
            .attr('font-size', `${scaledLinkFontSize}px`);
        }
        
        PerformanceMonitor.end('zoom-update');
      });

      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
          PerformanceMonitor.start('zoom-event');
          const transform = event.transform;
          setZoomTransform(transform);
          mainGroup.attr('transform', transform.toString());
          
          // If this is a user-initiated zoom (has sourceEvent), disable initial load auto-fit
          if (event.sourceEvent && isInitialLoad) {
            console.log('User interaction detected - disabling initial load auto-fit');
            setIsInitialLoad(false);
          }
          
          // OPTIMIZATION: Use RAF-throttled viewport and font updates
          throttledZoomUpdate(transform);
          PerformanceMonitor.end('zoom-event');
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

      // Calculate adaptive node sizing based on configuration and content - OPTIMIZED FOR MINIMAL PADDING
      const calculateNodeRadius = (node: GraphNode, textInfo: any) => {
        const baseRadius = nodeDiameter / 2;
        const confidenceBonus = node.confidence * (nodeDiameter * 0.3);
        const textSize = Math.max(textInfo.textWidth, textInfo.textHeight) / 2;
        // Reduced padding from 10px to just enough for text readability (6px)
        return Math.max(baseRadius + confidenceBonus, textSize + 6);
      };
      
      // Calculate text wrapping for all nodes to determine optimal sizing
      const nodeTextInfo = nodes.map(d => {
        // Dynamic max line length based on font size - smaller fonts can fit more characters
        const maxLineLength = Math.max(8, Math.min(20, 200 / fontSize));
        const { lines, maxWidth } = wrapText(d.name, maxLineLength);
        const estimatedWidth = maxWidth * (fontSize - 4) * 0.6; // Approximate character width
        const estimatedHeight = lines.length * (fontSize - 2); // Line height
        const textInfo = {
          lines,
          textWidth: estimatedWidth,
          textHeight: estimatedHeight
        };
        return {
          node: d,
          ...textInfo,
          radius: calculateNodeRadius(d, textInfo)
        };
      });

      // Calculate proper D3.js force parameters based on documented best practices
      const calculateForceParameters = () => {
        // D3.js documented best practices for force parameters
        
        // 1. Link Distance: Reduced for tighter node spacing
        const baseDistance = Math.max(80, Math.min(width, height) * 0.15); // Reduced from 150 and 0.2
        const linkDistance = baseDistance * forceStrength;
        
        // 2. Charge Force: Balanced repulsion for unified single-group layout
        const baseCharge = -200; // Further reduced for tighter unified layout with 5-8px spacing
        const chargeStrength = baseCharge * forceStrength;
        
        // 3. Collision Radius: Node radius + padding for consistent 5-8px spacing between all nodes
        const avgNodeRadius = nodeTextInfo.reduce((sum, info) => sum + info.radius, 0) / nodeTextInfo.length;
        const padding = 12; // 12px padding = consistent 12-15px spacing between all nodes in unified layout
        const collisionRadius = avgNodeRadius + padding;
        
        return {
          linkDistance,
          chargeStrength,
          collisionRadius: (d: GraphNode) => {
            // Find the correct node info for this specific node
            const nodeInfo = nodeTextInfo.find(info => info.node.id === d.id);
            const nodeRadius = nodeInfo?.radius || avgNodeRadius;
            const collisionRadius = nodeRadius + padding;
            
            // Debug log for first few nodes to verify 5-8px spacing calculation
            if (nodes.indexOf(d) < 3) {
              console.log(`🔍 Unified Layout - Node ${d.name}: visual=${nodeRadius.toFixed(1)}px + padding=${padding}px = collision=${collisionRadius.toFixed(1)}px (12-15px spacing)`);
            }
            
            return collisionRadius; // Individual node radius + padding
          }
        };
      };
      
      const forceParams = calculateForceParameters();
      
      // Apply layout-specific positioning
      const applyLayoutPositioning = () => {
        const centerX = width / 2;
        const centerY = height / 2;
        const padding = 80; // Keep nodes away from edges
        const usableWidth = width - (padding * 2);
        const usableHeight = height - (padding * 2);
        
        switch (layoutType) {
          case 'radial':
            // Position nodes in concentric circles - FULL VIEWPORT USAGE
            nodes.forEach((node, i) => {
              const degree = calculateNodeDegrees.get(node.id) || 0;
              const maxDegree = Math.max(...Array.from(calculateNodeDegrees.values()));
              const normalizedDegree = maxDegree > 0 ? degree / maxDegree : 0;
              
              // Use 80% of available viewport space instead of tiny 40%
              const maxRadius = Math.min(usableWidth, usableHeight) * 0.4;
              const radius = maxRadius * (0.3 + normalizedDegree * 0.7);
              const angle = (i / nodes.length) * 2 * Math.PI;
              
              node.x = centerX + radius * Math.cos(angle);
              node.y = centerY + radius * Math.sin(angle);
              node.fx = node.x;
              node.fy = node.y;
            });
            break;
            
          case 'grid':
            // Arrange nodes in a grid pattern - FULL VIEWPORT USAGE
            const cols = Math.ceil(Math.sqrt(nodes.length));
            const rows = Math.ceil(nodes.length / cols);
            const cellWidth = usableWidth / cols;
            const cellHeight = usableHeight / rows;
            
            nodes.forEach((node, i) => {
              const col = i % cols;
              const row = Math.floor(i / cols);
              
              node.x = padding + col * cellWidth + cellWidth / 2;
              node.y = padding + row * cellHeight + cellHeight / 2;
              node.fx = node.x;
              node.fy = node.y;
            });
            break;
            
          case 'circular':
            // Arrange nodes in a single circle - FULL VIEWPORT USAGE
            nodes.forEach((node, i) => {
              const angle = (i / nodes.length) * 2 * Math.PI;
              // Use 70% of available space for better edge-to-edge spread
              const radius = Math.min(usableWidth, usableHeight) * 0.35;
              
              node.x = centerX + radius * Math.cos(angle);
              node.y = centerY + radius * Math.sin(angle);
              node.fx = node.x;
              node.fy = node.y;
            });
            break;
            
          default: // force-directed - UNIFIED SINGLE TIGHT GROUP LAYOUT
            nodes.forEach((node, i) => {
              node.fx = null;
              node.fy = null;
              
              // CRITICAL FIX: ALL NODES START IN SAME SMALL AREA
              // No circular distribution, no group separation - ONLY tight clustering
              const tightAreaSize = 100; // Small starting area
              node.x = centerX + (Math.random() - 0.5) * tightAreaSize;
              node.y = centerY + (Math.random() - 0.5) * tightAreaSize;
              
              // Ensure nodes stay within viewport bounds
              node.x = Math.max(padding, Math.min(width - padding, node.x));
              node.y = Math.max(padding, Math.min(height - padding, node.y));
            });
            break;
        }
      };
      
      // Create simulation with adaptive force configuration
      const simulation = d3.forceSimulation<GraphNode>(nodes);
      
      // Store references for fitToView function
      nodesRef.current = nodes;
      simulationRef.current = simulation;
      
      // Apply different forces based on layout type
      if (layoutType === 'force-directed') {
        // CRITICAL FIX: TIGHT CLUSTERING WITH VISIBLE RELATIONSHIP LINES
        // Collision dominates, links are weak and short to maintain tight spacing
        simulation
          .force('link', d3.forceLink(links)
            .id(d => d.id)
            .distance(25) // Very short links to keep nodes close together
            .strength(0.1) // Very weak so collision detection dominates
          )
          .force('collision', d3.forceCollide()
            .radius(d => {
              // Find the correct node info for this specific node
              const nodeInfo = nodeTextInfo.find(info => info.node.id === d.id);
              const nodeRadius = nodeInfo?.radius || (nodeDiameter / 2);
              return nodeRadius + 7; // 7px padding = 12-15px spacing between nodes
            })
            .strength(1.0) // Maximum collision strength to prevent overlap - PRIORITY FORCE
            .iterations(1) // OPTIMIZATION: Further reduced iterations from 2 to 1 for better performance
          )
          .force('center', d3.forceCenter(width / 2, height / 2).strength(0.1)) // Very weak centering only
          .alphaDecay(0.05) // OPTIMIZATION: Even faster alpha decay (was 0.03, now 0.05) for quicker settling
          .velocityDecay(0.8); // High velocity decay to prevent excessive movement
          // Link force restored but weak - collision force dominates for tight spacing
      } else {
        // For fixed layouts, only use collision detection
        simulation
          .force('collision', d3.forceCollide()
            .radius(forceParams.collisionRadius) // Individual node radius + padding
            .strength(1.0) // Maximum collision strength
            .iterations(1) // OPTIMIZATION: Further reduced iterations from 2 to 1 for better performance
          )
          .alpha(0.3) // Higher alpha for better collision resolution
          .alphaDecay(0.05); // Slower decay for more separation time
        
        // Apply the specific layout positioning
        applyLayoutPositioning();
      }
        
      // Extended simulation duration for collision detection and spreading
      const getSimulationDuration = () => {
        if (layoutType === 'force-directed') {
          // OPTIMIZATION: Reduced simulation time for better performance
          const baseTime = Math.max(5000, nodes.length * 50); // Further reduced from 8000 and nodes * 100
          const maxTime = Math.min(10000, baseTime); // Further reduced max time from 15000 to 10000
          return maxTime;
        } else {
          // OPTIMIZATION: Reduced time for fixed layouts too
          return Math.max(2000, nodes.length * 30); // Further reduced from 3000 and nodes * 50
        }
      };

      console.log('🎯 CRITICAL FIX: TIGHT SINGLE ENTITY LAYOUT - MAXIMUM 8PX SPACING', {
        layoutType: layoutType,
        nodeCount: nodes.length,
        viewportSize: { width, height },
        layoutStrategy: 'ALL NODES TREATED AS SINGLE ENTITY',
        spacingRequirement: 'OPTIMIZED 12-15px between nodes with minimal node padding',
        fixImplemented: 'Reduced node padding, increased collision spacing',
        forceConfiguration: {
          collisionOnly: 'Individual node radius + 7px padding = 12-15px spacing',
          collisionStrength: '1.0 (maximum)',
          collisionIterations: '3 (multiple passes)',
          centeringStrength: '0.1 (very weak only)',
          linkForceREMOVED: 'ELIMINATED - was creating group separation',
          chargeForceREMOVED: 'ELIMINATED - was preventing tight clustering',
          alphaDecay: '0.01 (very slow for tight settling)',
          velocityDecay: '0.8 (high to prevent excessive movement)'
        },
        userForceStrength: forceStrength,
        simulationDuration: getSimulationDuration(),
        layoutApproach: {
          groupStrategy: 'Single unified group - ALL nodes treated together',
          spacingStrategy: 'Consistent 5-8px spacing via collision detection',
          noComponentSeparation: 'Removed disconnected component logic',
          allNodesAsOne: 'No matter how many nodes or connections'
        },
        nodeSize: `${nodeDiameter}px diameter (readable size maintained)`
      });

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
            .attr('stroke-width', Math.max(1.0, d.confidence * 1.5))
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
        .attr('r', (_, i) => nodeTextInfo[i]?.radius || nodeDiameter / 2)  // Use calculated adaptive radius
        .attr('fill', d => {
          // ENHANCEMENT: Highlight search matches and associated nodes
          if (searchQuery) {
            if (searchResults.directMatches.has(d.id)) {
              return isDarkMode ? '#1565C0' : '#0D47A1'; // Dark blue for direct search matches with theme support
            } else if (searchResults.associatedNodes.has(d.id)) {
              return isDarkMode ? '#26A69A' : '#00695C'; // Teal for associated nodes with theme support
            } else {
              return generateColor(d.type); // Normal color for visible but non-search nodes
            }
          }
          // Highlight focused node
          if (focusedNodeId === d.id) {
            return '#FF6B6B'; // Red for focused node
          }
          return generateColor(d.type);
        })
        .attr('stroke', d => {
          // ENHANCEMENT: Different strokes for direct matches vs associated nodes
          if (searchQuery) {
            if (searchResults.directMatches.has(d.id)) {
              return isDarkMode ? '#42A5F5' : '#1976D2'; // Professional blue stroke for direct search matches
            } else if (searchResults.associatedNodes.has(d.id)) {
              return '#1976D2'; // Deep blue stroke for associated nodes
            } else {
              return '#999'; // Gray stroke for non-search nodes
            }
          }
          if (focusedNodeId === d.id) {
            return '#FF0000'; // Red stroke for focused node
          }
          return '#fff';
        })
        .attr('stroke-width', d => {
          // ENHANCEMENT: Different stroke widths for direct matches vs associated nodes
          if (searchQuery) {
            if (searchResults.directMatches.has(d.id)) {
              return 3; // Moderate stroke for direct search matches
            } else if (searchResults.associatedNodes.has(d.id)) {
              return 2; // Medium stroke for associated nodes
            } else {
              return 1; // Thin stroke for non-search nodes
            }
          }
          if (focusedNodeId === d.id) {
            return 3; // Thick stroke for focused node
          }
          return 0.5;
        })
        .style('opacity', d => {
          // ENHANCEMENT: Dim non-search-related nodes when searching
          if (searchQuery) {
            if (searchResults.directMatches.has(d.id) || searchResults.associatedNodes.has(d.id)) {
              return 1.0; // Full opacity for search-related nodes
            } else {
              return 0.4; // Dimmed for non-search nodes
            }
          }
          return 1.0; // Full opacity when not searching
        })
        .style('cursor', 'pointer')
        .style('filter', d => {
          // ENHANCEMENT: Different glows for direct matches vs associated nodes
          if (searchQuery) {
            if (searchResults.directMatches.has(d.id)) {
              return 'drop-shadow(0 4px 8px rgba(255,215,0,0.6)) drop-shadow(0 0 12px rgba(255,140,0,0.4))'; // Enhanced gold glow for direct matches
            } else if (searchResults.associatedNodes.has(d.id)) {
              return 'drop-shadow(0 2px 4px rgba(79,195,247,0.4)) drop-shadow(0 0 8px rgba(25,118,210,0.3))'; // Blue glow for associated nodes
            } else {
              return 'none'; // No glow for non-search nodes
            }
          }
          if (focusedNodeId === d.id) {
            return 'drop-shadow(0 4px 8px rgba(255,107,107,0.4))'; // Red glow for focused node
          }
          return 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))';
        })
        .on('mouseenter', function(_, d) {
          const nodeIndex = nodes.findIndex(n => n.id === d.id);
          const originalRadius = nodeTextInfo[nodeIndex]?.radius || (nodeDiameter / 2);
          d3.select(this)
            .transition()
            .duration(200)
            .attr('r', originalRadius * 1.2)
            .attr('stroke-width', 2);
        })
        .on('mouseleave', function(_, d) {
          const nodeIndex = nodes.findIndex(n => n.id === d.id);
          const originalRadius = nodeTextInfo[nodeIndex]?.radius || (nodeDiameter / 2);
          const originalStrokeWidth = searchQuery ? 
            (searchResults.directMatches.has(d.id) ? 4 : 
             searchResults.associatedNodes.has(d.id) ? 2.5 : 1) : 
            (focusedNodeId === d.id ? 3 : 0.5);
          d3.select(this)
            .transition()
            .duration(200)
            .attr('r', originalRadius)
            .attr('stroke-width', originalStrokeWidth);
        })
        .on('click', function(_, d) {
          // Focus on clicked node
          setFocusedNodeId(focusedNodeId === d.id ? null : d.id);
        });

      // Node radius is already set correctly above

      // Create node labels group with proper text wrapping
      const nodeLabelGroup = mainGroup.append('g').attr('class', 'node-labels');
      
      // Create a group for each node to hold multiple text lines
      const nodeLabelGroups = nodeLabelGroup
        .selectAll('g')
        .data(nodeTextInfo)
        .join('g')
        .attr('class', 'node-label-group')
        .style('pointer-events', 'none')
        .style('opacity', d => {
          // ENHANCEMENT: Dim text for non-search-related nodes
          if (searchQuery) {
            if (searchResults.directMatches.has(d.id) || searchResults.associatedNodes.has(d.id)) {
              return 1.0; // Full opacity for search-related node labels
            } else {
              return 0.4; // Dimmed for non-search node labels
            }
          }
          return 1.0; // Full opacity when not searching
        });

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

      // Add enhanced tooltips for nodes
      node.append('title')
        .text((d, i) => {
          const nodeInfo = nodeTextInfo[i];
          const degree = calculateNodeDegrees.get(d.id) || 0;
          return `${d.name}\nType: ${d.type}\nConfidence: ${(d.confidence * 100).toFixed(1)}%\nConnections: ${degree}\nRadius: ${Math.round(nodeInfo?.radius || 0)}px`;
        });

      // Add tooltips for links with better entity name display
      link.append('title')
        .text(d => {
          const sourceName = typeof d.source === 'string' ? 
            (nodeIdToNameMap.get(d.source) || d.source) : 
            (d.source as GraphNode).name;
          const targetName = typeof d.target === 'string' ? 
            (nodeIdToNameMap.get(d.target) || d.target) : 
            (d.target as GraphNode).name;
          return `${d.relationship_type}\nFrom: ${sourceName}\nTo: ${targetName}\nConfidence: ${(d.confidence * 100).toFixed(1)}%`;
        });

      // OPTIMIZATION: Add drag behavior with RAF throttling for smoother performance
      const throttledDragUpdate = throttleRAF((event: any, d: GraphNode) => {
        d.fx = event.x;
        d.fy = event.y;
      });

      const dragHandler = d3.drag<SVGCircleElement, GraphNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', throttledDragUpdate) // OPTIMIZATION: Use throttled drag handler
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        });

      node.call(dragHandler);

      // OPTIMIZATION: Throttled tick updates with viewport culling for better performance
      let tickCount = 0;
      const throttledTick = throttleRAF(() => {
        try {
          PerformanceMonitor.start('tick-update');
          
          // OPTIMIZATION: Only update every 2nd tick for better performance
          tickCount++;
          if (tickCount % 2 !== 0) return;
          
          // OPTIMIZATION: Batch DOM updates with viewport culling
          const visibleNodes = nodes.filter(d => isNodeInViewport(d, zoomTransform, width, height));
          const visibleNodeIds = new Set(visibleNodes.map(d => d.id));
          
          // Update links (only for visible nodes)
          link
            .style('display', d => {
              const sourceVisible = visibleNodeIds.has((d.source as GraphNode).id);
              const targetVisible = visibleNodeIds.has((d.target as GraphNode).id);
              return (sourceVisible || targetVisible) ? 'block' : 'none';
            })
            .attr('x1', d => (d.source as GraphNode).x || 0)
            .attr('y1', d => (d.source as GraphNode).y || 0)
            .attr('x2', d => (d.target as GraphNode).x || 0)
            .attr('y2', d => (d.target as GraphNode).y || 0);

          // Update link labels (only for visible links)
          linkLabels
            .style('display', d => {
              const sourceVisible = visibleNodeIds.has((d.source as GraphNode).id);
              const targetVisible = visibleNodeIds.has((d.target as GraphNode).id);
              return (sourceVisible || targetVisible) ? 'block' : 'none';
            })
            .attr('x', d => ((d.source as GraphNode).x! + (d.target as GraphNode).x!) / 2)
            .attr('y', d => ((d.source as GraphNode).y! + (d.target as GraphNode).y!) / 2);

          // Update nodes (with viewport culling)
          node
            .style('display', d => isNodeInViewport(d, zoomTransform, width, height) ? 'block' : 'none')
            .attr('cx', d => d.x || 0)
            .attr('cy', d => d.y || 0);

          // Update node labels (with viewport culling)
          nodeLabelGroups
            .style('display', (_, i) => isNodeInViewport(nodeTextInfo[i].node, zoomTransform, width, height) ? 'block' : 'none')
            .attr('transform', (_, i) => `translate(${nodeTextInfo[i].node.x || 0}, ${nodeTextInfo[i].node.y || 0})`);
          
          PerformanceMonitor.end('tick-update');
        } catch (tickError) {
          console.error('Error in simulation tick:', tickError);
        }
      });
      
      // Update positions on simulation tick
      simulation.on('tick', throttledTick);

      const simulationDuration = getSimulationDuration();
      
      // Run simulation with adaptive duration for better layout
      setTimeout(() => {
        simulation.stop();
        console.log(`${layoutType} layout simulation stopped after ${simulationDuration}ms`);
        
        // Update node references after simulation completes
        nodesRef.current = nodes;
        
        // Auto-fit to view after simulation settles with improved timing
        setTimeout(() => {
          // Always auto-fit on initial load, or if user hasn't made manual zoom/pan adjustments
          const shouldAutoFit = isInitialLoad || 
            (Math.abs(zoomTransform.k - 1) < 0.1 && 
             Math.abs(zoomTransform.x) < 10 && 
             Math.abs(zoomTransform.y) < 10);
          
          if (shouldAutoFit) {
            console.log(`Auto-fitting to view after simulation completion - ${isInitialLoad ? 'initial load' : 'normal'}`);
            fitToView();
            
            // Mark initial load as complete after first auto-fit
            if (isInitialLoad) {
              setIsInitialLoad(false);
            }
          }
        }, 200); // Slight delay to ensure all positions are finalized
        console.log(`Final ${layoutType} layout: ${nodes.length} nodes, ${links.length} links`);
      }, simulationDuration);
      
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
  }, [visibleEntities, visibleRelationships, dimensions, isDarkMode, fontSize, nodeDiameter, forceStrength, layoutType, themeColors, generateColor, zoomTransform, searchQuery, searchResults, focusedNodeId, maxVisibleNodes]);

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
  }, [fetchStats, fetchAvailableDocuments]);

  // Auto-fetch entities when selectedDocument changes to ALL_ENTITIES
  useEffect(() => {
    if (selectedDocument === 'ALL_ENTITIES') {
      console.log('🔄 Auto-fetching entities for ALL_ENTITIES');
      fetchEntities('ALL_ENTITIES');
      fetchRelationships('ALL_ENTITIES');
    }
  }, [selectedDocument, fetchEntities, fetchRelationships]);

  // Update visualization when visible data changes
  useEffect(() => {
    if (visibleEntities.length > 0) {
      createVisualization();
      
      // Auto-fit after visualization is created and settled
      setTimeout(() => {
        // Always auto-fit on initial load, or if user hasn't made manual zoom/pan adjustments
        const shouldAutoFit = isInitialLoad || 
          (Math.abs(zoomTransform.k - 1) < 0.1 && 
           Math.abs(zoomTransform.x) < 10 && 
           Math.abs(zoomTransform.y) < 10);
        
        if (shouldAutoFit) {
          console.log(`Auto-fitting to view - ${isInitialLoad ? 'initial load' : 'data change'}`);
          fitToView();
          
          // Mark initial load as complete after first auto-fit
          if (isInitialLoad) {
            setIsInitialLoad(false);
          }
        }
      }, 1200); // Longer delay to allow simulation to complete
    }
  }, [visibleEntities, visibleRelationships, dimensions, isDarkMode, fontSize, nodeDiameter, forceStrength, layoutType, isInitialLoad]);

  // Keyboard shortcuts for zoom controls
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (visibleEntities.length === 0) return;
      
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
  }, [visibleEntities.length]);

  // Store zoom behavior reference
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  // Zoom control functions
  const zoomIn = () => {
    if (svgRef.current && zoomBehaviorRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(300).call(zoomBehaviorRef.current.scaleBy, 1.1);
    }
  };

  const zoomOut = () => {
    if (svgRef.current && zoomBehaviorRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(300).call(zoomBehaviorRef.current.scaleBy, 0.91);
    }
  };

  const resetZoom = () => {
    if (svgRef.current && zoomBehaviorRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(500).call(zoomBehaviorRef.current.transform, d3.zoomIdentity);
    }
  };

  const fitToView = () => {
    if (!svgRef.current || !zoomBehaviorRef.current) {
      console.warn('fitToView: SVG or zoom behavior not available');
      return;
    }

    const svg = d3.select(svgRef.current);
    const nodes = nodesRef.current;
    
    if (!nodes || nodes.length === 0) {
      console.warn('fitToView: No nodes available');
      return;
    }

    // Filter nodes with valid positions - essential for accurate bounds
    const validNodes = nodes.filter(node => 
      typeof node.x === 'number' && 
      typeof node.y === 'number' && 
      !isNaN(node.x) && 
      !isNaN(node.y)
    );

    if (validNodes.length === 0) {
      console.warn('fitToView: No nodes have valid positions yet. Waiting for simulation to settle...');
      
      // If simulation is still running, wait for it to settle
      if (simulationRef.current && simulationRef.current.alpha() > 0.1) {
        setTimeout(() => fitToView(), 100);
        return;
      } else {
        // Fallback: try to get positions from DOM elements
        const nodeElements = svg.selectAll('circle').nodes() as SVGCircleElement[];
        if (nodeElements.length === 0) {
          console.warn('fitToView: No rendered nodes found in DOM');
          return;
        }
        
        // Use DOM positions as fallback
        const domNodes = nodeElements.map((el, i) => ({
          x: parseFloat(el.getAttribute('cx') || '0'),
          y: parseFloat(el.getAttribute('cy') || '0'),
          id: nodes[i]?.id || `node-${i}`
        }));
        
        return fitToViewWithNodes(domNodes, svg);
      }
    }

    return fitToViewWithNodes(validNodes, svg);
  };

  // Extracted helper function for actual fit-to-view calculation
  const fitToViewWithNodes = (validNodes: any[], svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
    // Calculate bounds with proper node radius consideration
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    let hasValidBounds = false;

    validNodes.forEach(node => {
      const x = node.x;
      const y = node.y;
      const r = nodeDiameter / 2;
      
      minX = Math.min(minX, x - r);
      minY = Math.min(minY, y - r);
      maxX = Math.max(maxX, x + r);
      maxY = Math.max(maxY, y + r);
      hasValidBounds = true;
    });

    if (!hasValidBounds) {
      console.warn('fitToView: Unable to calculate valid bounds');
      return;
    }

    const contentWidth = maxX - minX;
    const contentHeight = maxY - minY;

    // Handle edge cases
    if (validNodes.length === 1) {
      // Single node: center it with moderate zoom
      const singleNode = validNodes[0];
      const transform = d3.zoomIdentity
        .translate(dimensions.width / 2 - singleNode.x, dimensions.height / 2 - singleNode.y)
        .scale(1.0);
      
      console.log('🎯 Fit-to-View (Single Node):', {
        node: { x: singleNode.x, y: singleNode.y },
        transform: { x: dimensions.width / 2 - singleNode.x, y: dimensions.height / 2 - singleNode.y, k: 1.0 }
      });
      
      svg.transition().duration(750).call(zoomBehaviorRef.current!.transform, transform);
      return;
    }

    // Handle very small content (all nodes clustered)
    const minContentSize = nodeDiameter * 2;
    const effectiveWidth = Math.max(contentWidth, minContentSize);
    const effectiveHeight = Math.max(contentHeight, minContentSize);

    // Calculate padding based on viewport and content size
    const viewportSize = Math.min(dimensions.width, dimensions.height);
    const basePadding = Math.max(30, viewportSize * 0.03); // Minimum 30px, or 3% of viewport
    const nodePadding = nodeDiameter * 1.5; // Extra space around nodes
    const adaptivePadding = Math.max(basePadding, nodePadding);

    // Calculate optimal scale with improved logic
    const availableWidth = Math.max(100, dimensions.width - adaptivePadding * 2);
    const availableHeight = Math.max(100, dimensions.height - adaptivePadding * 2);
    
    const scaleX = availableWidth / effectiveWidth;
    const scaleY = availableHeight / effectiveHeight;
    let optimalScale = Math.min(scaleX, scaleY);
    
    // Apply scale constraints
    const maxScale = Math.min(3.0, viewportSize / 200); // Adaptive max scale based on viewport
    const minScale = Math.max(0.05, Math.min(1.0, viewportSize / 2000)); // Adaptive min scale
    optimalScale = Math.max(minScale, Math.min(maxScale, optimalScale));

    // Calculate center position
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    
    // Calculate translation to center the content
    const translateX = dimensions.width / 2 - centerX * optimalScale;
    const translateY = dimensions.height / 2 - centerY * optimalScale;
    
    const transform = d3.zoomIdentity
      .translate(translateX, translateY)
      .scale(optimalScale);

    // Enhanced logging for debugging
    console.log('🎯 Enhanced Fit-to-View:', {
      totalNodes: validNodes.length,
      viewport: { width: dimensions.width, height: dimensions.height },
      contentBounds: { 
        minX: minX.toFixed(1), 
        minY: minY.toFixed(1), 
        maxX: maxX.toFixed(1), 
        maxY: maxY.toFixed(1),
        width: contentWidth.toFixed(1), 
        height: contentHeight.toFixed(1),
        effectiveWidth: effectiveWidth.toFixed(1),
        effectiveHeight: effectiveHeight.toFixed(1)
      },
      scaling: {
        scaleX: scaleX.toFixed(3),
        scaleY: scaleY.toFixed(3),
        optimal: optimalScale.toFixed(3),
        constraints: { min: minScale.toFixed(3), max: maxScale.toFixed(3) }
      },
      center: { x: centerX.toFixed(1), y: centerY.toFixed(1) },
      translation: { x: translateX.toFixed(1), y: translateY.toFixed(1) },
      padding: adaptivePadding.toFixed(1),
      utilization: {
        width: ((effectiveWidth * optimalScale) / dimensions.width * 100).toFixed(1) + '%',
        height: ((effectiveHeight * optimalScale) / dimensions.height * 100).toFixed(1) + '%'
      }
    });

    // Apply transform with smooth transition
    svg.transition()
      .duration(750)
      .ease(d3.easeQuadInOut)
      .call(zoomBehaviorRef.current!.transform, transform);
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
        window.location.href = '/meta-task.html';
        break;
      case 4:
        window.location.href = '/settings.html';
        break;
      case 5:
        // Already on knowledge graph page
        break;
      case 6:
        window.location.href = '/idc.html';
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
            value={5}
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
              label="Meta-Tasks" 
              id="tab-3"
              aria-controls="tabpanel-3"
            />
            <Tab 
              label="Settings" 
              id="tab-4"
              aria-controls="tabpanel-3"
            />
            <Tab 
              label="Knowledge Graph" 
              id="tab-5"
              aria-controls="tabpanel-4"
            />
            <Tab 
              label="IDC" 
              id="tab-6"
              aria-controls="tabpanel-5"
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

        {/* Search Control */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <SearchIcon sx={{ color: themeColors.text, fontSize: '18px' }} />
          <TextField
            size="small"
            placeholder="Search entities..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            sx={{
              '& .MuiOutlinedInput-root': {
                fontSize: '14px',
                height: '32px',
                minWidth: '200px',
                '& fieldset': {
                  borderColor: themeColors.border,
                },
                '&:hover fieldset': {
                  borderColor: themeColors.text,
                },
              },
              '& .MuiInputBase-input': {
                color: themeColors.text,
                padding: '8px 12px',
              },
            }}
          />
          {searchQuery && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'flex-start' }}>
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                <Chip
                  size="small"
                  label={`${searchResults.directMatches.size} direct`}
                  sx={{ 
                    backgroundColor: isDarkMode ? '#1565C0' : '#0D47A1', 
                    color: '#FFFFFF',
                    fontWeight: 'bold',
                    '& .MuiChip-deleteIcon': { color: '#FFFFFF' }
                  }}
                  variant="filled"
                  onDelete={() => setSearchQuery('')}
                />
                <Chip
                  size="small"
                  label={`${searchResults.associatedNodes.size} connected`}
                  sx={{ 
                    backgroundColor: isDarkMode ? '#26A69A' : '#00695C', 
                    color: '#FFFFFF',
                    fontWeight: 'bold'
                  }}
                  variant="filled"
                />
                <Chip
                  size="small"
                  label={`${searchResults.searchRelationships.size} links`}
                  sx={{ 
                    backgroundColor: isDarkMode ? '#66BB6A' : '#388E3C', 
                    color: '#FFFFFF',
                    fontWeight: 'bold'
                  }}
                  variant="filled"
                />
              </div>
              <Typography variant="caption" sx={{ 
                color: themeColors.text, 
                fontSize: '10px',
                opacity: 0.8,
                maxWidth: '250px',
                lineHeight: 1.2
              }}>
                🟡 Direct matches • 🔵 Connected nodes • 🟠 Active relationships
              </Typography>
            </div>
          )}
        </div>

        {/* Layout Type Control */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label htmlFor="layoutType" style={{ fontSize: '14px', fontWeight: '500', color: themeColors.text }}>Layout:</label>
          <select
            id="layoutType"
            value={layoutType}
            onChange={handleLayoutTypeChange}
            style={{ 
              padding: '6px 10px', 
              border: `1px solid ${themeColors.border}`, 
              borderRadius: '4px',
              fontSize: '14px',
              background: themeColors.surface,
              color: themeColors.text,
              minWidth: '120px'
            }}
          >
            <option value="force-directed">Force-Directed</option>
            <option value="radial">Radial</option>
            <option value="grid">Grid</option>
            <option value="circular">Circular</option>
          </select>
        </div>

        {/* Performance Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label style={{ fontSize: '14px', fontWeight: '500', color: themeColors.text }}>Max Nodes:</label>
          <select
            value={maxVisibleNodes}
            onChange={(e) => setMaxVisibleNodes(parseInt(e.target.value))}
            style={{ 
              padding: '6px 10px', 
              border: `1px solid ${themeColors.border}`, 
              borderRadius: '4px',
              fontSize: '14px',
              background: themeColors.surface,
              color: themeColors.text,
              minWidth: '80px'
            }}
          >
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
            <option value={200}>200</option>
            <option value={500}>500</option>
            <option value={1000}>All</option>
          </select>
          {entities.length > maxVisibleNodes && !searchQuery && (
            <Button
              size="small"
              variant="outlined"
              onClick={() => setMaxVisibleNodes(Math.min(maxVisibleNodes + 50, entities.length))}
              sx={{ fontSize: '12px', minWidth: 'auto', padding: '4px 8px' }}
            >
              +50 More
            </Button>
          )}
        </div>

        {/* Node Diameter Control */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label htmlFor="nodeDiameter" style={{ fontSize: '14px', fontWeight: '500', color: themeColors.text }}>Node Size:</label>
          <select
            id="nodeDiameter"
            value={nodeDiameter}
            onChange={handleNodeDiameterChange}
            style={{ 
              padding: '6px 10px', 
              border: `1px solid ${themeColors.border}`, 
              borderRadius: '4px',
              fontSize: '14px',
              background: themeColors.surface,
              color: themeColors.text,
              minWidth: '80px'
            }}
          >
            <option value={30}>Small</option>
            <option value={40}>Medium</option>
            <option value={50}>Large</option>
            <option value={65}>X-Large</option>
            <option value={80}>Huge</option>
          </select>
        </div>

        {/* Force Strength Control */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label htmlFor="forceStrength" style={{ fontSize: '14px', fontWeight: '500', color: themeColors.text }}>Spacing:</label>
          <input
            id="forceStrength"
            type="range"
            min="0.1"
            max="2.0"
            step="0.1"
            value={forceStrength}
            onChange={handleForceStrengthChange}
            style={{ 
              width: '80px',
              accentColor: '#2196f3'
            }}
            title={`Force Strength: ${forceStrength.toFixed(1)}x`}
          />
          <span style={{ fontSize: '12px', color: themeColors.secondaryText, minWidth: '30px' }}>
            {forceStrength.toFixed(1)}x
          </span>
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
        {visibleEntities.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginLeft: 'auto' }}>
            <div style={{ fontSize: '14px', color: themeColors.secondaryText, marginRight: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span>Zoom: {Math.round(zoomTransform.k * 100)}%</span>
              <span title={`Layout: ${layoutType}, Node Size: ${nodeDiameter}px, Spacing: ${forceStrength}x`}>
                📊 {layoutType.charAt(0).toUpperCase() + layoutType.slice(1).replace('-', ' ')}
              </span>
            </div>
            <button onClick={zoomIn} style={getButtonStyle()} title="Zoom In (+ key)">🔍+</button>
            <button onClick={zoomOut} style={getButtonStyle()} title="Zoom Out (- key)">🔍-</button>
            <button onClick={resetZoom} style={getButtonStyle()} title="Reset Zoom (Ctrl+R)">🔄</button>
            <button onClick={fitToView} style={getButtonStyle()} title="Fit to View (Ctrl+F)">📐</button>
            <button 
              onClick={() => {
                // EMERGENCY TIGHT LAYOUT - force maximum 8px spacing
                if (svgRef.current && visibleEntities.length > 1) {
                  const svg = d3.select(svgRef.current);
                  // CRITICAL FIX: START EMERGENCY NODES IN TIGHT AREA TOO
                  const emergencyNodes = visibleEntities.map((entity, i) => ({
                    id: entity.id,
                    name: entity.name,
                    type: entity.type,
                    confidence: entity.confidence,
                    // Start in small tight area near center - consistent with main layout
                    x: dimensions.width / 2 + (Math.random() - 0.5) * 100,
                    y: dimensions.height / 2 + (Math.random() - 0.5) * 100
                  }));
                  
                  // CRITICAL FIX: EMERGENCY FUNCTION ALSO ENFORCES 8PX MAX SPACING
                  const simulation = d3.forceSimulation(emergencyNodes)
                  .force('collision', d3.forceCollide()
                    .radius((d: any) => {
                      // Find the correct node radius for emergency tight spacing
                      const nodeInfo = nodesRef.current?.find(n => n.id === d.id);
                      if (nodeInfo && simulationRef.current) {
                        // Try to get radius from current node text info
                        const currentNodes = simulationRef.current.nodes();
                        const nodeIndex = currentNodes.findIndex(n => n.id === d.id);
                        if (nodeIndex >= 0) {
                          // Calculate radius similar to main calculation
                          const baseRadius = nodeDiameter / 2;
                          const confidenceBonus = d.confidence * (nodeDiameter * 0.3);
                          return Math.max(baseRadius + confidenceBonus, baseRadius) + 7; // 7px padding = 12-15px spacing
                        }
                      }
                      return nodeDiameter / 2 + 7; // Fallback with 7px padding = 12-15px spacing
                    })
                    .strength(1.0) // Maximum strength for collision detection
                    .iterations(3) // Standard iterations
                  )
                  .force('center', d3.forceCenter(dimensions.width / 2, dimensions.height / 2).strength(0.1)) // Very weak centering
                  .alpha(1.0)
                  .alphaDecay(0.01) // Very slow decay for tight settling
                  .velocityDecay(0.8); // High velocity decay to prevent excessive movement
                  // NO charge force - ONLY collision + weak centering for tight clustering

                  // Run for appropriate time for tight collision detection
                  setTimeout(() => {
                    simulation.stop();
                    createVisualization(); // Recreate visualization with tight positions
                    
                    // Auto-fit after tight layout is applied
                    setTimeout(() => {
                      fitToView();
                    }, 500);
                  }, 3000);
                }
              }}
              style={getButtonStyle('#ff9500')}
              title="Emergency Tight Layout - Force maximum 8px spacing"
            >
              💥
            </button>
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
      {visibleEntities.length > 0 ? (
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
              <div style={{ fontSize: '12px', color: themeColors.secondaryText, display: 'flex', alignItems: 'center', gap: '15px' }}>
                <span>
                  {visibleEntities.length}/{entities.length} entities • {visibleRelationships.length}/{relationships.length} relationships
                </span>
                {(visibleEntities.length < entities.length || visibleRelationships.length < relationships.length) && (
                  <Chip
                    size="small"
                    label={`Performance Mode: ${Math.round((1 - (visibleEntities.length / entities.length)) * 100)}% filtered`}
                    color="info"
                    variant="outlined"
                    sx={{ fontSize: '10px', height: '20px' }}
                  />
                )}
                {visibleRelationships.length > 0 && (
                  <span style={{ 
                    padding: '2px 6px', 
                    borderRadius: '3px',
                    background: visibleRelationships.length > 0 ? '#d4edda' : '#f8d7da',
                    color: visibleRelationships.length > 0 ? '#155724' : '#721c24',
                    fontSize: '11px'
                  }}>
                    {visibleRelationships.length === 0 ? 'No connections' : `${visibleRelationships.length} connections visible`}
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
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <strong style={{ fontSize: '12px', color: themeColors.text }}>Legend ({Array.from(new Set(visibleEntities.map(entity => entity.type).filter(type => type && typeof type === 'string'))).length} types visible):</strong>
                {isolatedEntities.length > 0 && (
                  <Button
                    size="small" 
                    variant="text"
                    onClick={() => setShowIsolatedNodes(!showIsolatedNodes)}
                    sx={{ fontSize: '10px', minWidth: 'auto', padding: '2px 6px' }}
                  >
                    {showIsolatedNodes ? 'Hide' : 'Show'} {isolatedEntities.length} Isolated
                  </Button>
                )}
              </div>
              {Array.from(new Set(visibleEntities.map(entity => entity.type).filter(type => type && typeof type === 'string'))).sort().map(type => (
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
            {visibleEntities.length > 0 && visibleRelationships.length === 0 && (
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

// OPTIMIZATION: Memoize the entire component to prevent unnecessary re-renders
export default memo(KnowledgeGraphViewer);