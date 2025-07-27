import React, { useState, useEffect, useRef, ErrorInfo, Component } from 'react';
import * as d3 from 'd3';

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
  
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions] = useState({ width: 800, height: 600 });

  // Color scheme for different entity types
  const entityColors: Record<string, string> = {
    'PERSON': '#ff7f0e',
    'ORG': '#2ca02c', 
    'CONCEPT': '#1f77b4',
    'LOCATION': '#d62728',
    'PRODUCT': '#9467bd',
    'default': '#7f7f7f'
  };

  // Fetch available document IDs
  const fetchAvailableDocuments = async () => {
    try {
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
        setAvailableDocuments(docIds);
      }
    } catch (err) {
      console.error('Error fetching document IDs:', err);
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

  // Fetch entities for a specific document
  const fetchEntities = async (documentId: string) => {
    if (!documentId) return;
    
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/knowledge-graph/entities/${documentId}`);
      if (response.ok) {
        const data = await response.json();
        setEntities(data.entities || []);
      } else {
        setError('Failed to fetch entities');
      }
    } catch (err) {
      setError('Error fetching entities');
      console.error('Entities fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Fetch relationships for a specific document
  const fetchRelationships = async (documentId: string) => {
    if (!documentId) return;
    
    try {
      const response = await fetch(`/api/v1/knowledge-graph/relationships/${documentId}`);
      if (response.ok) {
        const data = await response.json();
        setRelationships(data.relationships || []);
      } else {
        setError('Failed to fetch relationships');
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

      // Filter relationships to only include those with valid source/target entities
      const entityIds = new Set(nodes.map(n => n.id));
      const validRelationships = relationships.filter(rel => {
        const sourceValid = entityIds.has(rel.source_entity);
        const targetValid = entityIds.has(rel.target_entity);
        if (!sourceValid || !targetValid) {
          console.warn('Invalid relationship:', rel, 'Available entities:', Array.from(entityIds));
        }
        return sourceValid && targetValid;
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
      
      console.log('Processed nodes:', nodes.length);
      console.log('Processed links:', links.length);

      // Set up SVG with safe dimensions
      const width = Math.max(400, dimensions.width);
      const height = Math.max(300, dimensions.height);
      
      svg
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`)
        .style('background', 'white');

      // Create simulation with error handling
      const simulation = d3.forceSimulation<GraphNode>(nodes)
        .force('link', d3.forceLink<GraphNode, GraphLink>(links)
          .id(d => d.id)
          .distance(80)
          .strength(0.5)
        )
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(20));

      // Create links group
      const linkGroup = svg.append('g').attr('class', 'links');
      const link = linkGroup
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('stroke', '#999')
        .attr('stroke-opacity', 0.6)
        .attr('stroke-width', d => Math.max(1, d.confidence * 3));

      // Create link labels group
      const linkLabelGroup = svg.append('g').attr('class', 'link-labels');
      const linkLabels = linkLabelGroup
        .selectAll('text')
        .data(links)
        .join('text')
        .attr('text-anchor', 'middle')
        .attr('dy', -5)
        .attr('font-size', '10px')
        .attr('fill', '#666')
        .style('pointer-events', 'none')
        .text(d => d.relationship_type);

      // Create nodes group
      const nodeGroup = svg.append('g').attr('class', 'nodes');
      const node = nodeGroup
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('r', d => 8 + (d.confidence * 12))
        .attr('fill', d => entityColors[d.type] || entityColors.default)
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer');

      // Create node labels group
      const nodeLabelGroup = svg.append('g').attr('class', 'node-labels');
      const nodeLabels = nodeLabelGroup
        .selectAll('text')
        .data(nodes)
        .join('text')
        .attr('text-anchor', 'middle')
        .attr('dy', 4)
        .attr('font-size', '11px')
        .attr('font-weight', 'bold')
        .attr('fill', '#333')
        .style('pointer-events', 'none')
        .text(d => d.name.length > 12 ? d.name.slice(0, 12) + '...' : d.name);

      // Add tooltips
      node.append('title')
        .text(d => `${d.name}\nType: ${d.type}\nConfidence: ${(d.confidence * 100).toFixed(1)}%`);

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

          nodeLabels
            .attr('x', d => d.x || 0)
            .attr('y', d => d.y || 0);
        } catch (tickError) {
          console.error('Error in simulation tick:', tickError);
        }
      });

      // Stop simulation after a reasonable time
      setTimeout(() => {
        simulation.stop();
        console.log('Simulation stopped');
      }, 5000);
      
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
          .attr('font-size', '14px')
          .text('Visualization Error - Check Console');
      }
    }
  };

  // Initial load
  useEffect(() => {
    fetchStats();
    fetchAvailableDocuments();
  }, []);

  // Update visualization when data changes
  useEffect(() => {
    if (entities.length > 0) {
      createVisualization();
    }
  }, [entities, relationships, dimensions]);

  // Handle document selection
  const handleDocumentSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedDocument.trim()) {
      fetchEntities(selectedDocument.trim());
      fetchRelationships(selectedDocument.trim());
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2>Knowledge Graph Viewer</h2>
      
      {/* Stats Section */}
      {stats && (
        <div style={{ 
          background: '#f5f5f5', 
          padding: '15px', 
          borderRadius: '8px', 
          marginBottom: '20px' 
        }}>
          <h3>Graph Statistics</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
            <div><strong>Total Entities:</strong> {stats.total_entities}</div>
            <div><strong>Total Relationships:</strong> {stats.total_relationships}</div>
            <div><strong>Documents Processed:</strong> {stats.documents_processed}</div>
            <div><strong>Last Updated:</strong> {new Date(stats.last_updated).toLocaleString()}</div>
          </div>
          
          {Object.keys(stats.entity_types).length > 0 && (
            <div style={{ marginTop: '10px' }}>
              <strong>Entity Types:</strong>
              <div style={{ marginTop: '5px' }}>
                {Object.entries(stats.entity_types).map(([type, count]) => (
                  <span 
                    key={type} 
                    style={{ 
                      display: 'inline-block',
                      margin: '2px 5px',
                      padding: '2px 8px',
                      background: entityColors[type] || entityColors.default,
                      color: 'white',
                      borderRadius: '12px',
                      fontSize: '12px'
                    }}
                  >
                    {type}: {count}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Available Documents */}
      {availableDocuments.length > 0 && (
        <div style={{ marginBottom: '20px', background: '#e8f4f8', padding: '15px', borderRadius: '8px' }}>
          <h4 style={{ margin: '0 0 10px 0' }}>Available Documents:</h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {availableDocuments.map(docId => (
              <button
                key={docId}
                onClick={() => setSelectedDocument(docId)}
                style={{
                  padding: '6px 12px',
                  background: selectedDocument === docId ? '#007bff' : '#fff',
                  color: selectedDocument === docId ? 'white' : '#007bff',
                  border: '1px solid #007bff',
                  borderRadius: '20px',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                {docId}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Document Selection */}
      <form onSubmit={handleDocumentSubmit} style={{ marginBottom: '20px' }}>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
          <label htmlFor="documentId">Document ID:</label>
          {availableDocuments.length > 0 ? (
            <select
              id="documentId"
              value={selectedDocument}
              onChange={(e) => setSelectedDocument(e.target.value)}
              style={{ 
                padding: '8px 12px', 
                border: '1px solid #ddd', 
                borderRadius: '4px',
                minWidth: '250px'
              }}
            >
              <option value="">Select a document...</option>
              {availableDocuments.map(docId => (
                <option key={docId} value={docId}>{docId}</option>
              ))}
            </select>
          ) : (
            <input
              id="documentId"
              type="text"
              value={selectedDocument}
              onChange={(e) => setSelectedDocument(e.target.value)}
              placeholder="Enter document ID to view its knowledge graph"
              style={{ 
                padding: '8px 12px', 
                border: '1px solid #ddd', 
                borderRadius: '4px',
                minWidth: '300px'
              }}
            />
          )}
          <button 
            type="submit"
            disabled={loading || !selectedDocument.trim()}
            style={{
              padding: '8px 16px',
              background: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: loading ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? 'Loading...' : 'Load Graph'}
          </button>
          <button 
            type="button"
            onClick={() => {
              fetchAvailableDocuments();
              fetchStats();
            }}
            style={{
              padding: '8px 16px',
              background: '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Refresh
          </button>
        </div>
      </form>

      {/* Error Display */}
      {error && (
        <div style={{ 
          background: '#f8d7da', 
          color: '#721c24', 
          padding: '10px', 
          borderRadius: '4px',
          marginBottom: '20px'
        }}>
          Error: {error}
        </div>
      )}

      {/* Graph Visualization */}
      {entities.length > 0 ? (
        <VisualizationErrorBoundary onError={setError}>
          <div style={{ border: '1px solid #ddd', borderRadius: '8px', overflow: 'hidden' }}>
            <h3 style={{ margin: '0', padding: '10px', background: '#f8f9fa' }}>
              Knowledge Graph for Document: {selectedDocument}
            </h3>
            <svg ref={svgRef} style={{ display: 'block' }}></svg>
          
            {/* Legend */}
            <div style={{ padding: '10px', background: '#f8f9fa', borderTop: '1px solid #ddd' }}>
              <strong>Legend:</strong>
              {Object.entries(entityColors).filter(([type]) => type !== 'default').map(([type, color]) => (
                <span 
                  key={type}
                  style={{ 
                    display: 'inline-block',
                    margin: '0 10px 0 0',
                    fontSize: '12px'
                  }}
                >
                  <span 
                    style={{ 
                      display: 'inline-block',
                      width: '12px',
                      height: '12px',
                      background: color,
                      borderRadius: '50%',
                      marginRight: '4px',
                      verticalAlign: 'middle'
                    }}
                  ></span>
                  {type}
                </span>
              ))}
            </div>
          </div>
        </VisualizationErrorBoundary>
      ) : selectedDocument && !loading && (
        <div style={{ 
          textAlign: 'center', 
          padding: '40px', 
          color: '#666',
          border: '2px dashed #ddd',
          borderRadius: '8px'
        }}>
          No entities found for this document. Try ingesting the document first or check the document ID.
        </div>
      )}

      {!selectedDocument && !loading && (
        <div style={{ 
          textAlign: 'center', 
          padding: '40px', 
          color: '#666',
          border: '2px dashed #ddd',
          borderRadius: '8px'
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
    </div>
  );
};

export default KnowledgeGraphViewer;