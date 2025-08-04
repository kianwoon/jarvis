// Knowledge Graph Frontend Performance Optimizations
// Target: Smooth performance with ≤188 relationships

import React, { useMemo, useCallback, useState } from 'react';

// 1. PERFORMANCE MODE CONFIGURATION
export const PERFORMANCE_CONFIG = {
  // Force simulation optimizations
  simulation: {
    maxIterations: 150,  // Reduced from 300
    alphaDecay: 0.05,   // Faster convergence
    velocityDecay: 0.7,  // More damping
    
    forces: {
      charge: {
        strength: -300,
        distanceMax: 200,  // Limit force calculation distance
      },
      link: {
        distance: 50,
        strength: 0.5,
      },
      collision: {
        radius: 30,
        strength: 0.8,
      },
      center: {
        strength: 0.1,
      }
    }
  },
  
  // Rendering optimizations
  rendering: {
    enableCanvas: true,  // Use Canvas for >50 nodes
    enableCulling: true,  // Don't render off-screen nodes
    cullingMargin: 50,  // Pixels outside viewport
    
    // Level of detail
    lod: {
      minZoom: 0.3,
      maxZoom: 3,
      hideLabelsBelow: 0.5,
      hideRelationshipsBelow: 0.3,
      simplifyNodesBelow: 0.4,
    }
  },
  
  // Interaction optimizations
  interaction: {
    dragDebounce: 16,  // ~60fps
    zoomThrottle: 50,
    disableAnimationsDuringDrag: true,
    disableForcesDuringDrag: true,
  }
};

// 2. RELATIONSHIP FILTER COMPONENT
export const RelationshipFilter: React.FC<{
  relationships: any[];
  onFilter: (filtered: any[]) => void;
}> = ({ relationships, onFilter }) => {
  const [minConfidence, setMinConfidence] = useState(0.7);
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set());
  const [showCount, setShowCount] = useState(100);
  
  // Get unique relationship types
  const relationshipTypes = useMemo(() => {
    const types = new Map<string, number>();
    relationships.forEach(rel => {
      const type = rel.type || 'UNKNOWN';
      types.set(type, (types.get(type) || 0) + 1);
    });
    return Array.from(types.entries()).sort((a, b) => b[1] - a[1]);
  }, [relationships]);
  
  // Apply filters
  const applyFilters = useCallback(() => {
    let filtered = relationships;
    
    // Confidence filter
    filtered = filtered.filter(rel => 
      (rel.properties?.confidence || 0) >= minConfidence
    );
    
    // Type filter
    if (selectedTypes.size > 0) {
      filtered = filtered.filter(rel => 
        selectedTypes.has(rel.type || 'UNKNOWN')
      );
    }
    
    // Count limit
    filtered = filtered
      .sort((a, b) => 
        (b.properties?.confidence || 0) - (a.properties?.confidence || 0)
      )
      .slice(0, showCount);
    
    onFilter(filtered);
  }, [relationships, minConfidence, selectedTypes, showCount, onFilter]);
  
  return (
    <div className="relationship-filter">
      <h3>Relationship Filters</h3>
      
      <div className="filter-section">
        <label>
          Min Confidence: {minConfidence.toFixed(2)}
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={minConfidence}
            onChange={(e) => {
              setMinConfidence(parseFloat(e.target.value));
              applyFilters();
            }}
          />
        </label>
      </div>
      
      <div className="filter-section">
        <label>Max Relationships: {showCount}</label>
        <input
          type="range"
          min="10"
          max={relationships.length}
          step="10"
          value={showCount}
          onChange={(e) => {
            setShowCount(parseInt(e.target.value));
            applyFilters();
          }}
        />
      </div>
      
      <div className="filter-section">
        <h4>Relationship Types</h4>
        {relationshipTypes.map(([type, count]) => (
          <label key={type}>
            <input
              type="checkbox"
              checked={selectedTypes.has(type)}
              onChange={(e) => {
                const newTypes = new Set(selectedTypes);
                if (e.target.checked) {
                  newTypes.add(type);
                } else {
                  newTypes.delete(type);
                }
                setSelectedTypes(newTypes);
                applyFilters();
              }}
            />
            {type} ({count})
          </label>
        ))}
      </div>
      
      <button onClick={applyFilters}>Apply Filters</button>
    </div>
  );
};

// 3. PERFORMANCE MODE TOGGLE
export const PerformanceModeToggle: React.FC<{
  onToggle: (enabled: boolean) => void;
}> = ({ onToggle }) => {
  const [performanceMode, setPerformanceMode] = useState(true);
  
  return (
    <div className="performance-toggle">
      <label>
        <input
          type="checkbox"
          checked={performanceMode}
          onChange={(e) => {
            setPerformanceMode(e.target.checked);
            onToggle(e.target.checked);
          }}
        />
        Performance Mode
        {performanceMode && (
          <span className="performance-info">
            (Reduced animations, faster rendering)
          </span>
        )}
      </label>
    </div>
  );
};

// 4. OPTIMIZED FORCE SIMULATION
export function createOptimizedSimulation(nodes: any[], links: any[], config = PERFORMANCE_CONFIG) {
  const simulation = d3.forceSimulation(nodes)
    .alphaDecay(config.simulation.alphaDecay)
    .velocityDecay(config.simulation.velocityDecay);
  
  // Configure forces with performance settings
  simulation
    .force('link', d3.forceLink(links)
      .id((d: any) => d.id)
      .distance(config.simulation.forces.link.distance)
      .strength(config.simulation.forces.link.strength)
    )
    .force('charge', d3.forceManyBody()
      .strength(config.simulation.forces.charge.strength)
      .distanceMax(config.simulation.forces.charge.distanceMax)
    )
    .force('collision', d3.forceCollide()
      .radius(config.simulation.forces.collision.radius)
      .strength(config.simulation.forces.collision.strength)
    )
    .force('center', d3.forceCenter(0, 0)
      .strength(config.simulation.forces.center.strength)
    );
  
  // Stop after max iterations
  let iterations = 0;
  simulation.on('tick', () => {
    iterations++;
    if (iterations >= config.simulation.maxIterations) {
      simulation.stop();
    }
  });
  
  return simulation;
}

// 5. VIEWPORT-BASED CULLING
export function cullNodesAndLinks(
  nodes: any[], 
  links: any[], 
  viewport: { x: number; y: number; width: number; height: number },
  margin: number = 50
) {
  // Expand viewport by margin
  const expandedViewport = {
    x: viewport.x - margin,
    y: viewport.y - margin,
    width: viewport.width + margin * 2,
    height: viewport.height + margin * 2,
  };
  
  // Filter visible nodes
  const visibleNodes = new Set(
    nodes
      .filter(node => 
        node.x >= expandedViewport.x &&
        node.x <= expandedViewport.x + expandedViewport.width &&
        node.y >= expandedViewport.y &&
        node.y <= expandedViewport.y + expandedViewport.height
      )
      .map(node => node.id)
  );
  
  // Filter visible links (both endpoints must be visible)
  const visibleLinks = links.filter(link => 
    visibleNodes.has(link.source.id || link.source) &&
    visibleNodes.has(link.target.id || link.target)
  );
  
  return {
    nodes: nodes.filter(node => visibleNodes.has(node.id)),
    links: visibleLinks,
  };
}

// 6. DEBOUNCED/THROTTLED EVENT HANDLERS
export function useOptimizedHandlers(config = PERFORMANCE_CONFIG) {
  const dragHandler = useMemo(() => 
    debounce((event: any, d: any) => {
      // Handle drag
      d.fx = event.x;
      d.fy = event.y;
    }, config.interaction.dragDebounce),
    []
  );
  
  const zoomHandler = useMemo(() =>
    throttle((event: any) => {
      // Handle zoom
      const { transform } = event;
      // Apply transform
    }, config.interaction.zoomThrottle),
    []
  );
  
  return { dragHandler, zoomHandler };
}

// Helper functions
function debounce(func: Function, wait: number) {
  let timeout: NodeJS.Timeout;
  return function executedFunction(...args: any[]) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

function throttle(func: Function, limit: number) {
  let inThrottle: boolean;
  return function(...args: any[]) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}