import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Button,
  Grid,
  Alert
} from '@mui/material';
import {
  RadiatingToggle,
  RadiatingDepthControl,
  RadiatingProgress,
  RadiatingResultsViewer,
  RadiatingVisualization
} from './index';
import {
  RadiatingConfig,
  RadiatingProgress as RadiatingProgressType,
  RadiatingResults,
  RadiatingEntity,
  RadiatingRelationship
} from '../../types/radiating';

/**
 * Example component demonstrating how to use the Radiating Coverage components
 * This can be used as a reference or testing component
 */
const RadiatingExample: React.FC = () => {
  const [enabled, setEnabled] = useState(false);
  const [config, setConfig] = useState<RadiatingConfig>({
    enabled: true,
    maxDepth: 3,
    strategy: 'breadth-first',
    relevanceThreshold: 0.5,
    maxEntitiesPerLevel: 20,
    includeRelationships: true,
    autoExpand: false,
    cacheResults: true,
    timeoutMs: 30000
  });

  // Mock progress data for demonstration
  const [showProgress, setShowProgress] = useState(false);
  const mockProgress: RadiatingProgressType = {
    isActive: true,
    currentDepth: 2,
    totalDepth: 3,
    entitiesDiscovered: 45,
    relationshipsFound: 78,
    processedEntities: 32,
    queueSize: 13,
    elapsedTime: 12500,
    estimatedTimeRemaining: 8000,
    currentEntity: 'Machine Learning',
    status: 'traversing'
  };

  // Mock results data for demonstration
  const mockResults: RadiatingResults = {
    rootEntity: {
      id: 'root',
      name: 'Artificial Intelligence',
      type: 'concept',
      depth: 0,
      relevanceScore: 1.0
    },
    entities: [
      {
        id: 'root',
        name: 'Artificial Intelligence',
        type: 'concept',
        depth: 0,
        relevanceScore: 1.0
      },
      {
        id: 'ml',
        name: 'Machine Learning',
        type: 'concept',
        depth: 1,
        relevanceScore: 0.95,
        parentId: 'root'
      },
      {
        id: 'dl',
        name: 'Deep Learning',
        type: 'concept',
        depth: 1,
        relevanceScore: 0.92,
        parentId: 'root'
      },
      {
        id: 'nlp',
        name: 'Natural Language Processing',
        type: 'application',
        depth: 1,
        relevanceScore: 0.88,
        parentId: 'root'
      },
      {
        id: 'cv',
        name: 'Computer Vision',
        type: 'application',
        depth: 1,
        relevanceScore: 0.85,
        parentId: 'root'
      },
      {
        id: 'nn',
        name: 'Neural Networks',
        type: 'technique',
        depth: 2,
        relevanceScore: 0.82,
        parentId: 'dl'
      },
      {
        id: 'transformer',
        name: 'Transformers',
        type: 'architecture',
        depth: 2,
        relevanceScore: 0.78,
        parentId: 'nlp'
      },
      {
        id: 'cnn',
        name: 'Convolutional Networks',
        type: 'architecture',
        depth: 2,
        relevanceScore: 0.75,
        parentId: 'cv'
      }
    ],
    relationships: [
      {
        id: 'rel1',
        sourceId: 'root',
        targetId: 'ml',
        type: 'includes',
        weight: 0.95
      },
      {
        id: 'rel2',
        sourceId: 'root',
        targetId: 'dl',
        type: 'includes',
        weight: 0.92
      },
      {
        id: 'rel3',
        sourceId: 'root',
        targetId: 'nlp',
        type: 'application',
        weight: 0.88
      },
      {
        id: 'rel4',
        sourceId: 'root',
        targetId: 'cv',
        type: 'application',
        weight: 0.85
      },
      {
        id: 'rel5',
        sourceId: 'dl',
        targetId: 'nn',
        type: 'uses',
        weight: 0.82
      },
      {
        id: 'rel6',
        sourceId: 'nlp',
        targetId: 'transformer',
        type: 'implements',
        weight: 0.78
      },
      {
        id: 'rel7',
        sourceId: 'cv',
        targetId: 'cnn',
        type: 'implements',
        weight: 0.75
      }
    ],
    totalEntities: 8,
    totalRelationships: 7,
    maxDepthReached: 2,
    processingTime: 12500,
    strategy: 'breadth-first',
    relevanceThreshold: 0.5,
    timestamp: new Date()
  };

  const simulateRadiatingProcess = () => {
    setShowProgress(true);
    
    // Simulate completion after 3 seconds
    setTimeout(() => {
      setShowProgress(false);
    }, 3000);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>
        Radiating Coverage System Demo
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        This is a demonstration of the Universal Radiating Coverage System components.
        Toggle the system on and explore the various settings and visualizations.
      </Alert>

      <Grid container spacing={3}>
        {/* Toggle Section */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              1. Enable/Disable System
            </Typography>
            <RadiatingToggle
              onToggle={(isEnabled) => {
                setEnabled(isEnabled);
                console.log('Radiating toggled:', isEnabled);
              }}
              showLabel={true}
              showStatus={true}
            />
          </Paper>
        </Grid>

        {/* Configuration Section */}
        {enabled && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                2. Configure Settings
              </Typography>
              <RadiatingDepthControl
                onConfigChange={(newConfig) => {
                  setConfig(newConfig);
                  console.log('Config updated:', newConfig);
                }}
                compact={false}
              />
            </Paper>
          </Grid>
        )}

        {/* Progress Simulation */}
        {enabled && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                3. Progress Monitoring
              </Typography>
              
              {!showProgress ? (
                <Button
                  variant="contained"
                  onClick={simulateRadiatingProcess}
                >
                  Simulate Radiating Process
                </Button>
              ) : (
                <RadiatingProgress
                  progress={mockProgress}
                  onCancel={() => setShowProgress(false)}
                  compact={false}
                  showDetails={true}
                />
              )}
            </Paper>
          </Grid>
        )}

        {/* Results Viewer */}
        {enabled && !showProgress && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                4. Results Viewer
              </Typography>
              <RadiatingResultsViewer
                results={mockResults}
                onEntityClick={(entity) => {
                  console.log('Entity clicked:', entity);
                  alert(`Clicked on: ${entity.name} (${entity.type})`);
                }}
                onExploreEntity={(entity) => {
                  console.log('Explore entity:', entity);
                  alert(`Exploring: ${entity.name}`);
                }}
                compact={false}
              />
            </Paper>
          </Grid>
        )}

        {/* Visualization */}
        {enabled && !showProgress && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                5. Network Visualization
              </Typography>
              <RadiatingVisualization
                data={{
                  nodes: mockResults.entities.map(entity => ({
                    id: entity.id,
                    name: entity.name,
                    type: entity.type,
                    group: entity.depth,
                    radius: entity.relevanceScore,
                    color: ''
                  })),
                  links: mockResults.relationships.map(rel => ({
                    source: rel.sourceId,
                    target: rel.targetId,
                    value: rel.weight,
                    type: rel.type
                  }))
                }}
                width={800}
                height={500}
                onNodeClick={(node) => {
                  console.log('Visualization node clicked:', node);
                }}
              />
            </Paper>
          </Grid>
        )}

        {/* Configuration Display */}
        {enabled && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Current Configuration
              </Typography>
              <pre style={{ 
                backgroundColor: '#f5f5f5', 
                padding: '12px', 
                borderRadius: '4px',
                overflow: 'auto'
              }}>
                {JSON.stringify(config, null, 2)}
              </pre>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Container>
  );
};

export default RadiatingExample;