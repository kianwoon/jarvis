import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const AggregatorNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #8bc34a', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#8bc34a' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#8bc34a' }}>Aggregator</Typography>
      <Typography variant="body2">{data.label || 'Aggregator Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#8bc34a' }} />
  </Paper>
);

export default AggregatorNode;
