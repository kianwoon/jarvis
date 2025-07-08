import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const StateNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #2196f3', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#2196f3' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#2196f3' }}>State</Typography>
      <Typography variant="body2">{data.label || 'State Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#2196f3' }} />
  </Paper>
);

export default StateNode;
