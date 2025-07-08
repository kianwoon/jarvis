import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const CacheNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #ffc107', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#ffc107' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#ffc107' }}>Cache</Typography>
      <Typography variant="body2">{data.label || 'Cache Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#ffc107' }} />
  </Paper>
);

export default CacheNode;
