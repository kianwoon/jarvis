import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const RouterNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #3f51b5', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#3f51b5' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#3f51b5' }}>Router</Typography>
      <Typography variant="body2">{data.label || 'Router Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#3f51b5' }} />
  </Paper>
);

export default RouterNode;
