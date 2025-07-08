import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const ConditionNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #ff9800', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#ff9800' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#ff9800' }}>Condition</Typography>
      <Typography variant="body2">{data.label || 'If/Then'}</Typography>
    </Box>
    <Handle type="source" position={Position.Left} style={{ background: '#ff9800' }} />
    <Handle type="source" position={Position.Right} style={{ background: '#ff9800' }} />
  </Paper>
);

export default ConditionNode;