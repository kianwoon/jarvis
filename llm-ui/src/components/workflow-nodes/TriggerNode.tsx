import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const TriggerNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #e91e63', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#e91e63' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#e91e63' }}>Trigger</Typography>
      <Typography variant="body2">{data.label || 'Trigger Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#e91e63' }} />
  </Paper>
);

export default TriggerNode;
