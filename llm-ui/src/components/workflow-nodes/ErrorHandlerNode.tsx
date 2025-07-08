import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const ErrorHandlerNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #f44336', borderRadius: 2 }}>
    <Handle type="target" position={Position.Top} style={{ background: '#f44336' }} />
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#f44336' }}>ErrorHandler</Typography>
      <Typography variant="body2">{data.label || 'ErrorHandler Node'}</Typography>
    </Box>
    <Handle type="source" position={Position.Bottom} style={{ background: '#f44336' }} />
  </Paper>
);

export default ErrorHandlerNode;
