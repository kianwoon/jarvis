import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

const EndNode: React.FC<{ data: any }> = ({ data }) => (
  <Paper elevation={2} sx={{ padding: 2, minWidth: 120, border: '2px solid #f44336', borderRadius: 2, bgcolor: '#ffebee' }}>
    <Handle type="target" position={Position.Top} style={{ background: '#f44336' }} />
    <Box textAlign="center">
      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#f44336' }}>END</Typography>
      <Typography variant="body2">{data.label || 'Finish'}</Typography>
    </Box>
  </Paper>
);

export default EndNode;