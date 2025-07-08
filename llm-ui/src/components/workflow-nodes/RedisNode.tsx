import React from 'react';
import { Handle, Position } from 'reactflow';
import { Paper, Typography, Box } from '@mui/material';

interface RedisNodeProps {
  data: {
    label: string;
    operation?: string;
  };
}

const RedisNode: React.FC<RedisNodeProps> = ({ data }) => {
  return (
    <Paper elevation={2} sx={{ padding: 2, minWidth: 150, border: '2px solid #e53935', borderRadius: 2 }}>
      <Handle type="target" position={Position.Top} style={{ background: '#e53935' }} />
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#e53935' }}>
          Redis
        </Typography>
        <Typography variant="body2">{data.label || 'Redis Operation'}</Typography>
      </Box>
      <Handle type="source" position={Position.Bottom} style={{ background: '#e53935' }} />
    </Paper>
  );
};

export default RedisNode;