import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Psychology as ReasoningIcon
} from '@mui/icons-material';

// Helper function to parse content with <think> tags
interface ContentPart {
  type: 'content' | 'thinking';
  text: string;
}

function parseContentWithThinking(content: string): ContentPart[] {
  const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
  const parts: ContentPart[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = thinkRegex.exec(content)) !== null) {
    // Add content before the <think> tag
    if (match.index > lastIndex) {
      parts.push({
        type: 'content',
        text: content.substring(lastIndex, match.index)
      });
    }
    
    // Add the thinking content
    parts.push({
      type: 'thinking',
      text: match[1].trim()
    });
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add remaining content after the last <think> tag
  if (lastIndex < content.length) {
    parts.push({
      type: 'content',
      text: content.substring(lastIndex)
    });
  }
  
  return parts;
}

// Component to render message content with markdown and thinking sections
export const MessageContent = React.memo(({ content }: { content: string }) => {
  const [expandedThinking, setExpandedThinking] = useState<Set<number>>(new Set());
  const parts = parseContentWithThinking(content);
  
  const toggleThinking = (index: number) => {
    const newExpanded = new Set(expandedThinking);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedThinking(newExpanded);
  };

  return (
    <Box>
      {parts.map((part, index) => {
        if (part.type === 'content' && part.text.trim()) {
          return (
            <ReactMarkdown
              key={index}
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => (
                  <Typography variant="body1" component="div" sx={{ mb: 1 }}>
                    {children}
                  </Typography>
                ),
                h1: ({ children }) => (
                  <Typography variant="h4" component="h1" sx={{ mb: 2, mt: 2 }}>
                    {children}
                  </Typography>
                ),
                h2: ({ children }) => (
                  <Typography variant="h5" component="h2" sx={{ mb: 1.5, mt: 1.5 }}>
                    {children}
                  </Typography>
                ),
                h3: ({ children }) => (
                  <Typography variant="h6" component="h3" sx={{ mb: 1, mt: 1 }}>
                    {children}
                  </Typography>
                ),
                code: ({ className, children }) => {
                  const inline = !className?.includes('language-');
                  return inline ? (
                    <Box
                      component="code"
                      sx={{
                        fontFamily: 'monospace',
                        backgroundColor: 'action.hover',
                        color: 'text.primary',
                        padding: '2px 4px',
                        borderRadius: 1,
                        fontSize: 'inherit'
                      }}
                    >
                      {children}
                    </Box>
                  ) : (
                    <Box
                      component="pre"
                      sx={{
                        fontFamily: 'monospace',
                        backgroundColor: 'action.hover',
                        color: 'text.primary',
                        padding: '8px',
                        borderRadius: 1,
                        fontSize: '0.875rem',
                        overflow: 'auto',
                        margin: '8px 0'
                      }}
                    >
                      <code>{children}</code>
                    </Box>
                  );
                },
                ul: ({ children }) => (
                  <Box component="ul" sx={{ ml: 2, mb: 1 }}>
                    {children}
                  </Box>
                ),
                ol: ({ children }) => (
                  <Box component="ol" sx={{ ml: 2, mb: 1 }}>
                    {children}
                  </Box>
                ),
                li: ({ children }) => (
                  <Typography component="li" variant="body1" sx={{ mb: 0.5 }}>
                    {children}
                  </Typography>
                ),
                blockquote: ({ children }) => (
                  <Box
                    sx={{
                      borderLeft: '4px solid',
                      borderLeftColor: 'primary.main',
                      pl: 2,
                      ml: 1,
                      fontStyle: 'italic',
                      color: 'text.secondary'
                    }}
                  >
                    {children}
                  </Box>
                )
              }}
            >
              {part.text}
            </ReactMarkdown>
          );
        } else if (part.type === 'thinking') {
          const isExpanded = expandedThinking.has(index);
          return (
            <Box key={index} sx={{ my: 1 }}>
              <Accordion 
                expanded={isExpanded}
                onChange={() => toggleThinking(index)}
                sx={{ backgroundColor: 'background.default' }}
              >
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ReasoningIcon fontSize="small" color="primary" />
                    <Typography variant="body2" color="primary">
                      Reasoning
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => (
                        <Typography variant="body2" component="div" sx={{ mb: 1, color: 'text.secondary' }}>
                          {children}
                        </Typography>
                      ),
                      code: ({ className, children }) => {
                        const inline = !className?.includes('language-');
                        return inline ? (
                          <Box
                            component="code"
                            sx={{
                              fontFamily: 'monospace',
                              backgroundColor: 'action.selected',
                              color: 'text.secondary',
                              padding: '2px 4px',
                              borderRadius: 1,
                              fontSize: '0.8rem'
                            }}
                          >
                            {children}
                          </Box>
                        ) : (
                          <Box
                            component="pre"
                            sx={{
                              fontFamily: 'monospace',
                              backgroundColor: 'action.selected',
                              color: 'text.secondary',
                              padding: '8px',
                              borderRadius: 1,
                              fontSize: '0.8rem',
                              overflow: 'auto',
                              margin: '8px 0'
                            }}
                          >
                            <code>{children}</code>
                          </Box>
                        );
                      }
                    }}
                  >
                    {part.text}
                  </ReactMarkdown>
                </AccordionDetails>
              </Accordion>
            </Box>
          );
        }
        return null;
      })}
    </Box>
  );
});