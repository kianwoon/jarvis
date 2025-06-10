// Quick fix for OAuth modal - Add this to MCPMultiServerManager.tsx

// Replace the OAuth button onClick handler with this:
const handleOAuthButtonClick = async (server) => {
  try {
    // Fetch the server with sensitive data
    const response = await fetch(`/api/v1/mcp/servers/${server.id}?show_sensitive=true`);
    if (!response.ok) throw new Error('Failed to load server credentials');
    const serverWithCredentials = await response.json();
    
    console.log('Server with credentials:', serverWithCredentials);
    
    setEditingOAuthServer(serverWithCredentials);
    setOauthDialog(true);
  } catch (err) {
    console.error('Failed to load OAuth credentials:', err);
    setError('Failed to load OAuth credentials');
  }
};

// In the OAuth IconButton, replace onClick with:
onClick={(e) => {
  e.stopPropagation();
  handleOAuthButtonClick(server);
}}

// Alternative: If you can't modify MCPMultiServerManager.tsx,
// add this to OAuthCredentialsDialog.tsx in the useEffect:

useEffect(() => {
  // Add server ID to props first
  if (open && serverId) {
    const fetchFullCredentials = async () => {
      try {
        const response = await fetch(`/api/v1/mcp/servers/${serverId}?show_sensitive=true`);
        if (response.ok) {
          const data = await response.json();
          console.log('Fetched OAuth data:', data);
          
          if (data.oauth_credentials) {
            setCredentials(data.oauth_credentials);
            setJsonInput(JSON.stringify(data.oauth_credentials, null, 2));
          }
        }
      } catch (err) {
        console.error('Failed to fetch credentials:', err);
      }
    };
    
    fetchFullCredentials();
  }
}, [open, serverId]);