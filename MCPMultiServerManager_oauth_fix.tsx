// In MCPMultiServerManager.tsx, find this section around line 695-700:

// CURRENT CODE (BROKEN):
<IconButton
  size="small"
  onClick={(e) => {
    e.stopPropagation();
    setEditingOAuthServer(server);
    setOauthDialog(true);
  }}
>
  <VpnKeyIcon />
</IconButton>

// REPLACE WITH THIS (FIXED):
<IconButton
  size="small"
  onClick={async (e) => {
    e.stopPropagation();
    
    // Fetch the server with full OAuth credentials
    try {
      const response = await fetch(`/api/v1/mcp/servers/${server.id}?show_sensitive=true`);
      if (!response.ok) throw new Error('Failed to load server credentials');
      const serverWithCredentials = await response.json();
      
      setEditingOAuthServer(serverWithCredentials);
      setOauthDialog(true);
    } catch (err) {
      console.error('Failed to load OAuth credentials:', err);
      setError('Failed to load OAuth credentials');
    }
  }}
>
  <VpnKeyIcon />
</IconButton>