// Run this in your browser console to fix the OAuth modal temporarily

// First, find and click the OAuth button to open the modal
// Then run this code:

(async function fixOAuthModal() {
  // Get the server ID from the modal title or URL
  const serverId = 2; // Gmail MCP Server ID
  
  try {
    // Fetch the full credentials
    const response = await fetch(`/api/v1/mcp/servers/${serverId}?show_sensitive=true`);
    const data = await response.json();
    
    console.log('OAuth credentials:', data.oauth_credentials);
    
    // Fill in the form fields
    const fillField = (labelText, value) => {
      const labels = Array.from(document.querySelectorAll('label'));
      const label = labels.find(l => l.textContent.includes(labelText));
      if (label) {
        const input = label.parentElement.querySelector('input, textarea');
        if (input) {
          input.value = value || '';
          // Trigger React's onChange event
          const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
          nativeInputValueSetter.call(input, value || '');
          const ev = new Event('input', { bubbles: true });
          input.dispatchEvent(ev);
        }
      }
    };
    
    // Fill all fields
    if (data.oauth_credentials) {
      fillField('Client ID', data.oauth_credentials.client_id);
      fillField('Client Secret', data.oauth_credentials.client_secret);
      fillField('Access Token', data.oauth_credentials.access_token);
      fillField('Refresh Token', data.oauth_credentials.refresh_token);
      fillField('Project ID', data.oauth_credentials.project_id);
      
      console.log('✅ OAuth fields populated!');
    }
    
  } catch (err) {
    console.error('Failed to load credentials:', err);
  }
})();

// To reveal password fields, click the eye icons first, then run the script again