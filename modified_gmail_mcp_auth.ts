/**
 * Modified Gmail MCP authentication to support dynamic OAuth tokens
 * 
 * This shows how to modify the Gmail MCP server to accept tokens via:
 * 1. Tool parameters (per-request tokens)
 * 2. Environment variables (startup tokens)
 * 3. Dynamic refresh from external source
 */

import { OAuth2Client } from 'google-auth-library';
import fs from 'fs';
import path from 'path';

// Configuration paths
const CONFIG_DIR = path.join(os.homedir(), '.gmail-mcp');
const OAUTH_PATH = process.env.GMAIL_OAUTH_PATH || path.join(CONFIG_DIR, 'gcp-oauth.keys.json');
const CREDENTIALS_PATH = process.env.GMAIL_CREDENTIALS_PATH || path.join(CONFIG_DIR, 'credentials.json');

// OAuth2 configuration
let oauth2Client: OAuth2Client;
let globalOAuthConfig: any = null;

/**
 * Enhanced loadCredentials that supports multiple token sources
 */
async function loadCredentials() {
    try {
        // Load OAuth client configuration
        const keysContent = JSON.parse(fs.readFileSync(OAUTH_PATH, 'utf8'));
        const keys = keysContent.installed || keysContent.web;

        oauth2Client = new OAuth2Client(
            keys.client_id,
            keys.client_secret,
            "http://localhost:3000/oauth2callback"
        );

        // Try multiple token sources in priority order:
        
        // 1. Environment variables (highest priority)
        if (process.env.GMAIL_ACCESS_TOKEN) {
            console.log('Using OAuth tokens from environment variables');
            oauth2Client.setCredentials({
                access_token: process.env.GMAIL_ACCESS_TOKEN,
                refresh_token: process.env.GMAIL_REFRESH_TOKEN,
                token_type: 'Bearer',
                scope: 'https://www.googleapis.com/auth/gmail.modify'
            });
            return;
        }

        // 2. Credentials file (existing behavior)
        if (fs.existsSync(CREDENTIALS_PATH)) {
            console.log('Using OAuth tokens from credentials.json');
            const credentials = JSON.parse(fs.readFileSync(CREDENTIALS_PATH, 'utf8'));
            oauth2Client.setCredentials(credentials);
            return;
        }

        // 3. No credentials found
        console.log('No OAuth credentials found. Authentication required.');
    } catch (error) {
        console.error('Error loading credentials:', error);
        process.exit(1);
    }
}

/**
 * Create OAuth client with dynamic tokens
 */
function createDynamicOAuthClient(tokens?: any): OAuth2Client {
    // If tokens provided, create new client with those tokens
    if (tokens && tokens.access_token) {
        const client = new OAuth2Client(
            globalOAuthConfig?.client_id || process.env.GMAIL_CLIENT_ID,
            globalOAuthConfig?.client_secret || process.env.GMAIL_CLIENT_SECRET,
            "http://localhost:3000/oauth2callback"
        );
        
        client.setCredentials({
            access_token: tokens.access_token,
            refresh_token: tokens.refresh_token || globalOAuthConfig?.refresh_token,
            token_type: tokens.token_type || 'Bearer',
            scope: tokens.scope || 'https://www.googleapis.com/auth/gmail.modify'
        });
        
        return client;
    }
    
    // Otherwise use global client
    return oauth2Client;
}

/**
 * Modified tool handler that accepts OAuth tokens
 */
async function handleToolWithDynamicAuth(request: any) {
    const { name, arguments: args } = request.params;
    
    // Check if OAuth tokens are provided in the request
    let authClient = oauth2Client;
    
    if (args.oauth_token || args.google_access_token) {
        // Create a new OAuth client with the provided tokens
        authClient = createDynamicOAuthClient({
            access_token: args.oauth_token || args.google_access_token,
            refresh_token: args.oauth_refresh_token || args.google_refresh_token,
            token_type: 'Bearer'
        });
        
        // Remove OAuth fields from args before processing
        delete args.oauth_token;
        delete args.google_access_token;
        delete args.oauth_refresh_token;
        delete args.google_refresh_token;
    }
    
    // Initialize Gmail API with the appropriate auth client
    const gmail = google.gmail({ version: 'v1', auth: authClient });
    
    // Continue with normal tool processing...
    // (rest of the tool handler code)
}

/**
 * Token refresh handler
 */
async function refreshTokenIfNeeded(client: OAuth2Client) {
    try {
        const credentials = client.credentials;
        
        // Check if token is expired
        if (credentials.expiry_date && credentials.expiry_date <= Date.now()) {
            console.log('Token expired, refreshing...');
            
            // If we have a refresh token, use it
            if (credentials.refresh_token) {
                const { credentials: newCredentials } = await client.refreshAccessToken();
                client.setCredentials(newCredentials);
                
                // Optionally save the new tokens
                if (CREDENTIALS_PATH) {
                    fs.writeFileSync(CREDENTIALS_PATH, JSON.stringify(newCredentials));
                }
            } else {
                throw new Error('No refresh token available');
            }
        }
    } catch (error) {
        console.error('Error refreshing token:', error);
        throw error;
    }
}

/**
 * MCP Protocol Extension: OAuth update method
 */
async function handleOAuthUpdate(request: any) {
    const { access_token, refresh_token, expires_at } = request.params;
    
    if (access_token) {
        oauth2Client.setCredentials({
            access_token,
            refresh_token: refresh_token || oauth2Client.credentials.refresh_token,
            expiry_date: expires_at ? new Date(expires_at).getTime() : undefined,
            token_type: 'Bearer',
            scope: 'https://www.googleapis.com/auth/gmail.modify'
        });
        
        console.log('OAuth credentials updated via MCP protocol');
        
        return {
            content: [
                {
                    type: "text",
                    text: "OAuth credentials updated successfully"
                }
            ]
        };
    }
    
    throw new Error('No access token provided');
}

// Export the enhanced authentication functions
export {
    loadCredentials,
    createDynamicOAuthClient,
    handleToolWithDynamicAuth,
    refreshTokenIfNeeded,
    handleOAuthUpdate
};