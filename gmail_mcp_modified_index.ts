#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
    CallToolRequestSchema,
    ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { google } from 'googleapis';
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { OAuth2Client } from 'google-auth-library';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import http from 'http';
import open from 'open';
import os from 'os';
import {createEmailMessage} from "./utl.js";
import { createLabel, updateLabel, deleteLabel, listLabels, findLabelByName, getOrCreateLabel, GmailLabel } from "./label-manager.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Configuration paths
const CONFIG_DIR = path.join(os.homedir(), '.gmail-mcp');
const OAUTH_PATH = process.env.GMAIL_OAUTH_PATH || path.join(CONFIG_DIR, 'gcp-oauth.keys.json');
const CREDENTIALS_PATH = process.env.GMAIL_CREDENTIALS_PATH || path.join(CONFIG_DIR, 'credentials.json');

// OAuth2 configuration
let oauth2Client: OAuth2Client;
let oauthConfig: any = null;

/**
 * Enhanced loadCredentials that supports multiple token sources
 */
async function loadCredentials() {
    try {
        // Create config directory if it doesn't exist
        if (!process.env.GMAIL_OAUTH_PATH && !CREDENTIALS_PATH && !fs.existsSync(CONFIG_DIR)) {
            fs.mkdirSync(CONFIG_DIR, { recursive: true });
        }

        // Check for OAuth keys
        const localOAuthPath = path.join(process.cwd(), 'gcp-oauth.keys.json');
        let oauthPath = OAUTH_PATH;

        if (fs.existsSync(localOAuthPath)) {
            fs.copyFileSync(localOAuthPath, OAUTH_PATH);
            console.log('OAuth keys found in current directory, copied to global config.');
        }

        if (!fs.existsSync(OAUTH_PATH)) {
            console.error('Error: OAuth keys file not found. Please place gcp-oauth.keys.json in current directory or', CONFIG_DIR);
            process.exit(1);
        }

        const keysContent = JSON.parse(fs.readFileSync(OAUTH_PATH, 'utf8'));
        const keys = keysContent.installed || keysContent.web;

        if (!keys) {
            console.error('Error: Invalid OAuth keys file format.');
            process.exit(1);
        }

        // Store OAuth config for later use
        oauthConfig = keys;

        const callback = process.argv[2] === 'auth' && process.argv[3] 
            ? process.argv[3] 
            : "http://localhost:3000/oauth2callback";

        oauth2Client = new OAuth2Client(
            keys.client_id,
            keys.client_secret,
            callback
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
            const credentials = JSON.parse(fs.readFileSync(CREDENTIALS_PATH, 'utf8'));
            oauth2Client.setCredentials(credentials);
            console.log('Using OAuth tokens from credentials.json');
            return;
        }

        console.log('No OAuth credentials found. Please run with "auth" argument to authenticate.');
    } catch (error) {
        console.error('Error loading credentials:', error);
        process.exit(1);
    }
}

/**
 * Create OAuth client with dynamic tokens
 */
function createDynamicOAuthClient(tokens?: any): OAuth2Client {
    if (tokens && tokens.access_token) {
        const client = new OAuth2Client(
            tokens.client_id || process.env.GMAIL_CLIENT_ID || oauthConfig?.client_id,
            tokens.client_secret || process.env.GMAIL_CLIENT_SECRET || oauthConfig?.client_secret,
            "http://localhost:3000/oauth2callback"
        );
        
        client.setCredentials({
            access_token: tokens.access_token,
            refresh_token: tokens.refresh_token || oauth2Client.credentials.refresh_token,
            token_type: tokens.token_type || 'Bearer',
            scope: tokens.scope || 'https://www.googleapis.com/auth/gmail.modify',
            expiry_date: tokens.expiry_date
        });
        
        return client;
    }
    
    return oauth2Client;
}

/**
 * Refresh token if needed
 */
async function refreshTokenIfNeeded(client: OAuth2Client): Promise<void> {
    const credentials = client.credentials;
    
    // Check if token is expired (with 5 minute buffer)
    if (credentials.expiry_date && credentials.expiry_date <= Date.now() + 300000) {
        console.log('Token expired or expiring soon, refreshing...');
        
        if (credentials.refresh_token) {
            try {
                const { credentials: newCredentials } = await client.refreshAccessToken();
                client.setCredentials(newCredentials);
                
                // Log token refresh for external systems
                console.log('TOKEN_REFRESHED:', JSON.stringify({
                    access_token: newCredentials.access_token,
                    expiry_date: newCredentials.expiry_date
                }));
                
                // Update global client if it's the same
                if (client === oauth2Client && CREDENTIALS_PATH) {
                    fs.writeFileSync(CREDENTIALS_PATH, JSON.stringify(newCredentials));
                }
            } catch (error) {
                console.error('Failed to refresh token:', error);
                throw error;
            }
        } else {
            throw new Error('No refresh token available for token refresh');
        }
    }
}

// ... (keep all the existing type definitions and helper functions)

async function authenticate() {
    // ... (keep existing authenticate function)
}

// Main function
async function main() {
    await loadCredentials();

    if (process.argv[2] === 'auth') {
        await authenticate();
        console.log('Authentication completed successfully');
        process.exit(0);
    }

    // Server implementation
    const server = new Server({
        name: "gmail",
        version: "1.0.0",
        capabilities: {
            tools: {},
        },
    });

    // Tool handlers
    server.setRequestHandler(ListToolsRequestSchema, async () => ({
        tools: [
            // ... (keep all existing tool definitions)
        ],
    }));

    server.setRequestHandler(CallToolRequestSchema, async (request) => {
        const { name, arguments: args } = request.params;

        // Check if OAuth tokens are provided in the request
        let authClient = oauth2Client;
        
        if (args.google_access_token || args.oauth_token || args.access_token) {
            // Create a new OAuth client with the provided tokens
            authClient = createDynamicOAuthClient({
                access_token: args.google_access_token || args.oauth_token || args.access_token,
                refresh_token: args.google_refresh_token || args.oauth_refresh_token || args.refresh_token,
                client_id: args.google_client_id || args.client_id,
                client_secret: args.google_client_secret || args.client_secret,
            });
            
            // Remove OAuth fields from args before processing
            delete args.google_access_token;
            delete args.oauth_token;
            delete args.access_token;
            delete args.google_refresh_token;
            delete args.oauth_refresh_token;
            delete args.refresh_token;
            delete args.google_client_id;
            delete args.client_id;
            delete args.google_client_secret;
            delete args.client_secret;
        }

        // Refresh token if needed
        try {
            await refreshTokenIfNeeded(authClient);
        } catch (error) {
            console.error('Token refresh failed:', error);
            return {
                content: [{
                    type: "text",
                    text: `Authentication error: ${error.message}. Please provide valid OAuth credentials.`
                }],
            };
        }

        // Initialize Gmail API with the appropriate auth client
        const gmail = google.gmail({ version: 'v1', auth: authClient });

        // ... (rest of the existing tool handling code)
        // Keep all the existing switch cases and tool implementations
        // Just ensure they use the `gmail` client initialized above
    });

    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.log('Gmail MCP Server running with dynamic OAuth support');
}

main().catch((error) => {
    console.error('Server error:', error);
    process.exit(1);
});