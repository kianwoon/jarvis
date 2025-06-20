#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
    CallToolRequestSchema,
    ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { google } from 'googleapis';
import { OAuth2Client } from 'google-auth-library';

const server = new Server({
    name: "gmail",
    version: "1.0.0",
    capabilities: {
        tools: {},
    },
});

// Tools definition
server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
        {
            name: "search_emails",
            description: "Searches for emails using Gmail search syntax",
            inputSchema: {
                type: "object",
                properties: {
                    query: { type: "string", description: "Gmail search query" },
                    maxResults: { type: "number", description: "Max results to return" }
                },
                required: ["query"]
            }
        },
        {
            name: "get_latest_emails",
            description: "Retrieves the latest emails from Gmail inbox",
            inputSchema: {
                type: "object",
                properties: {
                    count: { type: "number", description: "Number of latest emails to retrieve (default: 10)" },
                    includeBody: { type: "boolean", description: "Whether to include email body content (default: false)" },
                    labelIds: { 
                        type: "array", 
                        items: { type: "string" },
                        description: "Filter by specific label IDs (e.g., ['INBOX'])" 
                    }
                },
                required: []
            }
        },
        {
            name: "find_email",
            description: "Advanced email search with full email context including from, to, cc, subject, and body",
            inputSchema: {
                type: "object",
                properties: {
                    sender: { type: "string", description: "Filter by sender email address" },
                    subject: { type: "string", description: "Filter by subject text (supports partial matches)" },
                    to: { type: "string", description: "Filter by recipient email address" },
                    cc: { type: "string", description: "Filter by CC email address" },
                    body: { type: "string", description: "Filter by email body content" },
                    hasAttachment: { type: "boolean", description: "Filter by presence of attachments" },
                    isUnread: { type: "boolean", description: "Filter by read/unread status" },
                    labelIds: { type: "array", items: { type: "string" }, description: "Filter by specific label IDs" },
                    dateAfter: { type: "string", description: "Filter emails after this date (YYYY/MM/DD or YYYY-MM-DD format)" },
                    dateBefore: { type: "string", description: "Filter emails before this date (YYYY/MM/DD or YYYY-MM-DD format)" },
                    maxResults: { type: "number", description: "Maximum number of results to return (default: 10)" }
                },
                required: []
            }
        },
        {
            name: "read_email",
            description: "Read a specific email by its message ID",
            inputSchema: {
                type: "object",
                properties: {
                    messageId: { type: "string", description: "ID of the email message to retrieve" }
                },
                required: ["messageId"]
            }
        },
        {
            name: "gmail_send",
            description: "Send a new email",
            inputSchema: {
                type: "object",
                properties: {
                    to: { type: "array", items: { type: "string" }, description: "List of recipient email addresses" },
                    subject: { type: "string", description: "Email subject" },
                    body: { type: "string", description: "Email body content" },
                    cc: { type: "array", items: { type: "string" }, description: "List of CC recipients" },
                    bcc: { type: "array", items: { type: "string" }, description: "List of BCC recipients" }
                },
                required: ["to", "subject", "body"]
            }
        },
        {
            name: "draft_email",
            description: "Create a new email draft",
            inputSchema: {
                type: "object",
                properties: {
                    to: { type: "array", items: { type: "string" }, description: "List of recipient email addresses" },
                    subject: { type: "string", description: "Email subject" },
                    body: { type: "string", description: "Email body content" },
                    cc: { type: "array", items: { type: "string" }, description: "List of CC recipients" },
                    bcc: { type: "array", items: { type: "string" }, description: "List of BCC recipients" }
                },
                required: ["to", "subject", "body"]
            }
        },
        {
            name: "gmail_send_draft",
            description: "Send an existing email draft",
            inputSchema: {
                type: "object",
                properties: {
                    draftId: { type: "string", description: "ID of the draft to send" }
                },
                required: ["draftId"]
            }
        },
        {
            name: "delete_email",
            description: "Permanently delete an email",
            inputSchema: {
                type: "object",
                properties: {
                    messageId: { type: "string", description: "ID of the email message to delete" }
                },
                required: ["messageId"]
            }
        },
        {
            name: "gmail_trash_message",
            description: "Move a message to trash",
            inputSchema: {
                type: "object",
                properties: {
                    messageId: { type: "string", description: "ID of the email message to trash" }
                },
                required: ["messageId"]
            }
        },
        {
            name: "gmail_get_thread",
            description: "Get all messages in a thread",
            inputSchema: {
                type: "object",
                properties: {
                    threadId: { type: "string", description: "ID of the thread to retrieve" }
                },
                required: ["threadId"]
            }
        },
        {
            name: "gmail_update_draft",
            description: "Update an existing draft",
            inputSchema: {
                type: "object",
                properties: {
                    draftId: { type: "string", description: "ID of the draft to update" },
                    to: { type: "array", items: { type: "string" }, description: "List of recipient email addresses" },
                    subject: { type: "string", description: "Email subject" },
                    body: { type: "string", description: "Email body content" },
                    cc: { type: "array", items: { type: "string" }, description: "List of CC recipients" },
                    bcc: { type: "array", items: { type: "string" }, description: "List of BCC recipients" }
                },
                required: ["draftId"]
            }
        },
        {
            name: "list_email_labels",
            description: "Retrieve all available Gmail labels",
            inputSchema: {
                type: "object",
                properties: {},
                required: []
            }
        },
        {
            name: "modify_email",
            description: "Modify email labels (move to different folders)",
            inputSchema: {
                type: "object",
                properties: {
                    messageId: { type: "string", description: "ID of the email message to modify" },
                    addLabelIds: { type: "array", items: { type: "string" }, description: "Label IDs to add" },
                    removeLabelIds: { type: "array", items: { type: "string" }, description: "Label IDs to remove" }
                },
                required: ["messageId"]
            }
        }
    ]
}));

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    
    // Common OAuth setup
    try {
        console.error(`[DEBUG] Setting up OAuth for tool: ${name}`);
        console.error(`[DEBUG] Client ID present: ${!!args.google_client_id}`);
        console.error(`[DEBUG] Access token present: ${!!args.google_access_token}`);
        
        const oauth2Client = new OAuth2Client(
            args.google_client_id,
            args.google_client_secret
        );
        
        oauth2Client.setCredentials({
            access_token: args.google_access_token,
            refresh_token: args.google_refresh_token
        });
        
        console.error(`[DEBUG] OAuth client configured successfully`);
        const gmail = google.gmail({ version: 'v1', auth: oauth2Client });
        console.error(`[DEBUG] Gmail API client created successfully`);
        
        if (name === "search_emails") {
            const query = args.query;
            const maxResults = args.maxResults || 10;
            
            const response = await gmail.users.messages.list({
                userId: 'me',
                q: query,
                maxResults: maxResults
            });
            
            const messages = response.data.messages || [];
            const results = await Promise.all(
                messages.map(async (msg) => {
                    const detail = await gmail.users.messages.get({
                        userId: 'me',
                        id: msg.id,
                        format: 'metadata',
                        metadataHeaders: ['Subject', 'From', 'Date']
                    });
                    const headers = detail.data.payload?.headers || [];
                    return {
                        id: msg.id,
                        subject: headers.find(h => h.name === 'Subject')?.value || '',
                        from: headers.find(h => h.name === 'From')?.value || '',
                        date: headers.find(h => h.name === 'Date')?.value || ''
                    };
                })
            );
            
            return {
                content: [{
                    type: "text",
                    text: results.length > 0 
                        ? results.map(r => `📧 ${r.subject}\n   From: ${r.from}\n   Date: ${r.date}\n   ID: ${r.id}`).join('\n\n')
                        : 'No emails found.'
                }]
            };
        }
        
        if (name === "get_latest_emails") {
            const count = args.count || 10;
            const includeBody = args.includeBody || false;
            const labelIds = args.labelIds || null;
            
            // Build the list request
            const listRequest = {
                userId: 'me',
                maxResults: count
            };
            
            // Add label filter if specified
            if (labelIds && labelIds.length > 0) {
                listRequest.labelIds = labelIds;
            }
            
            // Get the list of messages
            const response = await gmail.users.messages.list(listRequest);
            const messages = response.data.messages || [];
            
            // Fetch details for each message
            const emailDetails = await Promise.all(
                messages.map(async (msg) => {
                    const detail = await gmail.users.messages.get({
                        userId: 'me',
                        id: msg.id,
                        format: includeBody ? 'full' : 'metadata',
                        metadataHeaders: includeBody ? undefined : ['Subject', 'From', 'To', 'Date']
                    });
                    
                    const headers = detail.data.payload?.headers || [];
                    const subject = headers.find(h => h.name?.toLowerCase() === 'subject')?.value || '';
                    const from = headers.find(h => h.name?.toLowerCase() === 'from')?.value || '';
                    const to = headers.find(h => h.name?.toLowerCase() === 'to')?.value || '';
                    const date = headers.find(h => h.name?.toLowerCase() === 'date')?.value || '';
                    
                    let bodyPreview = '';
                    if (includeBody && detail.data.payload) {
                        // Simple body extraction for the simplified version
                        const extractBody = (payload) => {
                            if (payload.body?.data) {
                                return Buffer.from(payload.body.data, 'base64').toString('utf8');
                            }
                            if (payload.parts) {
                                for (const part of payload.parts) {
                                    if (part.mimeType === 'text/plain' && part.body?.data) {
                                        return Buffer.from(part.body.data, 'base64').toString('utf8');
                                    }
                                }
                            }
                            return '';
                        };
                        
                        const fullBody = extractBody(detail.data.payload);
                        bodyPreview = fullBody.length > 200 
                            ? fullBody.substring(0, 200) + '...' 
                            : fullBody;
                    }
                    
                    return {
                        id: msg.id,
                        subject,
                        from,
                        to,
                        date,
                        snippet: detail.data.snippet || '',
                        body: bodyPreview,
                        labels: detail.data.labelIds || []
                    };
                })
            );
            
            // Format the output
            const formattedEmails = emailDetails.map((email, index) => {
                const labelInfo = email.labels.length > 0 
                    ? `\n   Labels: ${email.labels.join(', ')}` 
                    : '';
                const bodyInfo = includeBody && email.body 
                    ? `\n   Preview: ${email.body}` 
                    : '';
                
                return `${index + 1}. 📧 ${email.subject}\n   From: ${email.from}\n   To: ${email.to}\n   Date: ${email.date}${labelInfo}\n   Snippet: ${email.snippet}${bodyInfo}\n   ID: ${email.id}`;
            }).join('\n\n');
            
            return {
                content: [{
                    type: "text",
                    text: emailDetails.length > 0 
                        ? `Latest ${emailDetails.length} emails:\n\n${formattedEmails}`
                        : 'No emails found.'
                }]
            };
        }
        
        if (name === "find_email") {
            // Build Gmail search query from parameters
            const queryParts = [];
            
            if (args.sender) {
                queryParts.push(`from:${args.sender}`);
            }
            
            if (args.subject) {
                queryParts.push(`subject:${args.subject}`);
            }
            
            if (args.to) {
                queryParts.push(`to:${args.to}`);
            }
            
            if (args.cc) {
                queryParts.push(`cc:${args.cc}`);
            }
            
            if (args.body) {
                queryParts.push(`"${args.body}"`);
            }
            
            if (args.hasAttachment === true) {
                queryParts.push('has:attachment');
            } else if (args.hasAttachment === false) {
                queryParts.push('-has:attachment');
            }
            
            if (args.isUnread === true) {
                queryParts.push('is:unread');
            } else if (args.isUnread === false) {
                queryParts.push('is:read');
            }
            
            if (args.labelIds && args.labelIds.length > 0) {
                args.labelIds.forEach((labelId) => {
                    queryParts.push(`label:${labelId}`);
                });
            }
            
            if (args.dateAfter) {
                const dateAfter = args.dateAfter.replace(/-/g, '/');
                queryParts.push(`after:${dateAfter}`);
            }
            
            if (args.dateBefore) {
                const dateBefore = args.dateBefore.replace(/-/g, '/');
                queryParts.push(`before:${dateBefore}`);
            }
            
            const query = queryParts.join(' ');
            
            if (!query) {
                return {
                    content: [{
                        type: "text",
                        text: "No search criteria provided. Please specify at least one search parameter."
                    }]
                };
            }
            
            // Search for emails using the constructed query
            const searchResponse = await gmail.users.messages.list({
                userId: 'me',
                q: query,
                maxResults: args.maxResults || 10
            });
            
            const messages = searchResponse.data.messages || [];
            
            if (messages.length === 0) {
                return {
                    content: [{
                        type: "text",
                        text: "No emails found matching the search criteria."
                    }]
                };
            }
            
            // Fetch full details for each email
            const emailDetails = await Promise.all(
                messages.map(async (msg) => {
                    const detail = await gmail.users.messages.get({
                        userId: 'me',
                        id: msg.id,
                        format: 'full'
                    });
                    
                    const headers = detail.data.payload?.headers || [];
                    const subject = headers.find(h => h.name?.toLowerCase() === 'subject')?.value || '';
                    const from = headers.find(h => h.name?.toLowerCase() === 'from')?.value || '';
                    const to = headers.find(h => h.name?.toLowerCase() === 'to')?.value || '';
                    const cc = headers.find(h => h.name?.toLowerCase() === 'cc')?.value || '';
                    const date = headers.find(h => h.name?.toLowerCase() === 'date')?.value || '';
                    const threadId = detail.data.threadId || '';
                    
                    // Extract email content - simplified version
                    const extractContent = (payload) => {
                        if (payload.body?.data) {
                            return Buffer.from(payload.body.data, 'base64').toString('utf8');
                        }
                        if (payload.parts) {
                            for (const part of payload.parts) {
                                if (part.mimeType === 'text/plain' && part.body?.data) {
                                    return Buffer.from(part.body.data, 'base64').toString('utf8');
                                }
                            }
                            // If no plain text, try HTML
                            for (const part of payload.parts) {
                                if (part.mimeType === 'text/html' && part.body?.data) {
                                    return Buffer.from(part.body.data, 'base64').toString('utf8');
                                }
                            }
                        }
                        return '';
                    };
                    
                    const body = extractContent(detail.data.payload);
                    
                    return {
                        id: msg.id,
                        threadId,
                        subject,
                        from,
                        to,
                        cc,
                        date,
                        body,
                        snippet: detail.data.snippet || '',
                        labels: detail.data.labelIds || [],
                        isUnread: detail.data.labelIds?.includes('UNREAD') || false
                    };
                })
            );
            
            // Format the detailed results
            const formattedEmails = emailDetails.map((email, index) => {
                const ccInfo = email.cc ? `\n   CC: ${email.cc}` : '';
                const labelInfo = email.labels.length > 0 ? `\n   Labels: ${email.labels.join(', ')}` : '';
                const statusInfo = email.isUnread ? ' [UNREAD]' : '';
                
                // Truncate body for display
                const bodyPreview = email.body.length > 500 
                    ? email.body.substring(0, 500) + '...\n   [Body truncated - use read_email for full content]'
                    : email.body;
                
                return `${index + 1}. 📧 ${email.subject}${statusInfo}
   From: ${email.from}
   To: ${email.to}${ccInfo}
   Date: ${email.date}${labelInfo}
   Thread ID: ${email.threadId}
   Message ID: ${email.id}
   
   Body:
   ${bodyPreview}`;
            }).join('\n\n' + '='.repeat(80) + '\n\n');
            
            return {
                content: [{
                    type: "text",
                    text: `Found ${emailDetails.length} email(s) matching search criteria:\n\n${formattedEmails}`
                }]
            };
        }
        
        if (name === "read_email") {
            const messageId = args.messageId;
            
            const detail = await gmail.users.messages.get({
                userId: 'me',
                id: messageId,
                format: 'full'
            });
            
            const headers = detail.data.payload?.headers || [];
            const subject = headers.find(h => h.name?.toLowerCase() === 'subject')?.value || '';
            const from = headers.find(h => h.name?.toLowerCase() === 'from')?.value || '';
            const to = headers.find(h => h.name?.toLowerCase() === 'to')?.value || '';
            const date = headers.find(h => h.name?.toLowerCase() === 'date')?.value || '';
            
            // Extract body
            const extractFullBody = (payload) => {
                if (payload.body?.data) {
                    return Buffer.from(payload.body.data, 'base64').toString('utf8');
                }
                if (payload.parts) {
                    for (const part of payload.parts) {
                        if (part.mimeType === 'text/plain' && part.body?.data) {
                            return Buffer.from(part.body.data, 'base64').toString('utf8');
                        }
                    }
                    // If no plain text, try HTML
                    for (const part of payload.parts) {
                        if (part.mimeType === 'text/html' && part.body?.data) {
                            return Buffer.from(part.body.data, 'base64').toString('utf8');
                        }
                    }
                }
                return '';
            };
            
            const body = extractFullBody(detail.data.payload);
            
            return {
                content: [{
                    type: "text",
                    text: `📧 Email Details:\n\nSubject: ${subject}\nFrom: ${from}\nTo: ${to}\nDate: ${date}\n\nBody:\n${body}`
                }]
            };
        }
        
        if (name === "gmail_send") {
            console.error(`[DEBUG] gmail_send tool called`);
            console.error(`[DEBUG] RAW ARGS: ${JSON.stringify(args)}`);
            console.error(`[DEBUG] Args keys: ${Object.keys(args)}`);
            console.error(`[DEBUG] Args type: ${typeof args}`);
            
            // Handle parameter extraction more carefully
            let { to, subject, body, cc, bcc } = args;
            
            // If body is undefined, check for common parameter variations
            if (!body) {
                body = args.message || args.content || args.text || "";
                console.error(`[DEBUG] Body was undefined, using fallback: ${body}`);
            }
            
            console.error(`[DEBUG] gmail_send called with: to=${JSON.stringify(to)}, subject=${subject}`);
            console.error(`[DEBUG] Body parameter: ${JSON.stringify(body)}`);
            console.error(`[DEBUG] Body length: ${body ? body.length : 'undefined'} characters`);
            
            // Ensure 'to' is an array
            const toArray = Array.isArray(to) ? to : [to];
            console.error(`[DEBUG] Converted to array: ${JSON.stringify(toArray)}`);
            
            // Create email message
            const messageParts = [
                `To: ${toArray.join(', ')}`,
                `Subject: ${subject}`
            ];
            
            if (cc && cc.length > 0) {
                messageParts.push(`Cc: ${cc.join(', ')}`);
            }
            if (bcc && bcc.length > 0) {
                messageParts.push(`Bcc: ${bcc.join(', ')}`);
            }
            
            messageParts.push('', body);
            const message = messageParts.join('\n');
            
            console.error(`[DEBUG] Complete message before encoding: ${JSON.stringify(message)}`);
            console.error(`[DEBUG] Message parts: ${JSON.stringify(messageParts)}`);
            
            const encodedMessage = Buffer.from(message).toString('base64')
                .replace(/\+/g, '-')
                .replace(/\//g, '_')
                .replace(/=+$/, '');
            
            console.error(`[DEBUG] Attempting to send email to Gmail API...`);
            console.error(`[DEBUG] Message size: ${encodedMessage.length} chars`);
            
            try {
                const response = await gmail.users.messages.send({
                    userId: 'me',
                    requestBody: {
                        raw: encodedMessage
                    }
                });
                
                console.error(`[DEBUG] Gmail API response: ${JSON.stringify(response.data)}`);
                console.error(`[DEBUG] Email sent successfully with ID: ${response.data.id}`);
                
                return {
                    content: [{
                        type: "text",
                        text: `✅ Email sent successfully!\nMessage ID: ${response.data.id}`
                    }]
                };
            } catch (gmailError) {
                console.error(`[ERROR] Gmail API call failed: ${gmailError.message}`);
                console.error(`[ERROR] Gmail error code: ${gmailError.code}`);
                console.error(`[ERROR] Gmail error details:`, gmailError);
                
                // Try to refresh token if it's an auth error
                if (gmailError.code === 401 || gmailError.message.includes('invalid_token')) {
                    console.error(`[DEBUG] Token seems expired, attempting refresh...`);
                    try {
                        const newTokens = await oauth2Client.refreshAccessToken();
                        console.error(`[DEBUG] Token refreshed successfully`);
                        
                        // Retry the send with new token
                        const retryResponse = await gmail.users.messages.send({
                            userId: 'me',
                            requestBody: {
                                raw: encodedMessage
                            }
                        });
                        
                        console.error(`[DEBUG] Retry successful: ${JSON.stringify(retryResponse.data)}`);
                        return {
                            content: [{
                                type: "text",
                                text: `✅ Email sent successfully (after token refresh)!\nMessage ID: ${retryResponse.data.id}`
                            }]
                        };
                    } catch (refreshError) {
                        console.error(`[ERROR] Token refresh failed: ${refreshError.message}`);
                        throw new Error(`Gmail authentication failed: ${refreshError.message}`);
                    }
                } else {
                    throw gmailError;
                }
            }
        }
        
        if (name === "draft_email") {
            const { to, subject, body, cc, bcc } = args;
            
            // Create email message
            const messageParts = [
                `To: ${to.join(', ')}`,
                `Subject: ${subject}`
            ];
            
            if (cc && cc.length > 0) {
                messageParts.push(`Cc: ${cc.join(', ')}`);
            }
            if (bcc && bcc.length > 0) {
                messageParts.push(`Bcc: ${bcc.join(', ')}`);
            }
            
            messageParts.push('', body);
            const message = messageParts.join('\n');
            
            const encodedMessage = Buffer.from(message).toString('base64')
                .replace(/\+/g, '-')
                .replace(/\//g, '_')
                .replace(/=+$/, '');
            
            const response = await gmail.users.drafts.create({
                userId: 'me',
                requestBody: {
                    message: {
                        raw: encodedMessage
                    }
                }
            });
            
            return {
                content: [{
                    type: "text",
                    text: `📝 Draft created successfully!\nDraft ID: ${response.data.id}`
                }]
            };
        }
        
        if (name === "gmail_send_draft") {
            const { draftId } = args;
            
            const response = await gmail.users.drafts.send({
                userId: 'me',
                requestBody: {
                    id: draftId
                }
            });
            
            return {
                content: [{
                    type: "text",
                    text: `✅ Draft sent successfully!\nMessage ID: ${response.data.id}`
                }]
            };
        }
        
        if (name === "delete_email") {
            const { messageId } = args;
            
            await gmail.users.messages.delete({
                userId: 'me',
                id: messageId
            });
            
            return {
                content: [{
                    type: "text",
                    text: `🗑️ Email permanently deleted!\nMessage ID: ${messageId}`
                }]
            };
        }
        
        if (name === "gmail_trash_message") {
            const { messageId } = args;
            
            await gmail.users.messages.trash({
                userId: 'me',
                id: messageId
            });
            
            return {
                content: [{
                    type: "text",
                    text: `🗑️ Email moved to trash!\nMessage ID: ${messageId}`
                }]
            };
        }
        
        if (name === "gmail_get_thread") {
            const { threadId } = args;
            
            const thread = await gmail.users.threads.get({
                userId: 'me',
                id: threadId,
                format: 'full'
            });
            
            const messages = thread.data.messages || [];
            const threadInfo = messages.map((msg, index) => {
                const headers = msg.payload?.headers || [];
                const subject = headers.find(h => h.name?.toLowerCase() === 'subject')?.value || '';
                const from = headers.find(h => h.name?.toLowerCase() === 'from')?.value || '';
                const date = headers.find(h => h.name?.toLowerCase() === 'date')?.value || '';
                const snippet = msg.snippet || '';
                
                return `📧 Message ${index + 1}:\n   Subject: ${subject}\n   From: ${from}\n   Date: ${date}\n   Preview: ${snippet}\n   ID: ${msg.id}`;
            }).join('\n\n');
            
            return {
                content: [{
                    type: "text",
                    text: `Thread ID: ${threadId}\nTotal messages: ${messages.length}\n\n${threadInfo}`
                }]
            };
        }
        
        if (name === "gmail_update_draft") {
            const { draftId, to, subject, body, cc, bcc } = args;
            
            // First get the existing draft
            const existingDraft = await gmail.users.drafts.get({
                userId: 'me',
                id: draftId
            });
            
            // Create updated message
            const messageParts = [];
            if (to) messageParts.push(`To: ${to.join(', ')}`);
            if (subject) messageParts.push(`Subject: ${subject}`);
            if (cc && cc.length > 0) messageParts.push(`Cc: ${cc.join(', ')}`);
            if (bcc && bcc.length > 0) messageParts.push(`Bcc: ${bcc.join(', ')}`);
            
            // If we have new content, use it; otherwise keep existing
            if (messageParts.length > 0 || body) {
                if (!to && existingDraft.data.message?.payload?.headers) {
                    const existingTo = existingDraft.data.message.payload.headers.find(h => h.name?.toLowerCase() === 'to')?.value;
                    if (existingTo) messageParts.push(`To: ${existingTo}`);
                }
                if (!subject && existingDraft.data.message?.payload?.headers) {
                    const existingSubject = existingDraft.data.message.payload.headers.find(h => h.name?.toLowerCase() === 'subject')?.value;
                    if (existingSubject) messageParts.push(`Subject: ${existingSubject}`);
                }
                
                messageParts.push('', body || 'Draft content');
                const message = messageParts.join('\n');
                
                const encodedMessage = Buffer.from(message).toString('base64')
                    .replace(/\+/g, '-')
                    .replace(/\//g, '_')
                    .replace(/=+$/, '');
                
                const response = await gmail.users.drafts.update({
                    userId: 'me',
                    id: draftId,
                    requestBody: {
                        message: {
                            raw: encodedMessage
                        }
                    }
                });
                
                return {
                    content: [{
                        type: "text",
                        text: `📝 Draft updated successfully!\nDraft ID: ${response.data.id}`
                    }]
                };
            }
            
            return {
                content: [{
                    type: "text",
                    text: `❌ No updates provided for draft`
                }]
            };
        }
        
        if (name === "list_email_labels") {
            const response = await gmail.users.labels.list({
                userId: 'me'
            });
            
            const labels = response.data.labels || [];
            const labelList = labels.map(label => 
                `• ${label.name} (ID: ${label.id})`
            ).join('\n');
            
            return {
                content: [{
                    type: "text",
                    text: `📂 Gmail Labels (${labels.length} total):\n\n${labelList}`
                }]
            };
        }
        
        if (name === "modify_email") {
            const { messageId, addLabelIds, removeLabelIds } = args;
            
            const modifyRequest = {};
            if (addLabelIds && addLabelIds.length > 0) {
                modifyRequest.addLabelIds = addLabelIds;
            }
            if (removeLabelIds && removeLabelIds.length > 0) {
                modifyRequest.removeLabelIds = removeLabelIds;
            }
            
            const response = await gmail.users.messages.modify({
                userId: 'me',
                id: messageId,
                requestBody: modifyRequest
            });
            
            const actions = [];
            if (addLabelIds && addLabelIds.length > 0) {
                actions.push(`Added labels: ${addLabelIds.join(', ')}`);
            }
            if (removeLabelIds && removeLabelIds.length > 0) {
                actions.push(`Removed labels: ${removeLabelIds.join(', ')}`);
            }
            
            return {
                content: [{
                    type: "text",
                    text: `✅ Email labels modified successfully!\nMessage ID: ${messageId}\n${actions.join('\n')}`
                }]
            };
        }
    } catch (error) {
        console.error(`[ERROR] Tool ${name} failed: ${error.message}`);
        console.error(`[ERROR] Stack trace: ${error.stack}`);
        
        // Ensure we always return a valid JSON-RPC response
        return {
            content: [{
                type: "text",
                text: `Error executing ${name}: ${error.message}`
            }]
        };
    }
    
    return {
        content: [{
            type: "text",
            text: `Unknown tool: ${name}`
        }]
    };
});

const transport = new StdioServerTransport();
await server.connect(transport);

// Force immediate output
process.stderr.write('Simple Gmail MCP Server running\n');

// Add process error handlers to catch any unhandled errors
process.on('uncaughtException', (error) => {
    console.error(`[UNCAUGHT] Exception: ${error.message}`);
    console.error(`[UNCAUGHT] Stack: ${error.stack}`);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error(`[UNHANDLED] Rejection at:`, promise, 'reason:', reason);
});