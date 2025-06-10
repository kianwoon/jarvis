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
        const oauth2Client = new OAuth2Client(
            args.google_client_id,
            args.google_client_secret
        );
        
        oauth2Client.setCredentials({
            access_token: args.google_access_token,
            refresh_token: args.google_refresh_token
        });
        
        const gmail = google.gmail({ version: 'v1', auth: oauth2Client });
        
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
            const { to, subject, body, cc, bcc } = args;
            
            console.error(`[DEBUG] gmail_send called with: to=${JSON.stringify(to)}, subject=${subject}`);
            
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
            
            const encodedMessage = Buffer.from(message).toString('base64')
                .replace(/\+/g, '-')
                .replace(/\//g, '_')
                .replace(/=+$/, '');
            
            console.error(`[DEBUG] Attempting to send email to Gmail API...`);
            const response = await gmail.users.messages.send({
                userId: 'me',
                requestBody: {
                    raw: encodedMessage
                }
            });
            
            console.error(`[DEBUG] Gmail API response: ${JSON.stringify(response.data)}`);
            return {
                content: [{
                    type: "text",
                    text: `✅ Email sent successfully!\nMessage ID: ${response.data.id}`
                }]
            };
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
        return {
            content: [{
                type: "text",
                text: `Error: ${error.message}`
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
console.error('Simple Gmail MCP Server running');