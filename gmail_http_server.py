#!/usr/bin/env python3
"""
HTTP Gmail Tool Server - Reliable alternative to Docker stdio MCP
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gmail HTTP Tool Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ToolRequest(BaseModel):
    tool: str
    parameters: Dict[str, Any]
    timestamp: str = None

class ToolResponse(BaseModel):
    success: bool
    result: Any = None
    error: str = None
    execution_time: float = None

async def execute_gmail_tool(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Gmail tool using Google APIs directly"""
    try:
        logger.info(f"[GMAIL-HTTP] Executing {tool_name}")
        logger.debug(f"[GMAIL-HTTP] Parameters: {json.dumps(params, indent=2)}")
        
        # Import Gmail API libraries
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        import base64
        
        # Extract OAuth credentials
        creds_data = {
            'token': params.get('google_access_token'),
            'refresh_token': params.get('google_refresh_token'),
            'client_id': params.get('google_client_id'),
            'client_secret': params.get('google_client_secret'),
            'token_uri': 'https://oauth2.googleapis.com/token'
        }
        
        # Create credentials object
        creds = Credentials.from_authorized_user_info(creds_data)
        
        # Refresh token if needed
        if creds.expired and creds.refresh_token:
            logger.info(f"[GMAIL-HTTP] Refreshing expired token for {tool_name}")
            creds.refresh(Request())
        
        # Build Gmail service
        service = build('gmail', 'v1', credentials=creds)
        logger.info(f"[GMAIL-HTTP] Gmail service created successfully")
        
        # Execute specific tool
        if tool_name == "gmail_send":
            return await _send_email(service, params)
        elif tool_name == "find_email":
            return await _find_email(service, params)
        elif tool_name == "search_emails":
            return await _search_emails(service, params)
        elif tool_name == "read_email":
            return await _read_email(service, params)
        else:
            return {"error": f"Tool {tool_name} not implemented in HTTP server"}
            
    except Exception as e:
        logger.error(f"[GMAIL-HTTP] Error executing {tool_name}: {str(e)}")
        logger.error(f"[GMAIL-HTTP] Error type: {type(e).__name__}")
        return {"error": f"Gmail API error: {str(e)}"}

async def _send_email(service, params: Dict[str, Any]) -> Dict[str, Any]:
    """Send email via Gmail API"""
    try:
        to_list = params.get('to', [])
        if isinstance(to_list, str):
            to_list = [to_list]
        
        subject = params.get('subject', '')
        body = params.get('body', '')
        cc_list = params.get('cc', [])
        bcc_list = params.get('bcc', [])
        
        logger.info(f"[GMAIL-SEND] Sending to: {to_list}")
        logger.info(f"[GMAIL-SEND] Subject: {subject}")
        
        # Create email message
        message_parts = [
            f"To: {', '.join(to_list)}",
            f"Subject: {subject}"
        ]
        
        if cc_list:
            message_parts.append(f"Cc: {', '.join(cc_list)}")
        if bcc_list:
            message_parts.append(f"Bcc: {', '.join(bcc_list)}")
        
        message_parts.extend(['', body])
        message = '\n'.join(message_parts)
        
        # Encode message
        encoded_message = base64.urlsafe_b64encode(message.encode('utf-8')).decode('utf-8')
        
        # Send via Gmail API
        send_result = service.users().messages().send(
            userId='me',
            body={'raw': encoded_message}
        ).execute()
        
        message_id = send_result.get('id')
        logger.info(f"[GMAIL-SEND] Email sent successfully with ID: {message_id}")
        
        return {
            "success": True,
            "message_id": message_id,
            "text": f"✅ Email sent successfully!\nMessage ID: {message_id}"
        }
        
    except Exception as e:
        logger.error(f"[GMAIL-SEND] Failed to send email: {str(e)}")
        return {"error": f"Failed to send email: {str(e)}"}

async def _find_email(service, params: Dict[str, Any]) -> Dict[str, Any]:
    """Find emails via Gmail API"""
    try:
        # Build search query
        query_parts = []
        
        if params.get('sender'):
            query_parts.append(f"from:{params['sender']}")
        if params.get('subject'):
            query_parts.append(f"subject:{params['subject']}")
        if params.get('from_'):  # Handle alternative parameter name
            query_parts.append(f"from:{params['from_']}")
        
        query = ' '.join(query_parts)
        max_results = params.get('maxResults', 10)
        
        logger.info(f"[GMAIL-FIND] Query: {query}")
        
        # Search messages
        search_result = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()
        
        messages = search_result.get('messages', [])
        logger.info(f"[GMAIL-FIND] Found {len(messages)} messages")
        
        # Get details for each message
        email_details = []
        for msg in messages[:5]:  # Limit to first 5 for performance
            msg_detail = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='full'
            ).execute()
            
            headers = msg_detail['payload'].get('headers', [])
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
            from_addr = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
            date = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
            
            email_details.append({
                'id': msg['id'],
                'subject': subject,
                'from': from_addr,
                'date': date,
                'snippet': msg_detail.get('snippet', '')
            })
        
        formatted_result = '\n\n'.join([
            f"📧 {email['subject']}\n   From: {email['from']}\n   Date: {email['date']}\n   ID: {email['id']}"
            for email in email_details
        ])
        
        return {
            "success": True,
            "emails_found": len(email_details),
            "text": formatted_result if email_details else "No emails found matching criteria"
        }
        
    except Exception as e:
        logger.error(f"[GMAIL-FIND] Failed to find emails: {str(e)}")
        return {"error": f"Failed to find emails: {str(e)}"}

async def _search_emails(service, params: Dict[str, Any]) -> Dict[str, Any]:
    """Search emails via Gmail API"""
    try:
        query = params.get('query', '')
        max_results = params.get('maxResults', 10)
        
        logger.info(f"[GMAIL-SEARCH] Query: {query}")
        
        # Search messages  
        search_result = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()
        
        messages = search_result.get('messages', [])
        logger.info(f"[GMAIL-SEARCH] Found {len(messages)} messages")
        
        # Format results
        if not messages:
            return {"success": True, "text": "No emails found"}
        
        # Get basic details
        results = []
        for msg in messages:
            msg_detail = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='metadata',
                metadataHeaders=['Subject', 'From', 'Date']
            ).execute()
            
            headers = msg_detail['payload'].get('headers', [])
            results.append({
                'id': msg['id'],
                'subject': next((h['value'] for h in headers if h['name'] == 'Subject'), ''),
                'from': next((h['value'] for h in headers if h['name'] == 'From'), ''),
                'date': next((h['value'] for h in headers if h['name'] == 'Date'), '')
            })
        
        formatted_result = '\n\n'.join([
            f"📧 {r['subject']}\n   From: {r['from']}\n   Date: {r['date']}\n   ID: {r['id']}"
            for r in results
        ])
        
        return {"success": True, "text": formatted_result}
        
    except Exception as e:
        logger.error(f"[GMAIL-SEARCH] Failed to search emails: {str(e)}")
        return {"error": f"Failed to search emails: {str(e)}"}

async def _read_email(service, params: Dict[str, Any]) -> Dict[str, Any]:
    """Read specific email via Gmail API"""
    try:
        message_id = params.get('messageId')
        if not message_id:
            return {"error": "messageId parameter required"}
        
        logger.info(f"[GMAIL-READ] Reading message: {message_id}")
        
        # Get message details
        message = service.users().messages().get(
            userId='me',
            id=message_id,
            format='full'
        ).execute()
        
        headers = message['payload'].get('headers', [])
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
        from_addr = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
        to_addr = next((h['value'] for h in headers if h['name'].lower() == 'to'), '')
        date = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
        
        # Extract body (simplified)
        body = message.get('snippet', 'No body content available')
        
        result_text = f"""📧 Email Details:

Subject: {subject}
From: {from_addr}
To: {to_addr}
Date: {date}

Body:
{body}"""
        
        return {"success": True, "text": result_text}
        
    except Exception as e:
        logger.error(f"[GMAIL-READ] Failed to read email: {str(e)}")
        return {"error": f"Failed to read email: {str(e)}"}

@app.post("/tools/{tool_name}")
async def execute_tool(tool_name: str, request: ToolRequest):
    """Execute a Gmail tool"""
    start_time = datetime.now()
    
    try:
        logger.info(f"[HTTP-SERVER] Received request for {tool_name}")
        
        # Execute the tool
        result = await execute_gmail_tool(tool_name, request.parameters)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        if "error" in result:
            logger.error(f"[HTTP-SERVER] {tool_name} failed: {result['error']}")
            return ToolResponse(
                success=False,
                error=result["error"],
                execution_time=execution_time
            )
        else:
            logger.info(f"[HTTP-SERVER] {tool_name} completed in {execution_time:.2f}s")
            return ToolResponse(
                success=True,
                result=result,
                execution_time=execution_time
            )
            
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"[HTTP-SERVER] {tool_name} exception: {str(e)}")
        return ToolResponse(
            success=False,
            error=str(e),
            execution_time=execution_time
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/tools")
async def list_tools():
    """List available tools"""
    return {
        "tools": [
            "gmail_send",
            "find_email", 
            "search_emails",
            "read_email"
        ]
    }

if __name__ == "__main__":
    logger.info("Starting Gmail HTTP Tool Server on port 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")