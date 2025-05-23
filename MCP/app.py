from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os, json, requests
from dotenv import load_dotenv
from datetime import datetime
import logging

app = FastAPI()

# Allow cross-origin requests from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend host
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Pydantic models for request/response validation
class ToolInvocation(BaseModel):
    name: str = Field(..., description="Name of the MCP tool to invoke")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

class ToolResponse(BaseModel):
    result: Any = Field(..., description="Result of the tool invocation")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

# Tool-specific argument models
class JiraListIssuesArgs(BaseModel):
    jql: str = Field(..., description="Jira JQL query")

class JiraCreateIssueArgs(BaseModel):
    project_key: str = Field(..., description="Jira project key")
    summary: str = Field(..., description="Issue summary")
    issue_type: str = Field(..., description="Issue type")
    description: Optional[str] = Field(None, description="Issue description")

class OutlookEventArgs(BaseModel):
    start_datetime: str = Field(..., description="Start datetime in ISO format")
    end_datetime: str = Field(..., description="End datetime in ISO format")
    subject: Optional[str] = Field(None, description="Event subject")
    attendees: Optional[List[str]] = Field(None, description="List of attendee emails")

class OutlookMessageArgs(BaseModel):
    folder: str = Field(..., description="Mail folder name")
    top: Optional[int] = Field(5, description="Number of messages to retrieve")

class OutlookSendMessageArgs(BaseModel):
    to: List[str] = Field(..., description="List of recipient emails")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")

# Initialize FastAPI app
app = FastAPI(
    title="MCP Server",
    description="MCP (Microservice Control Panel) Server for tool execution",
    version="1.0.0"
)

# CORS Configuration
allowed_origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
origins = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]

# Add default origins for development if none specified
if not origins:
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Allow all needed methods
    allow_headers=["*"],
)

# Load and validate manifest
def load_manifest():
    try:
        with open("manifest.json") as f:
            manifest = json.load(f)
            # Validate manifest structure
            if "tools" not in manifest:
                raise ValueError("Manifest must contain 'tools' array")
            return manifest
    except FileNotFoundError:
        logger.error("manifest.json not found")
        raise HTTPException(status_code=500, detail="Manifest file not found")
    except json.JSONDecodeError:
        logger.error("Invalid JSON in manifest.json")
        raise HTTPException(status_code=500, detail="Invalid manifest file")

manifest = load_manifest()

# Tool registry
tool_handlers = {
    "jira_list_issues": lambda args: handle_jira_list_issues(JiraListIssuesArgs(**args)),
    "jira_create_issue": lambda args: handle_jira_create_issue(JiraCreateIssueArgs(**args)),
    "outlook_list_events": lambda args: handle_outlook_list_events(OutlookEventArgs(**args)),
    "outlook_create_event": lambda args: handle_outlook_create_event(OutlookEventArgs(**args)),
    "outlook_list_messages": lambda args: handle_outlook_list_messages(OutlookMessageArgs(**args)),
    "outlook_send_message": lambda args: handle_outlook_send_message(OutlookSendMessageArgs(**args)),
    "get_datetime": lambda _: handle_get_datetime()
}

# Tool handler functions
def handle_jira_list_issues(args: JiraListIssuesArgs) -> Dict[str, Any]:
    try:
        url = f"{os.getenv('JIRA_URL')}/rest/api/2/search"
        auth = (os.getenv("JIRA_USER"), os.getenv("JIRA_TOKEN"))
        resp = requests.get(url, params={"jql": args.jql}, auth=auth)
        resp.raise_for_status()
        return {"result": resp.json()}
    except requests.RequestException as e:
        logger.error(f"Jira API error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def handle_jira_create_issue(args: JiraCreateIssueArgs) -> Dict[str, Any]:
    try:
        url = f"{os.getenv('JIRA_URL')}/rest/api/2/issue"
        auth = (os.getenv("JIRA_USER"), os.getenv("JIRA_TOKEN"))
        payload = {
            "fields": {
                "project": {"key": args.project_key},
                "summary": args.summary,
                "description": args.description or "",
                "issuetype": {"name": args.issue_type}
            }
        }
        resp = requests.post(url, json=payload, auth=auth)
        resp.raise_for_status()
        return {"result": resp.json()}
    except requests.RequestException as e:
        logger.error(f"Jira API error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def handle_outlook_list_events(args: OutlookEventArgs) -> Dict[str, Any]:
    try:
        graph_url = "https://graph.microsoft.com/v1.0/me"
        token = os.getenv("MS_GRAPH_TOKEN")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"{graph_url}/calendarview?startDateTime={args.start_datetime}&endDateTime={args.end_datetime}"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return {"result": resp.json()}
    except requests.RequestException as e:
        logger.error(f"Outlook API error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def handle_outlook_create_event(args: OutlookEventArgs) -> Dict[str, Any]:
    try:
        graph_url = "https://graph.microsoft.com/v1.0/me"
        token = os.getenv("MS_GRAPH_TOKEN")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"{graph_url}/events"
        body = {
            "subject": args.subject,
            "start": {"dateTime": args.start_datetime, "timeZone": "UTC"},
            "end": {"dateTime": args.end_datetime, "timeZone": "UTC"},
            "attendees": [{"emailAddress": {"address": e}, "type": "required"} for e in (args.attendees or [])]
        }
        resp = requests.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return {"result": resp.json()}
    except requests.RequestException as e:
        logger.error(f"Outlook API error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def handle_outlook_list_messages(args: OutlookMessageArgs) -> Dict[str, Any]:
    try:
        graph_url = "https://graph.microsoft.com/v1.0/me"
        token = os.getenv("MS_GRAPH_TOKEN")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"{graph_url}/mailFolders/{args.folder}/messages?$top={args.top}"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return {"result": resp.json()}
    except requests.RequestException as e:
        logger.error(f"Outlook API error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def handle_outlook_send_message(args: OutlookSendMessageArgs) -> Dict[str, Any]:
    try:
        graph_url = "https://graph.microsoft.com/v1.0/me"
        token = os.getenv("MS_GRAPH_TOKEN")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"{graph_url}/sendMail"
        mail = {
            "message": {
                "subject": args.subject,
                "body": {"contentType": "Text", "content": args.body},
                "toRecipients": [{"emailAddress": {"address": r}} for r in args.to]
            }
        }
        resp = requests.post(url, headers=headers, json=mail)
        resp.raise_for_status()
        return {"result": {"status": "sent"}}
    except requests.RequestException as e:
        logger.error(f"Outlook API error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def handle_get_datetime() -> Dict[str, str]:
    return {"datetime": datetime.now().isoformat()}

# API Endpoints
@app.get("/manifest", response_model=Dict[str, Any])
async def get_manifest():
    """Return the MCP manifest."""
    return manifest

@app.get("/datetime", response_model=Dict[str, str])
async def get_datetime():
    """Get the current datetime in ISO format."""
    return handle_get_datetime()

@app.post("/invoke", response_model=ToolResponse, responses={400: {"model": ErrorResponse}})
async def invoke(inv: ToolInvocation):
    """Invoke an MCP tool with the given arguments."""
    try:
        logger.info(f"Invoking tool: {inv.name} with args: {inv.arguments}")
        
        if inv.name not in tool_handlers:
            raise HTTPException(status_code=400, detail=f"Unknown tool '{inv.name}'")
        
        result = tool_handlers[inv.name](inv.arguments)
        logger.info(f"Tool {inv.name} executed successfully")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing tool {inv.name}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
