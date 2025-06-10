"""
OAuth Flow API endpoints for handling OAuth authorization directly in Jarvis
"""
from fastapi import APIRouter, HTTPException, Request, Query, Depends
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session
from typing import Optional
import secrets
import hashlib
import base64
import urllib.parse
import requests
import logging
from datetime import datetime

from app.core.db import SessionLocal, MCPServer

logger = logging.getLogger(__name__)
router = APIRouter()

# OAuth configuration
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Store state tokens temporarily (in production, use Redis)
oauth_states = {}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/gmail/authorize/{server_id}")
async def start_gmail_oauth(
    server_id: int,
    redirect_uri: str = Query(..., description="Redirect URI after authorization"),
    db: Session = Depends(get_db)
):
    """
    Start Gmail OAuth flow
    
    This endpoint initiates the OAuth flow by redirecting to Google's authorization page
    """
    # Get server configuration
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    if not server.oauth_credentials or not server.oauth_credentials.get("client_id"):
        raise HTTPException(
            status_code=400, 
            detail="OAuth client ID not configured. Please configure client credentials first."
        )
    
    client_id = server.oauth_credentials.get("client_id")
    
    # Generate state token for CSRF protection
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {
        "server_id": server_id,
        "redirect_uri": redirect_uri,
        "created_at": datetime.utcnow()
    }
    
    # Build authorization URL
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.send",
        "access_type": "offline",  # To get refresh token
        "prompt": "consent",  # Force consent to ensure refresh token
        "state": state
    }
    
    auth_url = f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"
    
    logger.info(f"Starting OAuth flow for server {server_id}, redirecting to Google")
    return RedirectResponse(url=auth_url)


@router.get("/gmail/callback")
async def gmail_oauth_callback(
    code: str = Query(..., description="Authorization code from Google"),
    state: str = Query(..., description="State token for CSRF protection"),
    error: Optional[str] = Query(None, description="Error from Google"),
    db: Session = Depends(get_db)
):
    """
    Handle OAuth callback from Google
    
    This endpoint receives the authorization code and exchanges it for tokens
    """
    # Handle errors from Google
    if error:
        logger.error(f"OAuth error from Google: {error}")
        return HTMLResponse(content=f"""
        <html>
            <body style="font-family: Arial, sans-serif; padding: 50px;">
                <h2>❌ Authorization Failed</h2>
                <p>Error: {error}</p>
                <p>Please close this window and try again.</p>
            </body>
        </html>
        """)
    
    # Verify state token
    if state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid state token")
    
    state_data = oauth_states.pop(state)
    server_id = state_data["server_id"]
    redirect_uri = state_data["redirect_uri"]
    
    # Get server configuration
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server or not server.oauth_credentials:
        raise HTTPException(status_code=404, detail="Server configuration not found")
    
    client_id = server.oauth_credentials.get("client_id")
    client_secret = server.oauth_credentials.get("client_secret")
    
    if not client_secret:
        raise HTTPException(
            status_code=400,
            detail="OAuth client secret not configured. Please configure both client ID and secret."
        )
    
    # Exchange authorization code for tokens
    try:
        token_response = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code"
            }
        )
        
        if not token_response.ok:
            logger.error(f"Token exchange failed: {token_response.text}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to exchange code for tokens: {token_response.text}"
            )
        
        tokens = token_response.json()
        
        # Update server with new tokens
        oauth_creds = server.oauth_credentials
        oauth_creds.update({
            "access_token": tokens.get("access_token"),
            "refresh_token": tokens.get("refresh_token"),
            "token_type": tokens.get("token_type", "Bearer"),
            "expires_in": tokens.get("expires_in"),
            "scope": tokens.get("scope"),
            "authorized_at": datetime.utcnow().isoformat()
        })
        
        server.oauth_credentials = oauth_creds
        server.updated_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Successfully updated OAuth tokens for server {server_id}")
        
        # Also update oauth_credentials table
        try:
            from sqlalchemy import text
            
            # Check if record exists
            check_sql = """
            SELECT id FROM oauth_credentials 
            WHERE mcp_server_id = :server_id AND service_name = 'gmail'
            """
            existing = db.execute(text(check_sql), {"server_id": server_id}).first()
            
            if existing:
                # Update existing record
                update_sql = """
                UPDATE oauth_credentials 
                SET access_token = :access_token,
                    refresh_token = :refresh_token,
                    token_expiry = NOW() + make_interval(secs => :expires_in),
                    updated_at = NOW()
                WHERE mcp_server_id = :server_id AND service_name = 'gmail'
                """
                db.execute(text(update_sql), {
                    "server_id": server_id,
                    "access_token": tokens.get("access_token"),
                    "refresh_token": tokens.get("refresh_token"),
                    "expires_in": tokens.get("expires_in", 3600)
                })
            else:
                # Insert new record  
                insert_sql = """
                INSERT INTO oauth_credentials 
                (mcp_server_id, service_name, client_id, client_secret, 
                 access_token, refresh_token, token_expiry, created_at, updated_at)
                VALUES 
                (:server_id, 'gmail', :client_id, :client_secret, 
                 :access_token, :refresh_token, NOW() + make_interval(secs => :expires_in), NOW(), NOW())
                """
                db.execute(text(insert_sql), {
                    "server_id": server_id,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "access_token": tokens.get("access_token"),
                    "refresh_token": tokens.get("refresh_token"),
                    "expires_in": tokens.get("expires_in", 3600)
                })
            
            db.commit()
            logger.info(f"Successfully updated oauth_credentials table for server {server_id}")
            
        except Exception as e:
            logger.error(f"Failed to update oauth_credentials table: {e}")
            # Don't fail the whole request if this fails
        
        # Return success page
        return HTMLResponse(content=f"""
        <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        padding: 50px;
                        background-color: #f0f0f0;
                    }}
                    .container {{
                        max-width: 600px;
                        margin: 0 auto;
                        background: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .success {{
                        color: #28a745;
                    }}
                    .info {{
                        background-color: #e7f3ff;
                        padding: 15px;
                        border-radius: 5px;
                        margin-top: 20px;
                    }}
                    button {{
                        background-color: #007bff;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 5px;
                        cursor: pointer;
                        margin-top: 20px;
                    }}
                    button:hover {{
                        background-color: #0056b3;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 class="success">✅ Gmail Authorization Successful!</h2>
                    <p>Your Gmail account has been successfully connected to Jarvis.</p>
                    
                    <div class="info">
                        <h4>What's next?</h4>
                        <ul>
                            <li>You can now use Gmail features in Jarvis</li>
                            <li>Try asking: "Get my latest Gmail emails"</li>
                            <li>Or: "Send an email to someone@example.com"</li>
                        </ul>
                    </div>
                    
                    <p><strong>Token Information:</strong></p>
                    <ul>
                        <li>Access Token: ✅ Received</li>
                        <li>Refresh Token: {"✅ Received" if tokens.get("refresh_token") else "❌ Not received (may need to revoke and re-authorize)"}</li>
                        <li>Expires in: {tokens.get("expires_in", "Unknown")} seconds</li>
                    </ul>
                    
                    <button onclick="window.close()">Close This Window</button>
                    
                    <script>
                        // Try to notify parent window if opened as popup
                        if (window.opener) {{
                            window.opener.postMessage({{
                                type: 'oauth_success',
                                server_id: {server_id}
                            }}, '*');
                        }}
                    </script>
                </div>
            </body>
        </html>
        """)
        
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gmail/revoke/{server_id}")
async def revoke_gmail_oauth(
    server_id: int,
    db: Session = Depends(get_db)
):
    """
    Revoke Gmail OAuth tokens
    
    This removes the stored tokens and optionally revokes them with Google
    """
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    if server.oauth_credentials and server.oauth_credentials.get("access_token"):
        # Optionally revoke token with Google
        token = server.oauth_credentials.get("access_token")
        try:
            revoke_response = requests.post(
                "https://oauth2.googleapis.com/revoke",
                params={"token": token}
            )
            if revoke_response.ok:
                logger.info(f"Successfully revoked token with Google for server {server_id}")
        except Exception as e:
            logger.warning(f"Failed to revoke token with Google: {e}")
    
    # Clear tokens from database
    if server.oauth_credentials:
        # Keep client ID and secret, only remove tokens
        oauth_creds = server.oauth_credentials
        oauth_creds.pop("access_token", None)
        oauth_creds.pop("refresh_token", None)
        oauth_creds.pop("expires_in", None)
        oauth_creds.pop("authorized_at", None)
        server.oauth_credentials = oauth_creds
        server.updated_at = datetime.utcnow()
        db.commit()
    
    # Clear from cache
    try:
        from app.core.oauth_token_manager import oauth_token_manager
        oauth_token_manager.invalidate_token(server_id, "gmail")
    except:
        pass
    
    return {"status": "success", "message": "OAuth tokens revoked"}


@router.get("/status/{server_id}")
async def get_oauth_status(
    server_id: int,
    db: Session = Depends(get_db)
):
    """
    Get OAuth status for a server
    """
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    oauth_creds = server.oauth_credentials or {}
    
    # Check if tokens exist and are valid
    has_tokens = bool(oauth_creds.get("access_token"))
    token_info = None
    
    if has_tokens:
        # Check token validity
        try:
            response = requests.get(
                f"https://oauth2.googleapis.com/tokeninfo",
                params={"access_token": oauth_creds.get("access_token")}
            )
            if response.ok:
                token_info = response.json()
        except:
            pass
    
    return {
        "server_id": server_id,
        "has_client_credentials": bool(oauth_creds.get("client_id") and oauth_creds.get("client_secret")),
        "has_tokens": has_tokens,
        "token_valid": token_info is not None,
        "token_info": token_info,
        "authorized_at": oauth_creds.get("authorized_at"),
        "scopes": token_info.get("scope", "").split() if token_info else []
    }