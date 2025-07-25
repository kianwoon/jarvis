"""
Simple datetime fallback for when MCP servers are unavailable
"""
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def get_current_datetime():
    """
    Simple fallback function to get current date and time
    Returns current datetime in a user-friendly format
    """
    try:
        current_time = datetime.now()
        
        # Format: "Wednesday, June 28, 2025 at 11:50 PM (UTC+8)"
        formatted_date = current_time.strftime("%A, %B %d, %Y at %I:%M %p")
        timezone_info = current_time.strftime("%Z") or "Local Time"
        
        result = {
            "current_datetime": f"{formatted_date} ({timezone_info})",
            "iso_format": current_time.isoformat(),
            "unix_timestamp": int(current_time.timestamp()),
            "date": current_time.strftime("%Y-%m-%d"),
            "time": current_time.strftime("%H:%M:%S"),
            "timezone": timezone_info,
            "day_of_week": current_time.strftime("%A"),
            "month": current_time.strftime("%B"),
            "year": current_time.year
        }
        
        logger.info(f"[FALLBACK] Datetime fallback returned: {result['current_datetime']}")
        return result
        
    except Exception as e:
        logger.error(f"[FALLBACK] Datetime fallback failed: {e}")
        # Even if everything fails, return something basic
        return {
            "current_datetime": "Unable to determine current time",
            "error": str(e)
        }