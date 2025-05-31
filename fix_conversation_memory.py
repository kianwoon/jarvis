#!/usr/bin/env python3
"""
Fix for Jarvis conversation memory issue

The problem:
1. Frontend sends 'session_id' but backend expects 'conversation_id'
2. Conversation history is not being passed to individual agents
3. RAG system doesn't include conversation history in the prompt

This script generates the necessary fixes.
"""

print("Jarvis Conversation Memory Fix")
print("=" * 50)

# Fix 1: Update the RAG endpoint to accept both session_id and conversation_id
fix1 = """
# In app/api/v1/endpoints/langchain.py, update the RAGRequest model:

class RAGRequest(BaseModel):
    question: str
    thinking: bool = False
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None  # Add this for backward compatibility
    use_langgraph: bool = True

# In the rag_endpoint function, update to use session_id if conversation_id is not provided:
@router.post("/rag")
def rag_endpoint(request: RAGRequest):
    # Use session_id as conversation_id if conversation_id is not provided
    conversation_id = request.conversation_id or request.session_id
    
    def stream():
        for chunk in rag_answer(
            request.question, 
            thinking=request.thinking, 
            stream=True,
            conversation_id=conversation_id,
            use_langgraph=request.use_langgraph
        ):
            yield chunk
    return StreamingResponse(stream(), media_type="application/json")
"""

# Fix 2: Update service.py to build conversation history into prompts
fix2 = """
# In app/langchain/service.py, add a function to get conversation history:

def get_conversation_history(conversation_id: str) -> str:
    \"\"\"Get formatted conversation history for a given conversation ID\"\"\"
    if not conversation_id:
        return ""
    
    try:
        # Simple in-memory cache for conversation history
        # In production, this should use Redis or a database
        if not hasattr(get_conversation_history, 'cache'):
            get_conversation_history.cache = {}
        
        history = get_conversation_history.cache.get(conversation_id, [])
        if not history:
            return ""
        
        # Format last 5 exchanges
        formatted = []
        for msg in history[-10:]:  # Last 10 messages (5 exchanges)
            role = "User" if msg.get("role") == "user" else "Assistant"
            formatted.append(f"{role}: {msg.get('content', '')}")
        
        return "\\n".join(formatted)
    except Exception as e:
        print(f"[ERROR] Failed to get conversation history: {e}")
        return ""

def store_conversation_message(conversation_id: str, role: str, content: str):
    \"\"\"Store a message in conversation history\"\"\"
    if not conversation_id:
        return
    
    try:
        if not hasattr(get_conversation_history, 'cache'):
            get_conversation_history.cache = {}
        
        if conversation_id not in get_conversation_history.cache:
            get_conversation_history.cache[conversation_id] = []
        
        get_conversation_history.cache[conversation_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 20 messages
        if len(get_conversation_history.cache[conversation_id]) > 20:
            get_conversation_history.cache[conversation_id] = get_conversation_history.cache[conversation_id][-20:]
    except Exception as e:
        print(f"[ERROR] Failed to store conversation message: {e}")

# Update rag_answer function to include conversation history:
def rag_answer(question: str, thinking: bool = False, stream: bool = False, conversation_id: str = None, use_langgraph: bool = True):
    # ... existing code ...
    
    # Store the user's question
    if conversation_id:
        store_conversation_message(conversation_id, "user", question)
    
    # Get conversation history
    conversation_history = get_conversation_history(conversation_id) if conversation_id else ""
    
    # Include conversation history in prompts
    if conversation_history:
        # For RAG queries, include history in the prompt
        history_context = f"\\n\\nPrevious conversation:\\n{conversation_history}\\n\\n"
        # Prepend to the prompt
        prompt = history_context + prompt
"""

# Fix 3: Update multi-agent system to pass conversation history to agents
fix3 = """
# In app/langchain/multi_agent_system_simple.py, update execute_agent to include conversation history:

async def execute_agent(self, agent_name: str, agent_data: Dict, query: str, context: Dict = None) -> AsyncGenerator[Dict, None]:
    # ... existing code ...
    
    # Add conversation history to context
    if context and "conversation_history" in context:
        context_str = f"\\n\\nCONTEXT:\\n"
        context_str += f"Previous conversation:\\n"
        for msg in context.get("conversation_history", [])[-6:]:  # Last 6 messages
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:200]  # Truncate long messages
            context_str += f"{role}: {content}\\n"
        context_str += f"\\n\\nAdditional context:\\n{json.dumps(context, indent=2)}"
    
    full_prompt = f\"\"\"{system_prompt}

USER QUERY: {query}{context_str}

Please provide a comprehensive response based on your role and expertise.\"\"\"
"""

# Fix 4: Update frontend to send conversation_id instead of session_id
fix4 = """
# In llm-ui/src/App.tsx, update the fetch request:

const res = await fetch('/api/v1/langchain/rag', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: backendMessageText,
    conversation_id: sessionId,  // Change from session_id to conversation_id
    thinking: false,  // or whatever the thinking state is
    use_langgraph: false  // Disabled due to Redis issues
  })
});
"""

print("\nProposed fixes:")
print("\n1. RAG Endpoint Fix (backward compatible):")
print(fix1)

print("\n2. Conversation History in Service:")
print(fix2)

print("\n3. Multi-Agent History Passing:")
print(fix3)

print("\n4. Frontend Fix:")
print(fix4)

print("\n\nImplementation Steps:")
print("1. Apply Fix 1 to make the backend accept both session_id and conversation_id")
print("2. Apply Fix 2 to add conversation history tracking and inclusion in prompts")
print("3. Apply Fix 3 to ensure multi-agent system passes history to individual agents")
print("4. Apply Fix 4 to update frontend to use the correct field name")
print("\nThis will ensure Jarvis remembers conversation context across messages!")