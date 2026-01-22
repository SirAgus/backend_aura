
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db, Thread, Message, MCPServer
from dependencies import get_current_user
from pydantic import BaseModel, HttpUrl
import json
import os
import httpx
from typing import List, Optional, Any, Dict
from mcp_manager import mcp_manager

router = APIRouter()

# --- SHARED LLM CONFIG ---
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://178.156.214.187:8080/v1/chat/completions")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None

class MCPServerCreate(BaseModel):
    name: str
    type: str # 'stdio' or 'sse'
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    env: Optional[Dict[str, str]] = None

class MCPServerResponse(MCPServerCreate):
    id: int
    enabled: bool

    class Config:
        from_attributes = True

@router.post("/completion")
@router.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """
    OpenAI-compatible chat completion endpoint with MCP support.
    """
    # 1. Get tools from MCP servers
    mcp_tools = await mcp_manager.get_tools_from_all_servers(db)
    
    # Merge with tools from request if any
    available_tools = mcp_tools
    if getattr(request, 'tools', None):
        available_tools.extend([t.dict() for t in request.tools])

    messages = [m.model_dump() for m in request.messages]
    
    payload = {
        "messages": messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": request.stream
    }
    
    if available_tools:
        payload["tools"] = available_tools
        payload["tool_choice"] = "auto"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # First LLM Call
            response = await client.post(LLAMA_API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Check for tool calls
            message = data["choices"][0]["message"]
            if message.get("tool_calls"):
                # Handle Tool Execution Loop
                tool_calls = message["tool_calls"]
                messages.append(message) # Add assistant's tool call message
                
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    
                    # Execute tool via MCP
                    try:
                        print(f"🛠️ Executing MCP tool: {function_name}")
                        content = await mcp_manager.call_tool(db, function_name, arguments)
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": function_name,
                            "content": json.dumps(content) if not isinstance(content, str) else content
                        })
                    except Exception as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": function_name,
                            "content": f"Error: {str(e)}"
                        })
                
                # Second LLM Call with tool results
                payload["messages"] = messages
                # Disable stream for tool calls second pass to keep it simple for now
                payload["stream"] = False 
                response = await client.post(LLAMA_API_URL, json=payload)
                return response.json()
            
            # Normal response (no tool calls or tools not requested)
            if not request.stream:
                return data
            else:
                # Re-stream if it was requested
                async def stream_generator():
                    async with client.stream("POST", LLAMA_API_URL, json=payload) as resp:
                        async for line in resp.aiter_lines():
                            if line:
                                yield line + "\n"
                return StreamingResponse(stream_generator(), media_type="text/event-stream")
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LLM/MCP Error: {str(e)}")

@router.post("/response")
async def quick_response(
    prompt: str,
    system_prompt: Optional[str] = "Eres un asistente inteligente y servicial. Responde de forma clara y directa.",
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """
    Simplified response endpoint. Returns just the text.
    """
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(LLAMA_API_URL, json=payload)
            data = response.json()
            # Extract content from OpenAI format
            reply = data["choices"][0]["message"]["content"]
            return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

# --- MCP SERVER MANAGEMENT ---

@router.post("/mcp/servers", response_model=MCPServerResponse)
def add_mcp_server(
    server: MCPServerCreate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can manage MCP servers")
    
    new_server = MCPServer(
        name=server.name,
        type=server.type,
        command=server.command,
        args=json.dumps(server.args) if server.args else None,
        url=server.url,
        env=json.dumps(server.env) if server.env else None
    )
    db.add(new_server)
    db.commit()
    db.refresh(new_server)
    
    # Pre-parse for response
    new_server.args = json.loads(new_server.args) if new_server.args else None
    new_server.env = json.loads(new_server.env) if new_server.env else None
    
    return new_server

@router.get("/mcp/servers", response_model=List[MCPServerResponse])
def list_mcp_servers(
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    servers = db.query(MCPServer).all()
    for s in servers:
        s.args = json.loads(s.args) if s.args else None
        s.env = json.loads(s.env) if s.env else None
    return servers

@router.delete("/mcp/servers/{server_id}")
def delete_mcp_server(
    server_id: int,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can manage MCP servers")
    
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    db.delete(server)
    db.commit()
    return {"status": "deleted"}
