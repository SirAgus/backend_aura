
import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from sqlalchemy.orm import Session
from database import MCPServer

class MCPManager:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self._exit_stack = None

    async def get_tools_from_all_servers(self, db: Session) -> List[Dict[str, Any]]:
        servers = db.query(MCPServer).filter(MCPServer.enabled == True).all()
        all_tools = []
        
        for server in servers:
            try:
                # This is a simplified version. In a real app, you'd want to cache sessions
                # and handle reconnections.
                tools = await self._get_server_tools(server)
                for tool in tools:
                    # Prefix tool name to avoid collisions
                    tool_name = f"{server.name}__{tool.name}"
                    all_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })
            except Exception as e:
                print(f"Error fetching tools from MCP server {server.name}: {e}")
                
        return all_tools

    async def _get_server_tools(self, server: MCPServer) -> List[Any]:
        if server.type == "stdio":
            args = json.loads(server.args) if server.args else []
            env = json.loads(server.env) if server.env else None
            server_params = StdioServerParameters(
                command=server.command,
                args=args,
                env={**os.environ, **(env or {})}
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return result.tools
        elif server.type == "sse":
            async with sse_client(server.url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return result.tools
        return []

    async def call_tool(self, db: Session, full_tool_name: str, arguments: Dict[str, Any]) -> Any:
        if "__" not in full_tool_name:
            raise ValueError("Invalid tool name format")
            
        server_name, tool_name = full_tool_name.split("__", 1)
        server = db.query(MCPServer).filter(MCPServer.name == server_name).first()
        
        if not server:
            raise ValueError(f"Server {server_name} not found")

        if server.type == "stdio":
            args = json.loads(server.args) if server.args else []
            env = json.loads(server.env) if server.env else None
            server_params = StdioServerParameters(
                command=server.command,
                args=args,
                env={**os.environ, **(env or {})}
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    return result.content
        elif server.type == "sse":
            async with sse_client(server.url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    return result.content
        
        return "Unsupported server type"

mcp_manager = MCPManager()
