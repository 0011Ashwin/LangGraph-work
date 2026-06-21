import os
import sys
import asyncio
import subprocess
import time
from typing import Annotated, Sequence, TypedDict, Any
from dotenv import load_dotenv

# Third-party libraries
import httpx
from pydantic import create_model, Field

# MCP libraries
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

# LangChain & LangGraph libraries
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load env variables
load_dotenv()

# Ensure API key is present
if not os.environ.get("GROQ_API_KEY"):
    print("[Error] GROQ_API_KEY environment variable is not set. Please add it to your .env file.")
    sys.exit(1)

# Definition of the graph state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def map_json_type(json_type: str) -> Any:
    """Helper to map JSON schema types to Python types for Pydantic."""
    if json_type == "integer":
        return int
    elif json_type == "number":
        return float
    elif json_type == "boolean":
        return bool
    elif json_type == "string":
        return str
    elif json_type == "array":
        return list
    elif json_type == "object":
        return dict
    return Any

def create_langchain_tool(session: ClientSession, mcp_tool: Any) -> StructuredTool:
    """Dynamically wraps an MCP tool into a LangChain StructuredTool."""
    name = mcp_tool.name
    description = mcp_tool.description
    schema = mcp_tool.inputSchema or {}

    # Build Pydantic model for validation
    fields = {}
    required_fields = schema.get("required", [])
    for param_name, param_info in schema.get("properties", {}).items():
        param_type = map_json_type(param_info.get("type"))
        param_desc = param_info.get("description", "")
        # If required, use Ellipsis (...) to indicate no default value
        default = ... if param_name in required_fields else None
        fields[param_name] = (param_type, Field(default=default, description=param_desc))

    args_schema = create_model(f"{name}_input", **fields)

    async def coroutine(**kwargs) -> str:
        # Invoke the tool on the MCP session
        result = await session.call_tool(name, arguments=kwargs)
        # Extract text content from result
        texts = []
        for content_block in result.content:
            if hasattr(content_block, "text"):
                texts.append(content_block.text)
            elif isinstance(content_block, dict) and "text" in content_block:
                texts.append(content_block["text"])
        return "\n".join(texts)

    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description=description,
        args_schema=args_schema
    )

async def run_agent():
    # 1. Start the HTTP weather server in the background
    weather_script = os.path.abspath("weather.py")
    print(f"[*] Starting weather.py server in background (transport=streamable-http)...")
    
    # We use sys.executable to ensure we run with the correct Python environment
    weather_process = subprocess.Popen(
        [sys.executable, weather_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    print("[*] Waiting for weather server to become ready on port 8081...")
    server_ready = False
    for i in range(15):
        try:
            # Pinging the streamable http mcp endpoint
            async with httpx.AsyncClient() as client:
                r = await client.get("http://127.0.0.1:8081/mcp", timeout=1.0)
                # Any response (even error status codes like 406) means the server is up and listening!
                server_ready = True
                break
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pass
        except Exception:
            # Any other exception means the server is probably running but returned something else
            server_ready = True
            break
        await asyncio.sleep(0.5)

    if not server_ready:
        print("[Error] Weather server did not start. Terminating process...")
        weather_process.terminate()
        return

    print("[+] Weather server is ready!")

    try:
        # 2. Connect to the math server using stdio transport
        math_script = os.path.abspath("math-server.py")
        print(f"[*] Connecting to math-server.py (transport=stdio)...")
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[math_script]
        )
        
        async with stdio_client(server_params) as (math_read, math_write):
            async with ClientSession(math_read, math_write) as math_session:
                await math_session.initialize()
                print("[+] Connected to Math-Server!")

                # 3. Connect to the weather server using streamable-http transport
                print(f"[*] Connecting to Weather-Server via streamable-http...")
                async with streamable_http_client("http://localhost:8081/mcp") as (weather_read, weather_write, _):
                    async with ClientSession(weather_read, weather_write) as weather_session:
                        await weather_session.initialize()
                        print("[+] Connected to Weather-Server!")

                        # 4. Discover and wrap tools
                        mcp_tools = []
                        
                        # Math tools
                        math_tools_response = await math_session.list_tools()
                        for t in math_tools_response.tools:
                            wrapped = create_langchain_tool(math_session, t)
                            mcp_tools.append(wrapped)
                            print(f"    - Wrapped Math tool: {t.name}")
                            
                        # Weather tools
                        weather_tools_response = await weather_session.list_tools()
                        for t in weather_tools_response.tools:
                            wrapped = create_langchain_tool(weather_session, t)
                            mcp_tools.append(wrapped)
                            print(f"    - Wrapped Weather tool: {t.name}")

                        # 5. Initialize the LLM and bind wrapped tools
                        llm = ChatGroq(model="llama-3.1-8b-instant")
                        llm_with_tools = llm.bind_tools(mcp_tools)

                        # Define LangGraph graph nodes
                        async def chatbot_node(state: AgentState):
                            system_msg = SystemMessage(
                                content="You are a helpful assistant with access to math and weather tools. "
                                        "Always trust the tool execution results and report them directly to the user."
                            )
                            message = await llm_with_tools.ainvoke([system_msg] + state["messages"])
                            return {"messages": [message]}

                        # Custom tool executor node
                        tools_by_name = {tool.name: tool for tool in mcp_tools}
                        async def tools_node(state: AgentState):
                            last_message = state["messages"][-1]
                            tool_messages = []
                            for tool_call in last_message.tool_calls:
                                tool_name = tool_call["name"]
                                tool_args = tool_call["args"]
                                print(f"\n[Tool Execution] Calling '{tool_name}' with args: {tool_args}")
                                tool_obj = tools_by_name[tool_name]
                                result = await tool_obj.ainvoke(tool_args)
                                print(f"[Tool Execution] Result: {result}")
                                tool_messages.append(
                                    ToolMessage(
                                        content=str(result),
                                        name=tool_name,
                                        tool_call_id=tool_call["id"]
                                    )
                                )
                            return {"messages": tool_messages}

                        # Conditional routing
                        def should_continue(state: AgentState):
                            last_message = state["messages"][-1]
                            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                                return "tools"
                            return END

                        # Build the graph
                        builder = StateGraph(AgentState)
                        builder.add_node("chatbot", chatbot_node)
                        builder.add_node("tools", tools_node)

                        builder.add_edge(START, "chatbot")
                        builder.add_conditional_edges(
                            "chatbot",
                            should_continue,
                            {
                                "tools": "tools",
                                END: END
                            }
                        )
                        builder.add_edge("tools", "chatbot")

                        graph = builder.compile()
                        print("\n[+] Agentic workflow graph built successfully!")

                        # 6. Run demo or interactive mode
                        if len(sys.argv) > 1 and sys.argv[1] == "--demo":
                            demo_queries = [
                                "What is the weather in Paris, and what is 534 multiplied by 82?",
                                "Subtract 99 from 450 and tell me the weather in Seattle."
                            ]
                            for query in demo_queries:
                                print(f"\n{'='*60}\n[Demo Query] {query}\n{'='*60}")
                                state = {"messages": [HumanMessage(content=query)]}
                                async for event in graph.astream(state):
                                    for node, output in event.items():
                                        if "messages" in output:
                                            last_msg = output["messages"][-1]
                                            if isinstance(last_msg, AIMessage) and last_msg.content:
                                                print(f"\n[Agent Response]:\n{last_msg.content}")
                        else:
                            print("\n=== Interactive MCP Chatbot (Type 'exit' to quit) ===")
                            while True:
                                try:
                                    user_input = input("\nYou: ")
                                    if user_input.strip().lower() in ("exit", "quit"):
                                        break
                                    if not user_input.strip():
                                        continue
                                    
                                    state = {"messages": [HumanMessage(content=user_input)]}
                                    async for event in graph.astream(state):
                                        for node, output in event.items():
                                            if "messages" in output:
                                                last_msg = output["messages"][-1]
                                                if isinstance(last_msg, AIMessage) and last_msg.content:
                                                    print(f"\nAgent: {last_msg.content}")
                                except (KeyboardInterrupt, EOFError):
                                    break

    finally:
        print("\n[*] Shutting down weather.py server process...")
        weather_process.terminate()
        try:
            weather_process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            weather_process.kill()
        print("[+] Gracefully shut down.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_agent())
