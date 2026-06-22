from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math-Server")

@mcp.tool()
def add(a:int, b:int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def subtract(a:int, b:int) -> int:
    """Subtract two numbers"""
    return a - b

@mcp.tool()
def multiply(a:int, b:int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def divide(a:int, b:int) -> int:
    """Divide two numbers"""
    return a / b

# The transport="stdio" argument tells the server to:
# Use standard input/output (stdio and standout) to receive and respond to tool function calls 
#Run MCP server on port 3001
if __name__ == "__main__":
    mcp.run(transport="stdio") # 
