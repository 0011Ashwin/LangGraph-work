from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather-Server", port=8081)

@mcp.tool()
async def get_weather(location:str) -> str:
    """Get the weather for a city"""
    return f"The weather in {location} is sunny."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    