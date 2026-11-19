# A real MCP server in eight lines (Example 27-21 of Section 27.7.1; the client cell of
# 00_chapter27_walkthrough.py, Example 27-22, launches this file as a subprocess and
# talks to it over stdio, the standard local transport).
from mcp.server import MCPServer                     # pip install mcp

server = MCPServer(name="weather")                   # a real MCP server

@server.tool()                                       # the decorator reads the type hints into a schema
def get_weather(city: str) -> dict:
    """Current weather for a city."""
    return {"city": city, "tempC": 18, "sky": "clear"}

if __name__ == "__main__":
    server.run()                                     # serve over stdio
