# Agentic AI work with Langgraph

Agentic AI applications built with LangGraph. This repository contains learning notebooks and a multi-server MCP agent that show how graphs, state, tools, and human review work together.
human 
## Definitions

- **Agentic AI** are AI agents that can work independently to perform a task. They decide what to do next, call tools when needed, and update shared state until the task is done.
- **LangGraph** is a framework for building agentic AI applications. You define a graph of nodes and edges. Nodes do work (call an LLM, run a tool, wait for a human). Edges decide the next step from the current state.
- **Graph API** is LangGraph's way to build that graph: a `State` schema, a `StateGraph`, nodes, edges (`START` / `END`), and a compiled runnable you can `invoke` or `stream`.
- **Agent state** is the shared data that flows through the graph. In these examples the main key is `messages`. The `add_messages` reducer appends new messages instead of overwriting the list.
- **Tool calling** lets the model request an external action (search, math, weather, human help). The graph routes those calls to a tools node and then back to the chatbot.
- **Human in the loop** pauses the graph with `interrupt` so a person can review or answer, then resumes with `Command`.
- **MCP (Model Context Protocol)** is how `client.py` talks to local tool servers (`math-server.py` over stdio, `weather.py` over HTTP) and wraps those tools for LangGraph.

## What is in this repository

| Path | Role |
| --- | --- |
| `1-BaiscChatbot/1-basicchatbot.ipynb` | Basic chatbot notebook using the LangGraph Graph API |
| `Human-assistant/humanintheloop.ipynb` | Human-in-the-loop notebook with tools, memory, and interrupt |
| `client.py` | MCP LangGraph agent that uses math and weather tools |
| `math-server.py` | MCP math tools server (stdio) |
| `weather.py` | MCP weather tools server (streamable HTTP on port 8081) |
| `requirements.txt` | Python dependencies |
| `pyporject.toml` | Project metadata |

## Setup

1. Use Python 3.12 (see `.python-version`).
2. Create a virtual environment and install dependencies:

   pip install -r requirements.txt

3. Create a `.env` file in the project root with:

   GROQ_API_KEY=your_groq_api_key

   For the human-in-the-loop notebook also set:

   TAVILY_API_KEY=your_tavily_api_key

4. Open the notebooks in Jupyter or VS Code, or run the MCP agent from the project root as shown below.

The notebooks and `client.py` load environment variables with `python-dotenv` and use Groq `llama-3.1-8b-instant` as the chat model.

---

## 1. Basic Chatbot (`1-BaiscChatbot/1-basicchatbot.ipynb`)

### Definition

This notebook is a minimal LangGraph chatbot. It does **not** call tools. It shows the Graph API pieces: typed state, a single LLM node, `START` / `END` edges, compile, visualize, and invoke.

It matches the original project note: a basic chatbot with LangGraph API work and agent state flow control.

### Workflow

```
START --> llmchatbot --> END
```

Step by step:

1. **Imports**  
   Load `StateGraph`, `START`, `END`, and `add_messages`.

2. **Define state**  
   `State` is a `TypedDict` with `messages: Annotated[list, add_messages]`.  
   The reducer appends each new message so conversation history is kept.

3. **Create the graph builder**  
   `graph_builder = StateGraph(State)`.

4. **Load the model**  
   `load_dotenv()` reads `.env`.  
   The notebook shows two equivalent ways to create the LLM:
   - `ChatGroq(model="llama-3.1-8b-instant")`
   - `init_chat_model("groq:llama-3.1-8b-instant")`

5. **Define the chatbot node**  
   `chatbot(state)` calls `llm.invoke(state["messages"])` and returns  
   `{"messages": [that reply]}`. LangGraph merges this into state with `add_messages`.

6. **Wire the graph**  
   - Add node `"llmchatbot"` bound to `chatbot`.  
   - Edge `START -> llmchatbot`.  
   - Edge `llmchatbot -> END`.  
   - `graph = graph_builder.compile()`.

7. **Visualize**  
   `graph.get_graph().draw_mermaid_png()` draws the linear flow (optional; needs graphviz extras).

8. **Run**  
   `graph.invoke({"messages": [("user", "Hi")]})`  
   State starts with a human message, the node produces an AI message, the graph ends.

### What you learn here

- How agent state is typed and updated.
- How a node reads state and returns a partial update.
- How `START` and `END` control a one-step conversation.

---

## 2. Human in the Loop (`Human-assistant/humanintheloop.ipynb`)

### Definition

This notebook is a tool-using agent that can stop and wait for a person. The model may:

- search the web with Tavily, or
- call `human_assistance`, which pauses the graph until a human resumes it.

A `MemorySaver` checkpointer stores state by `thread_id` so the same conversation can continue after the interrupt.

This is the second workflow that the original README listed as upcoming.

### Tools

- **`tavily_search(query)`** — web search via `langchain_tavily.TavilySearch` (2 results).
- **`human_assistance(query)`** — calls `interrupt({"query": query})` and later returns `human_response["data"]`.

The LLM is bound to both tools: `llm_with_tools = llm.bind_tools(tools)`.

### Workflow

```
START --> chatbot --(tools_condition)---> tools --> chatbot --> ...
                      |
                      +-- no tool calls --> END
```

When `human_assistance` runs, execution pauses inside the tools node until you resume.

Step by step:

1. **Load env and LLM**  
   Same Groq model as the basic notebook.

2. **Define state**  
   Again `messages` with `add_messages`.

3. **Register tools and the chatbot node**  
   `chatbot` invokes `llm_with_tools` on `state["messages"]`.  
   Parallel tool calling is avoided so an interrupt does not replay extra tool calls on resume.

4. **Add the tools node**  
   `ToolNode(tools=tools)` executes whatever tool the model requested.

5. **Connect edges**  
   - `START -> chatbot`  
   - Conditional edge from `chatbot` using `tools_condition` (tools vs end)  
   - `tools -> chatbot` so the model can use the tool result

6. **Compile with memory**  
   `graph = graph_builder.compile(checkpointer=memory)`  
   Checkpoints are required for interrupt and resume.

7. **Visualize**  
   Same mermaid PNG pattern as the basic notebook.

8. **First run (interrupt)**  
   Stream a user message that asks for expert help, with  
   `config = {"configurable": {"thread_id": "1"}}`.  
   The model should call `human_assistance`. The graph stops and prints the tool call.

9. **Resume (commented cells in the notebook)**  
   Build a human reply and send it back:

   human_response = "We, the experts are here to help! ..."
   human_command = Command(resume={"data": human_response})
   graph.stream(human_command, config, stream_mode="values")

   The `human_assistance` tool finishes, the chatbot node runs again, and the agent answers using the human text.

### What you learn here

- Binding tools and routing with `tools_condition` / `ToolNode`.
- Persisting a thread with `MemorySaver`.
- Pausing with `interrupt` and continuing with `Command(resume=...)`.

---

## 3. MCP math and weather agent (`client.py`, `math-server.py`, `weather.py`)

### Definition

A command-line agent that discovers tools from two MCP servers, wraps them as LangChain tools, and runs a LangGraph loop: chatbot → tools → chatbot until the model stops calling tools.

- **`math-server.py`** — FastMCP server named `Math-Server`. Tools: `add`, `subtract`, `multiply`, `divide`. Transport: **stdio**.
- **`weather.py`** — FastMCP server named `Weather-Server` on port **8081**. Tool: `get_weather(location)` (demo string: sunny). Transport: **streamable-http**.
- **`client.py`** — starts weather in the background, connects to both servers, wraps every MCP tool, builds the graph, then runs demo queries or an interactive chat.

`GROQ_API_KEY` must be set or `client.py` exits.

### Workflow

```
1. Start weather.py (subprocess, streamable-http, wait until :8081/mcp responds)
2. Connect math-server.py over stdio and initialize the MCP session
3. Connect weather over http://localhost:8081/mcp and initialize
4. list_tools on both sessions and wrap each tool as a LangChain StructuredTool
5. Bind tools to ChatGroq
6. Compile graph and run queries

Graph:

START --> chatbot --(has tool_calls?)--> tools --> chatbot --> ...
                      |
                      +-- no --> END
```

Step by step inside a turn:

1. **chatbot node** prepends a system message (use math and weather tools; trust tool results) and calls the tool-bound LLM.
2. **should_continue** looks at the last message. If there are `tool_calls`, go to `tools`; otherwise `END`.
3. **tools node** runs each requested MCP tool (`add`, `multiply`, `get_weather`, …) and appends `ToolMessage` results.
4. Control returns to **chatbot**, which answers the user from those results.

### How to run

From the project root, after `.env` is set:

- Interactive chat:

  python client.py

  Type a question, or `exit` / `quit` to stop.

- Fixed demo queries (weather + arithmetic):

  python client.py --demo

On Windows the client sets `WindowsSelectorEventLoopPolicy` before `asyncio.run`. On shutdown it terminates the weather subprocess.

Example questions the agent can handle:

- What is the weather in Paris, and what is 534 multiplied by 82?
- Subtract 99 from 450 and tell me the weather in Seattle.

---

## Suggested order

1. Run `1-BaiscChatbot/1-basicchatbot.ipynb` to see state, one node, and invoke.
2. Run `Human-assistant/humanintheloop.ipynb` to see tools, checkpoints, and interrupt/resume.
3. Run `client.py` to see the same graph pattern with live MCP servers.

## Requirements

See `requirements.txt`:

- langgraph
- langchain
- langsmith
- python-dotenv
- langchain-groq
- langchain_tavily
- mcp

The human-in-the-loop notebook also imports `langchain_google_genai` (optional alternative model) and `langchain_tavily` (required for search).