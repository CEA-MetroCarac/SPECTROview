import re

with open("docs/developer/ai_agent.md", "r") as f:
    text = f.read()

# Update Architecture Diagram
old_diagram = """    VMC -->|"delegates"| LLC["LLMClient (Model)"]
    VMC -->|"builds prompt"| PM["PromptManager"]
    PM -->|"loads .md"| FS["File System (prompts/, rules/, ...)"]
    LLC -->|"spawns"| LLW["LLMWorker (QThread)"]
    LLW -->|"HTTP stream"| OLL["Ollama (local)"]
    OLL -->|"token chunks"| LLW
    LLW -->|"signals"| LLC
    LLC -->|"signals"| VMC
    VMC -->|"result_ready"| VCP
    VCP -->|"plot_requested"| Main"""

new_diagram = """    VMC -->|"delegates"| LLC["LLMClient (Model)"]
    VMC -->|"builds prompt"| PM["PromptManager"]
    VMC -->|"starts"| MCP["FastMCP Server (mcp/server.py)"]
    PM -->|"loads .md"| FS["File System (prompts/, rules/, ...)"]
    LLC -->|"spawns (with MCP tools)"| LLW["LLMWorker (QThread)"]
    LLW -->|"HTTP stream"| OLL["Ollama/OpenAI (local/cloud)"]
    OLL -->|"tool calls / text chunks"| LLW
    LLW -->|"signals"| LLC
    LLC -->|"signals"| VMC
    VMC -->|"result_ready (ChatResult)"| VCP
    VCP -->|"plot_requested"| Main"""
text = text.replace(old_diagram, new_diagram)

# Update Module Structure
old_structure = """├── m_prompt_manager.py      # Prompt caching and assembly: loads and merges Markdown files based on intent
├── vm_chat.py               # ViewModel: delegates to PromptManager, history, response parsing"""

new_structure = """├── mcp/
│   └── server.py            # FastMCP Server defining the AI tools (plot_graph, query_dataframe, etc.)
├── m_prompt_manager.py      # Prompt caching and assembly: loads and merges Markdown files based on intent
├── vm_chat.py               # ViewModel: delegates to PromptManager, history, response parsing"""
text = text.replace(old_structure, new_structure)

# Update File Roles
old_file_roles = """| `m_conversation_store.py` | **Model** | Scans the history folder, lists saved conversations as lightweight summaries, loads conversations by ID. |
| `vm_chat.py` | **ViewModel** | Builds the system prompt using `PromptManager`. Manages conversation history. Parses the LLM's structured JSON response into a `ChatResult` object. Executes safe pandas operations. |"""
new_file_roles = """| `m_conversation_store.py` | **Model** | Scans the history folder, lists saved conversations as lightweight summaries, loads conversations by ID. |
| `mcp/server.py` | **Model** | FastMCP server that exposes internal SPECTROview resources (DataFrames, graphs) and tools (plotting, querying, statistics) to the LLM. |
| `vm_chat.py` | **ViewModel** | Builds the system prompt using `PromptManager`. Manages conversation history. Parses the LLM's tool calls into `ChatResult` objects. |"""
text = text.replace(old_file_roles, new_file_roles)

# Update System Prompt & LLM Contract
old_llm_contract_start = """The LLM is instructed to respond with **only a valid JSON object** (no markdown fences, no explanatory text) conforming to this schema:

```json"""
old_llm_contract_end = """  "answer_text": "plain text answer or null"
}
```"""
old_llm_contract_full = text[text.find(old_llm_contract_start):text.find(old_llm_contract_end) + len(old_llm_contract_end)]

new_llm_contract = """The AI agent uses **Tool Calling (Function Calling)** via the **Model Context Protocol (MCP)**.
Instead of relying on fragile JSON parsing, the LLM is provided with a strict schema of tools defined in `mcp/server.py` using `FastMCP`.

These tools include:
- `query_dataframe(query, df_name)`
- `get_statistics(columns, df_name)`
- `plot_graph(x, y, plot_style, z, filters, other_properties)`
- `update_graph(graph_id, properties)`
- `delete_graph(delete_all, graph_ids)`

The LLM decides which tool to call and passes the strongly-typed arguments. The `LLMClient` intercepts these tool calls and translates them into `ChatResult` objects."""
text = text.replace(old_llm_contract_full, new_llm_contract)

# Update Response Parsing & Error Handling
old_response_parsing = """`VMChat._parse_response()` handles several edge cases in the LLM output:

1. **Markdown fences**: Strips `` ```json `` / `` ``` `` that the model sometimes adds.
2. **JSON extraction**: If the top-level text isn't valid JSON, a regex searches for the first `{...}` block.
3. **Fallback**: If no valid JSON can be found, the raw text is returned as an `action: "answer"` so the user always sees something useful.
4. **Invalid DataFrame target**: Falls back to the active DataFrame, then to the first loaded DataFrame."""
new_response_parsing = """With the migration to **MCP Tool Calling**, response parsing is significantly more robust:

1. **Tool Execution**: When the LLM decides to perform an action, it emits a structured tool call (e.g., `plot_graph`).
2. **Mapping to ChatResult**: `vm_chat.py` extracts these tool calls from the message and wraps them in a `ChatResult` object.
3. **Fallback**: If the LLM just replies with text instead of a tool call, the text is returned as an `action: "answer"` so the user always sees something useful.
4. **Invalid DataFrame target**: The MCP tools handle fallback logic, resolving empty `df_name` arguments to the currently active DataFrame."""
text = text.replace(old_response_parsing, new_response_parsing)

with open("docs/developer/ai_agent.md", "w") as f:
    f.write(text)

