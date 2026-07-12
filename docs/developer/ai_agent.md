# **AI Data Chat**

The `AI Data Chat` is an optional, multi-provider AI chatbot that lets users query, filter, plot, and modify their data using natural language. It supports local inference via **Ollama**, cloud providers via the **OpenAI** SDK (OpenAI, DeepSeek, Gemini, and any OpenAI-compatible endpoint), and **Anthropic** Claude models. The agent communicates with the `Graphs` workspace to create, update, and delete plots in real time.

> The AI module is **fully optional**. If no AI package is installed or no provider is configured, the rest of SPECTROview is completely unaffected.

---

## **Prerequisites for Users**

### **Option 1: macOS**

1. **Install Ollama**
   Using Homebrew (recommended):
   ```bash
   brew install ollama
   ```
   *(Alternatively, download the macOS application directly from [Ollama's official website](https://ollama.com/download/mac).)*

2. **Start the Ollama Service**
   If you used Homebrew, start Ollama as a background service:
   ```bash
   brew services start ollama
   ```
   *(If you installed the Mac app, simply open the Ollama application from your Applications folder. You should see its icon in your menu bar.)*

### **Option 2: Windows**

1. **Install Ollama**
   Download the Windows installer from [Ollama's official website](https://ollama.com/download/windows) and run it.

2. **Start the Ollama Service**
   Ollama usually starts automatically after installation. If it doesn't, search for "Ollama" in the Start menu and open it. You should see the Ollama icon in your system tray (bottom right corner).

---

### **Common Steps (Both platforms)**

3. **Download the AI Model**
   SPECTROview uses `qwen2.5-coder:7b` by default. Open your terminal (or Command Prompt / PowerShell on Windows) and pull it:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```

4. **Install the Python Dependency**
   The AI features require the `ollama` Python package to communicate with the local service. From your SPECTROview project directory, run:
   ```bash
   pip install ollama
   # or install using the optional dependencies:
   pip install -e ".[ai]"
   ```

5. **Run SPECTROview**
   Start the application:
   ```bash
   python -m spectroview.main
   ```
   Click on the **AI Data Chat** button in the top toolbar. The chat panel will open, and the status bar should say **🟢 Ollama connected**.

---

## **Architecture Overview**

The AI feature follows the same strict **MVVM** pattern as the rest of the application. All AI-related code lives in the `spectroview/ai_agent/` package, isolated from the core workspaces.

```mermaid
graph TD
    TB["VMenuBar (toolbar button)"] -->|"ai_chat_requested"| Main["main.py"]
    Main -->|"lazy creation"| VCP["VChatPanel (View)"]
    VCP -->|"method call"| VMC["VMChat (ViewModel)"]
    VMC -->|"delegates"| LLC["LLMClient (Model)"]
    VMC -->|"builds prompt"| PM["PromptManager"]
    PM -->|"loads .md"| FS["File System (prompts/, rules/, ...)"]
    LLC -->|"spawns"| LLW["LLMWorker (QThread)"]
    LLW -->|"HTTP stream"| OLL["Ollama (local)"]
    OLL -->|"token chunks"| LLW
    LLW -->|"signals"| LLC
    LLC -->|"signals"| VMC
    VMC -->|"result_ready"| VCP
    VCP -->|"plot_requested"| Main
    Main -->|"create / update / delete"| VWG["VWorkspaceGraphs"]
```

---

## **Module Structure**

```
spectroview/ai_agent/
├── __init__.py              # Docstring only — keeps the module optional
├── config/                  # YAML configuration files (model.yaml, settings.yaml)
├── examples/                # Few-shot conversation examples (Markdown)
├── knowledge/               # Static domain facts (Markdown)
├── prompts/                 # Core identity, JSON schema, per-intent instructions (Markdown)
├── rules/                   # Behavioural constraints (Markdown)
├── templates/               # Reusable Markdown output formats (Markdown)
├── m_llm_client.py          # Model layer: LLM connections + QThread workers
│                            #   - LLMWorker (Ollama)
│                            #   - APIWorker (OpenAI-compatible)
│                            #   - AnthropicWorker (Anthropic SDK)
│                            #   - LLMClient (unified facade)
├── m_conversation.py        # Data model: single conversation (messages, save/load JSON)
├── m_conversation_store.py  # Conversation store: scan/list/load saved conversations
├── m_prompt_manager.py      # Prompt caching and assembly: loads and merges Markdown files based on intent
├── vm_chat.py               # ViewModel: delegates to PromptManager, history, response parsing
├── v_chat_panel.py          # View: floating chat dialog (QDialog)
└── v_history_dialog.py      # View: history browser dialog (QDialog)
```

### **File Roles**

| File | Layer | Responsibility |
|------|-------|---------------|
| `m_prompt_manager.py` | **Model** | Loads, caches, and assembles Markdown files (`prompts/`, `rules/`, etc.) into a cohesive system prompt. Handles intent detection. |
| `m_llm_client.py` | **Model** | Wraps Ollama, OpenAI SDK, and Anthropic SDK. Checks availability, lists models, spawns one of three background `QThread` workers depending on the selected provider. |
| `m_conversation.py` | **Model** | Represents a single conversation: add messages, auto-title, save/load as JSON. Skips saving empty conversations. |
| `m_conversation_store.py` | **Model** | Scans the history folder, lists saved conversations as lightweight summaries, loads conversations by ID. |
| `vm_chat.py` | **ViewModel** | Builds the system prompt using `PromptManager`. Manages conversation history. Parses the LLM's structured JSON response into a `ChatResult` object. Executes safe pandas operations. |
| `v_chat_panel.py` | **View** | Floating `QDialog` with chat bubbles, provider/model selector, timestamp display, status bar, and input field. Emits `plot_requested(dict)` when the AI suggests a plot or graph modification. |
| `v_history_dialog.py` | **View** | Browsable list of saved conversations sorted by most-recent-first. Supports open, rename, duplicate, and delete. |

---

## **Key Classes**

### **`LLMWorker` / `APIWorker` / `AnthropicWorker` — Background Threads**

**File**: `spectroview/ai_agent/m_llm_client.py`

Three `QThread` subclasses, one per backend. Each executes a single streaming chat request and emits token-by-token for live typing animation.

| Class | Backend | Package required |
|-------|---------|------------------|
| `LLMWorker` | Ollama (local) | `ollama` |
| `APIWorker` | OpenAI, DeepSeek, Gemini, Custom | `openai` |
| `AnthropicWorker` | Anthropic Claude | `anthropic` |

| Signal | Payload | Purpose |
|--------|---------|---------|
| `chunk_received` | `str` | Each streamed token fragment |
| `response_ready` | `str` | Full assembled response text |
| `error_occurred` | `str` | Error message if the backend is unreachable |

### **`LLMClient` — Connection Facade**

**File**: `spectroview/ai_agent/m_llm_client.py`

A lightweight facade that manages provider selection, API key storage, and worker lifecycle. Selects the appropriate worker class based on `self._provider`.

**Known providers** (defined in `API_PROVIDERS` dict):
- `Ollama` → `LLMWorker`
- `OpenAI`, `DeepSeek`, `Gemini`, `Custom` → `APIWorker`
- `Anthropic` → `AnthropicWorker`

**File**: `spectroview/ai_agent/m_llm_client.py`

A lightweight, non-Qt class that manages Ollama connectivity and worker lifecycle.

| Method | Purpose |
|--------|---------|
| `is_available()` | Returns `True` if both the `ollama` package and the local daemon are reachable (HTTP call to `/api/tags`) |
| `get_models()` | Returns sorted list of locally downloaded model names |
| `chat(model, messages, on_chunk, on_done, on_error)` | Spawns a background `LLMWorker`; cancels any previous in-flight request |
| `cancel()` | Terminates the current worker thread |
| `is_busy()` | Returns `True` while a worker is running |

**Default model**: `qwen2.5-coder:7b` (configurable via the UI combobox)

### **`PromptManager` — Prompt Assembly**

**File**: `spectroview/ai_agent/m_prompt_manager.py`

A caching manager that reads isolated Markdown sections from disk (`prompts/`, `rules/`, `knowledge/`, `examples/`) and concatenates them into the final system prompt based on user intent.

| Method | Purpose |
|--------|---------|
| `_detect_intent(msg)` | Infers intent (`"plotting"`, `"fitting"`, `"coding"`, `"chat"`) from keywords in the user's message. |
| `load_prompt(name)` | Loads a specific Markdown file (e.g., `load_prompt("system")`). Cached in memory. |
| `build_prompt(...)` | Assembles a complete system prompt from various markdown segments depending on the detected intent. |
| `clear_cache()` | Clears the prompt cache (useful if files change, though auto-reload is supported). |

### **`VMChat` — ViewModel**

**File**: `spectroview/ai_agent/vm_chat.py`

Manages one chat session linked to the loaded DataFrames. Follows the MVVM contract: the View calls public methods; the ViewModel responds exclusively through signals. Uses `PromptManager` to construct system prompts dynamically.

| Signal | Payload | Purpose |
|--------|---------|---------|
| `thinking_changed` | `bool` | `True` while the LLM is processing |
| `chunk_received` | `str` | Streaming token fragments for typing animation |
| `result_ready` | `ChatResult` | Parsed and validated action result |
| `error_occurred` | `str` | Human-readable error message |

#### **Public API**

| Method | Purpose |
|--------|---------|
| `set_dataframes(dfs, active_name)` | Update available DataFrames; clears history only when the set of DataFrame keys changes |
| `update_active_df_name(name)` | Switch active DataFrame without clearing history |
| `set_graphs(graphs)` | Update known open graphs (included in system prompt for context) |
| `set_model(model)` | Switch the Ollama model |
| `process_query(user_text)` | Send a user question to the LLM |
| `cancel()` | Abort in-progress request |
| `clear_history()` | Reset conversation history |

#### **Conversation History & Persistence**

Conversations are saved automatically as JSON files in the user-configured history folder (set via **Settings → AI tab**). The `VMChat` maintains an internal `MConversation` object that:

- Accumulates messages with ISO timestamps.
- Auto-titles itself from the first user message (up to 60 chars).
- Persists to disk after each AI response.
- Is **not cleared** when the user loads a new workspace file or switches DataFrames. History only resets when the user explicitly clicks **+ New Chat**.

The rolling context sent to the LLM is capped at the last **6 pairs** (12 messages) to prevent context overflow.

### **`ChatResult` — Parsed Response**

**File**: `spectroview/ai_agent/vm_chat.py`

A plain data object carrying the parsed LLM response back to the View:

| Field | Type | Purpose |
|-------|------|---------|
| `action` | `str` | `"filter"` \| `"statistics"` \| `"plot"` \| `"update"` \| `"delete"` \| `"answer"` |
| `explanation` | `str` | Human-readable description shown in the chat bubble |
| `dataframe` | `pd.DataFrame` \| `None` | Filtered DataFrame result |
| `text_summary` | `str` | Statistics text or plain-text answer |
| `plot_config` | `list[dict]` \| `None` | Plot configuration(s) for the Graphs workspace |
| `raw_response` | `str` | Full LLM text for debugging |

### **`VChatPanel` — View Dialog**

**File**: `spectroview/ai_agent/v_chat_panel.py`

A floating `QDialog` with the following layout:

```
┌─────────────────────────────────────────────────┐
│  🤖  AI Data Chat              [model combo] [⟳]│
│─────────────────────────────────────────────────│
│  🟢 Ollama connected    📊 2 DataFrame(s) loaded│
│─────────────────────────────────────────────────│
│                                                 │
│   ┌─[You]──────────────────────────────────┐    │
│   │ Show rows where fwhm_Si > 5            │    │
│   └────────────────────────────────────────┘    │
│                                                 │
│   ┌─[🤖 AI]───────────────────────────────┐    │
│   │ Found 42 rows matching `fwhm_Si > 5`  │    │
│   │ ┌────────┬──────┬──────┐              │    │
│   │ │ name   │ fwhm │ ...  │              │    │
│   │ └────────┴──────┴──────┘              │    │
│   └────────────────────────────────────────┘    │
│                                                 │
│─────────────────────────────────────────────────│
│  [Clear]  [Ask a question about your data…] [▶] │
└─────────────────────────────────────────────────┘
```

Helper widgets:
- **`_MessageBubble`**: Styled `QFrame` for user / assistant / error messages with role-dependent colors
- **`_ThinkingDots`**: Animated "Thinking..." label with cycling dots
- **`_DataFramePreview`**: Inline `QTableWidget` for displaying filtered DataFrames (capped at 50 rows, 15 columns) with a "Copy table" button
- **`_ChatLineEdit`**: `QLineEdit` subclass that emits `send_requested` on Enter key

---

## **System Prompt & LLM Contract**

The system prompt is highly modular and managed by the `PromptManager`. Instead of a massive hardcoded string, the prompt is assembled from granular Markdown files based on the detected intent (`chat`, `plotting`, `fitting`, `coding`). 

The `PromptManager` dynamically aggregates these markdown segments:
1. **Core Prompt (`prompts/system.md`, `chat.md`):** Identity and global instruction set.
2. **Context (injected by `VMChat`):** Loaded DataFrame schemas, sample values, active DataFrame name, and open graph summary.
3. **Rules (`rules/general.md`, `plotting.md`):** Behavioural constraints specific to the intent.
4. **Knowledge (`knowledge/features.md`):** Static facts about the software features.
5. **Examples (`examples/plotting_examples.md`):** Few-shot JSON examples for the specific intent.

The LLM is instructed to respond with **only a valid JSON object** (no markdown fences, no explanatory text) conforming to this schema:

```json
{
  "action": "filter | statistics | plot | update | delete | answer",
  "explanation": "short human-readable description",
  "target_dataframe": "name or null",
  "query": "pandas .query() expression or null",
  "stat_columns": ["col1", "col2"],
  "graph_update": {
    "graph_id": 1,
    "properties": { "ymin": 3.6, "plot_title": "New title", ... }
  },
  "graph_delete": {
    "delete_all": false,
    "graph_ids": [1, 2, 3]
  },
  "plot_config": [
    {
      "x": "column", "y": "column", "z": "column or null",
      "filters": ["pandas query string"],
      "plot_style": "point | scatter | box | bar | line | trendline | histogram | wafer | 2Dmap",
      "plot_title": "...", "xlabel": "...", "ylabel": "...",
      "xmin": null, "xmax": null, "ymin": null, "ymax": null,
      "color_palette": "jet | viridis | plasma | ...",
      "scatter_size": 70, "hist_bins": 20, "hist_kde": false,
      "plot_width": 480, "plot_height": 420, "dpi": 100
    }
  ],
  "answer_text": "plain text answer or null"
}
```

### **Safety Rules**

- Only `pd.DataFrame.query()` and `pd.DataFrame.describe()` are executed — **never** `eval()` or `exec()`.
- The LLM's `query` field is passed directly to `df.query()`, so only pandas expression syntax is allowed.
- If the JSON cannot be parsed, the raw text is displayed as an "answer" fallback.

---

## **Supported Actions**

### **1. Filter** (`action: "filter"`)

Applies a `pandas.query()` expression to the target DataFrame and shows the matching rows.

```
User: "Show rows where fwhm_Si > 5 and R² > 0.95"
→ action: "filter", query: "fwhm_Si > 5 and `R²` > 0.95"
→ Result: filtered DataFrame displayed inline as a table
```

### **2. Statistics** (`action: "statistics"`)

Runs `df[columns].describe()` on the specified columns. Falls back to all numeric columns if none are provided.

```
User: "Give me statistics for peak center and FWHM"
→ action: "statistics", stat_columns: ["center_Si", "fwhm_Si"]
→ Result: describe() output displayed as text
```

### **3. Plot** (`action: "plot"`)

Generates one or more plot configurations that are automatically applied to the Graphs workspace.

```
User: "Create a scatter plot of Slot vs center_Si colored by Zone"
→ action: "plot", plot_config: [{ x: "Slot", y: "center_Si", z: "Zone", plot_style: "scatter" }]
→ Result: plot created in the Graphs MDI area
```

**Comma-separated styles**: The LLM can return `"plot_style": "box, bar, point"` in a single entry. `VMChat._execute_plot()` expands this into 3 separate plot configs sharing the same axis/column settings.

### **4. Update** (`action: "update"`)

Modifies an existing graph by ID. The AI knows which graphs are open via the system prompt.

```
User: "Set the Y-axis range of graph 3 to [3.5, 4.2]"
→ action: "update", graph_update: { graph_id: 3, properties: { ymin: 3.5, ymax: 4.2 } }
→ Result: graph 3 is re-rendered with the new limits
```

### **5. Delete** (`action: "delete"`)

Closes one or more graphs. Supports `delete_all` or specific `graph_ids`.

```
User: "Delete all graphs except graph 5"
→ action: "delete", graph_delete: { delete_all: false, graph_ids: [1, 2, 3, 4, 6] }
→ Result: listed subwindows are closed
```

### **6. Answer** (`action: "answer"`)

Plain-text response for questions that don't map to data operations.

```
User: "What columns are available?"
→ action: "answer", answer_text: "The following columns are available: ..."
```

---

## **Data Flow: End-to-End**

```mermaid
sequenceDiagram
    participant User
    participant VCP as VChatPanel (View)
    participant VMC as VMChat (ViewModel)
    participant LLC as LLMClient (Model)
    participant Worker as LLMWorker (QThread)
    participant Ollama as Ollama (local)
    participant Main as main.py
    participant VWG as VWorkspaceGraphs

    User->>VCP: types question + Enter
    VCP->>VCP: _add_user_bubble(text)
    VCP->>VMC: process_query(text)
    VMC->>VMC: _build_system_prompt()
    VMC->>VMC: _build_messages(text)
    VMC-->>VCP: thinking_changed(True)
    VMC->>LLC: chat(model, messages, ...)
    LLC->>Worker: start()
    Worker->>Ollama: ollama.chat(stream=True)
    
    loop each token
        Ollama-->>Worker: chunk
        Worker-->>VMC: chunk_received(fragment)
        VMC-->>VCP: chunk_received(fragment)
        VCP->>VCP: update bubble text
    end

    Worker-->>VMC: response_ready(full_text)
    VMC->>VMC: _parse_response(full_text)
    VMC-->>VCP: result_ready(ChatResult)
    VMC-->>VCP: thinking_changed(False)

    alt action == "plot"
        VCP-->>Main: plot_requested(cfg)
        Main->>Main: _on_chat_plot_requested(cfg)
        Main->>VWG: create_plot_from_config(df_name, cfg)
    else action == "update"
        VCP-->>Main: plot_requested({_graph_update: ...})
        Main->>Main: _apply_graph_update(payload)
        Main->>VWG: vm.update_graph() + re-render
    else action == "delete"
        VCP-->>Main: plot_requested({_graph_delete: ...})
        Main->>Main: _apply_graph_delete(payload)
        Main->>VWG: close subwindows
    else action == "filter" / "statistics"
        VCP->>VCP: display DataFrame / text inline
    end
```

---

## **Integration with Graphs Workspace**

### **Lazy Initialization**

The chat panel is created on first use in `Main.open_ai_chat()`. This avoids importing the `ollama` package or building the UI at startup.

```python
# main.py
if self._chat_panel is None:
    self._chat_panel = VChatPanel(self)
    self._chat_panel.plot_requested.connect(self._on_chat_plot_requested)
```

### **DataFrame Synchronization**

The chat panel stays in sync with the Graphs workspace through three signal connections set up in `main.py`:

| Signal | Handler | Trigger |
|--------|---------|---------|
| `vm.dataframes_changed` | `sync_chat_dfs_full` | DataFrames added/removed — refreshes all DFs and graph info |
| `vm.dataframe_columns_changed` | `sync_chat_active` | Active DataFrame selection changed — preserves chat history |
| `vm.graphs_changed` | `sync_chat_graphs` | Graphs created/updated/deleted — refreshes graph IDs in system prompt |

Each time the panel is shown, a forced sync is performed to ensure current data is available.

### **Plot Config Normalization**

When the AI emits a `plot_config`, `main.py` normalizes the raw JSON before passing it to the Graphs workspace:

| Field | Normalization |
|-------|--------------|
| `y` | `str` → `[str]`, `None` → `[]` |
| Limit fields (`xmin`, `ymin`, ...) | `str`/`int`/`"null"` → `float` or `None` |
| Integer fields (`x_rot`, `scatter_size`, ...) | Cast to `int`, remove on failure |
| `filters` | `["expr1", "expr2"]` → `[{"expression": "expr1", "state": True}, ...]` |

### **Graph Update Flow**

For `action: "update"`, `main.py._apply_graph_update()`:

1. Retrieves the existing `MGraph` model by ID.
2. Normalizes the property types (same rules as above).
3. Calls `vm.update_graph(graph_id, props)` to update the model.
4. Re-renders the corresponding `VGraph` widget in the MDI area.

### **Graph Delete Flow**

For `action: "delete"`, `main.py._apply_graph_delete()`:

1. Reads `delete_all` and `graph_ids` from the payload.
2. Determines the set of graph IDs to close.
3. Closes the corresponding `QMdiSubWindow` instances (triggers cleanup via the `closed` signal).

---

## **Toolbar Entry Point**

The AI Chat button is added to the main toolbar in `VMenuBar`:

```python
# v_menubar.py
self.actionAIChat = self.addAction(
    QIcon(os.path.join(ICON_DIR, "llm_ai.png")),
    "AI Data Chat"
)
self.actionAIChat.triggered.connect(self.ai_chat_requested.emit)
```

The `ai_chat_requested` signal is connected to `Main.open_ai_chat()` in `setup_connections()`.

---

## **Optional Dependency Management**

The AI feature is gated behind an optional dependency:

```toml
# pyproject.toml
[project.optional-dependencies]
ai = ["ollama>=0.4", "openai>=1.0", "anthropic>=0.30"]
```

Install with:
```bash
pip install -e ".[ai]"
```

### **Import Guard Pattern**

```python
# m_llm_client.py
try:
    import ollama, openai, anthropic
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
```

```python
# main.py
try:
    from spectroview.ai_agent.v_chat_panel import VChatPanel, LLMClient
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
```

If dependencies are not installed:
- `LLMClient` is disabled
- The toolbar button shows an informational dialog
- No error is raised at startup

---

## **Response Parsing & Error Handling**

`VMChat._parse_response()` handles several edge cases in the LLM output:

1. **Markdown fences**: Strips `` ```json `` / `` ``` `` that the model sometimes adds.
2. **JSON extraction**: If the top-level text isn't valid JSON, a regex searches for the first `{...}` block.
3. **Fallback**: If no valid JSON can be found, the raw text is returned as an `action: "answer"` so the user always sees something useful.
4. **Invalid DataFrame target**: Falls back to the active DataFrame, then to the first loaded DataFrame.

---

## **Current Features (Graph Workspace)**

| Feature | Description |
|---------|-------------|
| **Natural language plot creation** | Ask for any of the 9 supported plot styles (point, scatter, box, bar, line, trendline, histogram, wafer, 2Dmap) with column, filter, and style options |
| **Multi-plot generation** | A single query can create multiple plots simultaneously |
| **Data filtering** | Filter DataFrames using natural language (translated to `pandas.query()` expressions) |
| **Descriptive statistics** | Request `describe()` statistics on specific columns |
| **Graph modification** | Update existing graph properties (axis limits, title, style, filters) by graph ID |
| **Graph deletion** | Delete specific graphs or all graphs by natural language instruction |
| **Multi-DataFrame support** | Switch between multiple loaded DataFrames; the AI knows which is active |
| **Conversation memory** | Multi-turn conversations with rolling 6-pair history; follow-up queries reuse context |
| **Conversation persistence** | Conversations survive file loads and tab switches. Only **+ New Chat** resets the history |
| **Conversation history browser** | Browse, search, open, rename, duplicate, and delete past conversations via the History dialog |
| **Streaming responses** | Token-by-token display with live typing animation |
| **Model selection** | Switch between locally downloaded Ollama models via the header combobox |
| **Inline data preview** | Filtered DataFrames displayed as interactive tables with copy-to-clipboard |
| **Context-aware prompts** | System prompt includes DataFrame schemas, sample values, and open graph configurations |


