# SPECTROview AI Agent — Developer Guide

How the **SPECTROview AI Agent** (`spectroview/ai_agent/`) is put together, and
where to change what. User-facing documentation lives in
`spectroview/resources/user_manual/ai_agent.md`; the architectural overview with
provider setup instructions is in `docs/developer/ai_agent.md`.

---

## Directory Structure

```text
spectroview/ai_agent/
│
├── __init__.py                    # Module docstring
├── vm_chat.py                     # ViewModel — prompt assembly + the agent loop
├── v_chat_panel.py                # View — Qt chat panel
├── v_history_dialog.py            # View — conversation history dialog
├── m_llm_client.py                # LLM backends (Ollama / OpenAI-compatible / Anthropic)
├── m_prompt_manager.py            # Loads, caches, assembles the .md fragments
├── m_conversation.py              # Conversation data model
├── m_conversation_store.py        # Conversation persistence (JSON files)
│
├── agent/                         # Qt-free core
│   ├── commands.py                #   CreatePlot / UpdatePlot / DeletePlots
│   └── ports.py                   #   AppContext protocol + RecordingContext fake
│
├── mcp/
│   ├── server.py                  # SPECTROview's own MCP server (tools + resources)
│   ├── hub.py                     # MCP client — one loop thread, N servers
│   └── config.py                  # servers.yaml -> ServerSpec
│
├── utils/
│   ├── safe_eval.py               # Sandboxed pandas expression evaluation
│   ├── df_summary.py              # Token-efficient column summaries
│   └── plot_utils.py              # Plot config normalisation + multi-style expansion
│
├── prompts/                       # Static instructions
│   ├── system.md                  #   Identity + dynamic-context placeholders
│   ├── chat.md                    #   General interaction rules
│   ├── plotting.md                #   Plotting instructions
│   └── system_small.md            #   Single compact prompt for small local models
│
├── rules/                         # Behavioural constraints
│   ├── general.md                 #   Universal rules
│   ├── plotting.md                #   Plot-specific rules
│   └── spectroview.md             #   Application conventions
│
├── knowledge/
│   └── features.md                # Supported formats, workspaces, plot styles
│
├── examples/                      # Few-shot examples
│   ├── plotting_examples.md       #   Full tier
│   └── examples_small.md          #   Small-model tier
│
└── config/
    ├── model.yaml                 # LLM/model defaults, context + token budgets
    ├── settings.yaml              # Prompt cache / auto-reload / debug logging
    └── servers.yaml               # Which MCP servers to connect
```

---

## Layers

```
VChatPanel (View)  ── signals ──  VMChat (ViewModel)
                                    ├── LLMClient   → Ollama / OpenAI-compat / Anthropic
                                    ├── PromptManager
                                    ├── MCPHub      → N MCP servers (config/servers.yaml)
                                    └── VMChatContext ── implements ──> AppContext
                                                                          ▲
                                            mcp/server.py tools ──────────┘
```

The MCP server never imports the ViewModel. It only sees an `AppContext`
(`agent/ports.py`): read the DataFrames/graphs, submit a command. That is what
lets the tools be tested against `RecordingContext`, and what would let the same
server run out-of-process later.

## The agent loop

One user message can take several LLM round trips:

```
process_query(text)
  └─ _build_messages()            system prompt + capped conversation history
  └─ LLMClient.chat(tools=…)      streams into _on_chunk, ends in _on_done
       └─ _on_done(text, tool_calls)
            ├─ no tool calls  → _emit_final_result() → ChatResult → View
            └─ tool calls     → _run_tool_call() each, via MCPHub
                               → append results as role="tool" messages
                               → chat() again (up to VMChat.MAX_AGENT_TURNS)
```

Graph tools do **not** draw anything. They submit a typed command
(`agent/commands.py`) to the context; `_emit_final_result()` drains the queue,
normalises it once (`utils/plot_utils.py`), and emits a `ChatResult`. The View
re-emits each config as `plot_requested`, and `main.py` applies it to the Graphs
workspace.

`_emit_final_result()` runs on **every** exit path — normal completion, the turn
cap, and a tool failure — so commands the model already queued are never lost
after it told the user they succeeded.

Conversation history is OpenAI-shaped everywhere. A `role="tool"` message is
only valid directly after the assistant message carrying its `tool_calls`, so
`MConversation.to_llm_messages()` widens the context window rather than slicing
between them, and the Anthropic backend translates the pair into
`tool_use`/`tool_result` blocks (`m_llm_client.anthropic_messages`).

---

## MCP servers

`MCPHub` owns one asyncio loop in a background thread and keeps a session open
per server in `config/servers.yaml`. Qt-thread code just calls
`hub.list_tools()` / `hub.call_tool(...)`.

Connecting another server is a YAML block — `in-process`, `stdio`, or `http`:

```yaml
- id: filesystem
  transport: stdio
  command: [npx, -y, "@modelcontextprotocol/server-filesystem", "~/data"]
  enabled: true
  tools: [read_file, list_directory]     # optional allowlist
```

Tool names stay unqualified while unique; a name offered by two servers becomes
`<id>__<tool>` for both. Keep the total tool count small — it is prompt tokens on
every turn, and small local models degrade quickly past a dozen or so, which is
what the per-server `tools:` allowlist is for.

> An external server's tool descriptions reach the model verbatim and are
> written by someone else. Only enable servers you trust.

## Context: pushed vs pulled

The system prompt always carries the cheap half — DataFrame names, shapes, and
every column name with its dtype — because a model that has not seen a column
name will invent one. The bulky half (sample values, row previews, full graph
configs) lives behind MCP **resources** and is fetched only when needed.

Since no LLM API has a native notion of a resource, the hub exposes reading one
as a synthetic `get_context(uri)` tool whose `uri` enum lists exactly the
resources currently available. Any server's resources appear there automatically.

---

## How the system prompt is assembled

`VMChat._build_system_prompt()` runs on every request:

```
1.  Dynamic context (computed fresh each turn)
    ├── DataFrame schemas (columns, dtypes, sample values, head(3) preview)
    ├── Active DataFrame name
    └── Open graphs (ID, style, x, y, z, filters, df)

2.  Static fragments via PromptManager.build_prompt(...)
    ├── Full tier:  prompts/{system,chat,plotting} + rules/{general,plotting,
    │               spectroview} + knowledge/features + examples/plotting_examples
    └── Small tier: prompts/system_small + rules/general + examples/examples_small

3.  Substitute {dataframes_section}, {active_df_info}, {graphs_info}
    → final system prompt
```

`str.replace()` is used rather than `str.format()` because the Markdown contains
literal `{` `}` braces in JSON examples.

**Which fragments load is decided in code**, in `_build_system_prompt()` — there
is no intent router. Add a fragment by creating the `.md` file and naming it in
the relevant `build_prompt(...)` call.

### Small-model tier

Local models below `small_model_param_threshold_b` (from `config/model.yaml`,
detected via `ollama show`) get a much shorter prompt, a smaller `num_ctx`, and a
capped conversation window. The user can force a tier from the panel's
Prompt-tier combo. See `docs/developer/ai_agent.md` for the rationale.

### Auto-reload during development

`auto_reload: true` in `config/settings.yaml` (the default) re-reads any `.md`
file whose mtime changed, so prompt edits take effect on the next message with no
application restart.

---

## How to add a tool

1. Add a decorated function in `mcp/server.py`:

```python
@mcp.tool()
def describe_peaks(model_name: str, df_name: str = "") -> str:
    """One-line summary the model reads. Args: documented here."""
    df = context.get_dataframe(df_name)
    ...
    return "…"
```

The docstring and type annotations *are* the schema the LLM sees — MCP derives
the JSON Schema from them. Prefer named, typed parameters over a catch-all dict:
a weak model reliably fills in a documented parameter and reliably fumbles an
undocumented convention.

2. Read the application only through `context` (the `AppContext`) — never import
   the ViewModel.
3. To change something, `context.submit(SomeCommand(...))` rather than touching
   the UI. A new command type means a dataclass in `agent/commands.py`, a branch
   in `VMChat._commands_to_configs`, and a handler in `main.py`.
4. Return an actionable string on failure ("… was NOT created; please retry"),
   so the loop can self-correct while it still has turns left.
5. Cover the schema in `tests/unit/ai_agent/test_ai_agent_schema.py` and the
   behaviour in `test_ai_agent_filter_validation.py` (both drive the tools
   through `RecordingContext` — no Qt, no LLM).
6. Mention it in `prompts/system.md`'s tool list.

## How to add a resource

Add an `@mcp.resource("spectroview://…")` function in `mcp/server.py`. It shows
up in `get_context`'s enum automatically — no client change. Use a resource,
rather than more prompt text, for anything bulky that most turns do not need.

## How to add a rule, knowledge file, or example

Create the `.md` file in the matching directory and add its stem to the
corresponding list in `VMChat._build_system_prompt()`. Files not named there are
never loaded. Keep `rules/` to constraints and `knowledge/` to facts.

---

## Configuration reference

### `config/model.yaml`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_provider` | str | `Ollama` | Backend when none configured at runtime |
| `default_model` | str | `qwen2.5-coder:7b` | Model for the default provider |
| `max_tokens` / `max_tokens_small` | int | `81920` / `4096` | Generation cap per tier |
| `max_context_messages` / `_small` | int\|null | `null` / `6` | History messages sent per tier |
| `request_timeout_seconds` | int\|null | `120` | Per-call timeout |
| `ollama_num_ctx_full` / `_small` | int | `16384` / `8192` | Ollama context window per tier |
| `ollama_think` | bool\|str | `false` | Hybrid-reasoning mode for e.g. qwen3 |
| `small_model_param_threshold_b` | float | `10.0` | Below this many B params ⇒ small tier |
| `column_detail_threshold` | int | `30` | Above this many columns, group them by prefix |

Runtime settings (API keys, selected provider/model) live in QSettings and always
win; this file only supplies defaults.

### `config/settings.yaml`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_cache` | bool | `true` | Cache `.md` contents in memory |
| `auto_reload` | bool | `true` | Re-read modified `.md` files automatically |
| `debug_prompt` | bool | `false` | Log the assembled system prompt |

---

## Tests

```bash
python -m pytest tests/unit/ai_agent -q
```

- `test_ai_agent_agent_loop.py` — the loop, driven by a scripted `FakeLLMClient`
  (round-trip counts, tool feedback, the turn cap, failure paths). Start here
  when changing `_on_done`.
- `test_ai_agent_conversation.py` — history → LLM message conversion, including
  the rule that a `role="tool"` message may never be split from its `tool_calls`.
- `test_ai_agent_mcp_hub.py` — server connection, tool-name qualification,
  allowlists, resources, and `servers.yaml` parsing.
- `test_ai_agent_anthropic_translation.py` — the OpenAI → Anthropic tool and
  message mapping.
- `test_ai_agent_schema.py` — the JSON Schema each tool exposes, and that the
  three plot-style lists have not drifted apart.
- `test_ai_agent_filter_validation.py` — filter dry-runs and property merging.
- `test_ai_agent_prompt_assembly.py` — both prompt tiers.
- `test_ai_agent_small_model_detection.py`, `_request_options.py`,
  `_reasoning_toggle.py`, `_safe_eval.py`, `_tls_errors.py`.

Nothing here needs a network, an API key, or a running Ollama.
