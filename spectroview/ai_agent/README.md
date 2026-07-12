# SPECTROview AI Agent — Developer Guide

This document describes the modular Prompt Engineering architecture of the **SPECTROview AI Agent** (`spectroview/ai_agent/`).

---

## Directory Structure

```text
spectroview/ai_agent/
│
├── __init__.py                    # Module docstring and exports
├── m_prompt_manager.py            # PromptManager — loads, caches, assembles .md files
├── m_llm_client.py                # LLM backend (Ollama, OpenAI, Gemini, Anthropic, ...)
├── m_conversation.py              # Conversation data model
├── m_conversation_store.py        # Conversation persistence (JSON files)
├── vm_chat.py                     # ViewModel — orchestrates everything (MVVM)
├── v_chat_panel.py                # View — Qt UI panel
├── v_history_dialog.py            # View — conversation history dialog
│
├── prompts/                       # Static prompt instructions (per intent)
│   ├── system.md                  #   Core identity + JSON response schema + dynamic placeholders
│   ├── chat.md                    #   General chat interaction rules
│   ├── plotting.md                #   Plotting-specific instructions
│   ├── fitting.md                 #   Peak fitting instructions
│   └── coding.md                  #   Python code generation instructions
│
├── rules/                         # Behavioural constraints
│   ├── general.md                 #   Universal rules (applies to all intents)
│   ├── plotting.md                #   Plot-specific rules
│   ├── python.md                  #   Code generation rules
│   ├── fitting.md                 #   Fitting-specific rules
│   └── spectroview.md             #   Application-specific conventions
│
├── knowledge/                     # Static domain knowledge
│   ├── features.md                #   Supported formats, workspaces, plot styles, peak models
│   ├── terminology.md             #   Glossary of SPECTROview terms
│   ├── shortcuts.md               #   Keyboard shortcuts reference
│   ├── faq.md                     #   Frequently asked questions
│   └── limitations.md             #   Known limitations and constraints
│
├── examples/                      # Few-shot examples for LLM context
│   ├── plotting_examples.md       #   Plot request → JSON response examples
│   ├── filtering_examples.md      #   Filter/statistics examples
│   └── workflow_examples.md       #   Multi-turn workflow examples
│
├── templates/                     # Reusable Markdown output templates
│   ├── report.md                  #   Analysis report template
│   ├── bug_report.md              #   Bug report template
│   └── release_note.md            #   Release note template
│
├── tools/                         # Reusable Python helper functions
│   ├── __init__.py
│   ├── dataframe_tool.py          #   Schema builders, safe_query, safe_describe
│   ├── plot_tool.py               #   Config normalisation, comma-style expansion
│   └── fitting_tool.py            #   Peak model registry, parameter validation
│
└── config/                        # Developer-facing configuration
    ├── model.yaml                 #   LLM provider/model defaults
    └── settings.yaml              #   Prompt assembly settings (cache, reload, examples)
```

---

## How Prompts Are Assembled

Every time the user sends a message, `VMChat._build_system_prompt()` runs:

```
1.  Compute dynamic context
    ├── DataFrame schemas (column names, types, sample values, preview)
    ├── Active DataFrame name
    └── Open graphs summary (Graph ID, style, x, y, z, filters)

2.  PromptManager.build_prompt(intent="chat", ...)
    ├── Load prompts/system.md          ← identity + JSON schema + {placeholders}
    ├── Load prompts/chat.md            ← action selection + memory rules
    ├── Load prompts/plotting.md        ← plot construction instructions
    ├── Load rules/general.md           ← universal safety/quality rules
    ├── Load rules/plotting.md          ← plot-specific rules
    ├── Load rules/spectroview.md       ← app-specific conventions
    └── Load knowledge/features.md     ← static feature facts

3.  static_prompt.format(
        dataframes_section=...,
        active_df_info=...,
        graphs_info=...,
    )
    → Final system prompt sent to the LLM
```

### Auto-Reload During Development

Set `auto_reload: true` in `config/settings.yaml` (default). Any `.md` file edited on disk will be automatically reloaded on the next request — no application restart required.

---

## How to Add New Rules

1. Create a new file in `rules/`, e.g. `rules/maps.md`
2. Follow the standard Markdown structure:

```markdown
# Purpose

One sentence describing what this rule file governs.

---

# Rules

- Rule 1.
- Rule 2.

---

# Constraints

- Hard constraint 1.
```

3. Load it in `VMChat._build_system_prompt()` by adding `"maps"` to the `rules` list:

```python
static_prompt = self._prompt_mgr.build_prompt(
    intent="chat",
    prompts=["system", "chat", "plotting"],
    rules=["general", "plotting", "spectroview", "maps"],  # ← added
    knowledge=["features"],
)
```

---

## How to Add New Knowledge

1. Create a new file in `knowledge/`, e.g. `knowledge/hardware.md`
2. Fill it with **facts only** (no instructions or constraints):

```markdown
# Purpose

Facts about supported SPECTROview-compatible hardware.

---

# Supported Spectrometers

| Manufacturer | Model | Format | Notes |
|---|---|---|---|
| Renishaw | inVia | .wdf | Single spectra and maps |
```

3. Include it in the prompt assembly:

```python
static_prompt = self._prompt_mgr.build_prompt(
    ...
    knowledge=["features", "hardware"],  # ← added
)
```

---

## How to Add New Examples

1. Create a new file in `examples/`, e.g. `examples/maps_examples.md`
2. Structure examples as User → AI Response pairs with fenced JSON blocks
3. Include it for the relevant intent:

```python
static_prompt = self._prompt_mgr.build_prompt(
    ...
    examples=["plotting_examples", "maps_examples"],  # ← added
)
```

---

## How to Add a New Prompt Module (Intent)

To add a new intent (e.g. `"statistics"`):

1. Create `prompts/statistics.md` and `rules/statistics.md`
2. Add defaults to `m_prompt_manager.py` in `_INTENT_DEFAULTS`:

```python
"statistics": {
    "prompts": ["system", "statistics"],
    "rules": ["general", "statistics", "spectroview"],
    "knowledge": ["features"],
    "examples": ["filtering_examples"],
},
```

3. Add detection keywords to `_INTENT_KEYWORDS`:

```python
"statistics": frozenset({"statistics", "describe", "mean", "std", "distribution"}),
```

4. Enable intent detection in `config/settings.yaml`:

```yaml
enable_intent_detection: true
```

---

## How to Add New Tools

1. Create a new file in `tools/`, e.g. `tools/map_tool.py`
2. Add helper functions with type hints and docstrings
3. Register exports in `tools/__init__.py`
4. Import and use in `VMChat` or other modules as needed

---

## Configuration Reference

### `config/model.yaml`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_provider` | str | `Ollama` | LLM backend when none configured at runtime |
| `default_model` | str | `qwen2.5-coder:7b` | Model name for the default provider |
| `temperature` | float | `0.2` | Sampling temperature (lower = more deterministic) |
| `max_tokens` | int | `8192` | Maximum response token count |
| `max_context_messages` | int\|null | `null` | Max conversation messages sent to LLM |
| `request_timeout_seconds` | int\|null | `120` | API call timeout |

### `config/settings.yaml`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_cache` | bool | `true` | Cache `.md` file contents in memory |
| `auto_reload` | bool | `true` | Reload modified `.md` files automatically |
| `enable_examples` | bool | `true` | Include few-shot examples in prompts |
| `enable_knowledge` | bool | `true` | Include knowledge files in prompts |
| `debug_prompt` | bool | `false` | Log the assembled system prompt to stdout |
| `enable_intent_detection` | bool | `false` | Auto-detect intent from user message |

---

## PromptManager API

```python
from spectroview.ai_agent.m_prompt_manager import PromptManager

mgr = PromptManager()                   # uses ai_agent/ as base_dir

# Load individual files
system = mgr.load_prompt("system")
rules  = mgr.load_rule("plotting")
kb     = mgr.load_knowledge("features")
ex     = mgr.load_example("plotting_examples")
tmpl   = mgr.load_template("report")

# Build a complete system prompt
prompt = mgr.build_prompt(
    intent="plotting",                  # or "chat", "fitting", "coding"
    user_message="create a box plot",   # used for intent detection if enabled
    prompts=["system", "chat", "plotting"],
    rules=["general", "plotting"],
    knowledge=["features"],
    examples=["plotting_examples"],
)

# Read configuration
model_cfg = mgr.model_config            # dict from config/model.yaml
settings  = mgr.settings               # dict from config/settings.yaml

# Invalidate cache
mgr.clear_cache()
```

---

## Backward Compatibility

This refactoring preserves **100% of the existing AI Agent functionality**:

- All 6 actions (`filter`, `statistics`, `plot`, `update`, `delete`, `answer`) work identically
- Multi-turn conversation memory is preserved
- All LLM providers (Ollama, OpenAI, Gemini, DeepSeek, Anthropic, Custom) work unchanged
- The application gracefully degrades if prompt files are missing (logs a warning, continues)
- Runtime settings (API keys, model selection) still use QSettings — YAML files are defaults only

---

## Development Workflow

1. **Edit a `.md` file** in `prompts/`, `rules/`, `knowledge/`, or `examples/`
2. **Send a chat message** in the AI Agent — the updated file is auto-reloaded
3. **Check `debug_prompt: true`** in `config/settings.yaml` to see the assembled prompt in logs
4. **Commit your changes** — the `.md` files are version-controlled alongside the code
