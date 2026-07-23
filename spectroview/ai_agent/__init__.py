"""
SPECTROview AI Agent module
---------------------------
AI-powered data chat and analysis assistant for SPECTROview.

Architecture
------------
A tool-calling agent. The LLM is offered a set of MCP tools describing what it
may do to the user's data; the ViewModel runs the request/tool/response loop
and forwards the resulting graph commands to the Graphs workspace.

    mcp/            – In-process MCP server exposing the SPECTROview tools
    prompts/        – Core identity and per-topic instructions
    rules/          – Behavioural constraints
    knowledge/      – Static domain facts
    examples/       – Few-shot conversation examples
    config/         – YAML configuration (model.yaml, settings.yaml)
    utils/          – Sandboxed pandas eval, DataFrame summaries, plot configs

Key classes
-----------
VMChat          – ViewModel: prompt assembly, the agent loop, tool dispatch
LLMClient       – LLM backend abstraction (Ollama / OpenAI-compatible / Anthropic)
MConversation   – Conversation data model with JSON persistence
PromptManager   – Loads, caches, and assembles the Markdown prompt fragments

Optional dependency
-------------------
``main.py`` guards the import of this package: if ``mcp``, ``ollama``,
``openai``, or ``anthropic`` is missing, the AI panel is disabled and the rest
of SPECTROview is unaffected.

Extending the AI
----------------
Prompt wording lives in the Markdown files under this package and needs no
Python change. Adding a *capability* means adding a tool in ``mcp/server.py``.
See ``README.md`` in this directory.
"""
