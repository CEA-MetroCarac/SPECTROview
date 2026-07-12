"""
SPECTROview AI Agent module
---------------------------
AI-powered data chat and analysis assistant for SPECTROview.

Architecture
------------
This module implements a modular Prompt Engineering architecture.
All prompts, rules, knowledge, and examples are stored as Markdown
files in subdirectories and loaded at runtime by ``PromptManager``.

    prompts/        – Core identity, JSON schema, per-intent instructions
    rules/          – Behavioural constraints (general, plotting, etc.)
    knowledge/      – Static domain facts (features, terminology, etc.)
    examples/       – Few-shot conversation examples
    templates/      – Reusable Markdown output formats
    tools/          – Reusable Python helper functions
    config/         – YAML configuration files

Key classes
-----------
VMChat          – ViewModel for the chat session (MVVM pattern)
LLMClient       – LLM backend abstraction (Ollama / cloud APIs)
MConversation   – Conversation data model with persistence
PromptManager   – Loads, caches, and assembles Markdown prompt files

Optional dependency
-------------------
All AI-related classes are optional.  If no LLM package is installed
(``ollama``, ``openai``, or ``anthropic``), the rest of the application
remains completely unaffected.  The VChatPanel view handles the
ImportError gracefully by disabling the AI menu item.

Extending the AI
----------------
To customise AI behaviour, edit the Markdown files under this package.
No Python source changes are required for prompt engineering.
See ``README.md`` in this directory for detailed instructions.
"""
