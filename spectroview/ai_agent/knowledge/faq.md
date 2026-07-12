# Purpose

This file contains frequently asked questions and their standard answers. The AI Agent uses this as a reference when users ask common questions about SPECTROview or the AI Agent itself.

---

# General Questions

## What can the AI Agent do?

The AI Agent can:
- **Filter data** — show rows matching conditions (e.g., "show rows where FWHM > 5")
- **Compute statistics** — mean, std, min, max, percentiles via `df.describe()`
- **Create plots** — any of the 9 supported styles directly in the Graphs workspace
- **Modify graphs** — update axis limits, titles, colors, filters on existing graphs
- **Delete graphs** — close specific or all open graphs
- **Answer questions** — explain data, suggest analysis steps, describe column contents

## What can the AI Agent NOT do?

The AI Agent cannot:
- Run arbitrary Python code (`eval`, `exec`) on your data
- Modify or overwrite spectral data or project files
- Trigger peak fitting — fitting is done in the Spectra/VBF workspace
- Access external networks or APIs
- Create new DataFrames by merging (use the Graphs workspace sidebar instead)

## How do I reset the conversation?

Click the **+ New Chat** button (new chat icon in the header) or press `Ctrl+Shift+A` to open a fresh conversation.

## Where are conversations saved?

Conversations are saved as JSON files in the folder configured in **Settings → AI tab → History Folder**. They can be browsed via the History button (📋 icon).

---

# Data Questions

## How does the AI know my column names?

Every time you send a message, the system automatically injects the current DataFrame schemas (column names, data types, sample values) into the AI's context. The AI reads this before responding.

## Can the AI query multiple DataFrames at once?

Currently, each action targets one DataFrame. Use the `target_dataframe` field to specify which one. The AI defaults to the currently active DataFrame.

## Why did the AI use the wrong DataFrame?

Make sure the correct DataFrame is selected in the Graphs workspace sidebar. The active DataFrame name is shown in the status bar of the AI chat panel.

---

# Plotting Questions

## Why didn't the AI set axis labels?

By design, the AI leaves axis labels (`xlabel`, `ylabel`) as `null` unless you explicitly request them. SPECTROview automatically generates optimal axis labels from the column names.

## How do I create multiple plots in one request?

Simply ask for multiple styles: "Create a box plot and a scatter plot of FWHM vs Slot". The AI will generate both in a single response.

## How do I update an existing graph instead of creating a new one?

Reference the graph by its ID: "Update graph 3 to use the viridis color palette". The AI uses `action: "update"` for modifications to existing graphs.

---

# Provider and Model Questions

## Which LLM providers are supported?

- **Ollama** (local, fully offline) — install via `pip install ollama` + `ollama serve`
- **OpenAI** (GPT-4o, GPT-4o-mini, etc.) — requires `pip install openai` + API key
- **Gemini** (Google) — requires `pip install openai` + Gemini API key
- **DeepSeek** — requires `pip install openai` + DeepSeek API key
- **Anthropic** (Claude) — requires `pip install anthropic` + API key
- **Custom** — any OpenAI-compatible endpoint

## Where do I configure my API key?

Go to **Settings → AI** tab and enter your API key for the selected provider. Keys are stored locally using Qt's secure settings storage.
