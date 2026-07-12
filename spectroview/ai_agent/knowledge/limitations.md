# Purpose

This file documents the known limitations of the SPECTROview AI Agent. The AI uses this information to set accurate user expectations and provide correct guidance when requests exceed its capabilities.

---

# Data Operations

## Permitted Operations Only

The AI Agent can only execute these safe data operations:

- `pandas.DataFrame.query(expression)` — row filtering
- `pandas.DataFrame.describe()` — descriptive statistics

**Not permitted:**
- `eval()`, `exec()`, or dynamic code execution
- Direct DataFrame mutation or column creation
- Merging or joining DataFrames
- File I/O (reading or writing files)

If a user requests an unsupported operation, respond with `action: "answer"` and explain the limitation.

## Context Window

The AI Agent sends only the **last N conversation messages** to the LLM (configurable, default: no cap but provider-dependent). For very long conversations, early context may be truncated. Users should start a new conversation for unrelated tasks.

---

# Plotting

## One DataFrame Per Plot

Each `plot_config` entry targets a single DataFrame. Multi-DataFrame overlay in one graph is not supported through the AI protocol.

## Spatial Plot Grouping

Spatial plots (`wafer`, `2Dmap`) cannot overlay multiple distinct groupings (e.g., multiple wafer IDs) on the same axes. Separate plot entries are required for each distinct group.

## No Live Graph Inspection

The AI knows the **configuration** of open graphs (style, x, y, z, filters) but cannot inspect the rendered image or pixel values. It cannot tell you "what the highest peak in graph 3 looks like".

---

# Fitting

## No Fitting Execution

The AI Agent cannot trigger fitting. Peak model fitting is performed exclusively in the **Spectra workspace** or via the **VBF engine**. The AI can only query and visualise existing fit results stored in DataFrames.

## No Spectral Data Access

The AI Agent only has access to DataFrames loaded in the **Graphs workspace**. Raw spectral data (intensity vs. wavenumber arrays) from the Spectra workspace is not directly queryable by the AI.

---

# Provider Limitations

## Ollama (Local)

- Requires Ollama daemon running (`ollama serve`)
- Model quality depends on the locally downloaded model size
- No internet connection required (privacy-preserving)

## Cloud Providers

- Require API keys stored in Settings
- Subject to provider rate limits and token quotas
- Response quality and speed vary by model and provider

## Response Parsing

If the LLM returns malformed JSON (despite instructions), the agent falls back to displaying the raw text as an `answer`. This may occur with smaller or less capable models.

---

# Security

## No File Access

The AI Agent has no ability to read or write files on the user's filesystem beyond what is explicitly loaded into SPECTROview.

## No Network Access

The AI Agent does not make external network requests on behalf of the user. All communication is between the application and the configured LLM provider only.
