## **AI Chat Agent**

The **SPECTROview AI Agent** is a built-in AI assistant that lets you query, filter, visualize, and modify your data using plain natural language — no coding required.

> The AI Chat Agent is **optional**. SPECTROview works fully without it. You only need to set it up if you want to use AI-powered features.

<div align="center">
  <img src="../user_manual_images/AI-agent/demo_ai_agent.gif" alt="AI Chat Agent demo" width="900"><br>
  <em>Talk to SPECTROview in plain language to filter data and create or customize plots.</em>
</div>

---

### **1. Opening the AI Chat Agent**

You can open the AI Chat panel in two ways:

- Click the **AI Chat Agent** button in the top menu bar.
- Use the keyboard shortcut **`Ctrl + Shift + A`**.

---

### **2. Supported Providers**

The AI Agent supports multiple LLM backends out of the box — no extra packages to install:

| Provider | Description | Requires |
|----------|-------------|----------|
| **Ollama (local)** | Fully local, no internet, no API key | [Ollama](https://ollama.com) installed and running |
| **OpenAI** | GPT-4o, GPT-4o-mini, etc. | API key |
| **DeepSeek** | Cost-effective, high-quality reasoning | DeepSeek API key |
| **Gemini** | Google Gemini models | Gemini API key |
| **Mistral** | Mistral AI models | Mistral API key |
| **Anthropic** | Claude models | Anthropic API key |
| **Custom** | Any OpenAI-compatible endpoint (e.g. an internal/institutional server) | Base URL + API key |

> **Note:** DeepSeek, Gemini, and other OpenAI-compatible providers all use the `openai` Python package as a generic networking client. You are **not** sending data to OpenAI — only the package is reused as a universal API tool.

---

### **3. Configuring a Provider and Model**

1. Inside the **Chat Composer** at the bottom of the panel, select an **AI Provider** from the dropdown (`Ollama (local)`, `OpenAI`, `Anthropic`, `Mistral`, `Gemini`, `DeepSeek`, `Groq`, or `Custom`).
2. For cloud providers, configure your **API key** in the **Settings** dialog (`Ctrl + Shift + S` → **AI** tab):
   - The **Custom** provider's **Custom API Key**, **Base URL**, and **Model Name** fields are shown by default.
   - Other cloud provider keys are organized under the collapsed **▸ Other cloud providers** section.
   - Use the **Test Connection** button directly in the **AI** tab to verify authentication, host connectivity, and model availability with immediate visual feedback (`✓ Connection successful` or detailed error messages).
3. Choose a **Model** from the editable model dropdown right next to the provider in the composer toolbar. If your model is not listed, you can type its identifier directly.
4. Click the **⟳ refresh** button next to the model dropdown to refresh available models.

> **Model Name & Label:** Some endpoints (custom ones especially) do not expose a model-listing API. Enter the model names you use in **Settings** → **AI** tab → **Model Name** as a comma-separated list (e.g. `model-a, model-b`). You can also specify a custom display **Label** (e.g. `ABC` or `Lab Model`) so that your endpoint appears by name in the composer's provider dropdown instead of generic `Custom`. Settings changes are applied immediately to any open chat dialog.

---

### **4. Automatic Small-Model Optimization**

Small local models (roughly under ~10 billion parameters — such as `qwen2.5:7b`, `gemma2:2b`, `phi3:mini`) are noticeably more sensitive to long system prompts. SPECTROview automatically inspects the parameter size of local Ollama models and activates a compact prompt tier, a targeted conversation window, and tuned generation parameters.

---

### **5. Modern Chat Composer & Interface Overview**

The chat interface features an intuitive, unified layout:

- **Top Header Bar**: A clean, single-row header bar containing the editable **Conversation Title** on the left (takes available width) and quick-action icon buttons on the right: **Chat History**, **New Chat**, and **Clear Chat**.
- **Conversation Thread**: Displays color-coded message cards (User, AI replies with formatted Markdown/tables/equations, and error alerts).
- **Embedded Composer Card**: A rounded composer box at the bottom containing:
  - An auto-expanding multi-line text input (press `Enter` to send, `Shift + Enter` for new lines).
  - An embedded bottom toolbar hosting the **AI Provider selector**, **Model selector**, **Refresh button**, **Voice dictation mic (`🎤`)**, and **Send ▶ / Stop ■ button**.

The chat panel shows messages in chronological order with clear color-coded bubbles:

- 🔵 **Blue bubble** — Your messages (User)
- 🟢 **Green bubble** — AI Agent replies
- 🔴 **Red bubble** — Error messages

Each bubble shows the **timestamp** (`YY-MM-DD HH:MM`) and action buttons:
- **📋 Copy** — Copy the message text to clipboard
- **↩ Reply** (AI messages only) — Reply specifically to that AI message

#### 🎤 Voice Dictation (Optional)

You can use the microphone button in the input bar to dictate your queries instead of typing them. This feature requires optional dependencies:
```bash
pip install SpeechRecognition pyaudio
```
If these are not installed, the microphone button will still appear but will show an error message when clicked instructing you to install them.

---

### **6. What the AI Agent Can Do**

The AI Agent has full awareness of your loaded DataFrames and open graphs. It can:

- **Filter data**: Show rows matching any condition
- **Compute statistics**: Mean, std, min, max, percentiles
- **Create plots**: Generate any of the 9 supported plot styles
- **Modify graphs**: Update axis ranges, titles, colors, filters
- **Delete graphs**: Remove specific or all open graphs
- **Answer questions**: Explain column contents, suggest analysis steps

The agent always knows your DataFrame names and every column name and type. It
looks up the rest — example values from a column, or a graph's exact current
settings — only when a question needs them, so ordinary requests stay fast.

---

### **7. Prompt Examples**

#### 🚀 Complex / Combined Prompts

The real strength of the agent is handling rich, multi-step requests in a
**single message** — analyze the data, create several graphs, and fine-tune
their appearance all at once.

**Analyze, then plot:**
```
Compute the mean and std of center_Si per Zone, then create a bar plot of the
per-Zone mean with error bars from the std, sorted from highest to lowest.
```
```
Find the wafer with the highest average FWHM_Si, filter to just that wafer, and
plot a wafer map of its FWHM_Si.
```

**Multiple graphs in one prompt:**
```
Create three graphs: a scatter of Slot vs center_Si colored by Zone, a box plot
of center_Si grouped by Zone, and a histogram of center_Si with 40 bins.
```
```
For each of FWHM_Si, center_Si and intensity_Si, plot a point plot vs Slot
colored by wafer.
```

**Plot and customize in the same request:**
```
Make a scatter plot of Slot vs center_Si colored by Zone, set the title to
"Si peak position", label the Y-axis "Raman shift (cm⁻¹)", set the Y range to
[520, 522], use the viridis palette, and turn on the grid.
```
```
Plot FWHM_Si vs temperature as a trendline, filter to Zone == "center", move the
legend to the upper right, and use a log scale on the X-axis.
```

**Everything together:**
```
Filter the data to R_squared > 0.95, then create a scatter plot and a box plot
of center_Si vs Zone side by side, colored by wafer, both with the title
"High-quality fits only" and the grid enabled.
```

---

#### 🧩 Simple / Single-Action Prompts

Good starting points that each do one thing — filter, summarize, plot, tweak, or
ask. Combine them freely into the complex prompts above.

```
Show rows where FWHM_Si > 5 and R_squared > 0.95
```
```
Give me statistics for center_Si and FWHM_Si by Zone
```
```
Create a scatter plot of Slot vs center_Si colored by Zone
```
```
Plot a wafer map of fwhm_Si for Slot 11
```
*(Note: Wafer and 2Dmap plots require your dataset to have spatial X and Y coordinate columns).*
```
Set the Y-axis range of graph 3 to [3.5, 4.2] and use the viridis palette
```
```
Delete graph 3
```
```
What columns are available in my data?
```

---

### **8. Conversation History**

The AI Agent **remembers your conversation** across multiple turns:

```
User: Show rows where FWHM > 5
AI:   Found 42 rows...

User: Now create a scatter plot of Slot vs center_Si for those rows
AI:   (creates scatter plot with FWHM > 5 filter applied)
```

- The chat history is **preserved** when you load new files or switch datasets — it only resets when you click **➕ New Chat**.
- Past conversations are automatically saved and can be reopened from the **History** panel (📋 button).
- Use **↩ Reply** on any AI message to specifically respond to that turn.

---

### **9. Tips & Best Practices**

- **Be specific about column names** — the AI knows your column names, but using the exact name avoids confusion.
- **Reference graph IDs** — when modifying or deleting graphs, check the graph ID shown in its title bar.
- **Start simple** — ask for a basic plot first, then refine it in follow-up messages.
- **Multiple plots at once** — request several styles in one prompt: *"Create a box and scatter plot of X vs Y"*.
- **Check the status bar** — a 🔴 red indicator means the connection is not working. Check your API key in Settings or verify Ollama is running.
- **Struggling local model?** — see [section 4](#4-automatic-small-model-optimization), keep requests to one or two plots per message, and prefer models that are explicitly documented as supporting tool/function calling.

---

### **10. Troubleshooting a "Connection error."**

If a cloud/custom provider replies with **"Connection error."**, the app couldn't establish the network connection at all (it is *not* a wrong model name or an expired key — those report differently). Common causes:

- **Wrong Base URL** — for a **Custom** provider, double-check the **Base URL** in Settings. It must be the endpoint's OpenAI-compatible root (e.g. `https://host/v1` or `https://host/openai/`), **not** the full `.../chat/completions` path.
- **Corporate / internal endpoint with a private certificate** — an on-premise endpoint (e.g. `https://…intra.company…/`) is often secured by a company certificate authority that Python doesn't trust by default, producing a hidden *certificate verify failed* error. SPECTROview handles this automatically by trusting your **operating system's certificate store** (via the `truststore` package, installed by default). If it still fails, point the app at your organization's certificate bundle by setting the environment variable **`SSL_CERT_FILE`** to your `.pem` file (e.g. `chain_bundle.pem`) before launching.
- **Off the network / VPN** — an internal endpoint is only reachable from the corporate network or VPN.

The error card now shows the underlying cause (e.g. `CERTIFICATE_VERIFY_FAILED`, `getaddrinfo failed`) beneath the summary, which tells you which of the above applies.

---

### **11. Keyboard Shortcuts**

| Shortcut | Action |
|----------|--------|
| `Ctrl + Shift + A` | Open AI Chat Agent |
| `Enter` | Send message |
| `Shift + Enter` | New line in input box |
