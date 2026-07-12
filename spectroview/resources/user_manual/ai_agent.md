## **AI Chat Agent**

The **SPECTROview AI Agent** is a built-in AI assistant that lets you query, filter, visualize, and modify your data using plain natural language — no coding required.

> The AI Chat Agent is **optional**. SPECTROview works fully without it. You only need to set it up if you want to use AI-powered features.

---

### **1. Opening the AI Chat Agent**

You can open the AI Chat panel in two ways:

- Click the 🤖 **AI Chat Agent** button in the top menu bar.
- Use the keyboard shortcut **`Ctrl + Shift + A`**.

---

### **2. Supported Providers**

The AI Agent supports multiple LLM backends:

| Provider | Description | Requires |
|----------|-------------|----------|
| **Ollama (local)** | Fully local, no internet, no API key | `pip install ollama` + Ollama installed |
| **OpenAI** | GPT-4o, GPT-4o-mini, etc. | `pip install openai` + API key |
| **DeepSeek** | Cost-effective, high-quality reasoning | `pip install openai` + DeepSeek API key |
| **Gemini** | Google Gemini models | `pip install openai` + Gemini API key |
| **Anthropic** | Claude models | `pip install anthropic` + Anthropic API key |
| **Custom** | Any OpenAI-compatible endpoint | `pip install openai` + custom URL + API key |

> **Note:** DeepSeek, Gemini, and other OpenAI-compatible providers all use the `openai` Python package as a generic networking client. You are **not** sending data to OpenAI — only the package is reused as a universal API tool.

---

### **3. Configuring a Provider**

1. Select a **Provider** from the dropdown on the top-left of the chat panel.
2. For cloud providers (OpenAI, DeepSeek, Gemini, Anthropic, Custom):
   - Enter your **API key** in the **Settings** panel (`Ctrl + Shift + S` → **AI** tab).
3. Select a **Model** from the model dropdown next to the provider.
4. Click the **⟳ refresh** icon to verify the connection. A green status indicator confirms it is working.

---

### **4. Chat Interface Overview**

The chat panel shows messages in chronological order with clear color-coded bubbles:

- 🔵 **Blue bubble** — Your messages (User)
- 🟢 **Green bubble** — AI Agent replies
- 🔴 **Red bubble** — Error messages

Each bubble shows the **timestamp** (`YY-MM-DD HH:MM`) and action buttons:
- **📋 Copy** — Copy the message text to clipboard
- **↩ Reply** (AI messages only) — Reply specifically to that AI message

---

### **5. What the AI Agent Can Do**

The AI Agent has full awareness of your loaded DataFrames and open graphs. It can:

- **Filter data**: Show rows matching any condition
- **Compute statistics**: Mean, std, min, max, percentiles
- **Create plots**: Generate any of the 9 supported plot styles
- **Modify graphs**: Update axis ranges, titles, colors, filters
- **Delete graphs**: Remove specific or all open graphs
- **Answer questions**: Explain column contents, suggest analysis steps

---

### **6. Prompt Examples**

#### 🔍 Data Filtering

```
Show rows where FWHM_Si > 5
```
```
Filter data where Zone == "center" and Slot < 10
```
```
Find samples where both R_squared > 0.95 and fwhm < 4
```
```
Show me all outliers where peak_intensity is more than 2 standard deviations above the mean
```

---

#### 📊 Statistics

```
Give me statistics for peak center and FWHM columns
```
```
What is the average FWHM by Zone?
```
```
Compare the mean and standard deviation of center_Si across all zones
```
```
How many unique samples are in the dataset?
```

---

#### 📈 Creating Plots

```
Create a scatter plot of Slot vs center_Si colored by Zone
```
```
Plot a box chart of FWHM grouped by wafer
```
```
Generate a point plot and a bar plot of peak_intensity vs Slot
```
```
Create a 2D map of center_Si with Slot on X and Zone on Y
```
```
Plot a histogram of FWHM values with 30 bins
```
```
Show a trendline of center_Si vs temperature
```

---

#### ✏️ Modifying Graphs

```
Set the Y-axis range of graph 3 to [3.5, 4.2]
```
```
Change the title of graph 1 to "FWHM Overview"
```
```
Update graph 5 to use the viridis color palette
```
```
Add a filter to graph 2: only show Zone == "edge"
```
```
Change the plot style of graph 4 to scatter
```

---

#### 🗑️ Deleting Graphs

```
Delete graph 3
```
```
Remove all graphs except graph 1
```
```
Close all open graphs
```

---

#### 💬 General Questions

```
What columns are available in my data?
```
```
Which columns contain numeric values?
```
```
What is the data type of the Zone column?
```

---

### **7. Conversation History**

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

### **8. Tips & Best Practices**

- **Be specific about column names** — the AI knows your column names, but using the exact name avoids confusion.
- **Reference graph IDs** — when modifying or deleting graphs, check the graph ID shown in its title bar.
- **Start simple** — ask for a basic plot first, then refine it in follow-up messages.
- **Multiple plots at once** — request several styles in one prompt: *"Create a box and scatter plot of X vs Y"*.
- **Check the status bar** — a 🔴 red indicator means the connection is not working. Check your API key in Settings or verify Ollama is running.

---

### **9. Keyboard Shortcuts**

| Shortcut | Action |
|----------|--------|
| `Ctrl + Shift + A` | Open AI Chat Agent |
| `Enter` | Send message |
| `Shift + Enter` | New line in input box |
