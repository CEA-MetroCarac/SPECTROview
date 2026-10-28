## **Settings & Preferences**

Access the **Settings** menu via the gear icon in the main toolbar (or press `Ctrl + Shift + S`). You can also click **About SPECTROview** in the top menu bar to jump directly to the **About** section.

<div align="center">
  <img src="../user_manual_images/Settings/settings.png" alt="Settings Panel" width="460"><br>
</div>

<br>

The `Settings Panel` features a modern two-panel interface with real-time search filtering:

- **Left Sidebar**:
  - **Search Bar**: Type any keyword (e.g. `working folder`, `tolerance`, `loss`, `mcp`, `python`, `citation`) to instantly filter the navigation list.
  - **Navigation Tabs**:
    1. **General** — SPECTROview Working Folder, portable configuration backup/restore, and update preferences.
    2. **Fit Settings** — Optimizer convergence parameters, robust loss function, peak polarity, and global peak limits.
    3. **AI Settings** — LLM provider credentials, connection testing, and local MCP server.
    4. **About** — Application version, build number, Python environment, runtime dependency versions, documentation, and citation.
- **Right Content Area**: Displays all options for the selected tab.
- **Footer**: Shows application version with standard **Save** and **Cancel** buttons.

> **Theme**: Interface theme selection (**Modern Dark**, **Modern Light**, **Classic Dark**, or **Classic Light**) is toggled directly from the **Theme** button in the top menu bar.

---

### **1. SPECTROview Working Folder & Preferences** *(General tab)*

Specify a single universal root folder for all of SPECTROview's templates, recipes, styles, AI chat history, and configuration backup. When you set it, five subfolders and a portable configuration file are created automatically inside it:

| Subfolder / File | Contents |
|------------------|----------|
| `fit_model/` | Custom fit models (JSON). The `Fit Model Builder` scans this folder and lists every model here for one-click reuse. |
| `plot_recipe/` | Plot Recipes — saved sets of full plot configurations (data bindings + style) for the `Graphs` workspace. |
| `plot_style/` | Style Templates — saved appearance-only styles that can be applied to any existing graph. |
| `ai_chat_history/` | AI chat conversation logs (JSON). Conversations are automatically saved here. |
| `template/` | General plot templates and workspace templates. |
| `config.json` | Portable configuration backup. Application settings are mirrored to this JSON file. |

- Click **Browse** to select a folder, or type the path directly.
- Click **Open** to open the configured folder directly in **Windows Explorer** or **macOS Finder** with one click.

> **Backup & PC Migration**: If you move to a new PC or want to make a backup, simply copy this single Working Folder to the new PC. In SPECTROview's Settings panel on the new machine, point the **SPECTROview Working Folder** to that folder: all fit models, plot recipes, style templates, AI chat history, and configuration (`config.json`) will be restored automatically!
>
> You can also use the **Export Config...** and **Restore Config...** buttons at any time to export or restore settings to/from a standalone JSON file.

#### **Application Updates**
- **Check for updates automatically on application startup**: When checked (default), SPECTROview will periodically check for new releases in the background upon application launch.

---

### **2. Fit Convergence Parameters** *(Fit Settings tab)*

These parameters control the behavior of the **Vectorized Batch Fit (`VBF`) engine** — the mathematical optimizer that performs curve fitting on your spectra. Adjusting these values directly affects fitting speed, precision, and convergence.

| Parameter | Setting | Default | Range | Description & Behavior |
|-----------|---------|---------|-------|------------------------|
| **Maximum Iterations** | `max_ite` | 200 | 1 – 10,000 | The maximum number of Levenberg-Marquardt iterations per spectrum. <br>• **Lower (50–100)**: Faster fitting for previews.<br>• **Higher (200+)**: Better convergence for complex spectra. |
| **x-tolerance** | `xtol` | 1×10⁻⁴ | 1×10⁻⁶ – 1×10⁻¹ | Relative tolerance for the parameter step size. Convergence requires: <br> mean(|&delta;<i>p</i>| / |<i>p</i>|) &lt; `xtol` |
| **f-tolerance** | `ftol` | 1×10⁻⁴ | 1×10⁻⁶ – 1×10⁻¹ | Relative tolerance for the cost function. Convergence requires: <br> |cost<sub>n</sub> - cost<sub>n-1</sub>| / |cost<sub>n</sub>| &lt; `ftol` |
| **Noise Threshold Coefficient** | `coef_noise` | 1.0 | 0 – 100 | Multiplier for the auto-estimated noise level (`coef_noise` × estimated noise amplitude). Masks noisy data and suppresses ghost peaks.<br>• **0**: Disabled<br>• **0.5–1.0**: Conservative<br>• **1.0–2.0**: Moderate (Default)<br>• **3.0+**: Aggressive<br><br>*Note: You can refer to the noise level displayed in the toolbar or activate the "Show noise level" checkbox within the `View Options` of the `SpectraViewer` to see the noise level directly on the spectra plot.* |
| **Loss Function** | `loss` | `linear` | `linear`, `soft_l1`, `huber` | Robust loss function for outlier resistance (e.g. cosmic ray spikes, sharp artifacts). <br>• **linear**: Standard least squares (default).<br>• **soft_l1**: Smooth L1 approximation $\rho(z) = 2(\sqrt{1+z}-1)$, robust to large spikes without manual masking.<br>• **huber**: Huber loss, quadratic for inliers and linear for residuals exceeding `f_scale`. |
| **Loss Scale** | `f_scale` | 1.0 | 1×10⁻⁴ – 10,000 | Margin scale between inliers and outliers for robust loss functions. Residuals are scaled as $(r / f\_scale)$ before evaluating $\rho'(z)$. |
| **Fit Negative Values** | - | Unchecked | - | When checked, negative intensity values (absorption or inverted peaks) are included in the fit. When unchecked, they are assigned zero weight. |

> **Note**: A spectrum is considered converged only when **both** `xtol` and `ftol` criteria are simultaneously satisfied.
>
> **Weighting Composition Order**: Hard exclusions (`fit_negative` and `coef_noise` noise-floor masking) are evaluated first to assign zero weight to excluded points. For active points, robust loss weighting (`soft_l1` or `huber`) is applied second, smoothly down-weighting outlier residuals via Iteratively Reweighted Least Squares (IRLS).

---

### **3. Global Peak Limits** *(Fit Settings tab)*

These parameters define **absolute boundary conditions** that are applied globally to all peaks across the application. When you add a new peak to a fit model, these limits are used to set the initial parameter bounds automatically.

> **Note**: You can check the "Limits" checkbox under the `PeakTable` to display the actual bounds of each peak.

| Limit | Default | Range | Description & Constraint |
|-------|---------|-------|--------------------------| 
| **Max Peak Shift** | 20 | 0 – 100 | Maximum allowed displacement of a peak center (`x0`) from its initial value. <br><i>x</i><sub>0</sub> - maxshift &le; <i>x</i><sub>0</sub><sup>fitted</sup> &le; <i>x</i><sub>0</sub> + maxshift |
| **Min Peak FWHM** | 0.1 | 0.01 – 500 | Minimum allowed Full Width at Half Maximum (`FWHM`). Prevents peaks from collapsing into delta-function artifacts. |
| **Max Peak FWHM** | 200 | 1 – 1,000 | Maximum allowed `FWHM`. Prevents peaks from broadening into flat baselines. |
| **Max Peak Intensity** | 100,000 | 1 – 1×10⁹ | Upper bound for peak amplitude. Prevents unrealistically high intensity values. |

---

### **4. AI Settings** *(AI Settings tab)*

The **AI Settings** tab configures the [AI Chat Agent](ai_agent.md). Nothing here is required to use the rest of SPECTROview.

#### **API Keys**

The fields for a **Custom** (OpenAI-compatible) endpoint are shown by default:

| Field | Description |
|-------|-------------|
| **Custom API Key** | The API key for your custom / internal endpoint. |
| **Base URL** | The endpoint's OpenAI-compatible root URL (e.g. `https://host/v1`). |
| **Model Name** | A comma-separated list of model names (e.g. `model-a, model-b`). These populate the model dropdown in the chat panel — handy for endpoints that don't expose a model-listing API. |
| **Display Label** | Label shown in the AI chat composer for this custom endpoint. |

Keys for the built-in cloud providers are tucked under the collapsible **▸ Other cloud providers** section (click to expand): **OpenAI**, **Anthropic**, **Gemini**, **DeepSeek**, and **Mistral**. All keys are stored masked and remembered between sessions.

- **Test Connection**: Select any provider from the dropdown and click **Test Connection** to verify API endpoint availability, credentials, and network connectivity without saving.

> **Where your keys are stored**: API keys go into your operating system's
> per-user settings store — the registry (`HKEY_CURRENT_USER\Software\CEA-Leti\SPECTROview`)
> on Windows, `~/Library/Preferences` on macOS. They are **never** written to a
> file inside the SPECTROview project folder, are not included in an exported
> workspace, and cannot be committed to a Git repository by accident.

#### **Local MCP Server**

Enable this only when a trusted MCP client on the same computer needs access
to the currently running SPECTROview session. The default endpoint is
`http://127.0.0.1:8765/mcp` and starts as soon as you select **OK**.

| Field | Default | Description |
|---|---|---|
| **Enable local MCP endpoint** | Off | Starts/stops the embedded MCP server. |
| **Port** | 8765 | Local Streamable HTTP port (1024–65535). Choose another if the port is busy. |

The bind address is fixed to localhost (`127.0.0.1`); it is not exposed to the
LAN or internet. Keep SPECTROview running while the client is connected.

---

### **5. About & Environment** *(About tab)*

The **About** tab provides comprehensive runtime and reference information:

- **Application Details**: Official SPECTROview logo, title, version number (`Version {VERSION}`), and build number (`Build: {BUILD_NUMBER}`).
- **Environment & Dependencies**: Dynamically lists the exact Python version, operating system architecture, and runtime package versions in use by SPECTROview (e.g. `PySide6` with Qt runtime version, `numpy`, `scipy`, `matplotlib`, `pandas`, `pybaselines`, `openpyxl`, `superqt`, `pyarrow`, AI SDKs, and data readers).
- **Check for Updates**: Direct action to check GitHub releases for new versions.
- **Documentation & Paper Citation**: Quick links to the online web documentation and the canonical SoftwareX peer-reviewed citation with DOI.
