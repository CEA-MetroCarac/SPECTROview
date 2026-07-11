## **Installation**

**SPECTROview** requires Python (versions 3.8 through 3.12). It can be easily installed and managed via your system's command-line interface (e.g., Command Prompt on Windows or Terminal on macOS/Linux).

### **1. From PyPI (Recommended)**
```bash
pip install spectroview
```

<div align="center">
  <img src="../user_manual_images/Installation/installation.gif" alt="Installation process" width="500"><br>
  <i>Successfully installing SPECTROview via pip.</i>
</div>

### **2. From GitHub (Latest Development Version)**
To install the latest development version directly from the source repository:
```bash
pip install git+https://github.com/CEA-MetroCarac/SPECTROview.git
```
> In systems where both Python 2 and Python 3 are installed, `pip3 install spectroview` should be used instead.

### **3. Launch and Update**

To launch **SPECTROview**, open your terminal or command prompt and execute:
```bash
spectroview
```

To update your installation to the latest release:
```bash
pip install --upgrade spectroview
```

To install or downgrade to a specific version (e.g., `26.28.1`):
```bash
pip install spectroview==26.28.1
```

---

### **4. Automatic Update Notifications**

**SPECTROview** automatically checks for new releases and notifies you with a non-intrusive banner at the top of the application window — no manual checking required.

#### How it works

- **On startup**, a background thread silently queries the [GitHub Releases page](https://github.com/CEA-MetroCarac/SPECTROview/releases) for the latest version.
- If a newer version is available, a **blue notification banner** appears at the top of the window:

```
🔔  A new version of SPECTROview is available:  26.29.0    [ 🔍 View changelog ]  [ Update later ]  [ Skip this version ]
```

> **Privacy Guarantee**: This check only sends a standard anonymous request to the public GitHub API to fetch the latest version number. **Absolutely no personal or usage data is collected or transmitted.**

#### Banner actions

| Button | Behaviour |
|--------|-----------|
| **🔍 View changelog** | Opens the GitHub release page in your browser to see what's new. |
| **Update later** | Hides the banner for the current session only. The banner will reappear on the next launch if the update is still available. |
| **Skip this version** | Hides the banner permanently for that specific version. The banner will never show again until an even newer version is released. |

> **Offline users**: If **SPECTROview** is installed on a machine without internet access, the update check fails silently — the application starts and works exactly as normal. No error is raised.
