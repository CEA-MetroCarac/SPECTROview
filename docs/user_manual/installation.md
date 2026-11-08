## **Installation**

Choose the **Desktop App** (recommended — no Python or coding setup required) or install as a **Python package (`pip`)**.

All installers are published on the official [**GitHub Releases**](https://github.com/CEA-MetroCarac/SPECTROview/releases) page. Your preferences, fit models, and templates are stored separately from the application folder, so upgrading keeps all your work safe.

---

### **1. Install SPECTROview**

Select your operating system below:

=== ":material-microsoft-windows: Windows"

    Go to [**GitHub Releases**](https://github.com/CEA-MetroCarac/SPECTROview/releases), expand **Assets** on the latest release, and choose one of the two Windows downloads:

    **Option A — Standard Installer (`...Windows_x64_Setup.exe`) *(Recommended)***

    1. Double-click the downloaded `.exe` file to install (no administrator password needed).
    2. Open **SPECTROview** from your **Start Menu** or **Desktop** shortcut.
    3. Future updates can be installed automatically inside the app with one click (**Update and Restart**).

    **Option B — Portable Version (`...Windows_x64_Portable.zip`) *(No installation)***

    1. Right-click the `.zip` file and choose **Extract All...** to any folder (including a USB stick or personal folder).
    2. Open the extracted folder and double-click **`SPECTROview.exe`**.

=== ":material-apple: macOS"

    Go to [**GitHub Releases**](https://github.com/CEA-MetroCarac/SPECTROview/releases), expand **Assets** on the latest release, and download the `.dmg` file for your Mac:

    - **Apple Silicon Macs (M1, M2, M3, M4 — most Macs since 2020):** download `...macOS_arm64.dmg`
    - **Older Intel Macs:** download `...macOS_x64.dmg`

    *(Not sure which Mac you have? Click the **Apple menu () → About This Mac** and check whether it lists **Chip: Apple ...** or **Processor: Intel**.)*

    **Steps to install:**

    1. Double-click the downloaded `.dmg` file.
    2. Drag **`SPECTROview`** onto the **`Applications`** folder icon.
    3. **First time opening only:** Right-click (or **Control-click**) **SPECTROview** inside **Applications** → click **Open** → click **Open** in the confirmation popup.

    !!! note "Why does macOS show a warning on first launch?"
        SPECTROview is a free open-source app and does not currently use a paid Apple Developer signature. If macOS says the developer cannot be verified, open **System Settings → Privacy & Security** and click **Open Anyway**. You only need to do this once per version.

=== ":material-linux: Linux"

    Go to [**GitHub Releases**](https://github.com/CEA-MetroCarac/SPECTROview/releases), expand **Assets** on the latest release, and choose your preferred format:

    **Option A — Ubuntu / Debian Installer (`...Linux_amd64.deb`) *(Recommended)***

    Double-click the downloaded `.deb` file to install it with your Software Center, or install from a terminal:

    ```bash
    sudo apt update
    sudo apt install ./SPECTROview_*_Linux_amd64.deb
    ```

    Then launch **SPECTROview** from your applications menu or by typing `spectroview`. Your settings remain safe in your home folder if you upgrade or uninstall later.

    **Option B — Portable AppImage (`...Linux_x86_64.AppImage`) *(No installation)***

    1. Right-click the file → **Properties → Permissions** → enable **Allow executing file as program** *(or run `chmod +x SPECTROview_*.AppImage`)*.
    2. Double-click the file to start **SPECTROview**.

    !!! tip "On Ubuntu 24.04 or newer"
        If double-clicking the AppImage does nothing, install FUSE once (`sudo apt install libfuse2t64`) or run `./SPECTROview_*.AppImage --appimage-extract-and-run`.

=== ":material-language-python: Python (pip)"

    If you already use Python (**Python &ge; 3.12**), install **SPECTROview** directly from [PyPI](https://pypi.org/project/spectroview/):

    ```bash
    pip install spectroview
    ```

    Launch **SPECTROview** from your terminal or Anaconda Prompt:

    ```bash
    spectroview
    ```

    <div align="center">
      <img src="../user_manual_images/Installation/installation.gif" alt="Installing SPECTROview with pip" width="500"><br>
      <i>Installing SPECTROview via pip.</i>
    </div>

    ??? tip "More Python commands (upgrade, specific version, development build)"

        - **Multiple Python versions installed?** Use `pip3 install spectroview` or `python -m pip install spectroview`.
        - **Upgrade to the latest release:** `pip install --upgrade spectroview`
        - **Install a specific version:** `pip install spectroview==26.46.1`
        - **Install the latest development code from GitHub:** `pip install git+https://github.com/CEA-MetroCarac/SPECTROview.git`
        - **Check installed version:** `pip show spectroview`
        - **Optional voice dictation (Python only):** `pip install SpeechRecognition pyaudio`

---

### **2. Using the Built-in AI Chat Agent**

The **AI Chat Agent** is already included — no extra installation is needed inside **SPECTROview**:

- **Cloud AI** *(OpenAI, Gemini, DeepSeek, Anthropic, etc.)*: enter your API key in **Settings → AI Settings**.
- **Private Local AI** *(runs offline on your own computer)*: install [Ollama](https://ollama.com) and start it before opening **SPECTROview**.

---

### **3. Automatic Updates**

**SPECTROview** checks for new versions in the background shortly after opening (or anytime when you click **Check for updates** in the top toolbar). When an update is ready, a bar appears at the top of the window:

<div align="center">
  <img src="../user_manual_images/Installation/update_banner.png" alt="Automatic update notification banner" width="820"><br>
  <i>Update banner shown when a newer version is available.</i>
</div>

- Click **Update and Restart** / **Download Update** to install or fetch the newest release.
- Click **Show changelog** to read what changed, **Update later** to dismiss for today, or **Skip this version** to stay on your current version until a future release.

| Your Version | What Happens When You Update |
|---|---|
| **Windows Installer** & **Python (`pip`)** | Downloads, checks the file automatically, updates **SPECTROview**, and restarts it for you. |
| **Linux AppImage** *(in a writable folder)* | Replaces itself safely and restarts **SPECTROview**. |
| **Windows Portable** & **macOS (`.dmg`)** | Downloads the new verified file and opens a simple prompt so you can extract or drag it into **Applications**. |
| **Ubuntu / Debian (`.deb`)** | Downloads the verified `.deb` file and lets you copy the install command in one click. |

!!! note "Offline friendly & private"
    No usage data or personal files are ever uploaded. If your computer is offline, startup update checks stay quiet so **SPECTROview** starts normally.

??? info "Security & corporate network details (for IT teams)"

    - **File integrity:** Every downloaded installer or package is checked against the official `SHA256SUMS.txt` checksum before anything runs.
    - **Trusted HTTPS connections (`GHSA-6q48-6c7w-9gw2`):** Network connections use your operating system's native certificate store first (supporting institutional/corporate CAs and custom `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` variables) and require a patched `certifi` bundle (`>= 2026.10.23`), refusing compromised eMudhra / emSign root CAs.
    - **Pre-release testing:** Optional release-candidate notifications can be toggled under **Settings → General → Application Updates**.

---

### **4. Common Questions & Troubleshooting**

??? question "macOS says “SPECTROview” cannot be opened (developer cannot be verified)"

    1. Open your **Applications** folder in **Finder**.
    2. Hold **Control** (or right-click) **`SPECTROview`** and choose **Open**.
    3. Click **Open** again on the popup dialog.

    *(Alternatively: open **System Settings → Privacy & Security**, scroll down, and click **Open Anyway**.)*

??? question "Windows: Should I download Setup.exe or Portable.zip?"

    Choose **`Setup.exe`** if you want Start Menu shortcuts and one-click automatic updates. Choose **`Portable.zip`** if you prefer no installation (or want to run from a USB drive) — make sure to **Extract All...** before double-clicking `SPECTROview.exe`.

??? question "AppImage does not open on Ubuntu / Debian"

    Minimal Ubuntu setups sometimes lack FUSE 2 or Qt X11 libraries. Run these once in a terminal:

    ```bash
    sudo apt update
    sudo apt install libfuse2t64 libxcb-cursor0 libxkbcommon-x11-0 libegl1 libgl1
    ```

    Or start the AppImage without FUSE: `./SPECTROview_*.AppImage --appimage-extract-and-run`.

??? question "Blank window or graphics crash when launching"

    Try launching with software graphics rendering before opening **SPECTROview**:

    - **Windows (PowerShell):** `$env:QT_QUICK_BACKEND="software"; spectroview`
    - **macOS / Linux:** `export QT_QUICK_BACKEND=software; spectroview`

??? question "Python errors (`pip: command not found`, permission error, or `spectroview: command not found`)"

    - If `pip` is not recognized, run `python -m pip install spectroview` (or `python3 -m pip install spectroview`).
    - If `spectroview` is not recognized after installing, launch with `python -m spectroview.main`.
    - Avoid system permission errors by creating a virtual environment first:

      ```bash
      python -m venv spectroview_env
      source spectroview_env/bin/activate    # macOS / Linux
      spectroview_env\Scripts\activate       # Windows
      pip install spectroview
      ```
