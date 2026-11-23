## **设置与偏好** { #settings-preferences }

您可以通过主工具栏中的齿轮图标打开 **设置** (Settings) 菜单（或按快捷键 `Ctrl + Shift + S`），也可以点击顶部菜单栏中的 **关于 SPECTROview** (About SPECTROview)，直接跳转至 **关于** (About) 部分。

<div align="center">
  <img src="../user_manual_images/Settings/settings.png" alt="设置面板" width="820"><br>
</div>

<br>

`Settings Panel` 采用现代化的双栏界面设计，支持实时搜索筛选：

- **左侧边栏**：
  - **搜索栏**：输入任意关键词（例如 `working folder`、`theme`、`language`、`tolerance`、`loss`、`mcp`、`python`、`citation`），即可即时过滤导航列表，并在页面中直接高亮匹配的选项且自动滚动定位至视图中。
  - **导航标签页**：
    1. **常规** (General) — SPECTROview 工作目录、界面主题、界面语言、便携式配置备份/恢复以及更新偏好设置。
    2. **拟合设置** (Fit Settings) — 优化器收敛参数、鲁棒损失函数、峰极性以及全局峰参数限制（配有专用曲线拟合图标）。
    3. **AI 设置** (AI Settings) — 大语言模型 (LLM) 服务商凭据、连接测试以及本地 MCP 服务器。
    4. **关于** — 应用程序版本、构建号、Python 运行环境、运行时依赖项版本、文档链接与论文引用信息。
- **右侧内容区**：显示当前所选标签页的全部配置选项。所有说明文本、环境详情及引用信息均支持鼠标选中，方便复制。
- **底部栏**：显示应用程序版本号，并提供标准的 **保存** (Save) 和 **取消** (Cancel) 按钮。

> **主题同步**：**界面主题** (Interface Theme) 既可以在 **常规** 标签页中选择（`Modern Dark`、`Modern Gray`、`Modern Light`、`Classic Dark` 或 `Classic Light`），也可以直接通过顶部菜单栏中的 **主题** (Theme) 按钮进行切换。两个选择器之间完全实时同步。

---

### **1. SPECTROview 工作目录与偏好设置** *(常规标签页)* { #1-spectroview-working-folder-preferences-general-tab }

指定一个统一的总根目录，用于集中存放 SPECTROview 的所有模板、峰位配置方案、配方、样式、AI 对话历史以及配置备份。设置完成后，系统将在该目录下自动创建六个子文件夹和一个便携式配置文件：

| 子文件夹 / 文件 | 内容说明 |
|------------------|----------|
| `fit_model/` | 自定义拟合模型（JSON）。`Fit Model Builder` 会扫描此文件夹并列出其中的所有模型，实现一键复用。 |
| `peak_profile/` | 自定义全局峰限制配置方案（JSON），按表征技术保存 `maxshift`、`minfwhm` 和 `maxfwhm` 预设。 |
| `plot_recipe/` | 绘图配方 (Plot Recipes) —— 为 `Graphs` 工作区保存的完整图表配置集合（数据绑定 + 样式）。 |
| `plot_style/` | 样式模板 (Style Templates) —— 仅保存外观样式，可应用于任意现有图表。 |
| `ai_chat_history/` | AI 对话记录日志（JSON）。对话内容会自动保存在此目录中。 |
| `template/` | 通用图表模板与工作区模板。 |
| `config.json` | 便携式配置备份。非机密应用程序设置会同步镜像到此 JSON 文件中。 |

- 点击 **浏览** (Browse) 选择目录，或直接输入路径。
- 点击 **打开** (Open) 即可一键在 **Windows 文件资源管理器** (Windows Explorer) 或 **macOS 访达** (macOS Finder) 中直接打开已配置的目录。

> **凭据安全**：在设置中输入的 AI 服务商 API 密钥会存储在当前用户专用的系统应用配置存储中（在 Windows、macOS 和 Linux 平台上均位于 `QSettings` 内部混淆命名空间下），并严格排除在 `config.json` 备份之外。这样既可避免频繁弹出系统的钥匙串 / KWallet 授权提示，也能防止您在共享或提交工作目录时意外泄露敏感凭据。
>
> **备份与电脑迁移**：如果您更换新电脑或希望进行备份，只需将这一个工作目录完整复制到新电脑即可。在新机器上打开 SPECTROview 的设置面板，将 **SPECTROview 工作目录** (SPECTROview Working Folder) 指向该目录，所有拟合模型、绘图配方、样式模板、AI 对话历史以及配置（`config.json`）都将自动恢复！
>
> 您也可以随时使用 **导出配置...** (Export Config...) 和 **恢复配置...** (Restore Config...) 按钮，将设置导出为独立的 JSON 文件或从中恢复配置。

#### **界面主题** { #interface-theme }

- **主题**：可在 **现代深色** (Modern Dark)、**现代中性灰** (Modern Gray)（默认）、**现代浅色** (Modern Light)、**经典深色** (Classic Dark) 或 **经典浅色** (Classic Light) 之间选择。选定后将立即更新当前界面主题，并与顶部菜单栏中的主题按钮保持完全同步。绘图画布主题独立于界面主题，默认为 **浅色模式** (Light Mode)；您可以随时通过光谱画布下方 `View Options` 工具栏中的 **图表主题** (Plot theme) 选择器将其切换为 **深色模式** (Dark Mode)（您的选择将在跨会话间自动保留）。

#### **界面语言** (Interface Language) { #interface-language }

- **语言** (Language)：可在 **English**（默认）、**Français**（法语）、**简体中文**、**Deutsch**（德语）和 **Русский**（俄语）之间选择。更改将**实时应用至整个应用程序**——包括菜单、工作区、视图、对话框、科学图表标签以及 AI 对话面板——无需重启软件。点击 **取消** 可将实时预览的语言更改回滚至之前保存的设置。所选语言通过 `QSettings` 和 `config.json` 在跨会话间持久保存，点击 **打开用户手册** (Open User Manual) 时将自动打开对应语言版本的文档（英文版为 `/user_manual/`，法文版为 `/fr/user_manual/`，德文版为 `/de/user_manual/`，俄文版为 `/ru/user_manual/`，简体中文版为 `/zh/user_manual/`）。

#### **应用程序更新** (Application Updates) { #application-updates }

- **应用启动时自动检查更新** (Check for updates automatically on application startup)：勾选后（默认开启），SPECTROview 会在启动时在后台定期检查 GitHub Releases（支持 HTTP `ETag` 缓存），并根据您的安装方式（`pip`、Windows 安装程序、便携版 ZIP、macOS DMG、Linux AppImage 或 `.deb`）提供相应的更新处理。
- **包含预览版 / 候选发布版更新** (Include pre-release / release candidate builds)：勾选后，检查更新时也会包含预发布标签（例如 `-rc.1` 或 `-beta`）。该选项默认关闭，以确保大多数用户仅接收正式稳定版本。

---

### **2. 拟合收敛参数** *(拟合设置标签页)* { #2-fit-convergence-parameters-fit-settings-tab }

这些参数用于控制 **向量化批量拟合（`VBF`）引擎**——即负责对光谱执行曲线拟合的数学优化器——的运行行为。调整这些数值会直接影响拟合速度、计算精度和收敛表现。

- **恢复默认** (Reset to Default)：点击标签页顶部的 **重置所有拟合设置** (Reset all fit settings to default) 按钮，即可立即将所有收敛参数及全局峰参数限制恢复为出厂默认值。

| 参数 | 配置键 | 默认值 | 取值范围 | 说明与行为 |
|------|--------|--------|----------|------------|
| **最大迭代次数** (Maximum Iterations) | `max_ite` | 200 | 10 – 10,000 | 每条光谱执行 Levenberg-Marquardt 优化的最大迭代次数。<br>• **较低（50–100）**：拟合速度更快，适合快速预览。<br>• **较高（200+）**：对复杂光谱可获得更佳的收敛效果。 |
| **自变量收敛容差** (x-tolerance) | `xtol` | 1×10⁻⁴ | 1×10⁻¹⁰ – 1.0 | 参数步长的相对容差。收敛需满足：<br> mean(|&delta;<i>p</i>| / |<i>p</i>|) &lt; `xtol` |
| **函数收敛容差** (f-tolerance) | `ftol` | 1×10⁻⁴ | 1×10⁻¹⁰ – 1.0 | 代价函数的相对容差。收敛需满足：<br> |cost<sub>n</sub> - cost<sub>n-1</sub>| / |cost<sub>n</sub>| &lt; `ftol` |
| **噪声阈值系数** (Noise Threshold Coefficient) | `coef_noise` | 0.0 | 0.0 – 10.0 | 自动估算噪声水平的乘数系数（`coef_noise` × 估算噪声幅值），用于屏蔽高噪声数据并抑制鬼峰。<br>• **0.0**：禁用（默认）<br>• **0.5–1.0**：保守<br>• **1.0–2.0**：适中<br>• **3.0+**：激进<br><br>*说明：您可以参考工具栏中显示的噪声水平，或在 `SpectraViewer` 的 `View Options` 中勾选“**显示噪声水平** (Show noise level)”复选框，直接在光谱图上查看噪声水平。* |
| **损失函数** (Loss Function) | `loss` | `linear` | `linear`, `soft_l1`, `huber` | 用于提高抗异常值（例如宇宙射线尖峰、尖锐伪影）能力的鲁棒损失函数。<br>• **linear**：标准最小二乘法（默认）。<br>• **soft_l1**：平滑 L1 近似 $\rho(z) = 2(\sqrt{1+z}-1)$，无需手动掩膜即可有效抵抗大幅度尖峰。<br>• **huber**：Huber 损失函数，对内点残差呈二次型增长，对超出 `f_scale` 的残差呈线性增长。 |
| **损失尺度** (Loss Scale) | `f_scale` | 1.0 | 0.01 – 100.0 | 鲁棒损失函数中划分内点与异常值的边际尺度。在计算 $\rho'(z)$ 之前，残差将按 $(r / f\_scale)$ 进行缩放。 |
| **拟合负值** (Fit Negative Values) | `fit_negative` | 未勾选 | - | 勾选后，负强度值（吸收峰或倒置峰）将参与拟合；取消勾选时，负强度数据点的权重将被设为零。 |

> **注意**：仅当 `xtol` 和 `ftol` 两个条件**同时**满足时，光谱才被判定为已收敛。
>
> **权重叠加顺序**：系统首先评估硬性排除条件（`fit_negative` 和 `coef_noise` 噪声底限掩膜），为被排除的数据点赋予零权重；随后再对其余有效数据点应用鲁棒损失加权（`soft_l1` 或 `huber`），通过迭代重加权最小二乘法 (IRLS) 平滑降低异常值残差的权重。

---

### **3. 全局峰参数限制** *(拟合设置标签页)* { #3-global-peak-limits-fit-settings-tab }

这些参数定义了全局应用于整个应用程序中所有峰的**绝对边界条件**。当您向拟合模型添加新峰时，系统会自动使用这些限制来初始化其参数上下界。您也可以在 `Fit Model Builder` 的 `Peaks` 分组中点击齿轮按钮直接跳转到此区域。

- **峰位配置方案** (Peak profile:)：可在内置表征技术预设（**Raman (cm⁻¹)**、**PL (eV)**、**PL (nm)**、**XRD (°)**）或您自己保存的配置方案之间快速切换。
- **保存 / 删除配置方案**：点击**峰位配置方案**下拉框旁的保存图标，即可将当前 `Max peak shift`、`Min peak FWHM` 和 `Max peak FWHM` 以自定义名称保存到 `<工作目录>/peak_profile/` 中；选择自定义配置方案后点击垃圾桶图标即可删除。手动修改任意限值后，下拉框会自动切换为**自定义** (Custom)。

> **注意**：您可以勾选 `PeakTable` 下方的“**限制范围** (Limits)”复选框，以显示每个峰的实际约束边界。

| 限制参数 | 默认值 | 取值范围 | 说明与约束条件 |
|----------|--------|----------|----------------|
| **最大峰位偏移** (Max Peak Shift) | 20 | 0 – 10,000 | 允许峰中心位置（`x0`）相对于其初始值的最大位移量。<br><i>x</i><sub>0</sub> - maxshift &le; <i>x</i><sub>0</sub><sup>fitted</sup> &le; <i>x</i><sub>0</sub> + maxshift |
| **最小峰半高宽** (Min Peak FWHM) | 0.1 | 0.0001 – 10,000 | 允许的最小半高全宽（`FWHM`），防止峰退化坍缩为脉冲状伪影（δ 函数）。 |
| **最大峰半高宽** (Max Peak FWHM) | 200 | 0.001 – 10,000 | 允许的最大半高全宽（`FWHM`），防止峰过度展宽为平坦的基线。 |

---

### **4. AI 设置** *(AI 设置标签页)* { #4-ai-settings-ai-settings-tab }

**AI 设置**标签页用于配置 [AI 智能体](ai_agent.md)。此处的配置并非必需，不设置也不影响 SPECTROview 其他功能的正常使用。

#### **API 密钥** { #api-keys }

默认情况下会显示**自定义** (Custom)（OpenAI 兼容）端点的配置字段：

| 字段 | 说明 |
|------|------|
| **自定义 API 密钥** (Custom API Key) | 自定义或内部服务商端点的 API 密钥。 |
| **基础 URL** (Base URL) | 该端点兼容 OpenAI 接口的根 URL（例如 `https://host/v1`）。 |
| **模型名称** (Model Name) | 以逗号分隔的模型名称列表（例如 `model-a, model-b`）。这些名称将填充至对话面板的模型下拉菜单中——对于未提供模型列表查询接口的端点尤为实用。 |
| **显示标签** (Display Label) | 在 AI 对话输入框中为该自定义端点显示的名称标签。 |

内置云端服务商的密钥收起在可折叠的 **▸ 其他云端服务商** (Other cloud providers) 区域中（点击即可展开）：**OpenAI**、**Anthropic**、**Gemini**、**DeepSeek** 和 **Mistral**。所有密钥均以掩膜形式存储，并在会话之间持久保留。

- **测试连接** (Test Connection)：从下拉列表中选择任意服务商，然后点击 **测试连接**，即可在不保存配置的情况下验证 API 端点的可用性、凭据有效性以及网络连通性。

> **密钥存储位置**：API 密钥保存在操作系统的当前用户设置存储中（在 `_credentials/` 下经混淆处理）——Windows 系统为注册表
> （`HKEY_CURRENT_USER\Software\CEA-Leti\SPECTROview`），
> macOS 系统为 `~/Library/Preferences`，Linux 系统为 `~/.config/CEA-Leti/SPECTROview.conf`。
> 它们**绝不会**写入 SPECTROview 项目目录中的任何文件，
> 也不会包含在导出的工作区或设置备份中，从而彻底避免被意外提交至 Git 仓库。

#### **本地 MCP 服务器** (Local MCP Server) { #local-mcp-server }

仅当同一台计算机上受信任的 MCP 客户端需要访问当前正在运行的 SPECTROview 会话时，才应启用此选项。默认端点为 `http://127.0.0.1:8765/mcp`，点击 **保存** 后即刻启动。

| 字段 | 默认值 | 说明 |
|------|--------|------|
| **启用本地 MCP 端点** (Enable local MCP endpoint) | 关闭 | 启动或停止内置 MCP 服务器。 |
| **端口** (Port) | 8765 | 本地流式 HTTP（Streamable HTTP）端口（1024–65535）。若该端口已被占用，请更换其他端口。 |

绑定地址固定为本地回环地址（`127.0.0.1`），不会暴露给局域网或互联网。客户端连接期间，请保持 SPECTROview 处于运行状态。

---

### **5. 关于与运行环境** *(关于标签页)* { #5-about-environment-about-tab }

**关于**标签页提供了全面的运行时信息与参考详情：

- **应用程序详情**：显示 SPECTROview 官方徽标、标题、版本号（`Version {VERSION}`）以及构建号（`Build: {BUILD_NUMBER}`）。
- **运行环境与依赖项** (Environment & Dependencies)：自动检测当前运行与打包模式（`Python Package (.whl / local)`、`Windows Installer (Velopack)`、`Windows Portable`、`macOS Application (.app)`、`Linux AppImage` 或 `Linux Debian Package (.deb)`）并显示各依赖项的版本：

  - **独立打包版**：显示打包时记录的冻结依赖项快照，以及内置 Python 和宿主机/构建系统的详细信息。
  - **已安装软件包（`.whl` / `pip` / 开发环境）**：动态检测当前本地 Python 环境中实际加载的实时软件包版本。
  - 列出 `PySide6`（含 Qt 运行时版本）、TLS 与安全组件（`certifi`、`truststore`）、科学计算栈（`numpy`、`scipy`、`matplotlib`、`pandas`、`pybaselines`）、导出引擎（`openpyxl`、`pyarrow`）、UI 扩展组件（`superqt`）以及 AI 对话 SDK。
- **检查更新** (Check for Updates)：直接触发检查 GitHub Releases 上是否有新版本发布。
- **论文引用**：提供 *SoftwareX* 同行评审论文的标准引用格式及 DOI，并附带 **复制** (Copy) 按钮，方便一键将引用信息复制到剪贴板。
