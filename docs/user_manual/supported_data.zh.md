## **支持的数据格式** { #supported-data-formats }

GitHub 仓库的 [`/examples`](https://github.com/CEA-MetroCarac/SPECTROview/tree/main/examples) 文件夹中提供了所有受支持数据类型的示例文件。用户可以下载这些示例文件，熟悉受支持的文件格式与数据结构，并借助真实数据练习使用 **SPECTROview**。

| 格式 | 扩展名 | 加载器 | 数据类型 |
|---|---|---|---|
| 分隔文本 | `.txt`, `.csv` | 自动检测分隔符与表头 | 单条光谱、二维图谱、晶圆图谱及时间分辨图谱 |
| Renishaw WiRE | `.wdf` | 内置二进制读取器 | 单条光谱、时间序列、线扫描剖面及二维图谱 |
| HORIBA LabSpec 6 | `.l6s`, `.l6m` | 内置二进制读取器 | 单条光谱 (`.l6s`) 与二维高光谱图谱 (`.l6m`) |
| HORIBA LabSpec 5 | `.ngs`, `.ngc` | 内置二进制读取器 | 单条光谱 (`.ngs`) 与压缩二维高光谱图谱 (`.ngc`) |
| SPC | `.spc` | 内置二进制读取器 | 单条光谱与二维图谱 |
| TRPL | `.dat` | 内置解析器 | 时间分辨光致发光衰减曲线 |
| 表格数据 | `.xlsx`, `.xls`, `.csv` | `Pandas` & `Openpyxl` | 用于绘图的 DataFrame |
| 拟合模型模板 | `.json` | **拟合模型构建器** (Fit Model Builder) | 峰模型与基线配置 |

### **1. `.wdf` (Renishaw)、`.spc`、`.l6s` / `.l6m` (HORIBA LabSpec 6) 与 `.ngs` / `.ngc` (HORIBA LabSpec 5) 格式** { #1-wdf-renishaw-spc-l6s-l6m-horiba-labspec-6-and-ngs-ngc-horiba-labspec-5-formats }

**SPECTROview** 原生支持多种光谱仪二进制格式，无需借助外部仪器厂商软件：

- **`.wdf`** — 使用 Renishaw 仪器配套 WiRE 软件采集的数据（离散光谱、时间/重复性序列、一维线扫描剖面以及二维高光谱图谱）。
- **`.l6s` 与 `.l6m`** — 使用 HORIBA LabSpec 6 软件采集的原生二进制光谱（`.l6s`）和二维光谱图像/图谱（`.l6m`），同时支持未压缩和压缩格式。
- **`.ngs` 与 `.ngc`** — 使用 HORIBA LabSpec 5 软件采集的原生二进制光谱（`.ngs`）和压缩二维光谱图像/图谱（`.ngc`）。
- **`.spc`** — 在 HORIBA 和 Thermo 仪器上使用 LabSpec 6 / T64000 / GRAMS 软件采集的数据（离散拉曼/PL 光谱及二维高光谱图谱）。

无论是离散光谱还是高光谱数据集，在打开或拖入主窗口时均会自动分配至对应工作区，无需预先进行格式转换。**SPECTROview** 还会解析并导入仪器**元数据**（激发波长、光栅、物镜、激光功率、曝光时间、累加次数、狭缝/共聚焦针孔尺寸、光谱单位、空间步长、采集时间戳，以及探测器型号、探测器温度、SWIFT 模式和信噪比等其他仪器参数）。

### **2. 光谱数据 (`.txt`, `.csv`)** { #2-spectroscopic-data-txt-csv }

**SPECTROview** 支持 TXT 或 CSV 格式的一维光谱数据。文件必须包含**两列数值数据**，列间可通过分号（`;`）、逗号（`,`）、制表符或空格分隔。文件开头的注释行（`#`）与列标题均为可选，软件会自动检测识别，从而确保无表头文件不会丢失第一个数据点：

- **第 1 列**：X 轴数值（例如，拉曼位移（cm⁻¹）、波长（nm）、2θ（度））。若 X 轴为降序排列，系统会自动按升序重新排序。
- **第 2 列**：对应的强度值（a.u.）。

### **3. 高光谱数据 (`.txt`, `.csv`)** { #3-hyperspectral-data-txt-csv }

**SPECTROview** 支持 TXT 或 CSV 格式的高光谱数据（二维载物台图谱、晶圆图谱以及多扫描时间分辨晶圆图谱），数据结构如下所示：

| X    | Y    | R1 | R2 | R3 | ... | Rn |
|------|------|----|----|----|-----|----|
| **X1** | **Y1** | i1 | i1 | i1 | ... | i1 |
| **X2** | **Y2** | i2 | i2 | i2 | ... | i2 |
| **...** | **...** | ... | ... | ... | ... | ... |
| **Xn** | **Yn** | in | in | in | ... | in |

- **第一行**（R1 → Rn）定义所有光谱共享的 X 轴数值（例如拉曼位移，单位 cm⁻¹）。
- **前两列**定义图谱中每条光谱的 X 和 Y 空间坐标。
- **其余各列**从第二行至最后一行包含每条光谱对应的强度值（I1 → In）。

> **注意**：原生的 Renishaw `.wdf` 图谱文件可直接加载到 **SPECTROview** 中，无需格式转换。从 Renishaw WiRE 导出的长表文本格式二维图谱（包含 `#X`、`#Y`、`#Wave`、`#Intensity` 列），可通过 `spectroview.api.io.convert_renishaw_map(input_path, output_path)` 转换为宽矩阵 `.txt` 格式。

### **4. 数据表 (Excel 或 CSV)** { #4-datasheet-excel-or-csv }

包含单个或多个工作表的 Excel 文件（`.xlsx`、`.xls`）以及 CSV 文件，均可直接导入至 **图表** (Graphs) 工作区，立即用于数据可视化与绘图。

### **5. SPECTROview 原生格式** { #5-native-spectroview-formats }

为便于保存分析流程并在之后恢复工作，**SPECTROview** 针对当前活动工作区使用特定的文件扩展名：

- **二维图谱** (Maps) 工作区 → `.maps`
- **光谱** (Spectra) 工作区 → `.spectra`
- **图表**工作区 → `.graphs`
- **拟合模型模板** → `.json`（拖放或点击 **加载** (Load) 导入至光谱与二维图谱工作区）
