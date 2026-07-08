## **Workspace: Graphs**

The `Graphs` workspace is exclusively dedicated to data visualization, engineered with a strong emphasis on simplicity, speed, and customization.

<div align="center">
  <img src="../user_manual_images/Graphs/ui_graph.gif" alt="Graphs Workspace UI" width="900"><br>
  <i>Overview of the Graphs Workspace interface.</i>
</div>
<br>

_______


### **1. Loading Data**

Datasets can be passed seamlessly from the `Spectra` and `Maps` workspaces, or imported directly from external Excel/CSV files. All available datasets are dynamically tracked and displayed in the dataset list widget.

![Dataset List](../user_manual_images/Graphs/df_list.png)<br>
The four available utility buttons allow you to:

- **View**: Inspect the data table natively.
- **Delete**: Remove the dataset from the workspace.
- **Save**: Export the dataset to Excel or CSV.
- **Refresh**: Dynamically reload the CSV/Excel file if it has been modified externally.

________


### **2. Add or Update a Plot**

1. Select your target dataset from the list.
2. Choose the appropriate columns for the X, Y, and Z axes using the provided dropdown menus.
3. Select your desired plot style (available styles: `scatter`, `point`, `bar`, `box`, `line`, `trendline`, `histogram`, `2Dmap`, `wafer`).
4. Define your plot labels, axis limits, and wafer diameter dimensions (if applicable).
5. Click **Add Plot** to generate the visualization.

<div align="center">
  <img src="../user_manual_images/Graphs/add_plot.gif" alt="Adding a Plot" width="800"><br>
  <i>Demonstration of adding a new plot or updating an existing one.</i>
</div>


______


### **3. Modifying an Existing Plot**

You can modify the **labels** of the axes and the **title** of the plot via the `Right Panel`. Click the **Update Plot** button to apply the changes instantly:

<div align="center">
  <img src="../user_manual_images/Graphs/labels.png" alt="Label Customization" width="300"><br>
  <i>Customize the selected plot via the Right Panel. Click the Update Plot button to apply changes.</i>
</div>

<br>

Click the **Customize** button to open the `Customize Dialog`, giving you deep, granular control over the Legend, Annotations, Axes, and general aesthetics.

<div align="center">
  <img src="../user_manual_images/Graphs/adjust_axis_limits.gif" alt="Adjusting Axis Limits" width="600"><br>
  <i>Adjusting the axis limits of the selected plot.</i>
</div>

<div align="center">
  <img src="../user_manual_images/Graphs/adjust_legends.gif" alt="Adjusting Legends" width="600"><br>
  <i>Adjusting legends and colors of the selected plot.</i>
</div>
<br>

<div align="center">
  <img src="../user_manual_images/Graphs/adjust_annotation.gif" alt="Adjusting Annotations" width="600"><br>
  <i>Adjusting the position and appearance of annotations on the selected plot.</i>
</div>
<br>

_____

### **4. Data Filtering**

You can dynamically filter the plotted data by applying boolean logic expressions in the **Filter** field using the format: `(column_name) (operator) (value)`.
> **Note**: String values must be enclosed in double quotes (`"text"`). Column headers containing spaces must be enclosed in backticks (`` `column name` ``).

<div align="center">
  <img src="../user_manual_images/Graphs/data_filter.png" alt="Data Filter" width="350"><br>
  <i>An example demonstrating how to apply filters to a dataset dynamically.</i>
</div>
<br>

| Filter Expression | Resulting Behavior |
|-------------------|---------| 
| `Confocal != "high"` | Excludes all data points where the "Confocal" column equals "high". |
| `Thickness == "1ML" or Thickness == "3ML"` | Includes only the data points where the "Thickness" column equals exactly "1ML" or "3ML". |
| `` `Laser Power` <= 5 `` | Includes data points where the "Laser Power" column is less than or equal to 5. |

_____

### **5. Plot-Style-Specific Customization**

The **Customize Dialog** (opened via the ⚙ button on any graph) contains a **"More Options"** tab that **adapts dynamically** to the active plot style, showing only the controls relevant to the current graph.

#### Trendline Plot

The `trendline` style offers powerful fitting controls:

| Option | Description |
|---|---|
| **Polynomial order** | Set the degree of the polynomial fit (1 = linear, 2 = quadratic, …, up to 10). |
| **Anchor point** | Constrain the fit to pass through a specific point. Choose **Origin (0, 0)** or enter a **custom (X₀, Y₀)**. |
| **Hue / Z axis** | When a Z column is selected, a separate trendline is fitted for each category group, each drawn in a distinct color. |
| **Equation table** | The fit results (equation and R² per group) are displayed in a table inside the dialog. Click **Copy** to export to the clipboard as tab-separated text (paste directly into Excel). |


#### Histogram Plot

The `histogram` style provides:

| Option | Description |
|---|---|
| **Bins** | Number of histogram bins (2–500, default 20). |
| **KDE overlay** | Overlay a smooth Kernel Density Estimate curve on the histogram. |
| **Fill style** | Choose **Filled** bars (default) or **Step** (outline only). |
| **Hue / Z axis** | When a Z column is selected, overlapping histograms are drawn per category with distinct colors. |

#### Other Plot Styles

| Plot Style | Available in More Options |
|---|---|
| `point` | Join data points toggle |
| `bar` | Error bar toggle |
| `wafer` | Statistics overlay toggle |

_____

### **6. Statistical Calculations & Fitting Details**

For statistical plot styles (`point`, `bar`, `box`, and `trendline`), SPECTROview automatically calculates essential metrics and visualizes the distribution of your data.

#### **Point and Bar Plots**

- **Central Value (Mean):** The primary data point (the circle marker in a `point` plot or the top of the bar in a `bar` plot) represents the **arithmetic mean** of all Y-values for a given X-category.
- **Error Bars (95% CI):** The error bars extending from the mean represent the **95% Confidence Interval** of the mean. This is calculated using the Standard Error of the Mean (SEM):
  `Error Bar = ± (1.96 × SEM)`
  where `SEM = Standard Deviation / sqrt(N)`.

#### **Box Plot**

The `box` plot natively displays the statistical distribution of the dataset based on standard five-number summaries:
- **Box Bounds (Q1 to Q3):** The main body of the box spans from the first quartile (25th percentile) to the third quartile (75th percentile), known as the Interquartile Range (IQR).
- **Median Line:** The horizontal line inside the box marks the **Median** (50th percentile).
- **Whiskers:** The whiskers extend to the furthest data points that are still within `1.5 × IQR` of the box edges.
- **Fliers (Outliers):** Any data points falling beyond the whiskers are explicitly drawn as individual outlier points.

#### **Trendline Estimation**

- **Standard Fit (No Anchor):** The trendline is calculated using an Ordinary Least Squares (OLS) linear regression to fit a polynomial of the specified degree:
  `y = p₀·xᵈ + p₁·xᵈ⁻¹ + ... + p_d`
  where `d` is the polynomial order.

- **Anchored Fit (Forced through a point (x₀, y₀)):** The coordinate system is shifted so that the anchor point becomes the origin (`x' = x - x₀`, `y' = y - y₀`). A polynomial without a constant term (zero intercept) is then fitted to the shifted variables:
  - **For Linear order (1):** The slope `m` is directly calculated as:
    `m = sum(x'_i * y'_i) / sum((x'_i)²)`
  - **For Higher orders (>1):** The shifted data is fitted with a zero intercept using a linear least squares solver.
  The fitted curve is then translated back to the original coordinate space.

#### **Trendline Confidence Intervals**

- **Confidence Level:** The shaded band around standard trendlines represents a **95% confidence interval** (estimated by Seaborn's regression plot module).
- **Estimation Method:** The confidence interval is estimated using a non-parametric bootstrapping procedure (resampling data points with replacement 1000 times, fitting a regression model to each bootstrap sample, and calculating the 2.5th and 97.5th percentiles of the predictions).
- **Anchored Fits:** Please note that confidence intervals are only computed and displayed for standard, unconstrained fits. If an anchor is enabled, the confidence interval band is omitted.
