## 6. Graphs Workspace

The Graphs Workspace is dedicated to data visualization, with an emphasis on simplicity and speed.

![Graphs Workspace](../user_manual_images/imageFile67.png)
*Figure 14: Graphs Workspace's interface. Control Panel on the right, Graph Viewer on the left.*

### 6.1 Loading Data

Datasets can be sent directly from other workspaces or loaded from Excel/CSV files. All loaded datasets are displayed in a list widget.
Buttons available: **View**, **Delete**, **Save**, **Refresh** (reloads CSV/Excel file if modified externally).

![Dataset List](../user_manual_images/imageFile69.png)
*Figure 15: List of loaded dataframe(s).*

### 6.2 Add a New Plot

1. Select dataset.
2. Select X, Y, Z columns from the dropdown menus.
3. Select plot style (scatter, point, bar, box, line, 2Dmap, wafer).
4. Specify labels, limits, and wafer size (if applicable).
5. Click **Add plot**.

![Add New Plot](../user_manual_images/imageFile70.png)

### 6.3 Modify Existing Plot

![Plot Widget](../user_manual_images/imageFile71.png)
![Plot Customize](../user_manual_images/imageFile72.png)
*Figure 16: Example of a plot widget. Click "Customize" to open the Customize Dialog.*

![Toolbar Buttons](../user_manual_images/imageFile73.png)

- Click **Customize** to open the Customize Dialog (Legend, Annotation, Axis, General).
- **Double-click** the legend box or annotation to edit its content directly.
- Adjust settings via the right ControlPanel and click **Update**.
![Update Button](../user_manual_images/imageFile75.png)

- Click **Copy** to copy the figure to the clipboard.
![Copy Button](../user_manual_images/imageFile76.png)

### 6.4 Data Filtering

Filter rows using a boolean expression in the **Filter** field: `(column_name) (operator) (value)`
String values must be enclosed in double quotes. Column headers with spaces must be enclosed in backticks (`` ` ``).

![Filter Example](../user_manual_images/imageFile77.png)
*Figure 17: Example of using filters.*

| Filter Expression | Meaning |
|-------------------|---------|
| `Confocal != "high"` | Values in "Confocal" not equal to "high" |
| `Thickness == "1ML" or Thickness == "3ML"` | "Thickness" equals 1ML or 3ML |
| `` `Laser Power` <= 5 `` | "Laser Power" is less than or equal to 5 |

### 6.5 Annotation Features

Customize annotations (lines and text) in the **MoreOptions** tab or by clicking directly on the plot.

![MoreOptions](../user_manual_images/imageFile79.png)
*Figure 18: MoreOptions Panel*

![Annotation Panel](../user_manual_images/imageFile80.png)
*Figure 19: Annotation (line & text) Customization Panel.*
