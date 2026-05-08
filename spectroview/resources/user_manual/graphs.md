## Graphs Workspace

The Graphs Workspace is exclusively dedicated to data visualization, engineered with a strong emphasis on simplicity, speed, and customization.

<div align="center">
  <img src="../user_manual_images/Graphs/ui_graph.gif" alt="ui_graph" width="800"><br>
  <i>Overview of the Graphs Workspace's UI.</i>
</div>
<br>

_______


### Loading Data

Datasets can be passed seamlessly from the Spectra and Maps workspaces, or imported directly from external Excel/CSV files. All available datasets are dynamically tracked and displayed in the dataset list widget.

![Dataset List](../user_manual_images/Graphs/df_list.png)<br>
Available four utility buttons include: 

- **View** : inspect the data table, 
- **Delete** : remove the dataset, 
- **Save** : export the dataset, 
- **Refresh** : dynamically reload the CSV/Excel file if it has been modified externally.

________


### Add a new plot or update existing plot:

1. Select your target dataset from the list.
2. Choose the appropriate columns for the X, Y, and Z axes using the provided dropdown menus.
3. Select your desired plot style (available styles: scatter, point, bar, box, line, 2Dmap, wafer).
4. Define your plot labels, axis limits, and wafer diameter dimensions (if applicable).
5. Click **Add Plot** to generate the visualization.

<div align="center">
  <img src="../user_manual_images/Graphs/add_plot.gif" alt="add_plot" width="800"><br>
  <i>Adding a new plot or updating existing plot.</i>
</div>


______


### Modifying an existing plot :

Modify the **labels** of axes, **title** of the plot can be done via the Right Panel as follows. Click to the "Update Plot" button to apply the changes:

<div align="center">
  <img src="../user_manual_images/Graphs/labels.png" alt="labels" width="300"><br>
  <i>Customize selected plot via the Right Panel. Click to Update plot button to apply the changes.</i>
</div>

<br>

Click **Customize** button to open the Customize Dialog, giving you deep control over the Legend, Annotations, Axes, and General aesthetics.

<div align="center">
  <img src="../user_manual_images/Graphs/adjust_axis_limits.gif" alt="adjust_axis_limits" width="600"><br>
  <i>Adjusting axis limits of the selected plot.</i>
</div>

<div align="center">
  <img src="../user_manual_images/Graphs/adjust_legends.gif" alt="adjust_legends" width="600"><br>
  <i>Adjusting legends/colors of the selected plot.</i>
</div>
<br>

<div align="center">
  <img src="../user_manual_images/Graphs/adjust_annotation.gif" alt="adjust_annotation" width="600"><br>
  <i>Adjusting position and appearance of annotations on the selected plot.</i>
</div>
<br>

_____

### Data Filtering

You can dynamically filter the plotted data by applying boolean logic expressions in the **Filter** field using the format: `(column_name) (operator) (value)`.
> **Note**: String values must be enclosed in double quotes (`"text"`). Column headers containing spaces must be enclosed in backticks (`` `column name` ``).

<div align="center">
  <img src="../user_manual_images/Graphs/data_filter.png" alt="data_filter" width="350"><br>
  <i>An example demonstrating how to apply filters to a dataset.</i>
</div>
<br>

| Filter Expression | Resulting Behavior |
|-------------------|---------|
| `Confocal != "high"` | Excludes all data points where the "Confocal" column equals "high". |
| `Thickness == "1ML" or Thickness == "3ML"` | Includes only the data points where the "Thickness" column equals exactly "1ML" or "3ML". |
| `` `Laser Power` <= 5 `` | Includes data points where the "Laser Power" column is strictly less than or equal to 5. |

