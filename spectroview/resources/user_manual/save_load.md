## Save & Load Workspace

SPECTROview utilizes dedicated native file formats to allow you to seamlessly pause, save, and resume your data analysis sessions without losing any configurations or fit results.

### Saving Your Work

To save your current progress, click the **Save** button located in the main application toolbar:

<div align="center">
  <img src="../user_manual_images/Save_Load/save_load_clear.png" alt="Save, Load, and Clear Buttons" width="120"><br>
</div>

The application will generate a specific file extension based on which workspace is currently active:

| Active Workspace | Native Extension | Stored Contents |
|-----------|-----------|----------|
| **Spectra** | `.spectra` | All loaded 1D spectra, custom baseline configurations, mathematical fit models, and aggregate fit results. |
| **Maps** | `.maps` | The raw 2D/wafer map data, instrument metadata, all associated spectra, and all point-by-point fit results. |
| **Graphs** | `.graphs` | All loaded mathematical datasets, customized plot configurations, and drawn annotations. |

### Loading Your Work

To resume a previous session, simply click the **Open** button in the main toolbar and select any previously saved `.spectra`, `.maps`, or `.graphs` file. SPECTROview will automatically detect the file format, reconstruct your dataset exactly as you left it, and instantly open the appropriate workspace tab.
