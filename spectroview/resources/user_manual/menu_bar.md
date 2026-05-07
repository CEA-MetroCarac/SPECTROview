## Menubar

A horizontal toolbar located at the top of the application provides quick access to essential features:

| Button | Function |
|--------|----------|
| ![Open](../user_manual_images/UI_Overview/open.png) | **Open**: Loads any supported data type. The application will automatically detect the file format and switch to the appropriate workspace. |
| ![Save](../user_manual_images/UI_Overview/save.png) | **Save**: Saves the active workspace to a dedicated file (`.maps`, `.spectra`, or `.graphs`), allowing you to effortlessly pause and resume your work later. |
| ![Clear](../user_manual_images/UI_Overview/clear.png) | **Clear**: Clears the currently active workspace, removing all loaded data to start a fresh session. |
| ![Convert](../user_manual_images/UI_Overview/convert.png) | To open **Hyperspectral Data Converter Tool** -> see bellow for more detail.|
| ![Calculation](../user_manual_images/UI_Overview/quick_calculation.png) | To Open **Quick Calculation Tool** -> see bellow for more detail. |
| ![Settings](../user_manual_images/UI_Overview/settings.png) | To open **Setting Panel** -> see bellow for more detail. |
| ![Theme](../user_manual_images/UI_Overview/theme.png) | **Theme Toggle**: Instantly switches the application interface between Dark and Light modes. |
| ![User Manual](../user_manual_images/UI_Overview/manual.png) | **User Manual**: Opens the integrated User Manual documentation viewer. |
| ![About](../user_manual_images/UI_Overview/about.png) | **About**: Displays version information, licensing, and details about the SPECTROview application. |


### Settings Panel
Opens a Settings Panel by clicking to icon "Settings" as described above. 

To convert text-exported hyperspectral data (2D maps) from Renishaw WiRE into a format natively supported by SPECTROview. Simply load your file(s) and click "Convert" to generate a new file with a `_converted` suffix.

![Hyperspectral Data Converter Tool](../user_manual_images/UI_Overview/ui_settings.png)

### Hyperspectral Data Converter Tool
Opens a utility by clicking to icon "Convert" as described above. 

To convert text-exported hyperspectral data (2D maps) from Renishaw WiRE into a format natively supported by SPECTROview. Simply load your file(s) and click "Convert" to generate a new file with a `_converted` suffix.

![Hyperspectral Data Converter Tool](../user_manual_images/UI_Overview/ui_converter.png)

### Quick Calculation Tool

Click to the "Calculator" icon to open  A suite suite of utility calculators : 

![Quick Calculation Tool](../user_manual_images/UI_Overview/ui_quick_calculation.png)

#### Calculation of laser spot size, depth of field and laser power density
This tool estimates the theoretical, diffraction-limited spatial resolution of your optical setup, including spot size, depth of focus, and laser power density.

- **Spot Size**: Calculated as $1.22 \times \lambda / \text{NA}$ ($\mu\text{m}$)
- **Depth of Focus**: Calculated as $4 \times n \times \lambda / \text{NA}^2$ ($\mu\text{m}$)

#### Calculation of penetration depth from absorption coefficient and laser wavelength.
This tool calculates the theoretical optical penetration depth of a laser into a specific material, derived from its complex refractive index (extinction coefficient).

- **Penetration Depth ($d$)**: Calculated as $\lambda / (4 \pi k)$ ($\text{nm}$)

#### Unit Converter
A rapid utility to convert between various standard spectroscopic units, ensuring flawless consistency during data analysis (e.g., converting wavelength in $\text{nm}$ to energy in $\text{eV}$, or calculating Raman shift in $\text{cm}^{-1}$ from excitation wavelengths).


