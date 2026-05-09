## Menu Bar

### Menu Bar Buttons
A horizontal toolbar located at the top of the application provides quick access to essential features:

| Button | Function |
|--------|----------|
| ![Open](../user_manual_images/menu_bar/open.png) | **Open**: Loads any supported data type. The application will automatically detect the file format and switch to the appropriate workspace. |
| ![Save](../user_manual_images/menu_bar/save.png) | **Save**: Saves the active workspace to a dedicated file (`.maps`, `.spectra`, or `.graphs`), allowing you to pause and resume your work effortlessly. |
| ![Clear](../user_manual_images/menu_bar/clear.png) | **Clear**: Clears the currently active workspace, removing all loaded data to start a fresh session. |
| ![Convert](../user_manual_images/menu_bar/convert.png) | **Convert**: Opens the Hyperspectral Data Converter Tool (see below for details). |
| ![Calculation](../user_manual_images/menu_bar/quick_calculation.png) | **Calculators**: Opens the Quick Calculation Tool suite (see below for details). |
| ![Settings](../user_manual_images/menu_bar/settings.png) | **Settings**: Opens the Settings Panel (see below for details). |
| ![Theme](../user_manual_images/menu_bar/theme.png) | **Theme Toggle**: Instantly switches the application interface between Dark and Light modes. |
| ![User Manual](../user_manual_images/menu_bar/manual.png) | **User Manual**: Opens the integrated User Manual documentation viewer. |
| ![About](../user_manual_images/menu_bar/about.png) | **About**: Displays version information, licensing, and details about the SPECTROview application. |


### Settings Panel
Click the **Settings** icon to open the Settings Panel, where you can:

- Adjust global fitting parameters.
- Define a default folder where all fit models (JSON format) are stored. The application will automatically scan this folder and load all models for easy selection in the fitting interface.

<div align="center">
  <img src="../user_manual_images/menu_bar/ui_settings.png" alt="Settings Panel UI" width="350"><br>
   <i>The Settings Panel interface.</i>
</div>


### Hyperspectral Data Converter Tool
Click the **Convert** icon to open this utility.

To convert hyperspectral data (2D maps) from Renishaw WiRE into a format natively supported by SPECTROview, load your file(s) and click **Convert**. The converted file will be saved with a `_converted` suffix in the same directory as the original file.

<div align="center">
  <img src="../user_manual_images/menu_bar/ui_converter.png" alt="Converter Tool UI" width="500"><br>
   <i>The Hyperspectral Data Converter Tool interface.</i>
</div>

### Quick Calculation Tool
Click the **Calculator** icon to open a suite of utility calculators:

<div align="center">
  <img src="../user_manual_images/menu_bar/ui_quick_calculation.png" alt="Quick Calculators UI" width="800"><br>
   <i>The Quick Calculators interface.</i>
</div>

#### 1. Laser Spot Size, Depth of Field, and Laser Power Density
This tool estimates the theoretical, diffraction-limited spatial resolution of your optical setup, including the spot size, depth of focus, and laser power density.

- **Spot Size**: Calculated as $1.22 \times \lambda / \text{NA}$ ($\mu\text{m}$)
- **Depth of Focus**: Calculated as $4 \times n \times \lambda / \text{NA}^2$ ($\mu\text{m}$)

#### 2. Penetration Depth
This tool calculates the theoretical optical penetration depth of a laser into a specific material, derived from its complex refractive index (extinction coefficient).

- **Penetration Depth ($d$)**: Calculated as $\lambda / (4 \pi k)$ ($\text{nm}$)

#### 3. Unit Converter
A rapid utility to convert between various standard spectroscopic units, ensuring flawless consistency during data analysis (e.g., converting wavelength in $\text{nm}$ to energy in $\text{eV}$, or calculating Raman shift in $\text{cm}^{-1}$ from excitation wavelengths).
