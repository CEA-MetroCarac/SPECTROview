import os

STYLE_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/style.py"
with open(STYLE_PATH, 'r') as f:
    content = f.read()

# 1. Decrease QSplitter width
content = content.replace("width: 8px;", "width: 4px;")
content = content.replace("height: 8px;", "height: 4px;")

# 2. Add pressed states for dark theme
dark_pressed_css = """
    QComboBox::drop-down:pressed, QSpinBox::up-button:pressed, QSpinBox::down-button:pressed, QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed {
        background: rgba(255, 255, 255, 0.15);
    }
"""

if "QSpinBox::up-button:pressed" not in content:
    # Insert dark theme pressed state after QSpinBox::down-button
    target_dark = """    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        background: transparent;
        border-left: 1px solid rgba(255, 255, 255, 0.08);
        width: 16px;
    }"""
    
    content = content.replace(target_dark, target_dark + dark_pressed_css)

# 3. Add pressed states for light theme
light_pressed_css = """
    QComboBox::drop-down:pressed, QSpinBox::up-button:pressed, QSpinBox::down-button:pressed, QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed {
        background: rgba(0, 0, 0, 0.10);
    }
"""

if "rgba(0, 0, 0, 0.10);" not in content.split("QSpinBox::up-button:pressed")[-1]:
    target_light = """    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        background: transparent;
        border-left: 1px solid rgba(0, 0, 0, 0.10);
        width: 16px;
    }"""
    
    content = content.replace(target_light, target_light + light_pressed_css)

with open(STYLE_PATH, 'w') as f:
    f.write(content)

print("Done")
