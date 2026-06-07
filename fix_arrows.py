import os

STYLE_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/style.py"
with open(STYLE_PATH, 'r') as f:
    content = f.read()

# Dark Theme
dark_patch_old = """    QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        subcontrol-origin: padding;
        background: transparent;
        border-left: 1px solid rgba(255, 255, 255, 0.08);
        width: 15px;
    }"""

dark_patch_new = """    QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        subcontrol-origin: padding;
        background: transparent;
        border-left: 1px solid rgba(255, 255, 255, 0.08);
        width: 16px;
    }
    QComboBox::down-arrow {
        image: url({ICON_DIR}/arrow-down.svg);
        width: 10px; height: 10px;
    }
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        image: url({ICON_DIR}/arrow-up.svg);
        width: 9px; height: 9px;
    }
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        image: url({ICON_DIR}/arrow-down.svg);
        width: 9px; height: 9px;
    }"""

# Light Theme
light_patch_old = """    QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        subcontrol-origin: padding;
        background: transparent;
        border-left: 1px solid rgba(0, 0, 0, 0.10);
        width: 15px;
    }"""

light_patch_new = """    QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        subcontrol-origin: padding;
        background: transparent;
        border-left: 1px solid rgba(0, 0, 0, 0.10);
        width: 16px;
    }
    QComboBox::down-arrow {
        image: url({ICON_DIR}/arrow-down-dark.svg);
        width: 10px; height: 10px;
    }
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        image: url({ICON_DIR}/arrow-up-dark.svg);
        width: 9px; height: 9px;
    }
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        image: url({ICON_DIR}/arrow-down-dark.svg);
        width: 9px; height: 9px;
    }"""

if "arrow-down.svg" not in content:
    content = content.replace(dark_patch_old, dark_patch_new)
    content = content.replace(light_patch_old, light_patch_new)
    
    with open(STYLE_PATH, 'w') as f:
        f.write(content)
    
print("Done")
