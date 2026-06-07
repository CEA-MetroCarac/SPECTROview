import os

STYLE_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/style.py"
with open(STYLE_PATH, 'r') as f:
    content = f.read()

# 1. Fix QSplitter handle width
# Dark
content = content.replace(
    "    QSplitter::handle:horizontal {\n        width: 3px;\n    }",
    "    QSplitter::handle:horizontal {\n        width: 8px;\n    }"
)
content = content.replace(
    "    QSplitter::handle:vertical {\n        height: 3px;\n    }",
    "    QSplitter::handle:vertical {\n        height: 8px;\n    }"
)
content = content.replace(
    "    QSplitter::handle {\n        background: rgba(255, 255, 255, 0.05);\n    }",
    "    QSplitter::handle {\n        background: transparent;\n    }"
)
# Light
content = content.replace(
    "    QSplitter::handle:horizontal {\n        width: 3px;\n    }",
    "    QSplitter::handle:horizontal {\n        width: 8px;\n    }"
)
content = content.replace(
    "    QSplitter::handle:vertical {\n        height: 3px;\n    }",
    "    QSplitter::handle:vertical {\n        height: 8px;\n    }"
)
content = content.replace(
    "    QSplitter::handle {\n        background: rgba(0, 0, 0, 0.05);\n    }",
    "    QSplitter::handle {\n        background: transparent;\n    }"
)

# 2. Fix SpinBox clickable areas
# Dark
dark_spinbox_fix = """    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        background: transparent;
        border-left: 1px solid rgba(255, 255, 255, 0.08);
        width: 16px;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        background: transparent;
        border-left: 1px solid rgba(255, 255, 255, 0.08);
        width: 16px;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        background: transparent;
        border-left: 1px solid rgba(255, 255, 255, 0.08);
        width: 16px;
    }"""

# Light
light_spinbox_fix = """    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        background: transparent;
        border-left: 1px solid rgba(0, 0, 0, 0.10);
        width: 16px;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        background: transparent;
        border-left: 1px solid rgba(0, 0, 0, 0.10);
        width: 16px;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        background: transparent;
        border-left: 1px solid rgba(0, 0, 0, 0.10);
        width: 16px;
    }"""

old_dark_buttons = """    QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        subcontrol-origin: padding;
        background: transparent;
        border-left: 1px solid rgba(255, 255, 255, 0.08);
        width: 16px;
    }"""

old_light_buttons = """    QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        subcontrol-origin: padding;
        background: transparent;
        border-left: 1px solid rgba(0, 0, 0, 0.10);
        width: 16px;
    }"""

content = content.replace(old_dark_buttons, dark_spinbox_fix)
content = content.replace(old_light_buttons, light_spinbox_fix)

with open(STYLE_PATH, 'w') as f:
    f.write(content)

print("Done")
