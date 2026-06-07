import os

STYLE_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/style.py"

with open(STYLE_PATH, 'r') as f:
    content = f.read()

# Add QScrollArea to QListWidget, QTreeWidget rule
if "QScrollArea" not in content:
    content = content.replace("QListWidget, QTreeWidget {", "QListWidget, QTreeWidget, QScrollArea {")
    # Also add QScrollArea to QTableWidget, QTableView just in case
    # wait, they have slightly different background logic, let's just use the QListWidget rule

    # We also need to style the scroll area's viewport so it doesn't draw a secondary border
    # Add a specific rule for QScrollArea widget
    viewport_style = """
    QScrollArea > QWidget > QWidget {
        background: transparent;
    }
"""
    if "QScrollArea > QWidget > QWidget" not in content:
        content = content.replace(
            "QListWidget, QTreeWidget, QScrollArea {",
            viewport_style + "\n    QListWidget, QTreeWidget, QScrollArea {"
        )

with open(STYLE_PATH, 'w') as f:
    f.write(content)

print("Done")
