import os

# 1. Update style.py
STYLE_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/style.py"
with open(STYLE_PATH, 'r') as f:
    style_content = f.read()

# Add #workspaceRightPanel to the dark theme rules
style_content = style_content.replace(
    "QListWidget, QTreeWidget, QScrollArea {",
    "QListWidget, QTreeWidget, QScrollArea, #workspaceRightPanel {"
)

with open(STYLE_PATH, 'w') as f:
    f.write(style_content)


# 2. Update v_workspace_spectra.py
SPECTRA_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/v_workspace_spectra.py"
with open(SPECTRA_PATH, 'r') as f:
    spectra_content = f.read()

if 'right_widget.setObjectName("workspaceRightPanel")' not in spectra_content:
    spectra_content = spectra_content.replace(
        "right_widget = QFrame()",
        "right_widget = QFrame()\n        right_widget.setObjectName(\"workspaceRightPanel\")"
    )
    with open(SPECTRA_PATH, 'w') as f:
        f.write(spectra_content)


# 3. Update v_workspace_maps.py
MAPS_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/v_workspace_maps.py"
with open(MAPS_PATH, 'r') as f:
    maps_content = f.read()

if 'panel.setObjectName("workspaceRightPanel")' not in maps_content:
    maps_content = maps_content.replace(
        "panel = QFrame()",
        "panel = QFrame()\n        panel.setObjectName(\"workspaceRightPanel\")"
    )
    # Also remove panel.setFrameShape(QFrame.StyledPanel) because our CSS handles the border
    maps_content = maps_content.replace("panel.setFrameShape(QFrame.StyledPanel)", "panel.setFrameShape(QFrame.NoFrame)")
    with open(MAPS_PATH, 'w') as f:
        f.write(maps_content)


# 4. Update v_workspace_graphs.py
GRAPHS_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/v_workspace_graphs.py"
with open(GRAPHS_PATH, 'r') as f:
    graphs_content = f.read()

if 'right_panel.setObjectName("workspaceRightPanel")' not in graphs_content:
    graphs_content = graphs_content.replace(
        "right_panel = QFrame()",
        "right_panel = QFrame()\n        right_panel.setObjectName(\"workspaceRightPanel\")"
    )
    graphs_content = graphs_content.replace("right_panel.setFrameShape(QFrame.StyledPanel)", "right_panel.setFrameShape(QFrame.NoFrame)")
    with open(GRAPHS_PATH, 'w') as f:
        f.write(graphs_content)

print("Done")
