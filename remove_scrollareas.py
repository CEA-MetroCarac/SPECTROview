import os
import re

# 1. Update v_workspace_maps.py
MAPS_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/v_workspace_maps.py"
with open(MAPS_PATH, 'r') as f:
    maps_content = f.read()

# In _create_maps_right_panel, remove QScrollArea
maps_patch_old = """    def _create_maps_right_panel(self):
        \"\"\"Create the right panel with map viewer and controls in a scroll area.\"\"\"
        # Main panel wrapper
        panel = QFrame()
        panel.setMaximumWidth(450)
        panel.setFrameShape(QFrame.StyledPanel)
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for all controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        # Scrollable content widget
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)"""

maps_patch_new = """    def _create_maps_right_panel(self):
        \"\"\"Create the right panel with map viewer and controls.\"\"\"
        # Main panel wrapper
        panel = QFrame()
        panel.setMaximumWidth(450)
        panel.setFrameShape(QFrame.StyledPanel)
        
        # Main layout (formerly scroll_content)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)"""

maps_content = maps_content.replace(maps_patch_old, maps_patch_new)

# Also remove the part where it adds scroll_area to main_layout
maps_footer_old = """        # Set scroll content and add to main panel
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        # ── Footer: count + progress bar + stop button (outside scroll area) ──
        footer_layout = QHBoxLayout()"""

maps_footer_new = """        # ── Footer: count + progress bar + stop button ──
        footer_layout = QHBoxLayout()"""

maps_content = maps_content.replace(maps_footer_old, maps_footer_new)
# change main_layout.addLayout(footer_layout) to layout.addLayout(footer_layout)
maps_content = maps_content.replace("main_layout.addLayout(footer_layout)", "layout.addLayout(footer_layout)")

with open(MAPS_PATH, 'w') as f:
    f.write(maps_content)

# 2. Update v_workspace_graphs.py
GRAPHS_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/v_workspace_graphs.py"
with open(GRAPHS_PATH, 'r') as f:
    graphs_content = f.read()

# _create_plot_tab
graphs_plot_old = """        # Wrap in QScrollArea
        scroll_area_plot = QScrollArea()
        scroll_area_plot.setWidgetResizable(True)
        scroll_area_plot.setWidget(tab_plot)
        
        return scroll_area_plot"""

graphs_plot_new = """        return tab_plot"""

graphs_content = graphs_content.replace(graphs_plot_old, graphs_plot_new)

# _create_more_options_tab
graphs_more_old = """        # Wrap in QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(tab_more)
        
        return scroll_area"""

graphs_more_new = """        return tab_more"""

graphs_content = graphs_content.replace(graphs_more_old, graphs_more_new)

with open(GRAPHS_PATH, 'w') as f:
    f.write(graphs_content)

print("Done")
