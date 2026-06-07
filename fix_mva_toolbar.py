import os

MVA_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/components/v_mva.py"

with open(MVA_PATH, 'r') as f:
    content = f.read()

# 1. Add imports if needed
if "QEvent" not in content or "QObject" not in content:
    content = content.replace(
        "from PySide6.QtCore import Qt, Signal",
        "from PySide6.QtCore import Qt, Signal, QObject, QEvent"
    )

# 2. Add ToolbarEventFilter to _PlotTab
patch = """
        class ToolbarEventFilter(QObject):
            def __init__(self, toolbar):
                super().__init__()
                self.toolbar = toolbar
            def eventFilter(self, obj, event):
                if event.type() == QEvent.PaletteChange:
                    action_dict = {action.text(): action for action in self.toolbar.actions() if action.text()}
                    for text, tooltip_text, image_file, name_of_method in self.toolbar.toolitems:
                        if text in action_dict and image_file is not None:
                            try:
                                icon = self.toolbar._icon(image_file + '.png')
                                action_dict[text].setIcon(icon)
                            except Exception:
                                pass
                return False
                
        self.toolbar_filter = ToolbarEventFilter(self.toolbar)
        self.toolbar.installEventFilter(self.toolbar_filter)
"""

if "ToolbarEventFilter" not in content:
    # insert after self.toolbar = NavigationToolbar2QT(self.canvas, self)
    content = content.replace(
        "self.toolbar = NavigationToolbar2QT(self.canvas, self)",
        "self.toolbar = NavigationToolbar2QT(self.canvas, self)\n" + patch
    )

with open(MVA_PATH, 'w') as f:
    f.write(content)

print("Done")
