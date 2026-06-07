import os
import re

FILTER_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/components/v_data_filter.py"

with open(FILTER_PATH, 'r') as f:
    content = f.read()

# 1. Reduce padx/pady to 2
if "layout_main.setContentsMargins(2, 2, 2, 2)" not in content:
    content = content.replace(
        "layout_main = QVBoxLayout(self)",
        "layout_main = QVBoxLayout(self)\n        layout_main.setContentsMargins(2, 2, 2, 2)"
    )

# 2. Fix QListWidget item spacing / checkbox clipping
# Ensure QSize is imported!
if "QSize" not in content:
    content = content.replace("from PySide6.QtCore import Qt, Signal, QStringListModel", "from PySide6.QtCore import Qt, Signal, QStringListModel, QSize")

# Fix sizeHint for _on_add_filter
content = content.replace(
    "item.setSizeHint(checkbox.sizeHint())",
    "size = checkbox.sizeHint()\n            item.setSizeHint(QSize(size.width(), max(size.height(), 24)))"
)

# Also fix set_filters
content = re.sub(
    r'(checkbox\.setChecked\(filter_data\.get\("state", False\)\)\n\s+)item\.setSizeHint\(checkbox\.sizeHint\(\)\)',
    r'\1size = checkbox.sizeHint()\n            item.setSizeHint(QSize(size.width(), max(size.height(), 24)))',
    content
)

# Add spacing to listwidget to make it look nicer
if "self.filter_listbox.setSpacing(2)" not in content:
    content = content.replace(
        "self.filter_listbox = QListWidget()",
        "self.filter_listbox = QListWidget()\n        self.filter_listbox.setSpacing(2)"
    )

with open(FILTER_PATH, 'w') as f:
    f.write(content)

print("Done")
