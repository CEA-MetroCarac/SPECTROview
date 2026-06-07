import os

# 1. Update style.py for #workspaceRightPanel margin
STYLE_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/style.py"
with open(STYLE_PATH, 'r') as f:
    style_content = f.read()

# I will add `margin-left: 3px;` to `#workspaceRightPanel`
# The easiest way is to add a specific rule for #workspaceRightPanel
if "#workspaceRightPanel {" not in style_content:
    style_content = style_content + """\n
    #workspaceRightPanel {
        margin-left: 3px;
    }
"""
    with open(STYLE_PATH, 'w') as f:
        f.write(style_content)

# 2. Update v_fit_model_builder.py margins
FIT_BUILDER = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/components/v_fit_model_builder.py"
with open(FIT_BUILDER, 'r') as f:
    fit_content = f.read()

old_margins = """        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)"""

new_margins = """        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(3, 3, 3, 3)"""

fit_content = fit_content.replace(old_margins, new_margins)

with open(FIT_BUILDER, 'w') as f:
    f.write(fit_content)

print("Done")
