import os

STYLE_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/style.py"
with open(STYLE_PATH, 'r') as f:
    content = f.read()

# Revert f-strings
content = content.replace('    return f"""', '    return """')

# At the end of the dark_glass_stylesheet and light_glass_stylesheet functions, replace {ICON_DIR}
content = content.replace('    """\n\n\ndef light_glass_stylesheet() -> str:', '    """.replace("{ICON_DIR}", ICON_DIR)\n\n\ndef light_glass_stylesheet() -> str:')
content = content.replace('    """\n\n\ndef setup_palette', '    """.replace("{ICON_DIR}", ICON_DIR)\n\n\ndef setup_palette')

with open(STYLE_PATH, 'w') as f:
    f.write(content)

print("Done")
