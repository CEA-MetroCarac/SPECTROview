import os

STYLE_PATH = "/Users/HoanLe/Documents/SPECTROview/spectroview/view/style.py"
with open(STYLE_PATH, 'r') as f:
    content = f.read()

# Make the return strings f-strings
content = content.replace('    return """', '    return f"""')

with open(STYLE_PATH, 'w') as f:
    f.write(content)

print("Done")
