import os

icons_dir = "/Users/HoanLe/Documents/SPECTROview/spectroview/resources/icons"
os.makedirs(icons_dir, exist_ok=True)

# Up arrow (SVG)
up_svg = """<svg width="12" height="12" viewBox="0 0 12 12" xmlns="http://www.w3.org/2000/svg">
  <polygon points="6,3 10,8 2,8" fill="white" fill-opacity="0.8"/>
</svg>"""

# Down arrow (SVG)
down_svg = """<svg width="12" height="12" viewBox="0 0 12 12" xmlns="http://www.w3.org/2000/svg">
  <polygon points="2,4 10,4 6,9" fill="white" fill-opacity="0.8"/>
</svg>"""

with open(os.path.join(icons_dir, "arrow-up.svg"), 'w') as f:
    f.write(up_svg)
    
with open(os.path.join(icons_dir, "arrow-down.svg"), 'w') as f:
    f.write(down_svg)

print("Created arrow SVGs")
