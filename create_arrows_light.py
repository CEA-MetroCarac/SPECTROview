import os

icons_dir = "/Users/HoanLe/Documents/SPECTROview/spectroview/resources/icons"

up_dark_svg = """<svg width="12" height="12" viewBox="0 0 12 12" xmlns="http://www.w3.org/2000/svg">
  <polygon points="6,3 10,8 2,8" fill="black" fill-opacity="0.8"/>
</svg>"""

down_dark_svg = """<svg width="12" height="12" viewBox="0 0 12 12" xmlns="http://www.w3.org/2000/svg">
  <polygon points="2,4 10,4 6,9" fill="black" fill-opacity="0.8"/>
</svg>"""

with open(os.path.join(icons_dir, "arrow-up-dark.svg"), 'w') as f:
    f.write(up_dark_svg)
    
with open(os.path.join(icons_dir, "arrow-down-dark.svg"), 'w') as f:
    f.write(down_dark_svg)

print("Created dark arrow SVGs")
