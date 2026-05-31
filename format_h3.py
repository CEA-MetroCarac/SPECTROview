import os
import re
import glob

directory = '/Users/HoanLe/Documents/SPECTROview/spectroview/resources/user_manual'
md_files = glob.glob(os.path.join(directory, '*.md'))

for file_path in md_files:
    if os.path.basename(file_path) == 'index.md':
        continue
        
    with open(file_path, 'r', encoding='utf-8') as f:
        # read content and split by lines to handle \n properly
        lines = f.read().split('\n')
        
    count = 1
    new_lines = []
    changed = False
    
    # Check if file has any H3 at all
    has_h3 = any(l.startswith('### ') for l in lines)
    if not has_h3:
        continue
        
    for line in lines:
        if line.startswith('### '):
            # Match bold header
            match_bold = re.match(r'^### \*\*(?:\d+\.)*\s*(.*?)\*\*$', line.strip())
            # Match normal header
            match_normal = re.match(r'^### (?:\d+\.)*\s*(.*)$', line.strip())
            
            if match_bold:
                title = match_bold.group(1)
                new_line = f'### **{count}. {title}**'
                if new_line != line: changed = True
                new_lines.append(new_line)
                count += 1
            elif match_normal and not line.strip().startswith('### **'):
                title = match_normal.group(1)
                new_line = f'### {count}. {title}'
                if new_line != line: changed = True
                new_lines.append(new_line)
                count += 1
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print(f'Updated {os.path.basename(file_path)}')

