import re
import sys

files = [
    '/Users/HoanLe/Documents/SPECTROview/spectroview/resources/user_manual/mva.md',
    '/Users/HoanLe/Documents/SPECTROview/spectroview/resources/user_manual/spectra_maps.md'
]

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        
    new_lines = []
    changed = False
    
    current_h3 = 0
    h4_count = 0
    
    for line in lines:
        if line.startswith('### '):
            # Parse the H3 number
            m = re.search(r'### \*\*(\d+)\.\s', line)
            if m:
                current_h3 = int(m.group(1))
            else:
                m2 = re.search(r'### (\d+)\.\s', line)
                if m2:
                    current_h3 = int(m2.group(1))
                else:
                    current_h3 = 0
            h4_count = 0
            new_lines.append(line)
        elif line.startswith('#### '):
            if current_h3 > 0:
                h4_count += 1
                # Remove existing numbers if any
                m_bold = re.match(r'^#### \*\*(?:\d+\.\d+\.)*\s*(.*?)\*\*$', line.strip())
                m_normal = re.match(r'^#### (?:\d+\.\d+\.)*\s*(.*)$', line.strip())
                
                if m_bold:
                    title = m_bold.group(1)
                    new_line = f'#### **{current_h3}.{h4_count}. {title}**'
                    if new_line != line: changed = True
                    new_lines.append(new_line)
                elif m_normal and not line.strip().startswith('#### **'):
                    title = m_normal.group(1)
                    new_line = f'#### {current_h3}.{h4_count}. {title}'
                    if new_line != line: changed = True
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print(f'Updated {file_path.split("/")[-1]}')

