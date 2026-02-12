
import os

utils_path = r'c:\Users\VL251876\Documents\Python\SPECTROview-1\spectroview\viewmodel\utils.py'

with open(utils_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.strip().startswith('def spectrum_to_dict'):
        skip = True
    
    if line.strip().startswith('def save_df_to_excel'):
        skip = False
        
    if not skip:
        # Also check if we are in the dict_to_spectrum part which we might have skipped partially if I didn't catch the start
        # but my logic captures from spectrum_to_dict start until save_df_to_excel start
        new_lines.append(line)

with open(utils_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Cleaned up {utils_path}, removed spectrum_to_dict and dict_to_spectrum.")
