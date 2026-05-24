import os
import re
import shutil
import subprocess

DATA_RAW_DIR = r"C:/Users/HP/OneDrive/Documents/Urdu-Speech-Ai/data/raw"
OLD_RAW_DIR = r"C:/Users/HP/OneDrive/Documents/Urdu-Speech-Ai/raw/raw"

os.makedirs(DATA_RAW_DIR, exist_ok=True)

# 1. Gather all MP3 files
mp3_files = []

def gather_mp3s(src_dir):
    if not os.path.exists(src_dir): return
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith('.mp3'):
                mp3_files.append(os.path.join(root, f))

gather_mp3s(DATA_RAW_DIR)
gather_mp3s(OLD_RAW_DIR)

date_pattern = re.compile(r"^(\d{1,2})[ _-]+(January|February|March|April|May|June|July|August|September|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)", re.IGNORECASE)

# Move all MP3s to their proper subfolder in DATA_RAW_DIR
print(f"Total MP3s found: {len(mp3_files)}")

for file_path in mp3_files:
    filename = os.path.basename(file_path)
    match = date_pattern.search(filename)
    if match:
        day = match.group(1).zfill(2)
        month = match.group(2).capitalize()
        folder_name = f"{day}_{month}"
    else:
        folder_name = "unsorted"
    
    target_dir = os.path.join(DATA_RAW_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)
    
    # If the file is already exactly where it should be, skip
    if os.path.abspath(file_path) == os.path.abspath(target_path):
        continue
        
    # If a file with the same name exists but it's a different path, skip to avoid overwrite?
    # Actually, we should move it, but if target exists, just remove source if it's the same file.
    if os.path.exists(target_path):
        # We can just skip and maybe remove the source if it's identical?
        if os.path.abspath(file_path) != os.path.abspath(target_path):
            pass # We'll just leave it or overwrite. Let's just skip moving if target exists.
    else:
        shutil.move(file_path, target_path)

print("Finished organizing.")
