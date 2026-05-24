# organize_audio_by_date.py
import os
import re
import shutil

# Mapping month names to numbers (optional, not needed for folder names)
MONTHS = {
    'January': '01', 'February': '02', 'March': '03', 'April': '04',
    'May': '05', 'June': '06', 'July': '07', 'August': '08',
    'September': '09', 'October': '10', 'November': '11', 'December': '12'
}

# Directory containing the audio files (adjust if needed)
RAW_DIR = r"C:/Users/HP/OneDrive/Documents/Urdu-Speech-Ai/raw/raw"

if not os.path.isdir(RAW_DIR):
    raise RuntimeError(f"Directory does not exist: {RAW_DIR}")

# Ensure an 'unsorted' directory exists for files without a recognizable date
UNSORTED_DIR = os.path.join(RAW_DIR, "unsorted")
os.makedirs(UNSORTED_DIR, exist_ok=True)

# Regex to capture a day and month name at the start of the filename (e.g., "14 August ...")
date_pattern = re.compile(r"^(\d{1,2})[ _-]+(January|February|March|April|May|June|July|August|September|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)", re.IGNORECASE)

for filename in os.listdir(RAW_DIR):
    file_path = os.path.join(RAW_DIR, filename)
    if not os.path.isfile(file_path):
        continue  # skip directories (including the ones we may create later)
    # Try to extract a date from the filename
    match = date_pattern.search(filename)
    if match:
        day = match.group(1).zfill(2)  # pad day with leading zero
        month = match.group(2).capitalize()
        folder_name = f"{day}_{month}"
    else:
        folder_name = "unsorted"
    target_dir = os.path.join(RAW_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)
    # If a file with the same name already exists, skip moving to avoid duplicates
    if os.path.exists(target_path):
        # Optionally you could rename, but spec says avoid duplicates, so we skip
        continue
    shutil.move(file_path, target_path)
    # Use a safe print that ignores characters not representable in the console encoding
    try:
        print(f"Moved {filename} -> {folder_name}/")
    except UnicodeEncodeError:
        print(f"Moved [unicode issue] -> {folder_name}/")
