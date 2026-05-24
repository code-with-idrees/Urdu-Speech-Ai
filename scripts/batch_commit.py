import subprocess
import os

def run_cmd(cmd):
    return subprocess.check_output(cmd, shell=True).decode('utf-8').strip()

# Find untracked files
all_untracked = run_cmd("git ls-files --others --exclude-standard data/raw").splitlines()
mp3_to_add = [f for f in all_untracked if f.endswith('.mp3')]

print(f"Total MP3s to batch commit: {len(mp3_to_add)}")

BATCH_SIZE = 50
# Determine starting batch num from git log
log_out = run_cmd("git log -n 1 --grep=\"data: upload batch\" --oneline")
import re
match = re.search(r'batch (\d+)', log_out)
if match:
    batch_num = int(match.group(1)) + 1
else:
    batch_num = 51

import tempfile

for i in range(0, len(mp3_to_add), BATCH_SIZE):
    batch = mp3_to_add[i:i+BATCH_SIZE]
    
    # Write batch to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        f.write('\n'.join(batch))
        temp_path = f.name
        
    try:
        # Use pathspec-from-file to safely add files regardless of quotes or special characters
        subprocess.run(['git', 'add', '--pathspec-from-file', temp_path], check=True)
        
        # Commit
        commit_msg = f"data: upload batch {batch_num} of audio files ({len(batch)} files)"
        print(f"Committing: {commit_msg}")
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        batch_num += 1
    finally:
        os.remove(temp_path)

print("Batch commits completed.")
