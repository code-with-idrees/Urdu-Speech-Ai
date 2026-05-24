import subprocess
import os
import re
import tempfile

BATCH_SIZE = 100  # number of files per batch

def run_cmd(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()

# Ensure a clean state: unstage everything
run_cmd('git reset')

# Find all untracked mp3 files under data/raw
all_untracked = run_cmd('git ls-files --others --exclude-standard data/raw').splitlines()
mp3_files = [f for f in all_untracked if f.lower().endswith('.mp3')]
print(f"Total untracked MP3 files: {len(mp3_files)}")

if not mp3_files:
    print("Nothing to commit.")
    exit(0)

batch_num = 1
for i in range(0, len(mp3_files), BATCH_SIZE):
    batch = mp3_files[i:i + BATCH_SIZE]
    # Write batch list to a temp file for safe pathspec handling
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tf:
        tf.write('\n'.join(batch))
        temp_path = tf.name
    try:
        # Stage the batch
        subprocess.run(['git', 'add', '--pathspec-from-file', temp_path], check=True)
        # Commit
        commit_msg = f"data: upload batch {batch_num} of {len(batch)} audio files"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        print(f"Committed batch {batch_num}: {len(batch)} files")
        # Push this batch
        subprocess.run(['git', 'push', 'origin', 'HEAD'], check=True)
        print(f"Pushed batch {batch_num} to remote")
    finally:
        os.remove(temp_path)
    batch_num += 1
print("All batches committed and pushed.")
