import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

# Fix console encoding on Windows to prevent UnicodeEncodeError
if sys.platform.startswith("win"):
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MANIFEST_FILE = PROJECT_ROOT / "data" / ".commit_manifest.json"

def run_git_command(args, cwd=PROJECT_ROOT):
    """Run a git command and return output and exit code."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True
        )
        return result.stdout.strip(), result.stderr.strip(), 0
    except subprocess.CalledProcessError as e:
        return e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else "", e.returncode

def get_current_branch():
    """Get the name of the current active branch."""
    stdout, _, code = run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    if code == 0:
        return stdout
    return "main"

def load_manifest():
    """Load the progress manifest to see which files have already been processed."""
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"committed_files": [], "completed_batches": 0}

def save_manifest(manifest):
    """Save the progress manifest."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(
        description="Batch Commit Audio Files to Git LFS to maximize commits and ensure reliable pushes."
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Number of files per commit batch")
    parser.add_argument("--push", action="store_true", help="Push to remote after each commit batch")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without committing or pushing")
    args = parser.parse_args()

    print("=" * 70)
    print(" >>> URDU SPEECH AI - BATCH COMMIT & PUSH TOOL")
    print("=" * 70)

    # 1. Verify Git LFS is tracking the files
    print("[*] Verifying Git LFS tracking rules...")
    gitattributes_path = PROJECT_ROOT / ".gitattributes"
    if not gitattributes_path.exists():
        print("[!] Warning: .gitattributes not found. Audio files might be tracked as standard Git objects instead of LFS!")
    else:
        print("[+] .gitattributes found and active.")

    # 2. Get list of files to process
    if not DATA_RAW.exists():
        print(f"[!] Error: Raw data directory '{DATA_RAW}' does not exist.")
        sys.exit(1)

    all_audio_files = sorted([
        f for f in DATA_RAW.glob("*")
        if f.is_file() and f.suffix.lower() in [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
    ])

    total_files = len(all_audio_files)
    if total_files == 0:
        print("[+] No audio files found in data/raw/ to commit!")
        sys.exit(0)

    total_size_bytes = sum(f.stat().st_size for f in all_audio_files)
    total_size_gb = total_size_bytes / (1024 ** 3)
    print(f"[+] Found {total_files} audio files in total.")
    print(f"[+] Total data size: {total_size_gb:.2f} GB")

    # 3. Load progress manifest
    manifest = load_manifest()
    committed_set = set(manifest.get("committed_files", []))
    
    # Filter files that haven't been committed yet
    files_to_commit = [f for f in all_audio_files if f.name not in committed_set]
    remaining_count = len(files_to_commit)

    print(f"[+] Already committed: {len(committed_set)} files.")
    print(f"[+] Remaining to commit: {remaining_count} files.")

    if remaining_count == 0:
        print("[+] All files have been successfully batch-committed!")
        sys.exit(0)

    # 4. Determine current branch
    branch = get_current_branch()
    print(f"[+] Current Git Branch: {branch}")
    print("=" * 70)

    if args.dry_run:
        print("[INFO] DRY RUN MODE ENABLED. No changes will be made.")

    # 5. Process in batches
    batch_size = args.batch_size
    batches = [files_to_commit[i:i + batch_size] for i in range(0, remaining_count, batch_size)]
    total_batches = len(batches)

    completed_batches = manifest.get("completed_batches", 0)

    for idx, batch in enumerate(batches):
        batch_idx = completed_batches + idx + 1
        batch_size_mb = sum(f.stat().st_size for f in batch) / (1024 ** 2)
        print(f"\n[+] Processing Batch {batch_idx}/{completed_batches + total_batches} ({len(batch)} files, {batch_size_mb:.2f} MB)")
        
        # Add files in batch (using force -f to bypass .gitignore)
        relative_paths = []
        for file_path in batch:
            rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
            relative_paths.append(rel_path)
            
            if not args.dry_run:
                # Add file with force to ignore .gitignore rule
                _, _, code = run_git_command(["add", "-f", rel_path])
                if code != 0:
                    print(f"  [!] Failed to add {rel_path}")
                    sys.exit(1)

        # Print quick preview of files in this batch
        print(f"  └── Adding files: {', '.join([f.name for f in batch[:3]])} ... (+ {len(batch) - 3} more)")

        # Commit batch
        commit_msg = f"data: upload batch {batch_idx} of audio files ({len(batch)} files, {batch_size_mb:.1f}MB)"
        if not args.dry_run:
            print(f"  [*] Committing batch {batch_idx}...")
            stdout, stderr, code = run_git_command(["commit", "-m", commit_msg])
            if code != 0:
                print(f"  [!] Commit failed: {stderr}")
                sys.exit(1)
            print(f"  [+] Committed successfully: {stdout.splitlines()[0] if stdout else ''}")
        else:
            print(f"  [Dry-Run] Would commit: \"{commit_msg}\"")

        # Push batch if requested
        if args.push:
            if not args.dry_run:
                print(f"  [*] Pushing batch {batch_idx} to remote...")
                stdout, stderr, code = run_git_command(["push", "origin", branch])
                if code != 0:
                    print(f"  [!] Push failed: {stderr}")
                    print("[!] Process halted. Fix the issue and re-run. Your progress is saved!")
                    sys.exit(1)
                print(f"  [+] Pushed successfully to origin/{branch}!")
            else:
                print(f"  [Dry-Run] Would push batch to origin/{branch}")

        # Update manifest
        if not args.dry_run:
            manifest["committed_files"].extend([f.name for f in batch])
            manifest["completed_batches"] = batch_idx
            save_manifest(manifest)

    print("\n" + "=" * 70)
    print(" >>> ALL BATCHES PROCESSED SUCCESSFULLY!")
    print(f" Total files: {total_files}")
    print(f" Completed batches: {completed_batches + total_batches}")
    if args.push:
        print("[+] All batches committed and pushed to remote.")
    else:
        print("[+] All batches committed locally. Run 'git push origin main' or run with --push to upload.")
    print("=" * 70)

if __name__ == "__main__":
    main()

