"""
Batch Audio Downloader — Urdu Speech AI Pipeline (Step 1)

Reads poet/video entries from config/channels.yaml and downloads the best
available audio as MP3 files into data/raw/. Maintains a manifest so that
already-downloaded videos are skipped on re-runs.

Usage:
    python src/batch_download.py                    # download all
    python src/batch_download.py --poet "Rahat Indori"  # single poet
    python src/batch_download.py --limit 5          # first 5 videos only
"""

import argparse
import re
from pathlib import Path

import yt_dlp

from utils import (
    load_channels,
    load_settings,
    get_data_paths,
    load_manifest,
    save_manifest,
    setup_logging,
    timestamp,
)

logger = setup_logging("batch_download", log_file="logs/batch_download.log")


# =============================================================================
# Helpers
# =============================================================================

def extract_video_id(url: str) -> str | None:
    """Extract the YouTube video ID from a URL."""
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/shorts\/)([a-zA-Z0-9_-]{11})",
        r"(?:embed\/)([a-zA-Z0-9_-]{11})",
    ]
    for pat in patterns:
        match = re.search(pat, url)
        if match:
            return match.group(1)
    return None


def download_single(url: str, output_dir: Path, settings: dict) -> dict | None:
    """
    Download audio from a single YouTube URL.

    Returns metadata dict on success, None on failure.
    """
    audio_cfg = settings.get("audio", {})

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_cfg.get("download_codec", "mp3"),
                "preferredquality": str(audio_cfg.get("download_quality", "192")),
            }
        ],
        "outtmpl": str(output_dir / "%(title)s__%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }

    ffmpeg_path = audio_cfg.get("ffmpeg_location")
    if ffmpeg_path:
        from utils import PROJECT_ROOT
        resolved = str((PROJECT_ROOT / ffmpeg_path).resolve())
        ydl_opts["ffmpeg_location"] = resolved

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                return None

            return {
                "video_id": info.get("id", ""),
                "title": info.get("title", ""),
                "duration_sec": info.get("duration", 0),
                "uploader": info.get("uploader", ""),
                "upload_date": info.get("upload_date", ""),
                "url": url,
                "downloaded_at": timestamp(),
            }
    except Exception as e:
        logger.error("Failed to download %s: %s", url, e)
        return None


# =============================================================================
# Main Pipeline
# =============================================================================

def run_batch_download(poet_filter: str | None = None, limit: int | None = None):
    """
    Download audio for all (or filtered) poets listed in channels.yaml.
    """
    settings = load_settings()
    data_paths = get_data_paths(settings)
    raw_dir = data_paths["raw_audio"]

    manifest_path = raw_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    poets = load_channels()
    if not poets:
        logger.warning(
            "No poets found in config/channels.yaml. "
            "Add YouTube URLs before running this script."
        )
        return

    download_count = 0
    skip_count = 0
    fail_count = 0

    for poet in poets:
        poet_name = poet.get("name", "Unknown")

        if poet_filter and poet_filter.lower() not in poet_name.lower():
            continue

        videos = poet.get("videos", [])
        logger.info("┌─ Poet: %s (%d videos)", poet_name, len(videos))

        for video in videos:
            url = video.get("url", "")
            if not url or "REPLACE_ME" in url:
                continue

            video_id = extract_video_id(url)
            if not video_id:
                logger.warning("│  ⚠ Could not extract video ID from: %s", url)
                fail_count += 1
                continue

            # Skip already downloaded
            if video_id in manifest:
                logger.info("│  ⏭ Already downloaded: %s", video.get("title", video_id))
                skip_count += 1
                continue

            # Check download limit
            if limit and download_count >= limit:
                logger.info("│  🛑 Reached download limit (%d)", limit)
                break

            logger.info("│  ⬇ Downloading: %s", video.get("title", url))
            meta = download_single(url, raw_dir, settings)

            if meta:
                meta["poet"] = poet_name
                meta["emotion_hint"] = video.get("emotion_hint", "")
                manifest[video_id] = meta
                save_manifest(manifest_path, manifest)
                download_count += 1
                logger.info("│  ✅ Saved: %s (%.1f min)", meta["title"], meta["duration_sec"] / 60)
            else:
                fail_count += 1

    logger.info("└─ Done: %d downloaded, %d skipped, %d failed", download_count, skip_count, fail_count)

    total_hours = sum(
        v.get("duration_sec", 0) for v in manifest.values()
    ) / 3600
    logger.info("   Total raw audio in manifest: %.1f hours", total_hours)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch download audio from YouTube for the Urdu Speech AI dataset."
    )
    parser.add_argument(
        "--poet",
        default=None,
        help="Download only for a specific poet (case-insensitive substring match).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of new videos to download in this run.",
    )
    args = parser.parse_args()

    run_batch_download(poet_filter=args.poet, limit=args.limit)
