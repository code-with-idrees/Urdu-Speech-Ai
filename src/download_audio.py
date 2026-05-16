import os
import argparse
import yt_dlp

def download_audio(url, output_path="data/raw"):
    """
    Downloads the best quality audio from a YouTube video and saves it as an MP3.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'quiet': False,
    }

    print(f"Downloading audio from: {url}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            print(f"Download completed successfully! File saved to {output_path}")
        except Exception as e:
            print(f"An error occurred during download: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download audio from a YouTube video.")
    parser.add_argument("url", help="The YouTube video URL")
    parser.add_argument("--output", default="data/raw", help="The directory to save the audio file")
    
    args = parser.parse_args()
    download_audio(args.url, args.output)
