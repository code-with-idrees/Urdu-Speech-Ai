import urllib.request
import zipfile
import os
import shutil

url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
zip_path = "ffmpeg.zip"
print("Downloading FFmpeg (this will take a moment)...")
urllib.request.urlretrieve(url, zip_path)

print("Extracting FFmpeg...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("ffmpeg_temp")

ffmpeg_exe = "ffmpeg_temp/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
ffprobe_exe = "ffmpeg_temp/ffmpeg-master-latest-win64-gpl/bin/ffprobe.exe"

try:
    if os.path.exists("ffmpeg.exe"):
        os.remove("ffmpeg.exe")
    shutil.move(ffmpeg_exe, "ffmpeg.exe")
    print("Successfully installed ffmpeg.exe!")
except Exception as e:
    print(f"Error moving ffmpeg.exe: {e}")

try:
    if os.path.exists("ffprobe.exe"):
        os.remove("ffprobe.exe")
    shutil.move(ffprobe_exe, "ffprobe.exe")
    print("Successfully installed ffprobe.exe!")
except Exception as e:
    pass

# Cleanup
os.remove(zip_path)
shutil.rmtree("ffmpeg_temp", ignore_errors=True)
print("Done!")
