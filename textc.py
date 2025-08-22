import os
import sys
import yt_dlp
import whisper

def download_from_url(url, output_path="downloads"):
    """Download audio from a YouTube/URL link using yt-dlp"""
    os.makedirs(output_path, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_path, "%(title)s.%(ext)s"),
        "quiet": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".m4a", ".mp3")
        return filename

def transcribe_audio(file_path, output_file="transcript.txt"):
    """Convert audio/video to text using Whisper"""
    model = whisper.load_model("base")  # you can use "small", "medium", or "large" for better accuracy
    result = model.transcribe(file_path)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    print(f"✅ Transcription saved to {output_file}")

if __name__ == "__main__":
    choice = input("Enter 1 for file, 2 for URL: ")
    
    if choice == "1":
        file_path = input("Enter the path to your audio/video file: ")
        transcribe_audio(file_path)
    
    elif choice == "2":
        url = input("Enter the YouTube/Audio/Video URL: ")
        downloaded_file = download_from_url(url)
        transcribe_audio(downloaded_file)
    
    else:
        print("Invalid choice")
