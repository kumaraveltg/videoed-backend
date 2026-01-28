from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import yt_dlp
import os
import shutil
import glob
import re
import unicodedata
from fastapi.responses import StreamingResponse 
from pydantic import BaseModel
import subprocess
import uuid

app = FastAPI()

class MergeRequest(BaseModel):
    files: list[str]
    output_name: str = "merged.mp4"

# ---- CORS (for React) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Accept-Ranges", "Content-Range", "Content-Length"]
)

# ---- Upload folder ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "videouploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- Serve uploaded videos ----
app.mount("/videos", StaticFiles(directory=UPLOAD_DIR), name="videos")

# -----------------------------
# Utility: safe filename
# -----------------------------
def safe_filename(name: str) -> str:
    # Convert Unicode → ASCII
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")

    # Remove invalid characters
    name = re.sub(r'[<>:"/\\|?*]', "", name)

    # Replace spaces with underscore
    name = name.replace(" ", "_")

    # Fallback
    if not name.lower().endswith(".mp4"):
        name = name + ".mp4"

    if not name:
        name = "video.mp4"

    return name

# -----------------------------
# Local file upload
# -----------------------------
@app.post("/upload/local")
async def upload_local(file: UploadFile = File(...)):
    filename = safe_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, filename)

    # ---- Save uploaded video ----
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ---- Extract audio using FFmpeg ----
    base_name = os.path.splitext(filename)[0]
    audio_filename = base_name + "_audio.m4a"   # or .mp3 if you want
    audio_path = os.path.join(UPLOAD_DIR, audio_filename)

    # FFmpeg command (no re-encode = fast)
    ffmpeg_cmd = (
        f'"{r"E:\ffmpeg\bin\ffmpeg.exe"}" '
        f'-y -i "{file_path}" -vn -acodec copy "{audio_path}"'
    )

    os.system(ffmpeg_cmd)

    # ---- Validate audio file ----
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        audio_filename = None  # extraction failed

    return {
        "message": "File uploaded successfully (video + audio separated)",
        "filename": filename,
        "video_url": f"http://localhost:8000/videos/{filename}",
        "audio_filename": audio_filename,
        "audio_url": f"http://localhost:8000/videos/{audio_filename}" if audio_filename else None
    }


# -----------------------------
# YouTube download
# -----------------------------
@app.post("/upload/youtube")
async def upload_youtube(url: str = Form(...)):
    try:
        # ---------- Templates ----------
        video_template = os.path.join(UPLOAD_DIR, "%(title).200s_video.%(ext)s")
        audio_template = os.path.join(UPLOAD_DIR, "%(title).200s_audio.%(ext)s")

        # ---------- VIDEO (merged MP4 - SAME AS BEFORE) ----------
        video_opts = {
            "outtmpl": video_template,
            "format": "bv*+ba/b",
            "merge_output_format": "mp4",
            "ffmpeg_location": r"E:\ffmpeg\bin",
            "noplaylist": True,
            "force_ipv4": True,
            "retries": 5,
            "quiet": True,
            "postprocessors": [{
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4"
            }],
        }

        # ---------- AUDIO ONLY (separate file) ----------
        audio_opts = {
            "outtmpl": audio_template,
            "format": "ba",
            "ffmpeg_location": r"E:\ffmpeg\bin",
            "noplaylist": True,
            "force_ipv4": True,
            "quiet": True,
            # Optional: convert to mp3
            # "postprocessors": [{
            #     "key": "FFmpegExtractAudio",
            #     "preferredcodec": "mp3",
            #     "preferredquality": "192",
            # }],
        }

        before_files = set(glob.glob(os.path.join(UPLOAD_DIR, "*")))

        # ---------- Download merged VIDEO ----------
        with yt_dlp.YoutubeDL(video_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        # ---------- Download AUDIO ONLY ----------
        with yt_dlp.YoutubeDL(audio_opts) as ydl:
            ydl.download([url])

        after_files = set(glob.glob(os.path.join(UPLOAD_DIR, "*")))
        new_files = list(after_files - before_files)

        # ---------- Find video + audio ----------
        video_file = next((f for f in new_files if f.lower().endswith(".mp4")), None)
        audio_file = next((f for f in new_files if f.lower().endswith((".m4a", ".webm", ".mp3"))), None)

        if not video_file:
            raise Exception("No merged MP4 video file found")

        # ---------- Sanitize video filename ----------
        original_video = os.path.basename(video_file)
        safe_video = safe_filename(original_video)
        safe_video_path = os.path.join(UPLOAD_DIR, safe_video)

        if safe_video != original_video:
            os.rename(video_file, safe_video_path)
        else:
            safe_video_path = video_file

        # ---------- Sanitize audio filename (NO .mp4 logic) ----------
        safe_audio_name = None
        if audio_file:
            original_audio = os.path.basename(audio_file)
            safe_audio_name = re.sub(r'[<>:"/\\|?* ]+', "_", original_audio)
            safe_audio_path = os.path.join(UPLOAD_DIR, safe_audio_name)

            if safe_audio_name != original_audio:
                os.rename(audio_file, safe_audio_path)
            else:
                safe_audio_path = audio_file

        return {
            "message": "YouTube video + audio downloaded successfully",
            "title": info.get("title"),
            "filename": safe_video,
            "video_url": f"http://localhost:8000/videos/{safe_video}",
            "audio_filename": safe_audio_name,
            "audio_url": f"http://localhost:8000/videos/{safe_audio_name}" if safe_audio_name else None,
            "file_size_mb": round(os.path.getsize(safe_video_path) / (1024 * 1024), 2)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})
 

@app.get("/video/list")
def list_videos():
    try:
        videos = []
        ALLOWED_EXTENSIONS = (".mp4", ".mp3", ".wav", ".aac", ".m4a", ".ogg", ".webm")
        for f in os.listdir(UPLOAD_DIR):
            if f.lower().endswith(ALLOWED_EXTENSIONS):
                videos.append({
                    "filename": f,
                    "video_url": f"http://localhost:8000/videos/{f}",
                    "file_size_mb": round(
                        os.path.getsize(os.path.join(UPLOAD_DIR, f)) / (1024 * 1024), 2
                    )
                })

        # Sort by newest first (optional but nice UX)
        videos.sort(
            key=lambda x: os.path.getmtime(os.path.join(UPLOAD_DIR, x["filename"])),
            reverse=True
        )

        return {
            "count": len(videos),
            "videos": videos
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.get("/video/{filename}")
def stream_video(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    def iterfile():
        with open(file_path, "rb") as f:
            yield from f

    file_size = os.path.getsize(file_path)

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": "video/mp4",
        "Cache-Control": "no-cache",
        "Content-Disposition": f'inline; filename="{filename}"'
    }

    return StreamingResponse(iterfile(), headers=headers)


def normalize_to_mp4(input_path: str, output_path: str):
    cmd = [
        r"E:\ffmpeg\bin\ffmpeg.exe",
        "-y",
        "-i", input_path,

        # Video normalize
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,"
               "pad=1280:720:(ow-iw)/2:(oh-ih)/2",
        "-r", "30",

        # Codecs
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.1",

        # Audio normalize (even if missing)
        "-c:a", "aac",
        "-b:a", "192k",

        # Important for audio-only inputs
        "-shortest",

        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg normalize failed:\n{result.stderr}")


@app.post("/video/merge")
def merge_videos(req: MergeRequest):
    try:
        normalized_paths = []

        # 1️⃣ Normalize all inputs
        for name in req.files:
            input_path = os.path.join(UPLOAD_DIR, name)

            if not os.path.exists(input_path):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"{name} not found"}
                )

            norm_name = f"norm_{uuid.uuid4().hex}.mp4"
            norm_path = os.path.join(UPLOAD_DIR, norm_name)

            normalize_to_mp4(input_path, norm_path)
            normalized_paths.append(norm_path)

        # 2️⃣ Create concat list
        list_file = os.path.join(UPLOAD_DIR, "merge_list.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for path in normalized_paths:
                f.write(f"file '{path.replace('\\', '/')}'\n")

        # 3️⃣ Merge normalized files
        output_path = os.path.join(UPLOAD_DIR, req.output_name)

        ffmpeg_cmd = [
                r"E:\ffmpeg\bin\ffmpeg.exe",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,

                # Force real concatenation
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",

                output_path
            ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"FFmpeg merge failed:\n{result.stderr}")

        if not os.path.exists(output_path):
            raise Exception("Merge failed: output not created")

        return {
            "message": "Videos merged successfully",
            "output": req.output_name,
            "video_url": f"http://localhost:8000/videos/{req.output_name}"
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
