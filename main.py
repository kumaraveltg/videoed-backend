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
from typing import List,Optional,Literal 
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles


app = FastAPI()

executor = ThreadPoolExecutor(max_workers=4)

class MergeRequest(BaseModel):
    files: list[str]
    output_name: str = "merged.mp4"

class TrimRequest(BaseModel): 
    start: float   # seconds
    end: float     # seconds

class MultiTrimRequest(BaseModel): 
    filename: str
    cuts: List[TrimRequest]   # parts to KEEP

 

class TextOverlay(BaseModel):
    text: str
    start: float
    end: float

    # positioning
    position: Literal["center", "top", "bottom", "topleft", "custom"] = "custom"
    x: Optional[float] = None
    y: Optional[float] = None

    # style
    fontsize: int = None
    fontcolor: str = None

class TextOverlayRequest(BaseModel):
    filename: str
    overlays: list[TextOverlay]

class FinalMergeRequest(BaseModel):
    main_video: str
    secondary_video: Optional[str] = None  # optional split-screen video
    audio: Optional[str] = None            # optional external audio
    text_overlays: Optional[List[TextOverlay]] = []
    output_name: str = "final_merged.mp4"
    split_mode: Literal["horizontal", "vertical", "none"] = "none"

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

UPLOAD_DIREC = r"E:\videoed\backend\ven\videouploads"
FFMPEG_PATH = r"E:\ffmpeg\bin\ffmpeg.exe"

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

def get_video_duration(file_path: str) -> float:
    cmd = [
        r"E:\ffmpeg\bin\ffprobe.exe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFprobe failed:\n{result.stderr}")

    return float(result.stdout.strip())

def delete_to_keep_ranges(delete_ranges: List[dict], total_duration: float) -> List[dict]:
    if not delete_ranges:
        return [{"start": 0, "end": total_duration}]

    delete_ranges.sort(key=lambda x: x["start"])
    keep_ranges = []

    current_start = 0.0

    for dr in delete_ranges:
        if dr["start"] > current_start:
            keep_ranges.append({"start": current_start, "end": dr["start"]})
        current_start = max(current_start, dr["end"])

    if current_start < total_duration:
        keep_ranges.append({"start": current_start, "end": total_duration})

    return keep_ranges
# -----------------------------
# Local file upload
# -----------------------------
@app.post("/upload/local")
async def upload_local(file: UploadFile = File(...)):
    filename = safe_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, filename)

    # ✅ OPTIMIZATION 1: Async file saving
    # Stream file in chunks instead of copying all at once
    async with aiofiles.open(file_path, 'wb') as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            await f.write(chunk)

    # ✅ OPTIMIZATION 2: Run thumbnails and audio extraction in PARALLEL
    # Instead of sequential (thumb → audio), do both at same time
    
    thumb_dir = os.path.join(UPLOAD_DIR, f"{filename}_thumbs")
    os.makedirs(thumb_dir, exist_ok=True)
    thumb_pattern = os.path.join(thumb_dir, "thumb_%04d.jpg")
    
    base_name = os.path.splitext(filename)[0]
    audio_filename = base_name + "_audio.m4a"
    audio_path = os.path.join(UPLOAD_DIR, audio_filename)

    # ✅ Run both FFmpeg tasks in parallel
    async def generate_thumbnails():
        """Async thumbnail generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            subprocess.run,
            [
                r"E:\ffmpeg\bin\ffmpeg.exe",
                "-y",
                "-i", file_path,
                "-vf", "fps=1,scale=160:-1",
                "-q:v", "5",  # ✅ Lower quality = faster
                "-threads", "2",  # ✅ Use 2 CPU cores
                thumb_pattern
            ],
            True,  # capture_output
            True   # text
        )

    async def extract_audio():
        """Async audio extraction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            subprocess.run,
            [
                r"E:\ffmpeg\bin\ffmpeg.exe",
                "-y",
                "-i", file_path,
                "-vn",
                "-acodec", "copy",
                audio_path
            ],
            True,  # capture_output
            True   # text
        )

    # ✅ CRITICAL: Run both tasks in parallel (saves ~50% time)
    thumb_result, audio_result = await asyncio.gather(
        generate_thumbnails(),
        extract_audio(),
        return_exceptions=True  # Don't fail if one errors
    )

    # Check results
    if isinstance(thumb_result, Exception) or thumb_result.returncode != 0:
        print(f"⚠️ Thumbnail generation failed: {thumb_result}")
    
    if isinstance(audio_result, Exception):
        print(f"⚠️ Audio extraction failed: {audio_result}")

    # ✅ OPTIMIZATION 3: Faster thumbnail collection
    # Use glob instead of listdir + filter
    import glob
    thumbnails = [
        f"http://localhost:8000/videos/{filename}_thumbs/{os.path.basename(f)}"
        for f in sorted(glob.glob(os.path.join(thumb_dir, "thumb_*.jpg")))
    ]

    # Validate audio file
    audio_exists = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
    if not audio_exists:
        audio_filename = None

    return {
        "message": "File uploaded successfully",
        "filename": filename,
        "video_url": f"http://localhost:8000/videos/{filename}",
        "audio_filename": audio_filename,
        "audio_url": f"http://localhost:8000/videos/{audio_filename}" if audio_filename else None,
        "thumbnails": thumbnails,
        "thumbnail_count": len(thumbnails)
    }

@app.post("/video/trim")
def trim_video_delete_mode(req: MultiTrimRequest):
    try:
        input_path = os.path.join(UPLOAD_DIR, req.filename)

        if not os.path.exists(input_path):
            return JSONResponse(status_code=404, content={"error": "File not found"})

        # 1️⃣ Get total duration
        total_duration = get_video_duration(input_path)

        # 2️⃣ Convert DELETE ranges to KEEP ranges
        delete_ranges = [{"start": c.start, "end": c.end} for c in req.cuts]

        keep_ranges = delete_to_keep_ranges(delete_ranges, total_duration)

        if not keep_ranges:
            return JSONResponse(
                status_code=400,
                content={"error": "Nothing left after delete"}
            )

        temp_files = []

        # 3️⃣ Trim KEEP parts
        for i, cut in enumerate(keep_ranges):
            temp_name = f"keep_part_{i}_{uuid.uuid4().hex}.mp4"
            temp_path = os.path.join(UPLOAD_DIR, temp_name)

            ffmpeg_cmd = [
                r"E:\ffmpeg\bin\ffmpeg.exe",
                "-y",

                "-ss", str(cut["start"]),
                "-to", str(cut["end"]),
                "-i", input_path,

                # FULL RESET
                "-map", "0",
                "-vf", "setpts=PTS-STARTPTS",
                "-af", "asetpts=PTS-STARTPTS",

                # Re-encode small chunks
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-c:a", "aac",

                temp_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            temp_files.append(temp_path)

        # 4️⃣ Create concat list
        list_file = os.path.join(
            UPLOAD_DIR, f"concat_{uuid.uuid4().hex}.txt"
        )

        with open(list_file, "w") as f:
            for file in temp_files:
                f.write(f"file '{file}'\n")

        # 5️⃣ Concat
        output_name = f"final_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(UPLOAD_DIR, output_name)

        concat_cmd = [
            r"E:\ffmpeg\bin\ffmpeg.exe",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path
        ]
        subprocess.run(concat_cmd, check=True)

        # 6️⃣ Cleanup
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(list_file):
            os.remove(list_file)

        return {
            "message": "Video delete-mode trim successful",
            "deleted_ranges": delete_ranges,
            "kept_ranges": keep_ranges,
            "output": output_name,
            "video_url": f"http://localhost:8000/videos/{output_name}"
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})




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
        #"-c:v", "libx264",
        "-c", "copy",
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
                "-c", "copy",
                #"-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                #"-c:a", "aac",
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


def get_text_xy_expr(pos: str, x: Optional[int], y: Optional[int]):
    if pos == "center":
        return "(w-text_w)/2", "(h-text_h)/2"
    elif pos == "top":
        return "(w-text_w)/2", "20"
    elif pos == "bottom":
        return "(w-text_w)/2", "h-text_h-20"
    elif pos == "topleft":
        return "20", "20"
    elif pos == "custom":
        return str(x or 50), str(y or 50)
    else:
        return "20", "20"  


@app.post("/video/add-text")
def add_text_overlay(req: TextOverlayRequest):

    input_path = os.path.join(UPLOAD_DIR, req.filename)

    if not os.path.exists(input_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    output_name = f"text_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(UPLOAD_DIR, output_name)

    draw_filters = []

    for o in req.overlays:
        # Escape special chars for ffmpeg
        safe_text = (
            o.text
            .replace("\\", r"\\")
            .replace(":", r"\:")
            .replace("'", r"\'")
        )

        x_expr, y_expr = get_text_xy_expr(o.position, o.x, o.y)

        draw_filters.append(
            f"drawtext=text='{safe_text}':"
            f"x={x_expr}:y={y_expr}:"
            f"fontsize={o.fontsize}:"
            f"fontcolor={o.fontcolor}:"
            f"box=1:boxcolor=black@0.4:"
            f"enable='between(t,{o.start},{o.end})'"
        )

    filter_complex = ",".join(draw_filters)

    ffmpeg_cmd = [
        r"E:\ffmpeg\bin\ffmpeg.exe",
        "-y",
        "-i", input_path,
        "-vf", filter_complex,
        "-c:v", "libx264",
        "-c:a", "copy",
        output_path
    ]

    print("FFMPEG CMD:", " ".join(ffmpeg_cmd))

    subprocess.run(ffmpeg_cmd, check=True)

    return {
        "message": "Text overlay added",
        "output": output_name,
        "video_url": f"http://localhost:8000/videos/{output_name}"
    }

class AudioModeRequest(BaseModel):
    filename: str              # input video
    audio_filename: Optional[str] = None  # optional audio
    mode: str                  # "mute" | "replace" | "mix"

@app.post("/video/audio-control")
def audio_control(req: AudioModeRequest):
    """
    Process video audio based on mode:
    - mute: Remove all audio
    - keep: Keep original video audio (no processing)
    - replace: Replace video audio with uploaded audio
    - mix: Mix video audio with uploaded audio
    """

    UPLOAD_DIREC = r"E:\videoed-backend\videouploads"
    FFMPEG_PATH = r"E:\ffmpeg\bin\ffmpeg.exe"

    input_video = os.path.join(UPLOAD_DIREC, req.filename)

    if not os.path.exists(input_video):
        return JSONResponse(status_code=404, content={"error": "Video not found"})

    audio_path = None
    if req.audio_filename:
        audio_path = os.path.join(UPLOAD_DIREC, req.audio_filename)
        if not os.path.exists(audio_path):
            return JSONResponse(status_code=404, content={"error": "Audio not found"})

    output_name = f"audio_{req.mode}_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(UPLOAD_DIREC, output_name)

    mode = req.mode.lower()

    # ---------------- KEEP ----------------
    if mode == "keep":
        # Just copy the video with its original audio (fast operation)
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_video,
            "-c:v", "copy",
            "-c:a", "copy",
            output_path
        ]

    # ---------------- MUTE ----------------
    elif mode == "mute":
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_video,
            "-c:v", "copy",
            "-an",  # Remove all audio streams
            output_path
        ]

    # ---------------- REPLACE ----------------
    elif mode == "replace":
        if not audio_path:
            return JSONResponse(status_code=400, content={"error": "audio_filename required for replace"})

        # Replace video audio with new audio
        # Loop audio if shorter than video, cut if longer
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_video,
            "-stream_loop", "-1",  # Loop audio infinitely
            "-i", audio_path,
            "-map", "0:v:0",       # Take video from first input
            "-map", "1:a:0",       # Take audio from second input
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",        # Audio bitrate
            "-shortest",           # Stop when shortest stream ends (video)
            output_path
        ]

    # ---------------- MIX ----------------
    elif mode == "mix":
        if not audio_path:
            return JSONResponse(status_code=400, content={"error": "audio_filename required for mix"})

        # Mix both audio streams
        # apad ensures both streams have same length
        filter_complex = (
            "[0:a]volume=0.7[a0];"      # Video audio at 70%
            "[1:a]volume=0.7,aloop=loop=-1:size=2e+09[a1];"  # Loop added audio
            "[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]"  # Mix and match to video length
        )

        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_video,
            "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path
        ]

    else:
        return JSONResponse(status_code=400, content={"error": "Invalid mode. Use mute | keep | replace | mix"})

    print("=" * 60)
    print(f"AUDIO CONTROL MODE: {mode.upper()}")
    print("=" * 60)
    print("FFMPEG CMD:", " ".join(ffmpeg_cmd))
    print("=" * 60)

    try:
        result = subprocess.run(
            ffmpeg_cmd, 
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ FFmpeg Success")
        
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg Error:")
        print(e.stderr)
        return JSONResponse(
            status_code=500, 
            content={"error": f"FFmpeg failed: {e.stderr[:200]}"}
        )

    return {
        "message": f"Audio {mode} successful",
        "output": output_name,
        "video_url": f"http://localhost:8000/videos/{output_name}"
    }

@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIREC, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename}

class SplitScreenRequest(BaseModel):
    top_video: str
    bottom_video: str
    audio_mode: str            # "top" | "bottom" | left| right "external" | "mute"
    audio_filename: str | None = None

@app.post("/video/split-screen")
def split_screen(req: SplitScreenRequest):

    top_path = os.path.join(UPLOAD_DIR, req.top_video)
    bottom_path = os.path.join(UPLOAD_DIR, req.bottom_video)

    if not os.path.exists(top_path) or not os.path.exists(bottom_path):
        return JSONResponse(status_code=404, content={"error": "Top or Bottom video not found"})

    audio_path = None
    if req.audio_mode == "external":
        if not req.audio_filename:
            return JSONResponse(status_code=400, content={"error": "audio_filename required"})
        audio_path = os.path.join(UPLOAD_DIR, req.audio_filename)
        if not os.path.exists(audio_path):
            return JSONResponse(status_code=404, content={"error": "External audio not found"})

    output_name = f"split_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(UPLOAD_DIR, output_name)

    # ---------------- VIDEO FILTER ----------------
    # Resize both to half height and stack vertically
    video_filter = (
        "[0:v]scale=1280:360[v0];"
        "[1:v]scale=1280:360[v1];"
        "[v0][v1]vstack=inputs=2[vout]"
    )

    # ---------------- AUDIO LOGIC ----------------
    if req.audio_mode == "top":
        audio_map = ["-map", "0:a:0"]
    elif req.audio_mode == "bottom":
        audio_map = ["-map", "1:a:0"]
    elif req.audio_mode == "external":
        audio_map = ["-map", "2:a:0"]
    elif req.audio_mode == "mute":
        audio_map = []
    else:
        return JSONResponse(status_code=400, content={"error": "Invalid audio_mode"})

    ffmpeg_cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", top_path,
        "-i", bottom_path,
    ]

    if req.audio_mode == "external":
        ffmpeg_cmd += ["-i", audio_path]

    ffmpeg_cmd += [
        "-filter_complex", video_filter,
        "-map", "[vout]",
    ]

    ffmpeg_cmd += audio_map

    ffmpeg_cmd += [
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",   # keep in sync safely
        output_path
    ]

    print("FFMPEG CMD:", " ".join(ffmpeg_cmd))
    subprocess.run(ffmpeg_cmd, check=True)

    return {
        "message": "Split screen video created",
        "output": output_name,
        "video_url": f"http://localhost:8000/videos/{output_name}"
    }

# -----------------------------
# Endpoint: Final Merge
# -----------------------------
@app.post("/video/final-merge")
def final_merge(req: FinalMergeRequest):
    try:
        inputs = []
        filter_complex = []

        main_path = os.path.join(UPLOAD_DIR, req.main_video)
        if not os.path.exists(main_path):
            return JSONResponse(status_code=404, content={"error": "Main video not found"})
        inputs.append(f"-i \"{main_path}\"")

        # Optional secondary video for split-screen
        if req.secondary_video and req.split_mode != "none":
            sec_path = os.path.join(UPLOAD_DIR, req.secondary_video)
            if not os.path.exists(sec_path):
                return JSONResponse(status_code=404, content={"error": "Secondary video not found"})
            inputs.append(f"-i \"{sec_path}\"")

        # Optional external audio
        if req.audio:
            audio_path = os.path.join(UPLOAD_DIR, req.audio)
            if not os.path.exists(audio_path):
                return JSONResponse(status_code=404, content={"error": "Audio file not found"})
            inputs.append(f"-i \"{audio_path}\"")

        # -----------------------------
        # Video filter
        # -----------------------------
        vout = "[vout]"
        if req.secondary_video and req.split_mode == "vertical":
            # stack vertically
            filter_complex.append(f"[0:v]scale=1280:360[v0];[1:v]scale=1280:360[v1];[v0][v1]vstack=inputs=2{vout}")
        elif req.secondary_video and req.split_mode == "horizontal":
            # stack horizontally
            filter_complex.append(f"[0:v]scale=640:720[v0];[1:v]scale=640:720[v1];[v0][v1]hstack=inputs=2{vout}")
        else:
            # single video
            vout = "0:v"

        # -----------------------------
        # Text overlays
        # -----------------------------
        if req.text_overlays:
            draw_filters = []
            for o in req.text_overlays:
                safe_text = o.text.replace("'", r"\'").replace(":", r'\:')
                x_expr, y_expr = get_text_xy_expr(o.position, o.x, o.y)
                draw_filters.append(
                    f"drawtext=text='{safe_text}':x={x_expr}:y={y_expr}:fontsize={o.fontsize}:fontcolor={o.fontcolor}:box=1:boxcolor=black@0.4:enable='between(t,{o.start},{o.end})'"
                )
            # attach drawtext to main video or stacked video
            drawtext_filter = ",".join(draw_filters)
            if req.secondary_video != None:
                filter_complex.append(f"{vout},{drawtext_filter}{vout}")
            else:
                filter_complex.append(f"[0:v]{drawtext_filter}{vout}")

        # -----------------------------
        # Audio mapping
        # -----------------------------
        audio_map = []
        if req.audio:
            # external audio mixed with main video
            audio_map = ["-map", f"{len(inputs)-1}:a:0", "-map", f"0:a:0", "-filter_complex", "[0:a][1:a]amix=inputs=2[aout]", "-map", "[aout]"]
        else:
            audio_map = ["-map", f"{0}:a:0"]

        # -----------------------------
        # Output
        # -----------------------------
        output_name = req.output_name
        output_path = os.path.join(UPLOAD_DIR, output_name)

        ffmpeg_cmd = f'{FFMPEG_PATH} -y {" ".join(inputs)}'
        if filter_complex:
            ffmpeg_cmd += f' -filter_complex "{";".join(filter_complex)}"'
        ffmpeg_cmd += " -c:v libx264 -c:a aac -b:a 192k -pix_fmt yuv420p "
        if audio_map:
            ffmpeg_cmd += " " + " ".join(audio_map)
        ffmpeg_cmd += f' "{output_path}"'

        print("FFMPEG CMD:", ffmpeg_cmd)
        subprocess.run(ffmpeg_cmd, shell=True, check=True)

        return {
            "message": "Final merge successful",
            "output": output_name,
            "video_url": f"http://localhost:8000/videos/{output_name}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})
    

# =====================================================
# MULTIPLE VIDEO INSERT MODELS
# =====================================================

class VideoInsertConfig(BaseModel):
    """Configuration for a single video insert/overlay"""
    insert_filename: str       # The video to insert
    start_time: float          # When insert appears (seconds)
    end_time: Optional[float] = None  # When insert disappears (auto-detect if None)
    x: int                     # X position (pixels from left)
    y: int                     # Y position (pixels from top)
    width: int                 # Insert width
    height: int                # Insert height
    opacity: float = 1.0       # 0.0 to 1.0
    volume: float = 0.5        # 0.0 to 1.0 (insert audio volume)
    z_index: int = 1           # Layer order (higher = on top)
    loop: bool = False         # Loop insert video if shorter than duration
    fade_in: float = 0.0       # Fade in duration (seconds)
    fade_out: float = 0.0      # Fade out duration (seconds)

class MultipleVideoInsertRequest(BaseModel):
    """Request to add multiple video inserts to main video"""
    main_video: str                           # Main/base video filename
    inserts: List[VideoInsertConfig]          # List of insert configurations
    output_name: Optional[str] = None         # Optional output filename
    keep_main_audio: bool = True              # Keep main video audio
    mix_insert_audio: bool = False            # Mix insert audio with main
    background_color: str = "black"           # Background color if needed

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe"""
    probe_cmd = [
        r"E:\ffmpeg\bin\ffprobe.exe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,avg_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]
    
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    stream = data['streams'][0]
    format_data = data.get('format', {})
    
    return {
        "width": int(stream['width']),
        "height": int(stream['height']),
        "duration": float(format_data.get('duration', stream.get('duration', 0))),
        "frame_rate": stream.get('avg_frame_rate', '30/1')
    }

def build_fade_filter(insert_label: str, fade_in: float, fade_out: float, 
                      start_time: float, end_time: float) -> str:
    """Build fade in/out filter for an insert"""
    filters = []
    
    if fade_in > 0:
        # Fade in at the start
        filters.append(f"fade=t=in:st={start_time}:d={fade_in}")
    
    if fade_out > 0:
        # Fade out before end
        fade_start = end_time - fade_out
        filters.append(f"fade=t=out:st={fade_start}:d={fade_out}")
    
    if filters:
        return f"[{insert_label}]{','.join(filters)}[{insert_label}_faded]"
    
    return None

# =====================================================
# MAIN ENDPOINT: MULTIPLE VIDEO INSERTS
# =====================================================

@app.post("/video/add-multiple-inserts")
def add_multiple_video_inserts(req: MultipleVideoInsertRequest):
    """
    Add multiple video inserts (Picture-in-Picture) to main video
    
    Features:
    - Multiple inserts at different times and positions
    - Z-index layering (control which video appears on top)
    - Custom positioning, sizing, and opacity
    - Audio mixing with volume control
    - Fade in/out transitions
    - Loop inserts if shorter than duration
    - Timeline-based insertion
    """
    
    try:
        # =====================================================
        # VALIDATION
        # =====================================================
        
        main_path = os.path.join(UPLOAD_DIR, req.main_video)
        if not os.path.exists(main_path):
            return JSONResponse(
                status_code=404, 
                content={"error": "Main video not found"}
            )
        
        # Get main video info
        main_info = get_video_info(main_path)
        main_duration = main_info['duration']
        
        print(f"📹 Main video: {req.main_video}")
        print(f"   Dimensions: {main_info['width']}x{main_info['height']}")
        print(f"   Duration: {main_duration:.2f}s")
        
        # Validate and process all inserts
        processed_inserts = []
        
        for idx, insert in enumerate(req.inserts):
            insert_path = os.path.join(UPLOAD_DIR, insert.insert_filename)
            
            if not os.path.exists(insert_path):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Insert video not found: {insert.insert_filename}"}
                )
            
            # Get insert video info
            insert_info = get_video_info(insert_path)
            
            # Auto-detect end_time if not provided
            if insert.end_time is None:
                insert.end_time = min(
                    insert.start_time + insert_info['duration'],
                    main_duration
                )
            
            # Validate times
            if insert.start_time >= main_duration:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Insert {idx+1} start_time ({insert.start_time}) exceeds main video duration ({main_duration})"}
                )
            
            if insert.end_time > main_duration:
                print(f"⚠️ Insert {idx+1} end_time adjusted from {insert.end_time} to {main_duration}")
                insert.end_time = main_duration
            
            processed_inserts.append({
                "config": insert,
                "path": insert_path,
                "info": insert_info,
                "index": idx
            })
            
            print(f"✅ Insert {idx+1}: {insert.insert_filename}")
            print(f"   Time: {insert.start_time:.2f}s → {insert.end_time:.2f}s")
            print(f"   Position: ({insert.x}, {insert.y})")
            print(f"   Size: {insert.width}x{insert.height}")
            print(f"   Z-Index: {insert.z_index}")
        
        # Sort inserts by z_index (lower z-index rendered first, appears behind)
        processed_inserts.sort(key=lambda x: x['config'].z_index)
        
        # =====================================================
        # GENERATE OUTPUT FILENAME
        # =====================================================
        
        if req.output_name:
            output_name = req.output_name
        else:
            output_name = f"multi_insert_{uuid.uuid4().hex}.mp4"
        
        output_path = os.path.join(UPLOAD_DIR, output_name)
        
        # =====================================================
        # BUILD FFMPEG FILTER COMPLEX
        # =====================================================
        
        input_files = ["-i", main_path]
        filter_parts = []
        audio_parts = []
        
        # Add all insert videos as inputs
        for proc_insert in processed_inserts:
            input_files.extend(["-i", proc_insert['path']])
        
        # Build video filters
        current_base = "0:v"  # Start with main video
        
        for proc_insert in processed_inserts:
            insert = proc_insert['config']
            input_idx = proc_insert['index'] + 1  # Main is 0, inserts start at 1
            
            # =====================================================
            # STEP 1: Scale insert to desired size
            # =====================================================
            
            insert_label = f"insert{proc_insert['index']}"
            scale_filter = f"[{input_idx}:v]scale={insert.width}:{insert.height}"
            
            # =====================================================
            # STEP 2: Add looping if needed
            # =====================================================
            
            insert_duration = insert.end_time - insert.start_time
            video_duration = proc_insert['info']['duration']
            
            if insert.loop and video_duration < insert_duration:
                # Loop the insert video
                loop_count = int(insert_duration / video_duration) + 1
                scale_filter += f",loop={loop_count}:1:0"
            
            scale_filter += f"[{insert_label}]"
            filter_parts.append(scale_filter)
            
            # =====================================================
            # STEP 3: Apply opacity if needed
            # =====================================================
            
            working_label = insert_label
            
            if insert.opacity < 1.0:
                opacity_label = f"{insert_label}_opacity"
                opacity_filter = (
                    f"[{working_label}]"
                    f"format=yuva420p,colorchannelmixer=aa={insert.opacity}"
                    f"[{opacity_label}]"
                )
                filter_parts.append(opacity_filter)
                working_label = opacity_label
            
            # =====================================================
            # STEP 4: Apply fade in/out if needed
            # =====================================================
            
            if insert.fade_in > 0 or insert.fade_out > 0:
                fade_label = f"{insert_label}_faded"
                fade_filters = []
                
                if insert.fade_in > 0:
                    fade_filters.append(f"fade=t=in:st=0:d={insert.fade_in}:alpha=1")
                
                if insert.fade_out > 0:
                    fade_start = insert_duration - insert.fade_out
                    fade_filters.append(f"fade=t=out:st={fade_start}:d={insert.fade_out}:alpha=1")
                
                fade_filter = (
                    f"[{working_label}]{','.join(fade_filters)}[{fade_label}]"
                )
                filter_parts.append(fade_filter)
                working_label = fade_label
            
            # =====================================================
            # STEP 5: Overlay on current base
            # =====================================================
            
            output_label = f"out{proc_insert['index']}"
            
            # Build overlay filter with time constraint
            overlay_filter = (
                f"[{current_base}][{working_label}]"
                f"overlay=x={insert.x}:y={insert.y}:"
                f"enable='between(t,{insert.start_time},{insert.end_time})'"
            )
            
            # Check if this is the last insert
            if proc_insert['index'] == len(processed_inserts) - 1:
                output_label = "vout"
            
            overlay_filter += f"[{output_label}]"
            filter_parts.append(overlay_filter)
            
            # Update current base for next overlay
            current_base = output_label
            
            # =====================================================
            # STEP 6: Audio handling
            # =====================================================
            
            if req.mix_insert_audio and insert.volume > 0:
                audio_label = f"audio{proc_insert['index']}"
                
                # Extract audio, adjust volume, and trim to insert duration
                audio_filter = (
                    f"[{input_idx}:a]"
                    f"volume={insert.volume},"
                    f"atrim=start={insert.start_time}:end={insert.end_time},"
                    f"asetpts=PTS-STARTPTS"
                    f"[{audio_label}]"
                )
                audio_parts.append(audio_label)
                filter_parts.append(audio_filter)
        
        # =====================================================
        # AUDIO MIXING
        # =====================================================
        
        audio_map = []
        
        if req.keep_main_audio and req.mix_insert_audio and len(audio_parts) > 0:
            # Mix main audio with all insert audios
            
            # Prepare main audio
            filter_parts.append("[0:a]volume=1.0[main_audio]")
            
            # Build amix with all audio sources
            all_audio_inputs = ["main_audio"] + audio_parts
            amix_filter = (
                f"[{']['.join(all_audio_inputs)}]"
                f"amix=inputs={len(all_audio_inputs)}:"
                f"duration=first:"
                f"dropout_transition=2"
                f"[aout]"
            )
            
            filter_parts.append(amix_filter)
            audio_map = ["-map", "[aout]"]
            
        elif req.keep_main_audio:
            # Keep only main video audio
            audio_map = ["-map", "0:a"]
            
        else:
            # No audio
            audio_map = ["-an"]
        
        # =====================================================
        # COMBINE FILTERS
        # =====================================================
        
        filter_complex = ";".join(filter_parts)
        
        # =====================================================
        # BUILD FFMPEG COMMAND
        # =====================================================
        
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-y",
            *input_files,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            *audio_map,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",  # Enable fast start for web playback
        ]
        
        # Add audio codec if audio is included
        if audio_map != ["-an"]:
            ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        
        ffmpeg_cmd.append(output_path)
        
        # =====================================================
        # EXECUTE FFMPEG
        # =====================================================
        
        print("=" * 80)
        print("MULTIPLE VIDEO INSERTS COMMAND")
        print("=" * 80)
        print(" ".join(ffmpeg_cmd))
        print("=" * 80)
        print("\nFILTER COMPLEX:")
        print(filter_complex)
        print("=" * 80)
        
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Multiple video inserts successful")
        
        return {
            "message": "Multiple video inserts added successfully",
            "output": output_name,
            "video_url": f"http://localhost:8000/videos/{output_name}",
            "inserts_count": len(processed_inserts),
            "main_video_duration": main_duration,
            "inserts_summary": [
                {
                    "filename": p['config'].insert_filename,
                    "time_range": f"{p['config'].start_time:.2f}s - {p['config'].end_time:.2f}s",
                    "position": f"({p['config'].x}, {p['config'].y})",
                    "size": f"{p['config'].width}x{p['config'].height}",
                    "z_index": p['config'].z_index
                }
                for p in processed_inserts
            ]
        }
        
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg Error:")
        print(e.stderr)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"FFmpeg failed: {e.stderr[:500]}",
                "full_error": e.stderr
            }
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

# =====================================================
# VIDEO INSERT (SPLIT & CONCATENATE)
# =====================================================

class VideoInsertAtPositionRequest(BaseModel):
    """Request to insert video(s) at specific positions in main video"""
    main_video: str                    # Main video filename
    inserts: List[dict]                # [{"filename": "clip.mp4", "position": 25.0}, ...]
    output_name: Optional[str] = None

@app.post("/video/insert-at-position")
def insert_video_at_position(req: VideoInsertAtPositionRequest):
    """
    Fast video insert using single-pass filter_complex.
    """
    
    temp_files = []  # Initialize at the top for cleanup
    
    try:
        main_path = os.path.join(UPLOAD_DIR, req.main_video)
        if not os.path.exists(main_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Main video not found"}
            )
        
        # Get main video info
        main_info = get_video_info(main_path)
        main_duration = main_info['duration']
        target_width = main_info['width']
        target_height = main_info['height']
        
        print(f"📹 Main video: {req.main_video}")
        print(f"   Duration: {main_duration:.2f}s")
        print(f"   Resolution: {target_width}x{target_height}")
        
        # =====================================================
        # STEP 1: Validate and sort inserts
        # =====================================================
        
        validated_inserts = []
        
        for insert in req.inserts:
            insert_path = os.path.join(UPLOAD_DIR, insert['filename'])
            
            if not os.path.exists(insert_path):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Insert video not found: {insert['filename']}"}
                )
            
            insert_info = get_video_info(insert_path)
            position = float(insert['position'])
            
            if position < 0 or position > main_duration:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid position {position}. Must be 0-{main_duration}"}
                )
            
            validated_inserts.append({
                'filename': insert['filename'],
                'path': insert_path,
                'position': position,
                'duration': insert_info['duration'],
                'width': insert_info['width'],
                'height': insert_info['height']
            })
            
            print(f"✅ Insert: {insert['filename']} at {position}s")
        
        # Sort by position
        validated_inserts.sort(key=lambda x: x['position'])
        
        # =====================================================
        # STEP 2: Build single-pass filter_complex
        # =====================================================
        
        # Prepare input files
        input_args = ["-i", main_path]
        insert_input_map = {}
        
        for idx, insert in enumerate(validated_inserts):
            input_args.extend(["-i", insert['path']])
            insert_input_map[insert['position']] = idx + 1
        
        # Build filter chain
        filter_parts = []
        segment_labels = []
        current_time = 0.0
        segment_idx = 0
        
        for insert in validated_inserts:
            insert_pos = insert['position']
            insert_input_idx = insert_input_map[insert_pos]
            
            # Main video segment BEFORE insert
            if insert_pos > current_time:
                duration = insert_pos - current_time
                
                main_segment_label = f"main{segment_idx}"
                filter_parts.append(
                    f"[0:v]trim=start={current_time}:end={insert_pos},setpts=PTS-STARTPTS,"
                    f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                    f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
                    f"[{main_segment_label}v]"
                )
                
                filter_parts.append(
                    f"[0:a]atrim=start={current_time}:end={insert_pos},asetpts=PTS-STARTPTS"
                    f"[{main_segment_label}a]"
                )
                
                segment_labels.append(f"[{main_segment_label}v][{main_segment_label}a]")
                segment_idx += 1
            
            # Insert video
            insert_segment_label = f"insert{segment_idx}"
            
            filter_parts.append(
                f"[{insert_input_idx}:v]scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
                f"[{insert_segment_label}v]"
            )
            
            filter_parts.append(
                f"[{insert_input_idx}:a]anull[{insert_segment_label}a]"
            )
            
            segment_labels.append(f"[{insert_segment_label}v][{insert_segment_label}a]")
            segment_idx += 1
            
            current_time = insert_pos
        
        # Remaining main video after last insert
        if current_time < main_duration:
            main_final_label = f"main{segment_idx}"
            
            filter_parts.append(
                f"[0:v]trim=start={current_time},setpts=PTS-STARTPTS,"
                f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
                f"[{main_final_label}v]"
            )
            
            filter_parts.append(
                f"[0:a]atrim=start={current_time},asetpts=PTS-STARTPTS"
                f"[{main_final_label}a]"
            )
            
            segment_labels.append(f"[{main_final_label}v][{main_final_label}a]")
        
        # Concatenate all segments
        concat_inputs = "".join(segment_labels)
        filter_parts.append(
            f"{concat_inputs}concat=n={len(segment_labels)}:v=1:a=1[outv][outa]"
        )
        
        # Combine all filters
        filter_complex = ";".join(filter_parts)
        
        # =====================================================
        # STEP 3: Single FFmpeg command
        # =====================================================
        
        if req.output_name:
            output_name = req.output_name
        else:
            output_name = f"inserted_{uuid.uuid4().hex}.mp4"
        
        output_path = os.path.join(UPLOAD_DIR, output_name)
        
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-y",
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            output_path
        ]
        
        print("=" * 80)
        print("SINGLE-PASS VIDEO INSERT")
        print("=" * 80)
        print(f"Segments to process: {len(segment_labels)}")
        print(f"Preset: ultrafast")
        print("=" * 80)
        
        # Execute
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Get final info
        final_info = get_video_info(output_path)
        final_duration = final_info['duration']
        
        print(f"✅ Single-pass insert complete!")
        print(f"   Original: {main_duration:.2f}s")
        print(f"   Final: {final_duration:.2f}s")
        
        return {
            "message": "Video insert successful",
            "output": output_name,
            "video_url": f"http://localhost:8000/videos/{output_name}",
            "original_duration": main_duration,
            "final_duration": final_duration,
            "resolution": f"{final_info['width']}x{final_info['height']}",
            "inserts_count": len(validated_inserts)
        }
    
    # ✅ HERE IS THE REQUIRED EXCEPT BLOCK
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg Error:")
        print(e.stderr[-2000:] if e.stderr else "No error output")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "FFmpeg processing failed",
                "details": e.stderr[-1000:] if e.stderr else "No details"
            }
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    
    # =====================================================
# IMAGE OVERLAY MODELS
# =====================================================

class ImageOverlay(BaseModel):
    """Single image overlay configuration"""
    image_filename: str           # Image file to overlay
    start: float                  # When image appears (seconds)
    end: float                    # When image disappears (seconds)
    x: int                        # X position (pixels from left)
    y: int                        # Y position (pixels from top)
    width: int                    # Image width
    height: int                   # Image height
    opacity: float = 1.0          # 0.0 to 1.0
    fade_in: float = 0.0          # Fade in duration (seconds)
    fade_out: float = 0.0         # Fade out duration (seconds)

class AddImageOverlaysRequest(BaseModel):
    """Request to add multiple image overlays to video"""
    filename: str                          # Main video filename
    overlays: List[ImageOverlay]           # List of image overlays
    output_name: Optional[str] = None

# =====================================================
# IMAGE OVERLAY ENDPOINT
# =====================================================

@app.post("/video/add-image-overlays")
def add_image_overlays(req: AddImageOverlaysRequest):
    """
    Add multiple image overlays to video.
    
    Features:
    - Multiple images at different times
    - Custom positioning and sizing
    - Opacity control
    - Fade in/out transitions
    - Timeline-based display
    
    Example:
    - Add logo in corner from 0-10s
    - Add watermark from 20-30s
    - Add sticker from 5-15s
    """
    
    try:
        # =====================================================
        # VALIDATION
        # =====================================================
        
        video_path = os.path.join(UPLOAD_DIR, req.filename)
        if not os.path.exists(video_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Video file not found"}
            )
        
        # Get video info
        video_info = get_video_info(video_path)
        video_duration = video_info['duration']
        
        print(f"📹 Video: {req.filename}")
        print(f"   Duration: {video_duration:.2f}s")
        print(f"   Resolution: {video_info['width']}x{video_info['height']}")
        
        # Validate image overlays
        validated_overlays = []
        
        for idx, overlay in enumerate(req.overlays):
            image_path = os.path.join(UPLOAD_DIR, overlay.image_filename)
            
            if not os.path.exists(image_path):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Image not found: {overlay.image_filename}"}
                )
            
            # Validate time range
            if overlay.start >= video_duration:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Overlay {idx+1} start time exceeds video duration"}
                )
            
            if overlay.end > video_duration:
                print(f"⚠️ Overlay {idx+1} end time adjusted to video duration")
                overlay.end = video_duration
            
            validated_overlays.append({
                'config': overlay,
                'path': image_path,
                'index': idx
            })
            
            print(f"✅ Image overlay {idx+1}: {overlay.image_filename}")
            print(f"   Time: {overlay.start:.2f}s → {overlay.end:.2f}s")
            print(f"   Position: ({overlay.x}, {overlay.y})")
            print(f"   Size: {overlay.width}x{overlay.height}")
        
        # =====================================================
        # BUILD FFMPEG FILTER COMPLEX
        # =====================================================
        
        # Start with video input
        filter_parts = []
        current_video_label = "0:v"
        
        for proc_overlay in validated_overlays:
            overlay = proc_overlay['config']
            input_idx = proc_overlay['index'] + 1  # Video is input 0
            
            overlay_label = f"img{proc_overlay['index']}"
            
            # =====================================================
            # STEP 1: Scale and format image
            # =====================================================
            
            scale_filter = (
                f"[{input_idx}:v]"
                f"scale={overlay.width}:{overlay.height},"
                f"format=yuva420p"  # Format with alpha channel
                f"[{overlay_label}_scaled]"
            )
            filter_parts.append(scale_filter)
            
            working_label = f"{overlay_label}_scaled"
            
            # =====================================================
            # STEP 2: Apply opacity
            # =====================================================
            
            if overlay.opacity < 1.0:
                opacity_label = f"{overlay_label}_opacity"
                opacity_filter = (
                    f"[{working_label}]"
                    f"colorchannelmixer=aa={overlay.opacity}"
                    f"[{opacity_label}]"
                )
                filter_parts.append(opacity_filter)
                working_label = opacity_label
            
            # =====================================================
            # STEP 3: Apply fade in/out
            # =====================================================
            
            if overlay.fade_in > 0 or overlay.fade_out > 0:
                fade_label = f"{overlay_label}_faded"
                fade_filters = []
                
                if overlay.fade_in > 0:
                    fade_filters.append(
                        f"fade=t=in:st=0:d={overlay.fade_in}:alpha=1"
                    )
                
                if overlay.fade_out > 0:
                    duration = overlay.end - overlay.start
                    fade_start = duration - overlay.fade_out
                    fade_filters.append(
                        f"fade=t=out:st={fade_start}:d={overlay.fade_out}:alpha=1"
                    )
                
                fade_filter = (
                    f"[{working_label}]{','.join(fade_filters)}[{fade_label}]"
                )
                filter_parts.append(fade_filter)
                working_label = fade_label
            
            # =====================================================
            # STEP 4: Overlay on video
            # =====================================================
            
            output_label = f"v{proc_overlay['index']}"
            
            # Build overlay filter with time constraint
            overlay_filter = (
                f"[{current_video_label}][{working_label}]"
                f"overlay=x={overlay.x}:y={overlay.y}:"
                f"enable='between(t,{overlay.start},{overlay.end})'"
            )
            
            # Check if this is the last overlay
            if proc_overlay['index'] == len(validated_overlays) - 1:
                output_label = "vout"
            
            overlay_filter += f"[{output_label}]"
            filter_parts.append(overlay_filter)
            
            # Update current video label for next overlay
            current_video_label = output_label
        
        # Combine filters
        filter_complex = ";".join(filter_parts)
        
        # =====================================================
        # BUILD FFMPEG COMMAND
        # =====================================================
        
        # Prepare inputs (video + all images)
        input_args = ["-i", video_path]
        for proc_overlay in validated_overlays:
            input_args.extend(["-loop", "1", "-i", proc_overlay['path']])
        
        # Generate output filename
        if req.output_name:
            output_name = req.output_name
        else:
            output_name = f"img_overlay_{uuid.uuid4().hex}.mp4"
        
        output_path = os.path.join(UPLOAD_DIR, output_name)
        
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-y",
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "0:a",  # Keep original audio
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-shortest",  # End when video ends
            output_path
        ]
        
        print("=" * 80)
        print("IMAGE OVERLAY COMMAND")
        print("=" * 80)
        print(" ".join(ffmpeg_cmd))
        print("=" * 80)
        print("\nFILTER COMPLEX:")
        print(filter_complex)
        print("=" * 80)
        
        # Execute FFmpeg
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Image overlays added successfully")
        
        return {
            "message": "Image overlays added successfully",
            "output": output_name,
            "video_url": f"http://localhost:8000/videos/{output_name}",
            "overlays_count": len(validated_overlays),
            "video_duration": video_duration,
            "overlays_summary": [
                {
                    "image": p['config'].image_filename,
                    "time_range": f"{p['config'].start:.2f}s - {p['config'].end:.2f}s",
                    "position": f"({p['config'].x}, {p['config'].y})",
                    "size": f"{p['config'].width}x{p['config'].height}",
                    "opacity": p['config'].opacity
                }
                for p in validated_overlays
            ]
        }
        
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg Error:")
        print(e.stderr)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"FFmpeg failed: {e.stderr[:500]}",
                "full_error": e.stderr
            }
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )