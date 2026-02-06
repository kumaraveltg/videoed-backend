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

app = FastAPI()

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

    # ---- Save uploaded video ----
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ===============================
    # 🎬 Generate timeline thumbnails
    # ===============================
    thumb_dir = os.path.join(UPLOAD_DIR, f"{filename}_thumbs")
    os.makedirs(thumb_dir, exist_ok=True)

    thumb_pattern = os.path.join(thumb_dir, "thumb_%04d.jpg")

    result = subprocess.run(
    [
        r"E:\ffmpeg\bin\ffmpeg.exe",
        "-y",
        "-i", file_path,
        "-vf", "fps=1,scale=160:-1",
        thumb_pattern
    ],
    capture_output=True,
    text=True
)

    if result.returncode != 0:
        print(f"FFmpeg thumbnail generation failed: {result.stderr}")

    # collect thumbnails
    thumbnails = []
    for f in sorted(os.listdir(thumb_dir)):
        if f.startswith("thumb_"):
            thumbnails.append(
                f"http://localhost:8000/videos/{filename}_thumbs/{f}"
            )

    # ===============================
    # 🔊 Extract audio (your original code)
    # ===============================
    base_name = os.path.splitext(filename)[0]
    audio_filename = base_name + "_audio.m4a"
    audio_path = os.path.join(UPLOAD_DIR, audio_filename)

    ffmpeg_audio_cmd = (
        f'"{r"E:\ffmpeg\bin\ffmpeg.exe"}" '
        f'-y -i "{file_path}" -vn -acodec copy "{audio_path}"'
    )

    os.system(ffmpeg_audio_cmd)

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        audio_filename = None

    return {
        "message": "File uploaded successfully",
        "filename": filename,
        "video_url": f"http://localhost:8000/videos/{filename}",
        "audio_filename": audio_filename,
        "audio_url": f"http://localhost:8000/videos/{audio_filename}" if audio_filename else None,
        "thumbnails": thumbnails
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

    UPLOAD_DIREC = r"E:\videoed\backend\ven\videouploads"
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

    # ---------------- MUTE ----------------
    if mode == "mute":
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_video,
            "-c:v", "copy",
            "-an",
            output_path
        ]

    # ---------------- REPLACE ----------------
    elif mode == "replace":
        if not audio_path:
            return JSONResponse(status_code=400, content={"error": "audio_filename required for replace"})

        # Loop audio so it always matches video length
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_video,
            "-stream_loop", "-1",
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]

    # ---------------- MIX ----------------
    elif mode == "mix":
        if not audio_path:
            return JSONResponse(status_code=400, content={"error": "audio_filename required for mix"})

        filter_complex = (
            "[0:a]volume=1.0[a0];"
            "[1:a]volume=1.0,apad[a1];"
            "[a0][a1]amix=inputs=2:duration=longest[aout]"
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
            output_path
        ]

    else:
        return JSONResponse(status_code=400, content={"error": "Invalid mode. Use mute | replace | mix"})

    print("FFMPEG CMD:", " ".join(ffmpeg_cmd))

    subprocess.run(ffmpeg_cmd, check=True)

    return {
        "message": f"Audio {mode} successful",
        "output": output_name,
        "video_url": f"http://localhost:8000/videos/{output_name}"
    }


class SplitScreenRequest(BaseModel):
    top_video: str
    bottom_video: str
    audio_mode: str            # "top" | "bottom" | "external" | "mute"
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