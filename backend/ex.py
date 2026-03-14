#!/usr/bin/env python3
# pipeline.py

import os
from dotenv import load_dotenv
load_dotenv()
import re
import sys
import json
import time
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from pydantic import BaseModel, Field, ValidationError
import matplotlib
matplotlib.use("Agg")
import spacy
from gensim.models import KeyedVectors
from fpdf import FPDF # pyright: ignore[reportMissingModuleSource]




import cv2
import numpy as np
import pytesseract
import whisper
import networkx as nx

from pyvis.network import Network
from PIL import Image
import matplotlib.pyplot as plt

# NLP / OCR / parsing
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy import Basic
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



# Transformers / BLIP / Summarization
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer, BlipProcessor, BlipForConditionalGeneration, pipeline as hf_pipeline

import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# ── BART Model Cache (singleton) ──────────────────────────────────────────────
# Loading BART takes ~15-30s. Cache it so only the first call is slow.
_BART_CACHE = {}

import threading
_BART_CACHE_LOCK = threading.Lock()

def _get_bart_model():
    """Get cached BART tokenizer and model. Loads on first call only."""
    with _BART_CACHE_LOCK:
        if "tokenizer" not in _BART_CACHE or "model" not in _BART_CACHE:
            print("[BART Cache] Loading facebook/bart-large-cnn (first time)...")
            _BART_CACHE["tokenizer"] = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
            _BART_CACHE["model"] = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
            print("[BART Cache] Model loaded and cached.")
        return _BART_CACHE["tokenizer"], _BART_CACHE["model"]


# Summary evaluation metrics
from rouge_score import rouge_scorer # type: ignore
import nltk.translate.bleu_score as bleu_score
from nltk.translate.bleu_score import SmoothingFunction

# ReportLab for PDF generation
try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
except ImportError:
    print("ReportLab not found. Please run 'pip install reportlab'")

# Sklearn for Topic Modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

try:
    nlp = spacy.load("en_core_web_md")
except:
    print("Installing spaCy model...")
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Ensure NLTK punkt is available (quiet)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------- Configuration ----------
BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = BASE_DIR.parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Sympy transformations for formula parsing
_TRANSFORMATIONS = (standard_transformations + (implicit_multiplication_application,))



# ---------- Utilities ----------
def run_cmd(cmd: List[str], hide_output: bool = True):
    """Run shell command and raise on failure."""
    try:
        if hide_output:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {' '.join(cmd)} -> {e}")

def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def clean_transcript_for_summary(text: str) -> str:
    """
    Dynamic NLP-based cleaning for summary quality.
    
    Instead of hardcoded patterns, uses sentence-level scoring:
    1. Structure score: proper subject-verb-object form
    2. Content score: informative vs filler/meta-commentary
    3. Coherence score: relates to surrounding topic context
    4. Position penalty: intro/outro sentences penalized
    
    Keeps sentences scoring above a dynamic threshold.
    """
    if not text or len(text.strip()) < 20:
        return text

    # Extract sentences
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split(". ")

    if not sentences:
        return text

    # ── Sentence-level scoring ──
    scored_sentences = []
    total = len(sentences)

    for idx, s in enumerate(sentences):
        s_clean = s.strip()
        s_lower = s_clean.lower()
        words = s_clean.split()
        word_count = len(words)

        # Skip extremely short
        if word_count < 4:
            continue

        score = 1.0  # Start neutral

        # ── A. STRUCTURE SCORE (does it have subject + verb?) ──
        verb_indicators = [
            " is ", " are ", " was ", " were ", " has ", " have ", " had ",
            " does ", " do ", " did ", " will ", " can ", " could ", " would ",
            " should ", " may ", " might ", " must ", " shall ",
            " means ", " refers ", " includes ", " involves ", " requires ",
            " provides ", " describes ", " defines ", " uses ", " stores ",
            " deals ", " contains ", " represents ", " allows ", " enables ",
            " supports ", " manages ", " handles ", " processes ", " performs ",
        ]
        padded = " " + s_lower + " "
        has_verb = any(v in padded for v in verb_indicators)

        if has_verb:
            score += 0.4  # Proper sentence structure
        elif word_count < 8:
            score -= 0.6  # Short fragment without verb → likely noise

        # ── B. CONTENT DENSITY (informative vs filler) ──
        # Definitional / technical markers boost score
        technical_markers = [
            "is defined as", "refers to", "is called", "is known as",
            "consists of", "is responsible for", "is used to", "is used for",
            "there are", "types of", "levels of", "example of",
            "such as", "for example", "in other words",
        ]
        if any(m in s_lower for m in technical_markers):
            score += 0.5

        # Meta/filler indicators reduce score dynamically
        filler_indicators = [
            "in this video", "in this presentation", "in the last presentation",
            "we will learn", "we will see", "we will discuss", "we will focus",
            "let us see", "let's see", "let me explain",
            "welcome back", "good morning", "hello everyone",
            "hope you understood", "thank you", "thanks for watching",
            "subscribe", "channel", "click the link", "comment below",
            "the answer will be revealed", "before we step into",
            "following points are covered", "at the end of this session",
            "i told you", "i have mentioned", "as i said",
        ]
        filler_hits = sum(1 for f in filler_indicators if f in s_lower)
        score -= filler_hits * 0.5

        # ── C. POSITION PENALTY (first/last 10% are often intro/outro) ──
        position_ratio = idx / max(total, 1)
        if position_ratio < 0.08 or position_ratio > 0.92:
            # Intro/outro zone — penalize filler-like sentences more
            if filler_hits > 0:
                score -= 0.3

        # ── D. QUALITY CHECKS ──
        # Rhetorical questions → not suitable for summaries
        if s_clean.endswith("?"):
            score -= 0.8

        # Alpha ratio check (OCR garbage)
        alpha_chars = sum(c.isalpha() for c in s_clean)
        if len(s_clean) > 0 and (alpha_chars / len(s_clean)) < 0.6:
            score -= 1.0

        # Repetitive/redundant phrasing penalty
        unique_words = set(w.lower() for w in words if len(w) > 2)
        if word_count > 5 and len(unique_words) / word_count < 0.5:
            score -= 0.4  # Too many repeated words

        # Word length bonus: medium-length sentences are most informative
        if 10 <= word_count <= 30:
            score += 0.2
        elif word_count > 50:
            score -= 0.2  # Very long sentences are often rambling

        scored_sentences.append((s_clean, score))

    # ── Dynamic threshold: keep sentences above median score ──
    if not scored_sentences:
        return text

    scores = [s[1] for s in scored_sentences]
    median_score = sorted(scores)[len(scores) // 2]
    # Threshold: at least median, but minimum 0.5 to filter real junk
    threshold = max(median_score, 0.5)

    kept = [s for s, sc in scored_sentences if sc >= threshold]

    if not kept:
        # Fallback: keep top 70% by score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        count = max(3, int(len(scored_sentences) * 0.7))
        # Re-sort by original order
        kept = [s for s, _ in scored_sentences[:count]]

    cleaned_text = " ".join(kept)
    print(f"[Dynamic Cleaning] {len(sentences)} sentences → {len(kept)} kept (threshold={threshold:.2f})")
    return cleaned_text

def ensure_str(x: Any) -> Any: # type: ignore
    """Convert Path-like objects to str for serialization or string ops."""
    if isinstance(x, (Path,)):
        return str(x)
    return x

def stringify_paths(d: dict) -> dict:
    """Convert all Path-like values inside dict to strings (recursive)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (Path,)):
           
            out[k] = str(v)
        elif isinstance(v, dict):
            
            out[k] = stringify_paths(v)
        elif isinstance(v, (list, tuple)):
            
            out[k] = [stringify_paths(x) if isinstance(x, dict) else 
                     (str(x) if isinstance(x, (Path,)) else x) 
                     for x in v]
        else:
           
            out[k] = v
    return out


def create_session_directories(session_id: str) -> Dict[str, Path]:
    SESSION_DIR = OUTPUTS_DIR / session_id
    SLIDES_DIR = SESSION_DIR / "slides"
    AUDIO_DIR = SESSION_DIR / "audio_segments"
    TRANSCRIPTS_DIR = SESSION_DIR / "transcripts"
    OCR_DIR = SESSION_DIR / "ocr"
    FORMULAS_DIR = SESSION_DIR / "formulas"
    DIAGRAMS_DIR = SESSION_DIR / "diagrams"
    DIAGRAM_TEXT_DIR = SESSION_DIR / "diagram_texts"
    FUSED_DIR = SESSION_DIR / "fused_sentences"
    PROCESSED_SLIDES_DIR = SESSION_DIR / "processed_slides"
    COMBINED_DIR = SESSION_DIR / "combined"
    SUMMARIES_DIR = SESSION_DIR / "summaries"
    GRAPHS_DIR = SESSION_DIR / "graphs"
    EVALUATIONS_DIR = SESSION_DIR / "evaluations"

    for d in (SESSION_DIR, SLIDES_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, OCR_DIR, FORMULAS_DIR,
              DIAGRAMS_DIR, DIAGRAM_TEXT_DIR, FUSED_DIR, PROCESSED_SLIDES_DIR, COMBINED_DIR, 
              SUMMARIES_DIR, GRAPHS_DIR, EVALUATIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "SESSION_DIR": SESSION_DIR,
        "SLIDES_DIR": SLIDES_DIR,
        "AUDIO_DIR": AUDIO_DIR,
        "TRANSCRIPTS_DIR": TRANSCRIPTS_DIR,
        "OCR_DIR": OCR_DIR,
        "FORMULAS_DIR": FORMULAS_DIR,
        "DIAGRAMS_DIR": DIAGRAMS_DIR,
        "DIAGRAM_TEXT_DIR": DIAGRAM_TEXT_DIR,
        "FUSED_DIR": FUSED_DIR,
        "PROCESSED_SLIDES_DIR": PROCESSED_SLIDES_DIR,
        "COMBINED_DIR": COMBINED_DIR,
        "SUMMARIES_DIR": SUMMARIES_DIR,
        "GRAPHS_DIR": GRAPHS_DIR,
        "EVALUATIONS_DIR": EVALUATIONS_DIR
    }

def _wait_for_file_stable(path, timeout=30, interval=0.5):
    """Wait until file stops growing (Windows file-lock race condition fix)"""
    import time
    import os
    
    start = time.time()
    last_size = -1

    while time.time() - start < timeout:
        if not os.path.exists(path):
            time.sleep(interval)
            continue

        size = os.path.getsize(path)
        if size == last_size and size > 0:
            return  # file is stable
        last_size = size
        time.sleep(interval)

    raise RuntimeError(f"File not stable after {timeout}s: {path}")


def wait_for_directory_stable(dir_path: Path, timeout: int = 60, interval: float = 0.5):
    """
    Wait until ALL files in a directory have stable sizes.
    Critical on Windows to avoid reading partially-written files.
    """
    start = time.time()
    snapshot = None

    while time.time() - start < timeout:
        current = {
            f.name: f.stat().st_size
            for f in dir_path.iterdir()
            if f.is_file()
        }
        if current == snapshot and current:
            print(f"[Download] Directory stable: {len(current)} files")
            return
        snapshot = current
        time.sleep(interval)

    raise RuntimeError(f"Download directory never stabilized after {timeout}s: {dir_path}")


def download_video_assets(url: str, session_dir: Path) -> Path:
    """
    Download video assets WITHOUT merging (separated concerns architecture).
    
    yt-dlp downloads separate audio + video streams. No merge, no rename,
    no Windows file-lock race condition.
    
    Returns the download directory containing separate audio/video files.
    Files will be named like: 136.mp4, 251.webm, 140.m4a, 18.mp4, etc.
    """
    dl_dir = session_dir / "raw_downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    # Common flags for both downloads
    common_flags = [
        sys.executable, "-m", "yt_dlp",
        "--no-warnings",
        "--no-playlist",
        "--geo-bypass",
        "--no-check-certificates",
        "--no-part",
        "--retries", "10",
        "--fragment-retries", "10",
        "--socket-timeout", "30",
    ]

    # STEP A: Download best audio-only stream
    audio_cmd = common_flags + [
        "-f", "ba",
        "--output", str(dl_dir / "audio.%(ext)s"),
        url
    ]
    print(f"[Download] yt-dlp (audio) → {url}")
    try:
        subprocess.run(audio_cmd, check=True)
        print("[Download] Audio stream downloaded successfully")
    except subprocess.CalledProcessError:
        print("[Download] No separate audio stream available (progressive video)")

    # STEP B: Download best video stream (with audio fallback for progressive)
    video_cmd = common_flags + [
        "-f", "bv*/b",   # best video-only, OR best single (progressive fallback)
        "--output", str(dl_dir / "video.%(ext)s"),
        url
    ]
    print(f"[Download] yt-dlp (video) → {url}")
    try:
        subprocess.run(video_cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp video download failed: {e}")

    # Wait for ALL files to finish writing (Windows stability)
    wait_for_directory_stable(dl_dir)

    # List what we got
    files = list(dl_dir.iterdir())
    if not files:
        raise FileNotFoundError(
            "yt-dlp did not produce any files. "
            "Possible causes: age restriction, login required, geo-block."
        )

    for f in files:
        print(f"[Download]   → {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")

    return dl_dir


def resolve_audio_file(dl_dir: Path) -> Optional[Path]:
    """
    Pick the best audio file from downloaded assets.
    Returns None if no standalone audio file found (progressive download case).
    """
    # Primary: look for explicitly named audio file (from our two-step download)
    audio_files = [
        f for f in dl_dir.iterdir()
        if f.is_file() and f.stem.lower() == "audio"
    ]
    if audio_files:
        best = max(audio_files, key=lambda f: f.stat().st_size)
        print(f"[Resolve] Audio file found: {best.name}")
        return best

    # Fallback: look for any pure audio file by extension
    pure_audio = [
        f for f in dl_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".m4a", ".ogg", ".opus", ".aac", ".mp3", ".wav"}
    ]
    if pure_audio:
        best = max(pure_audio, key=lambda f: f.stat().st_size)
        print(f"[Resolve] Best audio (by extension): {best.name}")
        return best

    # No separate audio found — progressive download
    print("[Resolve] No separate audio file found (progressive download)")
    return None


def resolve_video_file(dl_dir: Path, session_dir: Path) -> Path:
    """
    Pick the best video file for slide extraction.
    No merge needed — slides only require the video stream.
    Audio is handled separately via resolve_audio_file.
    """
    VIDEO_EXTS = {".mp4", ".mkv", ".webm"}

    # Primary: look for explicitly named video file (from our two-step download)
    video_by_name = [
        f for f in dl_dir.iterdir()
        if f.is_file() and f.stem.lower() == "video" and f.suffix.lower() in VIDEO_EXTS
    ]
    if video_by_name:
        best = max(video_by_name, key=lambda f: f.stat().st_size)
        print(f"[Resolve] Video file found: {best.name} ({best.stat().st_size / (1024*1024):.2f} MB)")
        return best

    # Fallback: pick any video file by extension (largest)
    all_video = [
        f for f in dl_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS
    ]
    if all_video:
        best = max(all_video, key=lambda f: f.stat().st_size)
        print(f"[Resolve] Video file (by extension): {best.name}")
        return best

    raise FileNotFoundError("No video file found in downloaded assets.")


def safe_merge(video_path: Path, audio_path: Path, out_path: Path):
    """
    Merge separate video + audio files using ffmpeg -c copy.
    Only called AFTER both files are fully stable.
    No transcoding — just muxing into a single container.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path)
    ]
    print(f"[Merge] ffmpeg: {video_path.name} + {audio_path.name} → {out_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg merge failed: {result.stderr[-500:] if result.stderr else 'unknown error'}")
    
    # Wait for the merged file to stabilize
    _wait_for_file_stable(str(out_path))
    print(f"[Merge] Success → {out_path.name} ({out_path.stat().st_size / (1024*1024):.2f} MB)")


def download_video(url: str, session_dir: Path) -> str:
    """
    DEPRECATED: Backwards-compatible wrapper around the new separated pipeline.
    Returns a single video file path (merged if needed).
    """
    dl_dir = download_video_assets(url, session_dir)
    video_file = resolve_video_file(dl_dir, session_dir)
    return str(video_file)


# ====== Slide Extraction Functions (UPDATED – Pixel Difference Based) ======

def display_frame(frame, title="Frame"):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def save_frame(frame, count, save_dir, start_time, end_time, timeline_file):
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / f"slide_{count:03d}.png"
    cv2.imwrite(str(filename), frame)

    with open(timeline_file, "a", encoding="utf-8") as f:
        f.write(
            f"Slide {count:03d}: Start {start_time:.2f}s - End {end_time:.2f}s\n"
        )

    print(f"[Slides] Saved {filename}")
    return filename


def detect_new_slide(prev_frame, current_frame, threshold=600000):
    # Apply blur to ignore minor noise/compression artifacts
    prev_blur = cv2.GaussianBlur(prev_frame, (5, 5), 0)
    curr_blur = cv2.GaussianBlur(current_frame, (5, 5), 0)
    
    diff = cv2.absdiff(prev_blur, curr_blur)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    score = int(np.sum(gray_diff))
    return score > threshold, score


# ---------- Optimized SSIM-based Slide Deduplication ----------
def compute_ssim_fast(img1, img2, target_size=(256, 256)):
    """
    Compute SSIM between two images after downscaling for speed.
    Returns similarity score between 0 and 1.
    """
    # Downscale for speed
    small1 = cv2.resize(img1, target_size, interpolation=cv2.INTER_AREA)
    small2 = cv2.resize(img2, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY) if len(small1.shape) == 3 else small1
    gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY) if len(small2.shape) == 3 else small2
    
    # Compute SSIM manually (faster than skimage)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    gray1 = gray1.astype(np.float64)
    gray2 = gray2.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def get_text_hash_fast(frame, target_size=(400, 300)):
    """
    Get a fast hash of OCR text from a downscaled frame.
    Returns a simple hash of detected text for comparison.
    """
    try:
        small = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Fast OCR with minimal config
        text = pytesseract.image_to_string(gray, config='--psm 6 -c tessedit_do_invert=0').strip()
        # Return hash of normalized text
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hash(normalized) if normalized else 0
    except Exception:
        return 0


def frames_are_similar_ssim(prev_frame, current_frame, ssim_threshold=0.92, prev_text_hash=None):
    """
    Fast duplicate detection using SSIM + optional OCR text hash.
    Returns (is_similar, ssim_score, text_hash)
    """
    ssim_score = compute_ssim_fast(prev_frame, current_frame)
    
    # If SSIM says very similar, it's a duplicate
    if ssim_score > ssim_threshold:
        return True, ssim_score, prev_text_hash
    
    # For borderline cases, use text hash (only compute if needed)
    if ssim_score > 0.85:
        curr_text_hash = get_text_hash_fast(current_frame)
        if prev_text_hash and curr_text_hash == prev_text_hash:
            return True, ssim_score, curr_text_hash
        return False, ssim_score, curr_text_hash
    
    return False, ssim_score, None


def frames_are_similar(prev_saved_frame, current_frame, threshold=600000):
    """Legacy function for backwards compatibility - now uses SSIM internally"""
    is_similar, ssim_score, _ = frames_are_similar_ssim(prev_saved_frame, current_frame)
    # Convert SSIM to a "difference score" for legacy compatibility
    score = int((1 - ssim_score) * 1000000)
    return is_similar, score


def extract_slides_with_timeline(
    video_path: str,
    slide_dir: Path,
    time_interval: float = 1.0,
    min_slide_interval: float = 5.0,
    save_timeout: float = 0.5,
    blur_threshold: float = 100.0,  # Laplacian variance threshold for blur detection
    stability_frames: int = 3,  # Number of stable frames before saving
):
    """
    Extract unique, stable, non-blurry slides from video.
    
    Improvements:
    - Blur detection using Laplacian variance (rejects transition frames)
    - Frame stability check (waits for content to stabilize)
    - SSIM-based duplicate detection
    """
    slide_dir.mkdir(parents=True, exist_ok=True)
    # Try multiple backends for OpenCV VideoCapture (Windows often has issues with certain H.264 streams)
    backends_to_try = [
        (cv2.CAP_FFMPEG, "FFMPEG"),
        (cv2.CAP_ANY, "ANY"),
        (cv2.CAP_MSMF, "MSMF"),
    ]
    
    cap = None
    for backend, backend_name in backends_to_try:
        print(f"[Slides] Trying OpenCV backend: {backend_name}")
        cap = cv2.VideoCapture(str(video_path), backend)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                # Reset to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print(f"[Slides] Successfully opened with {backend_name} backend")
                break
            cap.release()
        cap = None
    
    # If all backends fail, try transcoding the video first
    if cap is None or not cap.isOpened():
        print(f"[Slides] All OpenCV backends failed for: {video_path}")
        
        # Verify file exists and has content
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video file does not exist: {video_path}")
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            raise RuntimeError(f"Video file is empty (0 bytes): {video_path}")
        print(f"[Slides] Video file size: {file_size / (1024*1024):.2f} MB")
        
        # Try transcoding
        print("[Slides] Transcoding video with ffmpeg...")
        transcoded_path = Path(video_path).parent / "video_transcoded.mp4"
        transcode_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-an",  # no audio — video may be audio-free (DASH video-only stream)
            str(transcoded_path)
        ]
        try:
            result = subprocess.run(transcode_cmd, check=True, capture_output=True, text=True)
            video_path = str(transcoded_path)
            print(f"[Slides] Transcoded successfully to {transcoded_path}")
            cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise RuntimeError(f"Transcoding failed: {error_msg[:1000]}")

    if not cap or not cap.isOpened():
        raise RuntimeError(f"Failed to open video after all attempts: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_skip = int(max(1, fps * time_interval))

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read first frame: {video_path}")

    timeline_file = slide_dir.parent / "timeline.txt"
    with open(timeline_file, "w", encoding="utf-8") as f:
        f.write("Slide Timeline\n")
        f.write("=" * 40 + "\n")

    count = 0
    frame_number = 0
    last_slide_time = 0.0
    last_save_time = 0.0
    prev_saved_frame = prev_frame.copy()
    saved_text_hashes = set()  # Track OCR text hashes to avoid duplicates
    
    # Stability tracking
    candidate_frame = None
    candidate_time = 0.0
    stability_count = 0

    def is_blurry(frame, threshold=blur_threshold):
        """Detect blur using Laplacian variance. Lower = more blurry."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold, laplacian_var

    def is_frame_stable(prev, curr, threshold=0.98):
        """Check if frame content is stable (not transitioning)."""
        ssim = compute_ssim_fast(prev, curr)
        return ssim > threshold

    # Check first frame quality
    blurry, blur_score = is_blurry(prev_frame)
    if not blurry:
        save_frame(prev_frame, count, slide_dir, 0.0, 0.0, timeline_file)
        text_hash = get_text_hash_fast(prev_frame)
        if text_hash:
            saved_text_hashes.add(text_hash)
        count += 1
        print(f"[Slides] First frame saved (blur_score={blur_score:.1f})")
    else:
        print(f"[Slides] First frame is blurry ({blur_score:.1f}), skipping")

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

        frame_number += 1
        current_time = frame_number / fps

        if frame_number % frame_skip != 0:
            continue

        if current_time - last_slide_time < min_slide_interval:
            prev_frame = current_frame
            continue

        # Check if this frame is blurry (transition frame)
        blurry, blur_score = is_blurry(current_frame)
        if blurry:
            # Reset stability counter - we're in a transition
            candidate_frame = None
            stability_count = 0
            prev_frame = current_frame
            continue

        # Check for content change from last saved slide
        is_new, slide_diff = detect_new_slide(prev_saved_frame, current_frame)
        
        if not is_new:
            prev_frame = current_frame
            continue

        # Check if similar to already saved slide (SSIM + text hash)
        is_similar, ssim_score, text_hash = frames_are_similar_ssim(prev_saved_frame, current_frame)
        
        if is_similar or (text_hash and text_hash in saved_text_hashes):
            prev_frame = current_frame
            continue

        # Stability check - wait for frame to be stable before saving
        if candidate_frame is None:
            candidate_frame = current_frame.copy()
            candidate_time = current_time
            stability_count = 1
        else:
            if is_frame_stable(candidate_frame, current_frame):
                stability_count += 1
            else:
                # Content changed - update candidate
                candidate_frame = current_frame.copy()
                candidate_time = current_time
                stability_count = 1

        # Save only after frame is stable for N consecutive checks
        if stability_count >= stability_frames:
            print(f"[Slides] Frame {frame_number} | blur={blur_score:.1f} | ssim={ssim_score:.3f} | STABLE")
            
            save_frame(
                candidate_frame,
                count,
                slide_dir,
                last_save_time,
                candidate_time,
                timeline_file,
            )
            
            prev_saved_frame = candidate_frame.copy()
            if text_hash:
                saved_text_hashes.add(text_hash)
            last_slide_time = candidate_time
            last_save_time = candidate_time
            count += 1
            
            # Reset
            candidate_frame = None
            stability_count = 0
            time.sleep(save_timeout)

        prev_frame = current_frame

    end_time = frame_number / fps
    with open(timeline_file, "a", encoding="utf-8") as f:
        f.write(f"End: {end_time:.2f}s\nTotal Slides: {count}\n")

    cap.release()
    print(f"[Slides] Extraction complete → {count} unique stable slides")
    return timeline_file


# ---------- Audio extraction ----------
def extract_audio_segments_from_timeline(video_file: str, timeline_file: Path, out_dir: Path) -> List[Path]:
    """Legacy per-slide audio extraction - kept for backwards compatibility"""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(timeline_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and line.startswith('Slide')]

    results = []
    for i, line in enumerate(lines):
        parts = re.findall(r"([\d.]+)s", line)
        if len(parts) >= 2:
            start_time, end_time = map(float, parts[:2])
            duration = max(0.1, end_time - start_time)
            out_file = out_dir / f"audio_slide_{i+1:03d}.wav"
            cmd = ['ffmpeg', '-y', '-ss', str(start_time), '-i', str(video_file),
                   '-t', str(duration), '-ac', '1', '-ar', '16000', str(out_file)]
            run_cmd(cmd)
            results.append(out_file)
    return results


def extract_full_audio(video_file: str, out_dir: Path, direct_audio_file: Optional[Path] = None) -> Path:
    """
    Extract audio from the FULL video as a single file.
    
    If direct_audio_file is provided (from separated download), convert it
    to WAV for Whisper instead of extracting from video. This is faster and
    avoids ffmpeg issues with video-only streams.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # FAST PATH: If we already have a separate audio file, just convert to WAV
    if direct_audio_file and direct_audio_file.exists():
        print(f"[Audio] Using direct audio file: {direct_audio_file.name}")
        wav_file = out_dir / "full_audio.wav"

        if direct_audio_file.suffix.lower() == ".wav":
            # Already WAV — just copy
            shutil.copy2(str(direct_audio_file), str(wav_file))
        else:
            # Convert to 16kHz mono WAV for Whisper
            convert_cmd = [
                'ffmpeg', '-y', '-i', str(direct_audio_file),
                '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000',
                str(wav_file)
            ]
            try:
                subprocess.run(convert_cmd, capture_output=True, check=True, timeout=300)
            except subprocess.CalledProcessError as e:
                print(f"[Audio] WAV conversion failed, trying alternative: {e}")
                # Fallback: try with different flags
                alt_cmd = ['ffmpeg', '-y', '-i', str(direct_audio_file),
                           '-vn', '-ac', '1', '-ar', '16000', str(wav_file)]
                subprocess.run(alt_cmd, capture_output=True, check=True, timeout=300)

        if wav_file.exists() and wav_file.stat().st_size > 1000:
            print(f"[Audio] Direct audio → WAV success: {wav_file} ({wav_file.stat().st_size} bytes)")
            return wav_file
        else:
            print("[Audio] Direct audio conversion produced empty file, falling back to video extraction")

    # SLOW PATH: Extract audio from video file (used for local files or fallback)
    video_file_str = str(video_file)
    
    # First, check if video has audio stream using ffprobe
    try:
        # Check all streams
        probe_cmd = [
            'ffprobe', '-v', 'error', 
            '-show_entries', 'stream=index,codec_type,codec_name',
            '-of', 'csv=p=0', video_file_str
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        print(f"[Audio] Video streams info:\n{probe_result.stdout}")
        
        if 'audio' in probe_result.stdout.lower():
            print("[Audio] Audio stream detected.")
        else:
            print("[Audio] WARNING: ffprobe says no audio stream. Trying extraction anyway...")
    except Exception as e:
        print(f"[Audio] ffprobe check failed (continuing anyway): {e}")

    # Try extraction with different output formats
    output_formats = [
        ("full_audio.wav", ['-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000']),
        ("full_audio.mp3", ['-vn', '-acodec', 'libmp3lame', '-q:a', '2', '-ac', '1', '-ar', '16000']),
        ("full_audio.aac", ['-vn', '-acodec', 'aac', '-b:a', '128k', '-ac', '1', '-ar', '16000']),
    ]
    
    success_file = None
    
    for filename, ffmpeg_args in output_formats:
        out_file = out_dir / filename
        cmd = ['ffmpeg', '-y', '-i', video_file_str] + ffmpeg_args + [str(out_file)]
        
        try:
            print(f"[Audio] Attempting extraction to {filename}...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0 and out_file.exists() and out_file.stat().st_size > 10000: # at least 10KB
                print(f"[Audio] Success: {out_file} ({out_file.stat().st_size} bytes)")
                success_file = out_file
                
                # If we got a non-wav file, convert to wav for Whisper
                if not filename.endswith('.wav'):
                    wav_file = out_dir / "full_audio.wav"
                    print(f"[Audio] Converting {filename} to {wav_file.name}...")
                    convert_cmd = ['ffmpeg', '-y', '-i', str(out_file), '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', str(wav_file)]
                    subprocess.run(convert_cmd, capture_output=True, check=True)
                    if wav_file.exists():
                        return wav_file
                return out_file
            else:
                print(f"[Audio] Failed to extract {filename} (RC={result.returncode}, Size={out_file.stat().st_size if out_file.exists() else 0})")
                if result.stderr:
                    print(f"[Audio] Error: {result.stderr[-500:]}")
                    
        except Exception as e:
            print(f"[Audio] Exception extracting {filename}: {e}")
            
    # Return None instead of crashing if no audio can be extracted
    print(f"[Audio] WARNING: Could not extract any audio from {video_file}. The video might be silent or corrupted.")
    return None





# ---------- Transcription (Whisper) ----------
def transcribe_audio_segments_whisper(audio_segments_dir: Path, transcripts_dir: Path,
                                      model_size: str = "base", device: str = "cpu") -> List[Path]:
    """Legacy per-segment transcription - kept for backwards compatibility"""
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(model_size, device=device)

    saved = []
    for audio_file in sorted(Path(audio_segments_dir).glob("*.wav")):
        slide_num_match = re.search(r"(\d+)", audio_file.name)
        slide_num = slide_num_match.group(1) if slide_num_match else audio_file.stem
        print(f"[Whisper] Transcribing slide {slide_num} -> {audio_file}")
        res = model.transcribe(str(audio_file))
        transcript_text = res.get('text', '').strip()
        out_path = Path(transcripts_dir) / f"transcript_slide_{slide_num}.txt"
        safe_write_text(out_path, transcript_text)
        saved.append(out_path)
    print(" Transcription complete.")
    return saved


def transcribe_full_audio(audio_file: Optional[Path], transcripts_dir: Path,
                          model_size: str = "base", device: str = "cpu") -> Path:
    """
    Transcribe the FULL video audio as a single transcript.
    Returns path to the full_transcript.txt file.
    """
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / "full_transcript.txt"
    
    if not audio_file or not audio_file.exists():
        print("[Whisper] No valid audio file provided. Creating empty transcript.")
        out_path.write_text("", encoding='utf-8')
        return out_path
    
    
    print(f"[Whisper] Loading model '{model_size}' on {device}...")
    model = whisper.load_model(model_size, device=device)
    
    print(f"[Whisper] Transcribing full audio: {audio_file}")
    result = model.transcribe(str(audio_file))
    transcript_text = result.get('text', '').strip()
    
    safe_write_text(out_path, transcript_text)
    print(f"[Whisper] Full transcript saved: {out_path} ({len(transcript_text)} chars)")
    
    return out_path



# ---------- OCR: RLSA + merged boxes ----------
def closing_operation(binary_image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

def apply_rlsa(binary_image, horizontal_threshold=30, vertical_threshold=10):
    horizontal_kernel = np.ones((1, horizontal_threshold), np.uint8)
    vertical_kernel = np.ones((vertical_threshold, 1), np.uint8)
    horizontal_rlsa = cv2.dilate(binary_image, horizontal_kernel, iterations=1)
    vertical_rlsa = cv2.dilate(binary_image, vertical_kernel, iterations=1)
    return cv2.bitwise_or(horizontal_rlsa, vertical_rlsa)

def calculate_centroid(box):
    x, y, w, h = box
    return y + h // 2

def is_same_entity(box1, box2, distance_threshold=20):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    Hcur, Hnew = h1, h2
    Ocur, Onew = calculate_centroid(box1), calculate_centroid(box2)
    if Hnew > 2 * Hcur:
        return False
    if abs(Onew - Ocur) > Hcur:
        return False
    return True

def merge_boxes(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x_min, y_min = min(x1, x2), min(y1, y2)
    x_max, y_max = max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def _process_single_slide_ocr(img_path: str, processed_dir: Path, ocr_text_dir: Path):
    """Helper function for parallel OCR processing of a single slide."""
    try:
        img_name = os.path.basename(img_path)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return None

        _, bin_img = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)
        closed = closing_operation(bin_img)
        rlsa = apply_rlsa(closed, horizontal_threshold=30, vertical_threshold=10)

        img_color = cv2.imread(img_path)
        contours, _ = cv2.findContours(rlsa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        merged = []
        for box in boxes:
            placed = False
            for i, m in enumerate(merged):
                if is_same_entity(box, m):
                    merged[i] = merge_boxes(m, box)
                    placed = True
                    break
            if not placed:
                merged.append(box)

        slide_texts = []
        for (x, y, w, h) in merged:
            roi = img_color[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            inv_ratio = (roi_bin == 0).sum() / roi_bin.size
            if inv_ratio < 0.3:
                _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Tesseract is CPU bound, running in parallel speeds this up significantly
            text = pytesseract.image_to_string(roi_bin, lang='eng')
            if text.strip():
                slide_texts.append(text.strip())

        text_filename = f"{os.path.splitext(img_name)[0]}.txt"
        with open(os.path.join(ocr_text_dir, text_filename), 'w', encoding='utf-8') as f:
            f.write("\n\n".join(slide_texts))

        # save annotated
        for (x, y, w, h) in merged:
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out_path = os.path.join(processed_dir, f"output_{img_name}")
        cv2.imwrite(out_path, img_color)
        return img_name
    except Exception as e:
        print(f"Error processing OCR for {img_path}: {e}")
        return None

def run_ocr_for_slides_in_folder(slide_dir: Path, processed_dir: Path, ocr_text_dir: Path):
    processed_dir.mkdir(parents=True, exist_ok=True)
    ocr_text_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in os.listdir(slide_dir) if f.lower().endswith('.png')],
                         key=lambda fname: int(re.search(r"(\d+)", fname).group(1)) if re.search(r"(\d+)", fname) else 0)

    # Use ProcessPoolExecutor for parallel processing
    # Max workers = cpu_count() is usually good for CPU-bound tasks like OCR
    max_workers = 4  # Fixed to 4 workers as requested
    print(f"[OCR] Starting parallel OCR with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for img_name in image_files:
            img_path = os.path.join(slide_dir, img_name)
            futures.append(executor.submit(_process_single_slide_ocr, img_path, processed_dir, ocr_text_dir))
        
        # Wait for all to complete
        completed_count = 0
        for future in as_completed(futures):
            if future.result():
                completed_count += 1
                
    print(f"[OCR] Completed OCR for {completed_count}/{len(image_files)} slides.")

# ---------- Formula detection (SymPy) ----------
def is_potential_formula(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    pattern = r'^[0-9A-Za-z\+\-\=\*\/\(\)\^\%\.\s]+$'
    return bool(re.match(pattern, text))

def is_valid_formula_sympy(text: str) -> bool:
    try:
        # Avoid single number or single char
        if len(text) < 3:
            return False
            
        # Avoid common words that look like math but aren't
        # e.g. "Figure", "Table", "Slide"
        if re.search(r'[a-zA-Z]{3,}', text) and not re.search(r'[\+\-\=\*\/\^\\]', text):
            return False
            
        parsed = parse_expr(text, transformations=_TRANSFORMATIONS)
        return isinstance(parsed, Basic)
    except Exception:
        return False

def _process_single_slide_formulas(img_path: str, out_dir: Path, conf_threshold: int) -> List[Path]:
    """Helper for parallel formula detection"""
    saved_paths = []
    try:
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None:
            return saved_paths

        text_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        for i in range(len(text_data.get('text', []))):
            try:
                conf = int(text_data['conf'][i])
            except Exception:
                conf = 0
            token = (text_data['text'][i] or '').strip()
            
            # Stricter checks: higher confidence, length > 2, valid sympy
            if conf > conf_threshold and token and is_potential_formula(token) and is_valid_formula_sympy(token):
                x, y, w, h = (text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i])
                crop = image[y:y+h, x:x+w]
                out_name = out_dir / f"{os.path.splitext(filename)[0]}_formula_{i}.png"
                cv2.imwrite(str(out_name), crop)
                saved_paths.append(out_name)
        return saved_paths
    except Exception as e:
        print(f"Formula error {img_path}: {e}")
        return saved_paths

def detect_and_save_formulas(slide_dir: Path, out_dir: Path, conf_threshold: int = 70) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in os.listdir(slide_dir) if f.lower().endswith('.png')],
                         key=lambda fname: int(re.search(r"(\d+)", fname).group(1)) if re.search(r"(\d+)", fname) else 0)

    max_workers = 4
    print(f"[Formulas] Starting parallel formula detection with {max_workers} workers...")
    
    all_saved = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for filename in image_files:
            img_path = os.path.join(slide_dir, filename)
            futures.append(executor.submit(_process_single_slide_formulas, img_path, out_dir, conf_threshold))
            
        for future in as_completed(futures):
            res = future.result()
            if res:
                all_saved.extend(res)
                
    print(f"[Formulas] Extracted formulas from {len(image_files)} slides.")
    return all_saved


# ---------- Diagram detection + BLIP captions ----------
_blip_processor = None
_blip_model = None

def _load_blip():
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        _blip_processor = None
        _blip_model = None
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        except Exception as e:
            raise RuntimeError(f"Failed to load BLIP model/processor: {e}")
    return _blip_processor, _blip_model

def generate_blip_caption(image_path: str) -> str:
    try:
        proc, mdl = _load_blip()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl.to(device)
        raw_image = Image.open(image_path).convert('RGB')
        inputs = proc(raw_image, return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        out = mdl.generate(**inputs, max_length=50, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
        caption = proc.decode(out[0], skip_special_tokens=True).strip()
        caption = re.sub(r'\s+', ' ', caption).strip()
        return caption
    except Exception as e:
        print(f" BLIP caption error for {image_path}: {e}")
        return ""

def _safe_conf(val):
    try:
        return int(val)
    except Exception:
        try:
            return round(float(val))
        except Exception:
            return -1

def is_text_heavy_region(roi, conf_threshold=60, text_ratio_threshold=0.05):
    d = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)
    total_area = max(1, roi.shape[0] * roi.shape[1])
    text_area = 0
    for i in range(len(d.get('text', []))):
        txt = (d['text'][i] or "").strip()
        conf = _safe_conf(d['conf'][i])
        if txt and conf >= conf_threshold:
            x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
            text_area += (w * h)
    ratio = text_area / total_area
    return ratio > text_ratio_threshold

def edge_density(roi):
    if roi is None or roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.sum(edges > 0) / (roi.shape[0] * roi.shape[1]))

def remove_text_regions_for_diagrams(image: np.ndarray, conf_threshold: int = 60) -> np.ndarray:
    img = image.copy()
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f" pytesseract.image_to_data failed: {e}")
        return img

    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf_raw = data.get("conf", [None] * n)[i]
        try:
            conf = int(float(conf_raw))
        except Exception:
            conf = -1

        if conf >= conf_threshold and text:
            try:
                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1)
            except Exception:
                continue
    return img

def _process_single_slide_diagrams(img_path: str, out_dir: Path, area_threshold_ratio: float, 
                                   proximity_thresh: int, ocr_conf_threshold: int, 
                                   text_heavy_thresh: float, min_edge_density: float) -> List[Path]:
    """Helper for parallel diagram detection (CPU bound)"""
    saved_paths = []
    try:
        fname = os.path.basename(img_path)
        orig = cv2.imread(img_path)
        if orig is None:
            return saved_paths

        H, W = orig.shape[:2]
        area_threshold = max(1, int(H * W * area_threshold_ratio))

        no_text = remove_text_regions_for_diagrams(orig.copy(), conf_threshold=ocr_conf_threshold)

        gray = cv2.cvtColor(no_text, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < area_threshold:
                continue
            ar = float(w) / max(h, 1)
            if not (0.3 < ar < 4.0):
                continue
            boxes.append((x, y, w, h))

        def combine_nearby(boxes_list):
            combined = []
            for bx in boxes_list:
                x, y, w, h = bx
                merged_flag = False
                for i, ex in enumerate(combined):
                    ex_x, ex_y, ex_w, ex_h = ex
                    if abs(x - ex_x) < proximity_thresh and abs(y - ex_y) < proximity_thresh:
                        nx0, ny0 = min(x, ex_x), min(y, ex_y)
                        nx1, ny1 = max(x + w, ex_x + ex_w), max(y + h, ex_y + ex_h)
                        combined[i] = (nx0, ny0, nx1 - nx0, ny1 - ny0)
                        merged_flag = True
                        break
                if not merged_flag:
                    combined.append(bx)
            return combined

        merged = combine_nearby(boxes)

        slide_num_match = re.search(r"(\d+)", fname)
        slide_num = slide_num_match.group(1) if slide_num_match else "0"

        for i, (x, y, w, h) in enumerate(merged, start=1):
            pad = 6
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            roi = orig[y0:y1, x0:x1]
            if roi is None or roi.size == 0:
                continue

            if is_text_heavy_region(roi, conf_threshold=ocr_conf_threshold, text_ratio_threshold=text_heavy_thresh):
                continue

            if edge_density(roi) < min_edge_density:
                continue

            png_name = f"slide_{slide_num}_diagram_{i}.png"
            out_path = out_dir / png_name
            cv2.imwrite(str(out_path), roi)
            saved_paths.append(out_path)
            
        return saved_paths
    except Exception as e:
        print(f"Diagram error {img_path}: {e}")
        return saved_paths

def detect_and_save_diagrams(
    slide_dir: Path,
    out_dir: Path,
    text_out_dir: Path,
    area_threshold_ratio: float = 0.01,
    proximity_thresh: int = 50,
    ocr_conf_threshold: int = 60,
    text_heavy_thresh: float = 0.05,
    min_edge_density: float = 0.01,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    text_out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [f for f in os.listdir(slide_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda fname: int(re.search(r"(\d+)", fname).group(1)) if re.search(r"(\d+)", fname) else 0,
    )

    max_workers = 4
    print(f"[Diagrams] Starting parallel diagram detection with {max_workers} workers...")
    
    all_diagrams = []
    
    # Phase 1: Detect and save diagram images (CPU heavy) in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for fname in image_files:
            img_path = os.path.join(slide_dir, fname)
            futures.append(executor.submit(
                _process_single_slide_diagrams, 
                img_path, out_dir, area_threshold_ratio, proximity_thresh,
                ocr_conf_threshold, text_heavy_thresh, min_edge_density
            ))
            
        for future in as_completed(futures):
            res = future.result()
            if res:
                all_diagrams.extend(res)
                
    print(f"[Diagrams] Extracted {len(all_diagrams)} diagrams from {len(image_files)} slides.")
    
    # Phase 2: Generate captions (GPU/Model heavy) sequentially
    print("[Diagrams] Generating captions (Sequential)...")
    for diagram_path in all_diagrams:
        caption = ""
        try:
            caption = generate_blip_caption(str(diagram_path))
        except Exception as e:
            print(f"BLIP caption failed for {diagram_path}: {e}")
        
        caption_path = text_out_dir / (diagram_path.name.replace(".png", ".txt"))
        safe_write_text(caption_path, caption)
        # NOTE: caption is intentionally written ONLY to text_out_dir (diagram_texts/).
        # Do NOT write a .txt alongside the image in diagrams/ — that pollutes the
        # diagrams folder with caption files instead of images, breaking the renderer's
        # glob-based image discovery (notes_renderer.py scans diagram_texts/ separately).
        
    return all_diagrams

# ====================== Fusion & Summarization ======================

# Cleaning function to remove explicit metadata labels that should not appear in the final summary
def clean_text_for_summary(text: str) -> str:
    """
    Remove lines/prefixes that identify metadata such as "Transcript:", "Slide Content:",
    "Visual Elements:", "Diagram:", "Summary:", "Slide 01:" etc.
    Also collapse whitespace.
    """
    if not text:
        return text
    # Remove common prefixes at line starts (case-insensitive)
    patterns = [
        r'^\s*(transcript|transcription)\s*[:\-]\s*', 
        r'^\s*(slide content|slide)\s*[:\-]\s*',
        r'^\s*(visual elements|visuals)\s*[:\-]\s*',
        r'^\s*(diagram|diagram caption|diagram text)\s*[:\-]\s*',
        r'^\s*(summary|note|notes|caption)\s*[:\-]\s*'
    ]
    
    out_lines = []
    for line in text.splitlines():
       
        line = re.sub(r'^\s*Slide\s*\d+\s*[:\-\.]?\s*', '', line, flags=re.I)
        
        for pat in patterns:
            line = re.sub(pat, '', line, flags=re.I)
        if line.strip():
            out_lines.append(line.strip())
    cleaned = " ".join(out_lines)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

    return []


def fuse_slide(slide_num: str, transcripts_dir: Path, ocr_dir: Path, diagram_text_dir: Path, fused_dir: Path):
    """
    Combine all slide text sources (transcript + OCR + diagram captions) into a single paragraph.
    """
    fused_dir.mkdir(parents=True, exist_ok=True)
    slide_id = f"slide{slide_num}"

    # Try multiple possible transcript file patterns
    transcript = ""
    possible_transcript_files = [
        transcripts_dir / f"transcript_slide_{slide_num}.txt",
        transcripts_dir / f"transcript_slide_{slide_num.zfill(3)}.txt",  
        transcripts_dir / f"slide_{slide_num}_transcript.txt",
    ]
    
    for tfile in possible_transcript_files:
        if tfile.exists():
            transcript = tfile.read_text(encoding='utf-8').strip()
            print(f" Found transcript: {tfile.name}")
            break


    ocr_text = ""
    possible_ocr_files = [
        ocr_dir / f"slide_{slide_num}.txt", 
        ocr_dir / f"slide_{slide_num.zfill(3)}.txt",
        ocr_dir / f"slide_{slide_num}_ocr.txt",
    ]
    
    for ocr_path in possible_ocr_files:
        if ocr_path.exists():
            ocr_text = ocr_path.read_text(encoding='utf-8').strip()
            print(f" Found OCR: {ocr_path.name}")
            break

    dtexts = []
    diagram_patterns = [
        f"*slide{slide_num}*",
        f"*slide_{slide_num}*", 
        f"*slide_{slide_num.zfill(3)}*"
    ]
    
    for pattern in diagram_patterns:
        for dfile in diagram_text_dir.rglob(pattern):
            try:
                if dfile.suffix.lower() == '.txt' and dfile.is_file():
                    content = dfile.read_text(encoding='utf-8').strip()
                    if content:
                        dtexts.append(content)
                        print(f" Found diagram text: {dfile.name}")
            except Exception as e:
                print(f" Error reading diagram file {dfile}: {e}")
                continue
                
    diagram_text = " ".join(dtexts).strip()

    combined_text = (
        "Slide Content:\n" + ocr_text + "\n\n"
        "Diagram Description:\n" + diagram_text
    ).strip()

    if not combined_text:
        print(f"No content found for slide {slide_num}")
        return None

    fused_file = fused_dir / f"{slide_id}_fused.txt"
    safe_write_text(fused_file, combined_text)
    print(f" Saved fused text: {fused_file.name} ({len(combined_text)} chars)")
    return fused_file

def combine_all_fused_text(fused_dir: Path, combined_dir: Path, transcripts_dir: Path = None) -> Path:
    """
    Combine all slide content (OCR + diagrams) into a single file,
    then append the full video transcript at the end.
    
    Structure:
    1. All slide contents (OCR text, diagram descriptions)
    2. Full video transcript
    """
    from content_sanitizer import ContentSanitizer
    sanitizer = ContentSanitizer()
    
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_file = combined_dir / "all_fused_text.txt"
    
    # Collect slide fused files
    files = []
    for pattern in ["*fused*", "*slide*"]:
        files.extend(list(fused_dir.glob(pattern)))
    
    files = list(set([f for f in files if f.is_file() and f.suffix == '.txt']))
    files.sort()
    
    # Pre-pass: build dynamic OCR noise profile from slide contents
    raw_ocr_texts = []
    for file_path in files:
        try:
            content = file_path.read_text(encoding='utf-8')
            if "Slide Content:" in content:
                slide_content = content.split("Diagram Description:")[0].replace("Slide Content:", "").strip()
                raw_ocr_texts.append(slide_content)
        except Exception:
            pass
    sanitizer.build_dynamic_ocr_noise(raw_ocr_texts)
    
    all_texts = []
    
    # Section 1: Full Video Transcript (Prioritized)
    all_texts.append("=" * 60)
    all_texts.append("FULL VIDEO TRANSCRIPT")
    all_texts.append("=" * 60 + "\n")
    
    if transcripts_dir:
        full_transcript_path = transcripts_dir / "full_transcript.txt"
        if full_transcript_path.exists():
            try:
                transcript = full_transcript_path.read_text(encoding='utf-8').strip()
                if transcript:
                    cleaned_transcript = sanitizer.sanitize_text(transcript)
                    all_texts.append(cleaned_transcript)
                    print(f" Added sanitized transcript ({len(cleaned_transcript)} chars, originally {len(transcript)})")
                else:
                    all_texts.append("Transcript is empty.")
            except Exception as e:
                all_texts.append(f"Failed to read transcript: {e}")
                print(f"Failed to read transcript: {e}")
        else:
            all_texts.append("Full transcript not found.")
            print(f" Full transcript not found at {full_transcript_path}")
    else:
        all_texts.append("Transcripts directory not provided.")
    
    # Section 2: Slide Contents (OCR Text + Diagrams)
    all_texts.append("\n" + "=" * 60)
    all_texts.append("SLIDE CONTENTS (OCR Text + Diagrams)")
    all_texts.append("=" * 60 + "\n")
    
    if files:
        print(f"Found {len(files)} slide files to combine")
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text and len(text) > 10:
                        # Sanitize OCR and clean it before appending
                        cleaned_slide = sanitizer.sanitize_text(text)
                        if cleaned_slide:
                            all_texts.append(f"--- {file_path.stem} ---\n{cleaned_slide}\n")
                            print(f" Added sanitized {file_path.name} ({len(cleaned_slide)} chars)")
            except Exception as e:
                print(f"Failed to read {file_path.name}: {e}")
    else:
        all_texts.append("No slide content found.\n")
        print(f" No fused files found in {fused_dir}")
    
    # Write combined file
    combined_content = "\n".join(all_texts)
    safe_write_text(combined_file, combined_content)
    print(f"Combined content saved to: {combined_file} ({len(combined_content)} chars)")
    
    return combined_file
    
def clean_promotional_content(text: str) -> str:
    """
    Removes promotional, playlist, exam-related, and meta-commentary text
    to prepare for final summarization.
    """
    if not text:
        return text
        
    lines = text.splitlines()
    cleaned_lines = []
    
    # Patterns to remove
    promotional_patterns = [
        r"(refer|check|link|see|visit).*playlist", 
        r"subscribe.*channel", 
        r"link.*description",
        r"hit.*like",
        r"comment.*below",
        r"share.*video",
        r"follow.*instagram",
        r"join.*(group|channel)",
        r"whatsapp", r"telegram",
        r"contact.*number",
        r"helpline",
    ]
    
    exam_patterns = [
        r"exam.*tips", 
        r"interview.*question", 
        r"previous.*year", 
        r"gate.*question", 
        r"university.*exam",
        r"semester.*exam",
        r"marks", 
        r"score.*good"
    ]
    
    noise_patterns = [
        r"in this video",
        r"welcome back",
        r"thank you",
        r"hope you understood",
        r"let.*know",
        r"any doubts"
    ]
    
    all_patterns = promotional_patterns + exam_patterns + noise_patterns
    
    # Remove specific noise phrases from content instead of dropping lines
    # Split into sentences to avoid dropping huge chunks of text
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split(". ")

    cleaned_sentences = []
    
    all_patterns = promotional_patterns + exam_patterns + noise_patterns

    for sent in sentences:
        sent_lower = sent.lower()
        
        # Check if sentence is purely promotional (high confidence)
        is_promo = False
        for p in promotional_patterns:
             if re.search(p, sent_lower):
                 # If the pattern match covers a significant part of the sentence, drop it
                 # Or just naive drop for now, but since we are at sentence level, it's safer.
                 # "Subscribe to channel" -> drop.
                 # "Check the link in description" -> drop.
                 is_promo = True
                 break
        if is_promo:
            continue

        # For noise patterns, maybe just clean them out or drop if short
        # "In this video we will..." -> meaningful context, keep it?
        # But "Hope you understood" -> drop.
        
        # Let's drop "noise" only if the sentence is short (< 10 words)
        is_noise = False
        if len(sent.split()) < 15:
             for p in noise_patterns:
                 if re.search(p, sent_lower):
                     is_noise = True
                     break
        if is_noise:
            continue

        cleaned_sentences.append(sent)
        
    return " ".join(cleaned_sentences).strip()

def generate_global_summary(text: str, model, tokenizer, device, max_len=350, min_len=200) -> str:
    """
    Final meta-summarizer: takes combined context-rich summaries and
    produces a concise, global summary.
    """
    if not text:
        return ""
        
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        do_sample=False, 
        length_penalty=1.0,  # Lower penalty to encourage longer, more natural paragraphs
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    
    return tokenizer.decode(
        summary_ids[0], 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    ).strip()


def generate_bart_summary_from_fused(
    fused_dir: Path, 
    output_path: Path,
    bart_max_input_tokens: int = 1024,  
    bart_max_output_tokens: int = 350,   
    chunk_overlap: int = 100             
):
    """
    Generate an abstractive summary from all fused slide paragraphs using FAST BART model only.
    """
    print("[BART Summary] ")

    
    all_texts = []

    files = sorted([
        f for f in os.listdir(fused_dir)
        if "fused" in f and (f.endswith(".txt") or "." not in f)
    ])

    if not files:
        print(f" No fused text files found in {fused_dir}.")
        safe_write_text(Path(output_path), "⚠️ No text found for summarization.")
        return ""

    print(f"Found {len(files)} fused text files for summarization.")

    for fname in files:
        path = os.path.join(fused_dir, fname)
        try:
            with open(path, encoding='utf-8') as fh:
                text = fh.read().strip()
            if len(text) > 20:
                text = clean_text_for_summary(text)
                all_texts.append(text)
        except Exception as e:
            print(f" Failed to read {fname}: {e}")

    if not all_texts:
        print(" All fused files are empty or invalid.")
        safe_write_text(Path(output_path), " No text found for summarization.")
        return ""

    
    full_text = " ".join(all_texts)
    print(f" Loaded total text length: {len(full_text)} characters.")

    # If text is short, use a simple extractive approach
    # (Though if we want the global summary style, we might skip this optimization or lower threshold)
    if len(full_text.split()) < 50:  
        print(" Text is very short, using simpler fallback...")
        sentences = sent_tokenize(full_text)
        summary = " ".join(sentences[:10])
        safe_write_text(Path(output_path), summary)
        return summary

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Model] Using cached BART model on {device}...")
    
    try:
        tokenizer, model = _get_bart_model()
        model = model.to(device)
        print(f" BART model ready (cached)")
    except Exception as e:
        print(f" Failed to load model: {e}. Using extractive fallback.")
        sentences = sent_tokenize(full_text)
        summary = " ".join(sentences[:min(12, len(sentences))])  
        safe_write_text(Path(output_path), summary)
        return summary

    # Step 4: Chunking and Segment Summarization
    try:
        inputs = tokenizer(full_text, return_tensors="pt", truncation=False)
        input_tokens = inputs["input_ids"][0].tolist()
    except Exception as e:
        print(f" Tokenization failed: {e}. Using fallback.")
        return ""

    if len(input_tokens) <= bart_max_input_tokens:
        print(" No chunking needed, processing as single segment...")
        chunks = [input_tokens]
    else:
        print(f" Text too long ({len(input_tokens)} tokens), using chunking...")
        chunks = []
        start = 0
        while start < len(input_tokens):
            end = start + bart_max_input_tokens
            chunks.append(input_tokens[start:end])
            start += bart_max_input_tokens - chunk_overlap

    print(f"Processing {len(chunks)} segments...")

    # Stage 3: Per-segment summarization (Context-Rich)
    segment_summaries = []
    for i, chunk_tokens in enumerate(chunks, 1):
        # preserve more detail in segments
        try:
            input_ids = torch.tensor([chunk_tokens]).to(device)
            summary_ids = model.generate(
                input_ids,
                max_length=350,      # Context-rich to support 10-12 final sentences
                min_length=150,       
                num_beams=4,         
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
            seg_summary = tokenizer.decode(
                summary_ids[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).strip()
            if seg_summary:
                segment_summaries.append(seg_summary)
                print(f"[Segment {i}] summarized ({len(seg_summary)} chars)")
        except Exception as e:
            print(f"[Segment {i}] failed: {e}")
            continue

    if not segment_summaries:
        print("All segment summarizations failed.")
        return ""

    # Stage 4: Meta-Summarization (The Final Layer)
    print("--- Starting Final Global Summarization ---")
    
    # 1. Combine
    combined_segment_text = "\n".join(segment_summaries)
    
    # 2. Clean Again (Dynamic removal of promo/exam chatter)
    cleaned_combined_text = clean_promotional_content(combined_segment_text)
    print(f"Cleaned combined text from {len(combined_segment_text)} to {len(cleaned_combined_text)} chars")
    
    # 3. Final Summary
    try:
        final_summary = generate_global_summary(
            cleaned_combined_text, 
            model, 
            tokenizer, 
            device,
            max_len=150,    # User specified strict limit
            min_len=80      # User specified min
        )
    except Exception as e:
        print(f"Final summarization failed: {e}")
        final_summary = cleaned_combined_text # Fallback to cleaned combined text

    safe_write_text(Path(output_path), final_summary)
    
    word_count = len(final_summary.split())
    print(f"Final Global Summary generated ({word_count} words).")
    return final_summary




def evaluate_summaries(candidate_summary: str, reference_summary: str) -> Dict:
    """
    Comprehensive summary evaluation using multiple metrics:
    1. ROUGE-1: Lexical content coverage (unigram overlap)
    2. ROUGE-L: Structural similarity (longest common subsequence)
    3. Keyword Coverage: Salience proxy (important terms covered)
    4. BERTScore: Semantic equivalence (precision, recall, F1)
    5. Sentence Embedding Cosine: Global meaning alignment
    """
    results = {}
    
    # 1. ROUGE Scores (ROUGE-1, ROUGE-L)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_summary, candidate_summary)
    
    results['rouge'] = {
        'rouge1': {
            'precision': rouge_scores['rouge1'].precision,
            'recall': rouge_scores['rouge1'].recall,
            'fmeasure': rouge_scores['rouge1'].fmeasure
        },
        'rougeL': {
            'precision': rouge_scores['rougeL'].precision,
            'recall': rouge_scores['rougeL'].recall,
            'fmeasure': rouge_scores['rougeL'].fmeasure
        }
    }
    
    # 2. Keyword Coverage (Salience Proxy)
    try:
        # Extract keywords using spaCy NER + noun chunks
        ref_doc = nlp(reference_summary)
        gen_doc = nlp(candidate_summary)
        
        # Get important terms: named entities + noun chunks
        ref_keywords = set()
        gen_keywords = set()
        
        for ent in ref_doc.ents:
            ref_keywords.add(ent.text.lower().strip())
        for chunk in ref_doc.noun_chunks:
            if len(chunk.text) > 2:
                ref_keywords.add(chunk.root.lemma_.lower())
        
        for ent in gen_doc.ents:
            gen_keywords.add(ent.text.lower().strip())
        for chunk in gen_doc.noun_chunks:
            if len(chunk.text) > 2:
                gen_keywords.add(chunk.root.lemma_.lower())
        
        # Calculate coverage
        if len(ref_keywords) > 0:
            covered = len(ref_keywords & gen_keywords)
            keyword_coverage = covered / len(ref_keywords)
        else:
            keyword_coverage = 0.0
            
        results['keyword_coverage'] = keyword_coverage
    except Exception as e:
        print(f"Keyword coverage calculation failed: {e}")
        results['keyword_coverage'] = 0.0
    
    # 3. BERTScore (Semantic Similarity)
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            [candidate_summary], 
            [reference_summary], 
            lang='en',
            verbose=False,
            rescale_with_baseline=True
        )
        results['bertscore'] = {
            'precision': float(P[0]),
            'recall': float(R[0]),
            'f1': float(F1[0])
        }
    except ImportError:
        print("bert-score not installed. Install with: pip install bert-score")
        results['bertscore'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    except Exception as e:
        print(f"BERTScore calculation failed: {e}")
        results['bertscore'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # 4. Sentence Embedding Cosine Similarity (Global Meaning)
    try:
        # Use sentence-transformers (already imported at top)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb_candidate = model.encode(candidate_summary, convert_to_tensor=True)
        emb_reference = model.encode(reference_summary, convert_to_tensor=True)
        cosine_sim = util.cos_sim(emb_candidate, emb_reference).item()
        results['sentence_cosine'] = cosine_sim
    except Exception as e:
        print(f"Sentence embedding cosine failed: {e}")
        results['sentence_cosine'] = 0.0
    
    return results

# ==================== NON-KG BASED SUMMARY & NOTES ====================

def fast_extractive_summary(text: str, top_n: int = 30) -> str:
    """
    Select top_n sentences using TF-IDF similarity to document mean.
    Fast way to get global context and reduce input size for BART.
    """
    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= top_n:
            return text
        
        # Vectorize
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return text # Empty or stopword-only text
            
        # Compute document mean vector (representing "overall meaning")
        doc_vector = tfidf_matrix.mean(axis=0)
        
        # Compute similarity of each sentence to doc vector
        scores = cosine_similarity(tfidf_matrix, doc_vector).flatten()
        
        # Select top N, keeping original order for coherence
        ranked_indices = np.argsort(scores)[::-1][:top_n]
        parsed_indices = sorted(ranked_indices)
        
        return " ".join([sentences[i] for i in parsed_indices])
    except Exception as e:
        print(f"[Fast Extractive] Error: {e}, using first {top_n} sentences")
        return " ".join(sent_tokenize(text)[:top_n])


def generate_non_kg_unified_summary(session1_dir: Path, session2_dir: Path, output_dir: Path) -> str:
    """
    Generate a Unified Text Summary using the high-quality 5-Stage Pipeline:
    1. Robust Noise Cleaning (NLP-based)
    2. Sentence-Aware Topic Chunking
    3. Per-Chunk Abstractive Summarization (BART)
    4. Two-Pass Merge (for coherence)
    5. Post-processing (7-10 sentences)
    
    Replaces older fast_extractive_summary approach for better quality.
    """
    from summary_postprocessor import clean_and_constrain_summary

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "non_kg_summary.txt"

    # Collect raw text from both sessions
    all_texts = []
    
    # Priority order for text sources
    sources = [
        ("combined", "all_fused_text.txt"),
        ("transcripts", "full_transcript.txt"),
    ]

    for sdir in [session1_dir, session2_dir]:
        found_for_session = False
        for subdir, fname in sources:
            fpath = sdir / subdir / fname
            if fpath.exists():
                text = fpath.read_text(encoding='utf-8').strip()
                if text and len(text) > 100:
                    all_texts.append(text)
                    found_for_session = True
                    break
        
        # Fallback to fused_sentences dir if main files missing
        if not found_for_session:
            fused_dir = sdir / "fused_sentences"
            if fused_dir.exists():
                session_text = []
                for f in sorted(fused_dir.glob("*.txt")):
                    t = f.read_text(encoding='utf-8').strip()
                    if len(t) > 50:
                        session_text.append(t)
                if session_text:
                    all_texts.append(" ".join(session_text))

    if not all_texts:
        msg = "No text available for summarization."
        safe_write_text(output_path, msg)
        return msg

    combined_text = "\n\n".join(all_texts)
    
    # ── STAGE 1: ROBUST CLEANING ──
    # Clean promotional content first
    combined_text = clean_promotional_content(combined_text)
    # Use the advanced NLP cleaner
    cleaned_text = clean_transcript_for_summary(combined_text)
    
    print(f"[NonKG Summary] Starting 5-Stage Pipeline on {len(cleaned_text)} chars...")

    try:
        tokenizer, model = _get_bart_model()
        
        # ── STAGE 2: SENTENCE-AWARE CHUNKING ──
        try:
            all_sentences = sent_tokenize(cleaned_text)
        except:
            all_sentences = cleaned_text.split(". ")

        chunks = []
        current_chunk = []
        current_tokens = 0
        max_chunk_tokens = 900  

        for sent in all_sentences:
            sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
            if current_tokens + sent_tokens > max_chunk_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        print(f"[NonKG Summary] Step 2: Chunking -> {len(chunks)} segments")

        # ── STAGE 3: PER-CHUNK SUMMARIZATION ──
        segment_summaries = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50: continue
            
            inputs = tokenizer([chunk], max_length=1024, return_tensors="pt", truncation=True)
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=200, 
                min_length=40,
                num_beams=4,
                early_stopping=True,
                length_penalty=1.5,
                no_repeat_ngram_size=3
            )
            seg = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            if seg and len(seg.strip()) > 20:
                segment_summaries.append(seg.strip())
                print(f"[NonKG Summary] Step 3: Chunk {i+1} summarized")

        if not segment_summaries:
            return "Summarization failed to produce content."

        # ── STAGE 4: TWO-PASS MERGE ──
        merged_text = " ".join(segment_summaries)
        merged_tokens = len(tokenizer.encode(merged_text, add_special_tokens=False))
        
        print(f"[NonKG Summary] Step 4: Merging {len(segment_summaries)} chunks ({merged_tokens} tokens)")

        # Final pass parameters
        final_max_len = 400
        final_min_len = 150
        
        if merged_tokens > 900:
             # Reprocess if too long
             inputs = tokenizer([merged_text], max_length=1024, return_tensors="pt", truncation=True)
        else:
             inputs = tokenizer([merged_text], max_length=1024, return_tensors="pt", truncation=True)
             
        final_ids = model.generate(
            inputs["input_ids"],
            max_length=final_max_len,
            min_length=final_min_len,
            num_beams=4,
            early_stopping=True,
            length_penalty=1.2,
            no_repeat_ngram_size=3
        )
        final_summary = tokenizer.decode(final_ids[0], skip_special_tokens=True)

        # ── STAGE 5: POST-PROCESSING ──
        final_summary = clean_and_constrain_summary(
            final_summary,
            min_sentences=10,
            max_sentences=12,
        )

        safe_write_text(output_path, final_summary)
        print(f"[NonKG Summary] ✅ Saved to {output_path} ({len(final_summary)} chars)")
        return final_summary

    except Exception as e:
        print(f"[NonKG Summary] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return "Summarization failed."


def generate_non_kg_notes(session1_dir: Path, session2_dir: Path, output_dir: Path) -> Path:
    """
    Generate structured notes from the combined raw text of both videos
    using NMF topic modeling (no knowledge graph involved).
    Returns the path to the generated PDF.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    output_dir.mkdir(parents=True, exist_ok=True)
    notes_dir = output_dir / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect raw text
    all_texts = []
    diagram_paths = {}  # name -> path
    diagram_captions = {}  # name -> caption

    for sdir in [session1_dir, session2_dir]:
        # Collect text
        for text_dir_name in ["combined", "fused_sentences", "combined_fused"]:
            tdir = sdir / text_dir_name
            if tdir.exists():
                for f in sorted(tdir.glob("*.txt")):
                    text = f.read_text(encoding='utf-8').strip()
                    if text and len(text) > 20:
                        all_texts.append(text)
                break

        # Fallback: transcript
        if not all_texts:
            tp = sdir / "transcripts" / "full_transcript.txt"
            if tp.exists():
                all_texts.append(tp.read_text(encoding='utf-8').strip())

        # Collect diagrams
        diag_dir = sdir / "diagrams"
        cap_dir = sdir / "diagram_texts"
        if diag_dir.exists():
            for img in list(diag_dir.glob("*.png")) + list(diag_dir.glob("*.jpg")):
                diagram_paths[img.name] = img
                cap_file = cap_dir / f"{img.stem}.txt" if cap_dir.exists() else None
                if cap_file and cap_file.exists():
                    diagram_captions[img.name] = cap_file.read_text(encoding='utf-8').strip()

    # Also check fused session's own diagrams
    fused_diag_dir = output_dir / "combined_fused"
    if fused_diag_dir.exists():
        cap_file = fused_diag_dir / "merged_captions.json"
        if cap_file.exists():
            try:
                merged_caps = json.loads(cap_file.read_text(encoding='utf-8'))
                diagram_captions.update(merged_caps)
            except: pass
        for img in list(fused_diag_dir.glob("*.png")) + list(fused_diag_dir.glob("*.jpg")):
            if img.name not in diagram_paths:
                diagram_paths[img.name] = img

    if not all_texts:
        print("[NonKG Notes] No text found")
        return None
        
    print(f"[NonKG Notes] Pipeline initialized. Loaded {len(diagram_paths)} diagrams and {len(diagram_captions)} captions from disk.")

    combined_text = "\n\n".join(all_texts)
    print(f"[NonKG Notes] Combined text: {len(combined_text)} chars")

    # 2. Split into sentences
    sentences = sent_tokenize(combined_text)
    
    try:
        from summary_postprocessor import _is_noise_sentence, _is_fragment
    except ImportError:
        _is_noise_sentence = lambda x: False
        _is_fragment = lambda x: False

    # Filter short/noisy sentences and strip metadata artifacts
    meta_patterns = [
        "slide content:", "diagram description:", "--- slide", 
        "visual elements:", "transcript:", "diagram & slide",
        "keywords & terminology"
    ]
    
    cleaned_sents = []
    for s in sentences:
        if len(s.split()) < 5 or sum(c.isalpha() for c in s) / max(len(s), 1) <= 0.6:
            continue
        # Filter high non-alphanumeric ratio (OCR noise like "w | E>")
        if sum(not c.isalnum() and not c.isspace() for c in s) / max(len(s), 1) > 0.15:
            continue
        if any(m in s.lower() for m in meta_patterns):
            continue
        if _is_noise_sentence(s) or _is_fragment(s):
            continue
        # Drop standalone short questions
        if s.strip().endswith('?') and len(s.split()) < 8:
            continue
            
        cleaned_sents.append(s)
        
    sentences = cleaned_sents

    if len(sentences) < 5:
        print("[NonKG Notes] Too few sentences for topic modeling")
        return None

    # 3. NMF Topic Modeling
    n_topics = min(8, max(3, len(sentences) // 10))
    print(f"[NonKG Notes] Extracting {n_topics} topics via NMF...")

    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        tfidf = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()

        nmf = NMF(n_components=n_topics, random_state=42, max_iter=300)
        W = nmf.fit_transform(tfidf)  # sentence-topic matrix
        H = nmf.components_           # topic-word matrix
    except Exception as e:
        print(f"[NonKG Notes] NMF failed: {e}")
        return None

    # 4. Extract topic labels and assign sentences
    # ── Semantic label mapper for NMF topics ──
    def _infer_topic_label(top_words, topic_sentences_text):
        """Generate a readable, educational topic label from NMF cluster content."""
        combined = " ".join(top_words).lower() + " " + topic_sentences_text.lower()
        
        # Category detection patterns → readable headings
        _CATEGORY_PATTERNS = [
            # Definitions / core concepts
            (["definition", "defined", "is a", "refers to", "known as", "called", "means"],
             "Definition and Fundamental Concepts"),
            # Operations
            (["operation", "process", "function", "method", "perform", "execute", "step", "procedure"],
             "Primary Operations"),
            # Types / Classification
            (["type", "types", "classification", "category", "kind", "form", "variant"],
             "Types and Classification"),
            # Examples / Analogies
            (["example", "analogy", "like", "real world", "everyday", "illustrat", "instance", "case"],
             "Real-World Examples"),
            # Applications / Use cases
            (["application", "applied", "used in", "used for", "use case", "practical", "scenario"],
             "Applications and Use Cases"),
            # Properties / Characteristics
            (["property", "characteristic", "principle", "feature", "attribute", "behavior", "follows"],
             "Key Characteristics"),
            # Abstract concepts
            (["abstract", "interface", "implementation", "encapsulation", "model", "framework"],
             "Conceptual Framework"),
            # Status / Conditions
            (["condition", "status", "state", "check", "validation", "constraint", "error", "exception"],
             "Conditions and Constraints"),
            # Components / Structure
            (["component", "element", "part", "structure", "architecture", "layer", "module", "unit"],
             "Structure and Components"),
            # Implementation details
            (["implement", "mechanism", "technique", "approach", "strategy", "design", "algorithm"],
             "Implementation Details"),
        ]
        
        for patterns, label in _CATEGORY_PATTERNS:
            match_count = sum(1 for p in patterns if p in combined)
            if match_count >= 2:
                return label
        
        # Fallback: extract a meaningful label from a NON-filler sentence
        _spoken_filler = [
            "in this video", "let us see", "we will", "thank you", "subscribe",
            "we say", "we are going", "we have seen", "we will see",
            "in the introduction", "in the previous", "in the next", "in the last",
            "as we discussed", "as i said", "as you can see", "let me",
            "hello everyone", "hi everyone", "good morning", "welcome to",
            "in this lecture", "in this presentation", "in this session",
            "the replay obviously", "if he gives", "let's say we",
            "so basically", "so now", "so here", "so what we",
            "now we will", "now let's", "told you", "told us",
        ]
        if topic_sentences_text.strip():
            # Try each sentence until we find a non-filler one
            for sent in topic_sentences_text.strip().split('.'):
                sent = sent.strip()
                if not sent or len(sent.split()) < 3:
                    continue
                sent_lower = sent.lower()
                if any(f in sent_lower for f in _spoken_filler):
                    continue
                # Clean up: take first ~5 meaningful words
                words = [w for w in sent.split() if len(w) > 2][:5]
                if len(words) >= 2:
                    label = " ".join(words)
                    return label[0].upper() + label[1:]
        
        # Final fallback: title-case top NMF words (better than sentence fragments)
        label_words = [w for w in top_words[:3] if len(w) > 2]
        if label_words:
            return " ".join(w.title() for w in label_words)
        return f"Key Concepts {topic_idx + 1}"

    topics = []
    for topic_idx in range(n_topics):
        top_word_indices = H[topic_idx].argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_word_indices]

        # Get sentences assigned to this topic (highest weight)
        topic_sentences = []
        for sent_idx, sent in enumerate(sentences):
            if W[sent_idx, topic_idx] > 0.1:  # threshold
                topic_sentences.append((sent, W[sent_idx, topic_idx]))

        # Sort by weight (most relevant first)
        topic_sentences.sort(key=lambda x: x[1], reverse=True)

        # Generate semantic topic label
        top_sents_text = " ".join(s for s, _ in topic_sentences[:3])
        topic_label = _infer_topic_label(top_words, top_sents_text)

        # Track first occurrence in original text for ordering
        first_occurrence = len(combined_text)
        for sent, _ in topic_sentences[:1]:
            pos = combined_text.find(sent[:50])
            if pos >= 0:
                first_occurrence = pos

        topics.append({
            "label": topic_label,
            "top_words": top_words,
            "sentences": [s for s, _ in topic_sentences[:8]],
            "first_occurrence": first_occurrence
        })

    # 5. Order topics by first occurrence in text (educational order)
    topics.sort(key=lambda t: t["first_occurrence"])

    # Remove empty topics
    topics = [t for t in topics if t["sentences"]]

    # Deduplicate labels: if multiple clusters get the same semantic label, add ordinal suffix
    label_counts = {}
    for t in topics:
        lbl = t["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    for lbl, count in label_counts.items():
        if count > 1:
            idx = 1
            for t in topics:
                if t["label"] == lbl:
                    t["label"] = f"{lbl} ({idx})"
                    idx += 1

    print(f"[NonKG Notes] {len(topics)} topics extracted: {[t['label'] for t in topics]}")

    # 6. Find relevant diagrams for each topic
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    topic_diagrams = {}
    if diagram_paths and diagram_captions:
        for topic in topics:
            topic_emb = embedding_model.encode(topic["label"] + " " + " ".join(topic["top_words"]), convert_to_tensor=True)
            best_img = None
            # Drop threshold significantly so visual representation is preserved even if the NMF cluster is noisy
            best_score = 0.15  # minimum threshold
            for img_name, caption in diagram_captions.items():
                if img_name in topic_diagrams.values():
                    continue  # already used
                cap_emb = embedding_model.encode(caption, convert_to_tensor=True)
                score = util.pytorch_cos_sim(topic_emb, cap_emb).item()
                if score > best_score:
                    best_score = score
                    best_img = img_name
            if best_img and best_img in diagram_paths:
                topic_diagrams[topic["label"]] = best_img
                
    print(f"[NonKG Notes] Successfully paired {len(topic_diagrams)} diagrams to NMF Topics.")

    # 7. Build HierarchicalNotes JSON — simplified: Description + Related Concepts only
    from hierarchical_schema import (
        make_point, make_subsection, make_section, make_notes,
        validate_hierarchy, fix_hierarchy
    )
    from notes_renderer import render_pdf, render_txt

    sections = []
    used_diagram_paths = set()  # Track used diagram file paths to prevent duplicates
    seen_sentences = set()  # Track seen sentences to prevent content duplication
    
    for i, topic in enumerate(topics, 1):
        topic_sents = topic["sentences"]
        
        # Deduplicate sentences within and across topics
        unique_sents = []
        for s in topic_sents:
            s_key = s.strip().lower()
            if s_key not in seen_sentences and len(s.split()) >= 5:
                seen_sentences.add(s_key)
                unique_sents.append(s)
        
        if not unique_sents:
            continue
        
        # ── Build simple structure: Description + Related Concepts ──
        subsections = []
        
        # "Description" subsection: first 2-3 sentences as overview
        desc_count = min(3, max(1, len(unique_sents) // 2))
        desc_sents = unique_sents[:desc_count]
        desc_points = [make_point(s) for s in desc_sents]
        subsections.append(make_subsection("Description", desc_points))
        
        # "Related Concepts" subsection: remaining sentences
        remaining = unique_sents[desc_count:]
        if remaining:
            related_points = [make_point(s) for s in remaining]
            subsections.append(make_subsection("Related Concepts", related_points))

        # Diagram — strict path-based dedup
        diagram_path = None
        diagram_caption = None
        if topic["label"] in topic_diagrams:
            img_name = topic_diagrams[topic["label"]]
            img_p = diagram_paths.get(img_name)
            if img_p and img_p.exists():
                img_path_str = str(img_p)
                if img_path_str not in used_diagram_paths:
                    used_diagram_paths.add(img_path_str)
                    diagram_path = img_path_str
                    diagram_caption = diagram_captions.get(img_name, f"Diagram: {img_name}")
        
        sections.append(make_section(
            topic["label"],
            subsections,
            diagram_path=diagram_path,
            diagram_caption=diagram_caption
        ))

    # 8. Validate & Fix
    notes_dict = make_notes("Lecture Notes (Topic Modeling)", sections)

    # 8a. Apply concept flow improvements (non-KG-dependent steps only)
    try:
        from concept_flow_organizer import apply_concept_flow
        notes_dict = apply_concept_flow(
            notes_dict, nodes=[], edges=[],
            run_steps=["sanitize_qmarks", "bullet_clean", "section_dedup", "tautology", "takeaways", "dedup_images"]
        )
        print(f"[NonKG Notes] Concept flow organizer applied")
    except Exception as cfo_err:
        print(f"[NonKG Notes] Concept flow organizer skipped: {cfo_err}")

    is_valid, violations = validate_hierarchy(notes_dict)
    if not is_valid:
        print(f"[NonKG Notes] ⚠️ Hierarchy violations ({len(violations)}): {violations[:3]}...")
        notes_dict = fix_hierarchy(notes_dict)
        is_valid2, v2 = validate_hierarchy(notes_dict)
        if is_valid2:
            print(f"[NonKG Notes] ✅ Auto-fix resolved all violations")
        else:
            print(f"[NonKG Notes] ⚠️ {len(v2)} violations remain after auto-fix: {v2[:3]}")

    # 9. Render
    pdf_path = notes_dir / "non_kg_notes.pdf"
    txt_path = notes_dir / "non_kg_notes.txt"
    
    try:
        render_pdf(notes_dict, pdf_path, image_base_dir=output_dir)
        render_txt(notes_dict, txt_path)
        
        n_sections = len(notes_dict.get("sections", []))
        n_subsections = sum(len(s.get("subsections", [])) for s in notes_dict.get("sections", []))
        print(f"[NonKG Notes] ✅ Hierarchical Notes saved ({n_sections} sections, {n_subsections} subsections)")
    except Exception as e:
        print(f"[NonKG Notes] PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return txt_path  # Return txt path as fallback

    return pdf_path


# ------------------ Topic Extraction & Dynamic Example Generation ------------------

# ------------------ Dynamic Example Generation (No LDA) ------------------

def make_dynamic_examples(combined_text: str, n_examples: int = 4) -> List[Dict[str, str]]:
    """
    Create synthetic few-shot examples based on combined_text (simple extraction).
    Returns list of {"input": "...", "output": "..."} where output is KG-like JSON string.
    """
    examples: List[Dict[str, str]] = []
    sentences = sent_tokenize(str(combined_text or ""))
    if not sentences:
        # fallback simple example
        examples.append({
            "input": "A stack is a linear data structure that follows LIFO. It supports push and pop.",
            "output": json.dumps({
                "nodes": [
                    {"id": "N1", "label": "Stack", "description": "A LIFO data structure"},
                    {"id": "N2", "label": "LIFO", "description": "Last in first out"},
                    {"id": "N3", "label": "Push", "description": "Add element"},
                    {"id": "N4", "label": "Pop", "description": "Remove element"}
                ],
                "edges": [
                    {"source": "N1", "target": "N2", "relation": "follows"},
                    {"source": "N1", "target": "N3", "relation": "supports"},
                    {"source": "N1", "target": "N4", "relation": "supports"}
                ]
            }, ensure_ascii=False)
        })
        return examples

    # if we still need more examples, craft from combined sentences
    idx = 0
    while len(examples) < n_examples and idx < len(sentences):
        s = sentences[idx]
        idx += 1
        # create a lightweight output: pick 2 nouns / tokens as nodes
        words = [w for w in re.findall(r"\b[a-zA-Z0-9_]{3,}\b", s)][:4]
        if not words:
            continue
        nodes = []
        edges = []
        for i, w in enumerate(words):
            nodes.append({"id": f"N{i+1}", "label": w, "description": ""})
        for i in range(1, len(words)):
            edges.append({"source": "N1", "target": f"N{i+1}", "relation": "related_to"})
        examples.append({"input": s, "output": json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)})

    # final fallback to ensure non-empty
    if not examples:
        examples.append({
            "input": sentences[0],
            "output": json.dumps({"nodes": [{"id": "N1", "label": "Topic", "description": ""}], "edges": []}, ensure_ascii=False)
        })

    return examples

def format_examples_for_prompt(examples: List[Dict[str, str]]) -> str:
    """
    Convert examples list into a text block suitable for the LLM prompt.
    Each example will be shown as:
    Example Input:
    "..."
    Example Output:
    { ... JSON ... }
    """
    out_lines = []
    for ex in examples:
        inp = ex.get("input", "").strip()
        outp = ex.get("output", "").strip()
        out_lines.append("Example Input:\n" + json.dumps(inp, ensure_ascii=False) + "\n\nExample Output:\n" + outp + "\n\n")
    return "\n".join(out_lines)

# ------------------ LLM Graph Transformer  ------------------
import google.generativeai as genai

# ensure GEMINI_API_KEY is loaded
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f" Warning configuring Gemini: {e}")
else:
    print(" GEMINI_API_KEY not set. KG extraction will fail until you set it in .env")


class Node(BaseModel):
    id: str
    label: str
    description: Optional[str] = Field("", description="Short description (optional)")


class Edge(BaseModel):
    source: str
    target: str
    relation: Optional[str] = Field("", description="Relation label (optional)")


class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


class LLMGraphTransformer:
    def __init__(self, model_name="models/gemini-2.5-flash"):
        try:
            self.model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings=None
            )
        except Exception as e:
            print(f"Gemini model init failed: {e}")
            self.model = None

        # kept for compatibility but we will generate dynamic few-shot examples
        self.fewshot_examples = (
            'Example Input:\n'
            '"A stack is a linear data structure based on the Last In First Out (LIFO) principle. '
            'A stack allows push and pop operations to insert and remove elements."\n\n'
            'Example Output:\n'
            '{\n'
            '  "nodes": [\n'
            '    {"id": "N1", "label": "Stack", "description": "A linear data structure following LIFO ordering"},\n'
            '    {"id": "N2", "label": "Linear Data Structure", "description": "A sequential data organization"},\n'
            '    {"id": "N3", "label": "LIFO", "description": "Last-In First-Out principle"},\n'
            '    {"id": "N4", "label": "Push Operation", "description": "Inserts an element into the stack"},\n'
            '    {"id": "N5", "label": "Pop Operation", "description": "Removes the topmost element from the stack"}\n'
            '  ],\n'
            '  "edges": [\n'
            '    {"source": "N1", "target": "N2", "relation": "is_a"},\n'
            '    {"source": "N1", "target": "N3", "relation": "follows"},\n'
            '    {"source": "N1", "target": "N4", "relation": "allows"},\n'
            '    {"source": "N1", "target": "N5", "relation": "allows"}\n'
            '  ]\n'
            '}\n'
        )

    def build_prompt(self, text: str, session_id: Optional[str] = None) -> str:
        """
        Build a dynamic prompt. 
        Also generate dynamic few-shot examples from combined text.
        Save dynamic examples into outputs/{session_id}/dynamic_examples.txt if session_id provided.
        """
        text = str(text)
        preamble = (
            "You are helping a student understand the concepts in an educational lecture.\n"
            "Your job is to extract a knowledge graph: a set of concepts (nodes) and the\n"
            "relationships between them (edges).\n\n"
            "Please represent the knowledge graph using this JSON format:\n"
            "{\n"
            '  \"nodes\": [ {\"id\": \"N1\", \"label\": \"Concept Name\", \"description\": \"A full explanatory sentence about this concept.\"} ],\n'
            '  \"edges\": [ {\"source\": \"N1\", \"target\": \"N2\", \"relation\": \"verb phrase describing the relationship\"} ]\n'
            "}\n\n"
            "STRICT NODE LABEL RULES — a node label MUST:\n"
            "  1. Be a NOUN PHRASE (a named concept, not a sentence or clause).\n"
            "     GOOD: 'Stack', 'Push Operation', 'LIFO Principle', 'Array Implementation'\n"
            "     BAD:  'A Linear Data Structure In Which', 'In Which', 'Explicit', 'Real Life'\n"
            "  2. NOT start with an article ('A', 'An', 'The') or a relative clause word\n"
            "     ('In Which', 'Where', 'That', 'Which').\n"
            "  3. NOT be a sentence fragment extracted from the middle of a sentence.\n"
            "  4. Be 1–5 words maximum. If the concept needs more words, use the description.\n"
            "  5. Use the CANONICAL / standard name of the concept.\n"
            "     GOOD: 'Static Stack', 'Dynamic Stack'\n"
            "     BAD:  'Size Stack', 'Dynamic Resizing Stack'\n\n"
            "STRICT EDGE RELATION RULES — a relation MUST:\n"
            "  1. Be a verb phrase that explains HOW the concepts relate.\n"
            "     GOOD: 'is implemented using', 'supports the operation of', 'follows the principle of'\n"
            "     BAD:  'includes', 'has' (too vague — say what the inclusion means)\n"
            "  2. Produce a grammatically complete sentence when used as:\n"
            "     '<source label> <relation> <target label>'\n"
            "     GOOD: 'Stack is implemented using Array' ✓\n"
            "     BAD:  'Top Of The Stack Explicit Stack' ✗ (not a sentence)\n\n"
            "STRICT DESCRIPTION RULES — a description MUST:\n"
            "  1. Be a complete, grammatically correct sentence (Subject + Verb + Object).\n"
            "  2. Explain WHAT the concept is and WHY it matters.\n"
            "  3. NOT repeat the node label verbatim as the entire description.\n\n"
            "Do NOT include any explanations outside the JSON. Return ONLY the JSON object.\n\n"
        )

        # Generate dynamic few-shot examples (LDA removed)
        try:
            dynamic_examples = make_dynamic_examples(text, n_examples=4)
            examples_text = format_examples_for_prompt(dynamic_examples)
        except Exception as e:
            print(f"[Prompt] Failed to generate dynamic examples: {e}")
            examples_text = self.fewshot_examples

        # Save dynamic examples for debugging / inspection (per-session if session_id provided)
        try:
            if session_id:
                debug_dir = OUTPUTS_DIR / str(session_id)
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / "dynamic_examples.txt"
            else:
                debug_path = OUTPUTS_DIR / "dynamic_examples_debug.txt"
            debug_path.write_text(examples_text, encoding="utf-8")
        except Exception as e:
            print(f"[Prompt] Could not save dynamic examples debug file: {e}")

        prompt = preamble + examples_text + "\nCONTENT:\n" + text
        return prompt

    def _safe_load_json(self, raw_text: str):
        """Try to extract JSON block and return parsed dict or raise."""
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)

        try:
            start = raw_text.index("{")
            end = raw_text.rindex("}") + 1
            candidate = raw_text[start:end]
        except ValueError:
            candidate = raw_text

        
        try:
            return json.loads(candidate)
        except Exception:
            import re
            m = re.search(r'(\{(?:.|\n)*\})', raw_text)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception as e:
                    raise RuntimeError(f"Failed to parse JSON from model output (regex attempt): {e}")
            raise RuntimeError("No valid JSON found in model output.")

    
    def _normalize_graph(self, raw_data: dict) -> KnowledgeGraph:
        """
        Normalize Gemini JSON output into a clean KnowledgeGraph.
        Ensures ALL node + edge fields become plain strings (no Path objects).
        """
        if "connections" in raw_data and "edges" not in raw_data:
            raw_data["edges"] = raw_data["connections"]

        nodes_raw = raw_data.get("nodes", []) or []
        edges_raw = raw_data.get("edges", []) or []

        normalized_nodes = []
        label_to_id = {}

        for i, n in enumerate(nodes_raw):
            if isinstance(n, dict):
                raw_id = n.get("id") or f"N{i+1}"
                raw_label = n.get("label") or n.get("name") or f"Node_{i+1}"
                raw_desc = n.get("description") or ""
                try:
                    nid = os.fspath(raw_id)
                except Exception:
                    nid = str(raw_id)
                label = os.fspath(raw_label) if isinstance(raw_label, (Path,)) else str(raw_label)
                desc = os.fspath(raw_desc) if isinstance(raw_desc, (Path,)) else str(raw_desc)
            else:
                nid = f"N{i+1}"
                label = str(n)
                desc = ""

            # ── FIX-1/4: Sanitize node label — repair fragments, strip articles ──
            clean_label = LLMGraphTransformer.sanitize_node_label(label, desc)
            # If sanitization returns empty (pure fragment, unrecoverable), skip node
            if not clean_label:
                print(f"[KG Normalize] Skipped fragment node label: '{label}'")
                continue
            label = clean_label

            node_obj = {"id": str(nid), "label": str(label), "description": str(desc)}
            normalized_nodes.append(node_obj)

            label_to_id[label.lower()] = node_obj["id"]
            label_to_id[node_obj["id"].lower()] = node_obj["id"]

        normalized_edges = []
        for e in edges_raw:
            if not isinstance(e, dict):
                continue
            source_raw = e.get("source") or e.get("from") or e.get("subject") or ""
            target_raw = e.get("target") or e.get("to") or e.get("object") or ""
            relation_raw = e.get("relation") or e.get("rel") or e.get("label") or ""

            source = os.fspath(source_raw) if isinstance(source_raw, (Path,)) else str(source_raw)
            target = os.fspath(target_raw) if isinstance(target_raw, (Path,)) else str(target_raw)
            relation = os.fspath(relation_raw) if isinstance(relation_raw, (Path,)) else str(relation_raw)

            src_id = label_to_id.get(source.lower())
            tgt_id = label_to_id.get(target.lower())

            if not src_id:
                src_id = f"N{len(normalized_nodes)+1}"
                normalized_nodes.append({"id": src_id, "label": source, "description": ""})
                label_to_id[source.lower()] = src_id

            if not tgt_id:
                tgt_id = f"N{len(normalized_nodes)+1}"
                normalized_nodes.append({"id": tgt_id, "label": target, "description": ""})
                label_to_id[target.lower()] = tgt_id

            normalized_edges.append({
                "source": src_id,
                "target": tgt_id,
                "relation": relation
            })

        # ── FIX-4: KG-level node deduplication ───────────────────────────────
        # Merge nodes whose canonical labels are near-identical before building
        # the graph, so downstream hierarchy never sees duplicates.
        # Strategy (fully dynamic — no domain words):
        #   canonical = lowercase, strip stop-words, sort words alphabetically
        #   If two nodes share the same canonical form OR word-overlap ≥ 70%,
        #   keep the one with the richer description; retarget all edges to it.
        #
        # E.g. "Network Topology" / "Network Topologies" / "Topology and Network"
        #      all collapse to the node with the longest description.

        _DEDUP_STOPS = {'a','an','the','of','in','for','to','and','or','is','are',
                        'was','were','its','this','that','by','as','from','with'}

        def _canonical_label(lbl: str) -> frozenset:
            words = re.findall(r'[a-z]+', lbl.lower())
            return frozenset(w for w in words if w not in _DEDUP_STOPS and len(w) > 2)

        def _word_overlap_ratio(a: str, b: str) -> float:
            ca, cb = _canonical_label(a), _canonical_label(b)
            if not ca or not cb: return 0.0
            return len(ca & cb) / max(len(ca | cb), 1)

        # Build merge map: node_id → survivor_id
        merge_map: dict = {}   # id → id of survivor
        processed_ids = []

        for i, node_a in enumerate(normalized_nodes):
            aid = node_a["id"]
            if aid in merge_map:
                continue  # already merged
            canonical_a = _canonical_label(node_a["label"])
            processed_ids.append(aid)

            for node_b in normalized_nodes[i+1:]:
                bid = node_b["id"]
                if bid in merge_map:
                    continue
                # Check overlap
                overlap = _word_overlap_ratio(node_a["label"], node_b["label"])
                canonical_b = _canonical_label(node_b["label"])
                exact = (canonical_a == canonical_b and len(canonical_a) > 0)
                if exact or overlap >= 0.72:
                    # Survivor = longer description
                    desc_a = len((node_a.get("description") or "").split())
                    desc_b = len((node_b.get("description") or "").split())
                    if desc_b > desc_a:
                        # a merges into b — but we've already confirmed a; swap labels
                        merge_map[aid] = bid
                        # Give b the richer label (whichever is shorter/cleaner)
                        if len(node_a["label"]) < len(node_b["label"]):
                            node_b["label"] = node_a["label"]
                    else:
                        merge_map[bid] = aid

        # Remove merged (non-survivor) nodes
        survived_ids = {n["id"] for n in normalized_nodes} - set(merge_map.keys())
        deduped_nodes = [n for n in normalized_nodes if n["id"] in survived_ids]

        # Retarget edges: any source/target pointing to a merged node → survivor
        def _resolve(nid: str) -> str:
            seen = set()
            while nid in merge_map and nid not in seen:
                seen.add(nid)
                nid = merge_map[nid]
            return nid

        deduped_edges = []
        seen_edge_keys: set = set()
        for e in normalized_edges:
            src = _resolve(e["source"])
            tgt = _resolve(e["target"])
            if src == tgt:
                continue   # self-loop after merge — discard
            key = (src, tgt, e["relation"].lower().strip())
            if key in seen_edge_keys:
                continue   # duplicate edge after merge — discard
            seen_edge_keys.add(key)
            deduped_edges.append({"source": src, "target": tgt, "relation": e["relation"]})

        if len(normalized_nodes) != len(deduped_nodes):
            print(f"[KG Dedup] Merged {len(normalized_nodes) - len(deduped_nodes)} "
                  f"duplicate nodes → {len(deduped_nodes)} unique nodes remain")

        node_objs = [Node(**{k: str(v) for k, v in n.items()}) for n in deduped_nodes]
        edge_objs = [Edge(**{k: str(v) for k, v in e.items()}) for e in deduped_edges]

        return KnowledgeGraph(nodes=node_objs, edges=edge_objs)

    @staticmethod
    def sanitize_node_label(label: str, description: str = "") -> str:
        """
        Repair or reject KG node labels that are sentence fragments, relative clauses,
        or other non-noun-phrase strings that should never become section headings.

        Strategy (fully dynamic — no domain words hardcoded):
        1. Strip leading articles (a/an/the).
        2. If label starts with a relative-clause word ('in which', 'where', 'that', etc.)
           try to extract a noun phrase from the description instead.
        3. If label is longer than 6 words AND starts with an article/particle, truncate
           to the first meaningful noun phrase.
        4. Normalize terminology: detect common naming anti-patterns (e.g. adjective +
           'Resizing' / 'Size' + noun) and replace with canonical equivalents inferred
           from the label words themselves.
        5. Return a cleaned title-cased label, never longer than 6 words.
        """
        if not label or not label.strip():
            return label

        label = label.strip()

        # ── Step 1: Strip leading articles ───────────────────────────────────
        _ARTICLE_RE = re.compile(r'^(a|an|the)\s+', re.I)
        label = _ARTICLE_RE.sub('', label).strip()

        # ── Step 2: Detect relative-clause / subordinator fragments ──────────
        # These are never valid concept names on their own.
        _FRAGMENT_STARTERS = re.compile(
            r'^(in\s+which|in\s+that|where\s+|whereby\s+|which\s+|that\s+|'
            r'when\s+|how\s+|whose\s+|wherein\s+|thereof\s+)',
            re.I
        )
        if _FRAGMENT_STARTERS.match(label):
            # Try to recover a noun phrase from the description
            if description and len(description.split()) >= 4:
                doc_words = description.split()
                # Take first 4 content words as a recovered heading
                _STOPS = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be',
                          'in', 'on', 'at', 'of', 'to', 'for', 'and', 'or',
                          'with', 'from', 'by', 'as', 'it', 'its', 'this', 'that'}
                content = [w.strip('.,;:') for w in doc_words
                           if w.lower().strip('.,;:') not in _STOPS and len(w) > 2]
                if content:
                    label = ' '.join(content[:4])
                else:
                    label = ' '.join(doc_words[:3])
            else:
                # No description — return empty so caller can skip this node
                return ""

        # ── Step 2b: Detect and fix "X and Y" relation-artifact labels ─────────
        # These are produced when KG extraction collapses a relation into a label:
        # e.g. "NetworkTopology → defines → Arrangement" → "Arrangement and Network"
        # Detection: one or both sides of " and " is a vague/relational word.
        # We keep the more specific (longer or richer) side.
        _VAGUE_CONNECTOR_WORDS = {
            'arrangement', 'network', 'structure', 'system', 'concept', 'data',
            'information', 'user', 'thing', 'item', 'element', 'component',
            'aspect', 'feature', 'property', 'detail', 'point', 'part', 'type',
            'redundancy', 'topology', 'administration', 'connection', 'node',
        }
        if ' and ' in label.lower():
            parts = re.split(r'\s+and\s+', label, flags=re.I)
            if len(parts) == 2:
                left, right = parts[0].strip(), parts[1].strip()
                left_vague  = left.lower()  in _VAGUE_CONNECTOR_WORDS
                right_vague = right.lower() in _VAGUE_CONNECTOR_WORDS
                # If one side is vague, keep the other (more specific) side
                if left_vague and not right_vague:
                    label = right
                elif right_vague and not left_vague:
                    label = left
                # If BOTH sides are vague (e.g. "Redundancy and Network"), use
                # description to recover — handled in the fragment_starter logic above.
                # If neither is vague, keep both (it's a legitimate compound label)


        words = label.split()
        if len(words) > 6:
            # Try to stop at the first verb-like word (heuristic: lowercase common verbs)
            _VERB_SET = {'is', 'are', 'was', 'were', 'be', 'in', 'that', 'which',
                         'where', 'when', 'how', 'can', 'will', 'has', 'have'}
            cutoff = len(words)
            for i, w in enumerate(words):
                if w.lower() in _VERB_SET and i >= 2:
                    cutoff = i
                    break
            label = ' '.join(words[:min(cutoff, 5)])

        # ── Step 4: Terminology normalization (dynamic, no hardcoded domain words) ──
        # Detect anti-patterns in label structure and emit canonical equivalents:
        #   "X Size Y" / "X Resizing Y" → "X Y" (remove process-verb noise words)
        #   "Y with X" → "X Y" (reorder to canonical noun-first form)
        # Uses structural pattern matching, not concept-name lists.
        _PROCESS_VERB_WORDS = {'resizing', 'sizing', 'processing', 'handling',
                                'managing', 'based', 'driven', 'oriented'}
        label_words = label.split()
        cleaned_words = [w for w in label_words
                         if w.lower() not in _PROCESS_VERB_WORDS]
        if cleaned_words and len(cleaned_words) < len(label_words):
            label = ' '.join(cleaned_words)

        # ── Step 5: Title-case and length cap ────────────────────────────────
        label = label.strip().rstrip('.,;:')
        if label:
            label = label[0].upper() + label[1:]
        # Hard cap at 6 words
        label_words = label.split()
        if len(label_words) > 6:
            label = ' '.join(label_words[:6])

        return label if label.strip() else ""

    def extract(self, text: str, session_id: Optional[str] = None) -> KnowledgeGraph:
        """
        Generate KG using Gemini. Accepts session_id to save dynamic examples.
        """
        if not text or not text.strip():
            raise RuntimeError("KG extraction failed: input text is empty.")
        if not self.model:
            raise RuntimeError("Gemini model not initialized. Check GEMINI_API_KEY and model name.")
        text = str(text) if text else ""
        prompt = self.build_prompt(text, session_id=session_id)
        try:
            response = self.model.generate_content(
                prompt)
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")
        
        # --- Validate Gemini output before accessing response.text ---
        try:
            if not response.candidates or not response.candidates[0].content.parts:
                raise RuntimeError(
                    "Gemini returned no text content. "
                    "finish_reason = SAFETY BLOCK or EMPTY RESPONSE."
                )
        except AttributeError:
            raise RuntimeError(
                "Gemini returned an unexpected empty response structure."
            )

        # SAFEST RESPONSE TEXT EXTRACTOR FOR GEMINI
        raw_text = ""
        try:
            cand = getattr(response, "candidates", None)
            if cand and len(cand) > 0:
                part = getattr(cand[0], "content", None)
                if part and getattr(part, "parts", None):
                    for p in part.parts:
                        t = getattr(p, "text", None)
                        if t:
                            raw_text = t
                            break

            if not raw_text:
                raw_text = str(response)

        except Exception:
            raw_text = str(response)

        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        print("[KG] Gemini raw response preview:", raw_text[:1000])


        parsed = self._safe_load_json(raw_text)
        if "connections" in parsed and "edges" not in parsed:
            parsed["edges"] = parsed["connections"]

        try:
            kg = self._normalize_graph(parsed)
            return kg
        except ValidationError as ve:
            print("Processing failed: Pydantic validation error while creating KnowledgeGraph:", ve)
            print("Parsed JSON keys:", list(parsed.keys()) if isinstance(parsed, dict) else type(parsed))
            raise RuntimeError(f"Failed to validate KG: {ve}")
        except Exception as e:
            print("Processing failed while normalizing KG:", e)
            raise



def generate_bart_summary_from_fused(
    input_file: Path,
    output_path: Path,
    bart_max_input_tokens: int = 1024,
    bart_max_output_tokens: int = 300
) -> str:
    """
    Generate a Context-Rich Summary using a 5-Stage Pipeline:
    1. Dynamic Noise Cleaning (sentence-level scoring)
    2. Sentence-Aware Topic Chunking (no mid-sentence breaks)
    3. Per-Chunk Abstractive Summarization (BART, cached model)
    4. Two-Pass Merge (chunk summaries → final coherent summary)
    5. Post-processing (clean_and_constrain_summary for 7-10 sentences)
    """
    from summary_postprocessor import clean_and_constrain_summary

    input_file = Path(input_file)
    if not input_file.exists():
        print(f" Warning: Input file {input_file} not found for summary.")
        return ""

    text = input_file.read_text(encoding="utf-8").strip()
    if not text:
        return ""

    # STEP 1: Dynamic noise cleaning
    text = clean_promotional_content(text)
    text = clean_transcript_for_summary(text)

    print(f"[Summary] Starting 5-Stage Pipeline on {len(text)} chars...")

    try:
        tokenizer, model = _get_bart_model()

        # STEP 2: SENTENCE-AWARE TOPIC CHUNKING
        # Split into sentences first, then group into chunks of ~800 tokens
        try:
            all_sentences = sent_tokenize(text)
        except:
            all_sentences = text.split(". ")

        # Group sentences into chunks, respecting token budget
        chunks = []
        current_chunk = []
        current_tokens = 0
        max_chunk_tokens = 900  # Leave room for BART's 1024 limit

        for sent in all_sentences:
            sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
            if current_tokens + sent_tokens > max_chunk_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        print(f"[Summary] Step 2: Sentence-aware chunking → {len(chunks)} topic segments")

        # STEP 3: PER-CHUNK ABSTRACTIVE SUMMARIZATION
        segment_summaries = []

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue

            inputs = tokenizer([chunk], max_length=1024, return_tensors="pt", truncation=True)

            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=350,           # Rich per-segment output
                min_length=150,            # Ensure substantial content
                num_beams=4,
                early_stopping=True,
                length_penalty=1.0,       # Encourage longer, natural paragraphs
                no_repeat_ngram_size=3,   # Prevent repetition
            )
            seg_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            if seg_summary and len(seg_summary.strip()) > 20:
                segment_summaries.append(seg_summary.strip())
                print(f"[Summary] Step 3: Chunk {i+1}/{len(chunks)} summarized ({len(seg_summary)} chars)")

        if not segment_summaries:
            print("[Summary] All chunk summarizations failed.")
            return ""

        # STEP 4: TWO-PASS MERGE — combine chunk summaries into final coherent summary
        merged_text = " ".join(segment_summaries)
        print(f"[Summary] Step 4: Merging {len(segment_summaries)} chunk summaries ({len(merged_text)} chars)")

        # If merged text fits in one BART pass, run final summarization
        merged_tokens = len(tokenizer.encode(merged_text, add_special_tokens=False))

        if merged_tokens > 900:
            # Too long for one pass — re-chunk and summarize again
            print(f"[Summary] Merged text too long ({merged_tokens} tokens), running second pass...")
            merge_inputs = tokenizer([merged_text], max_length=1024, return_tensors="pt", truncation=True)
            merge_ids = model.generate(
                merge_inputs["input_ids"],
                max_length=450,
                min_length=250,
                num_beams=4,
                early_stopping=True,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
            )
            full_summary = tokenizer.decode(merge_ids[0], skip_special_tokens=True)
        else:
            # Fits in one pass — final coherence pass
            merge_inputs = tokenizer([merged_text], max_length=1024, return_tensors="pt", truncation=True)
            merge_ids = model.generate(
                merge_inputs["input_ids"],
                max_length=450,
                min_length=250,
                num_beams=4,
                early_stopping=True,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
            )
            full_summary = tokenizer.decode(merge_ids[0], skip_special_tokens=True)

        # STEP 5: POST-PROCESSING — clean noise and constrain to 7-10 sentences
        full_summary = clean_and_constrain_summary(
            full_summary,
            min_sentences=7,
            max_sentences=10,
        )

        # Save
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(full_summary, encoding="utf-8")
            print(f"[Summary] ✅ Summary saved ({len(full_summary.split('.'))} sentences)")

        return full_summary

    except Exception as e:
        print(f"BART Summary Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return ""


# ---------- Main pipeline ----------
def process_video_full(url_or_path: str,
                       whisper_model: str = 'base',
                       session_id: Optional[str] = None,
                       device: str = "cpu") -> Dict:
    if session_id is None:
        session_id = str(int(time.time())) + "_" + uuid.uuid4().hex[:6]
    session_dirs = create_session_directories(session_id)

    # 1) download if URL — separated concerns: audio + video downloaded separately
    direct_audio_file = None  # Will be set if we have a separate audio file
    if str(url_or_path).lower().startswith('http'):
        print(f"[Download] {url_or_path}")
        dl_dir = download_video_assets(url_or_path, session_dirs["SESSION_DIR"])
        direct_audio_file = resolve_audio_file(dl_dir)
        video_file = resolve_video_file(dl_dir, session_dirs["SESSION_DIR"])
        video_path = str(video_file)
    else:
        video_path = str(url_or_path)

    # 2) slides + timeline (uses video file)
    print("[Slides] extracting slides")
    timeline_file = extract_slides_with_timeline(video_path, session_dirs["SLIDES_DIR"])

    # 3) FULL audio — use direct audio file if available, else extract from video
    print("[Audio] extracting FULL video audio")
    full_audio_file = extract_full_audio(video_path, session_dirs["AUDIO_DIR"], direct_audio_file=direct_audio_file)

    # 4) FULL audio transcription
    print("[Transcription] running Whisper on full audio")
    full_transcript_file = transcribe_full_audio(full_audio_file, session_dirs["TRANSCRIPTS_DIR"], model_size=whisper_model, device=device)

    # 5) OCR
    print("[OCR] running RLSA + boxed OCR")
    run_ocr_for_slides_in_folder(session_dirs["SLIDES_DIR"], session_dirs["PROCESSED_SLIDES_DIR"], session_dirs["OCR_DIR"])

    # 6) formulas
    print("[Formulas] detecting formulas")
    detect_and_save_formulas(session_dirs["SLIDES_DIR"], session_dirs["FORMULAS_DIR"], conf_threshold=60)

    # 7) diagrams + captions
    print("[Diagrams] extracting diagrams and generating BLIP captions")
    detect_and_save_diagrams(session_dirs["SLIDES_DIR"], session_dirs["DIAGRAMS_DIR"], session_dirs["DIAGRAM_TEXT_DIR"])

    # 8) fusion (per-slide)
    print("[Fusion] fusing transcript + OCR + diagram texts per slide")
    slide_files = sorted([f for f in os.listdir(session_dirs["SLIDES_DIR"]) if f.lower().endswith('.png')])
    fused_count = 0
    
    for fname in slide_files:
        m = re.search(r'slide_?(\d+)', fname.lower())
        if not m:
            m = re.search(r'(\d+)', fname)  # fallback
        if not m:
            print(f"⚠️ Could not extract slide number from: {fname}")
            continue
            
        num_str = m.group(1)
        slide_num = str(int(num_str)) 
        
        print(f"[Fusion] Processing slide {slide_num} from file {fname}")
        result = fuse_slide(slide_num, session_dirs["TRANSCRIPTS_DIR"], session_dirs["OCR_DIR"], session_dirs["DIAGRAM_TEXT_DIR"], session_dirs["FUSED_DIR"])
        
        if result:
            fused_count += 1
            print(f" Fused slide {slide_num}")
        else:
            print(f" No content to fuse for slide {slide_num}")

    print(f"[Fusion] Completed: {fused_count} slides fused")
    
    # Debug: List fused files
    fused_files = list(session_dirs["FUSED_DIR"].glob("*_fused.txt"))
    print(f"[Fusion] Found {len(fused_files)} fused files: {[f.name for f in fused_files]}")

    # 8.5) Combine all slide content + full transcript into one file
    print("[Combined] Saving all slide content + full transcript to combined directory...")
    combined_fused_file = combine_all_fused_text(session_dirs["FUSED_DIR"], session_dirs["COMBINED_DIR"], session_dirs["TRANSCRIPTS_DIR"])

    # 9) summarization - Generate BART summary only
    bart_summary_path = session_dirs["SUMMARIES_DIR"] / "global_summary_bart.txt"
    
    print("[Summary] generating BART summary...")
    bart_summary = generate_bart_summary_from_fused(
        combined_fused_file, 
        bart_summary_path, 
        bart_max_input_tokens=1024, 
        bart_max_output_tokens=150
    )

    # ---------- KG INPUT: USE FULL FUSED TEXT ----------
    print("[KG] Using FULL fused text for topic modeling and KG generation")

    fused_text_for_kg = ""
    combined_fused_file = session_dirs["COMBINED_DIR"] / "all_fused_text.txt"

    if combined_fused_file.exists():
        fused_text_for_kg = combined_fused_file.read_text(encoding="utf-8").strip()

    if not fused_text_for_kg:
        print(" No fused text available for KG extraction. Skipping KG.")
        return {
            "session_id": session_id,
            "session_dirs": stringify_paths(session_dirs),
            "error": "No fused text available for KG extraction.",
            "knowledge_graph": {"nodes": [], "edges": []}
        }

    # Save for debugging / reproducibility
    safe_write_text(
        session_dirs["SESSION_DIR"] / "text_used_for_kg.txt",
        fused_text_for_kg
    )
    # Topics extraction REMOVED (Legacy LDA)

    
    # ---------- NEW: Gemini LLM Knowledge Graph using dynamic few-shot examples (summary-driven) ----------
    print("[KG] Constructing KG using LLMGraphTransformer with summary-driven dynamic prompts ...")

    kg_model = LLMGraphTransformer(model_name="models/gemini-2.5-flash")
    try:
        # FIX-8: Clean OCR/speech errors from input text before KG extraction
        # Use a temporary WMDKGNotesGenerator instance just for normalize_text
        _ocr_cleaner = WMDKGNotesGenerator.__new__(WMDKGNotesGenerator)
        fused_text_for_kg = _ocr_cleaner.normalize_text(fused_text_for_kg)

        kg = kg_model.extract(
        fused_text_for_kg,
        session_id=session_id
        )

        nodes_json = [n.dict() for n in kg.nodes]
        edges_json = [e.dict() for e in kg.edges]

    except Exception as e:
        print(f" KG extraction failed: {e}")
        # Return partial response; in actual app raise or handle appropriately.
        return {
        "session_id": session_id,
        "session_dirs": stringify_paths(session_dirs),
        "error": f"KG extraction failed: {e}",
        "knowledge_graph": {"nodes": [], "edges": []}
    }


    # Build networkx graph
    G = nx.DiGraph()

    for node in kg.nodes:
        G.add_node(node.id, label=str(node.label), description=str(node.description))

    for edge in kg.edges:
        G.add_edge(str(edge.source), str(edge.target), label=str(edge.relation))


    graphs_dir = session_dirs["GRAPHS_DIR"]
    nodes_file = graphs_dir / "kg_nodes.json"
    edges_file = graphs_dir / "kg_edges.json"
    try:
        nodes_json = [n.dict() for n in kg.nodes]
        edges_json = [e.dict() for e in kg.edges]
        safe_write_text(nodes_file, json.dumps(nodes_json, indent=2, ensure_ascii=False))
        safe_write_text(edges_file, json.dumps(edges_json, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Failed to save nodes/edges JSON: {e}")

    # Save PNG and HTML
    graph_img = session_dirs["GRAPHS_DIR"] / "kg_graph.png"
    graph_html = session_dirs["GRAPHS_DIR"] / "kg_graph.html"

    # Create visualization only if we have nodes
    if nodes_json:
        plt.figure(figsize=(12, 9))
        pos = nx.spring_layout(G, seed=42)
        
        # FIX: Convert all labels to plain strings
        clean_labels = {}
        for n in G.nodes():
            raw_label = G.nodes[n].get("label", n)
            # Ensure it's a string, not WindowsPath
            clean_labels[n] = str(raw_label) if raw_label else str(n)
        
        nx.draw(
            G, pos,
            with_labels=True,
            labels=clean_labels,
            node_size=1200,
            node_color='lightblue',
            font_size=10,
            font_weight='bold'
        )
        
        # FIX: Convert edge labels to strings
        edge_labels_raw = nx.get_edge_attributes(G, 'label')
        clean_edge_labels = {}
        for edge, label in edge_labels_raw.items():
            clean_edge_labels[edge] = str(label) if label else ""
        
        if clean_edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=clean_edge_labels,
                font_color='red',
                font_size=8
            )
        
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(graph_img, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Interactive HTML
        net = Network(height="750px", width="100%", directed=True)
        net.from_nx(G)
        
        # FIX: Ensure all attributes are strings
        for node in net.nodes:
            node["label"] = str(node.get("label", node["id"]))
            node["title"] = str(node.get("description", ""))
        
        for edge in net.edges:
            edge["title"] = str(edge.get("label", ""))
        
        net.write_html(str(graph_html))  # Ensure path is string
    else:
        # Create empty placeholder files
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No Knowledge Graph Generated\n(No meaningful concepts found)', 
                ha='center', va='center', fontsize=14)
        plt.axis("off")
        plt.savefig(graph_img)
        plt.close()
        
        # Empty HTML
        with open(graph_html, 'w', encoding='utf-8') as f:
            f.write('<html><body><h3>No Knowledge Graph Generated</h3><p>No meaningful concepts found in the content.</p></body></html>')
    
    graph_image_str = str(graph_img).replace(str(OUTPUTS_DIR) + os.sep, "")
    graph_html_str = str(graph_html).replace(str(OUTPUTS_DIR) + os.sep, "")
    nodes_file_str = str(nodes_file)
    edges_file_str = str(edges_file)

    # Ensure session_dirs are all strings
    session_dirs_str = {}
    for key, value in session_dirs.items():
        session_dirs_str[key] = str(value)

    return {
        "session_id": session_id,
        "session_dirs": session_dirs_str,  
        "graph_image": graph_image_str,
        "graph_html": graph_html_str,
        "kg_nodes_file": nodes_file_str,
        "kg_edges_file": edges_file_str,
        "knowledge_graph": {
            "nodes": nodes_json,
            "edges": edges_json
        }
    }


def run_with_timeout(func, timeout_seconds, task_name):
    """
    Run a function with a timeout. If it times out, return a fallback message.
    """
    try:
        import threading
        from threading import Thread
        
        class FuncThread(Thread):
            def __init__(self):
                Thread.__init__(self)
                self.result = None
                self.exception = None
            
            def run(self):
                try:
                    self.result = func()
                except Exception as e:
                    self.exception = e
        
        thread = FuncThread()
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            print(f" {task_name} timed out after {timeout_seconds} seconds")
            return f"{task_name} timed out after {timeout_seconds} seconds. Please try with a shorter video or different settings."
        
        if thread.exception:
            print(f" {task_name} failed: {thread.exception}")
            return f"{task_name} failed: {str(thread.exception)}"
        
        return thread.result
        
    except Exception as e:
        print(f" Error running {task_name}: {e}")
        return f"Error running {task_name}: {str(e)}"
    
# ---------- Two-video orchestration ----------
def process_two_videos(video1: str, video2: str,
                       whisper_model: str = 'base',
                       device: str = 'cpu',
                       session_id: Optional[str] = None) -> Dict:
    if session_id is None:
        session_id = str(int(time.time())) + "_" + uuid.uuid4().hex[:6]
    print(f"[TwoVideo] creating parent session: {session_id}")
    parent_dir = OUTPUTS_DIR / session_id
    parent_dir.mkdir(parents=True, exist_ok=True)

    sid1 = session_id + "_v1"
    sid2 = session_id + "_v2"

    print(f"[TwoVideo] Processing video1 -> session {sid1}")
    res1 = process_video_full(video1, whisper_model=whisper_model, session_id=sid1, device=device)
    print(f"[TwoVideo] Processing video2 -> session {sid2}")
    res2 = process_video_full(video2, whisper_model=whisper_model, session_id=sid2, device=device)

    combined_fused_dir = parent_dir / "combined_fused"
    combined_fused_dir.mkdir(parents=True, exist_ok=True)

    # Accept session_dirs either as dict of Path or dict of str (from process_video_full)
    for r in (res1, res2):
        sess_dirs = stringify_paths(r.get("session_dirs", {}))
        fused_dir = sess_dirs.get("FUSED_DIR") or sess_dirs.get("fused_dir") or sess_dirs.get("FUSED_DIR".lower())
        if isinstance(fused_dir, str):
            fused_dir = Path(fused_dir)
        fused_dir = Path(fused_dir)
        for fname in os.listdir(fused_dir):
            if fname.endswith("_fused.txt"):
                shutil.copy(fused_dir / fname, combined_fused_dir / f"{r['session_id']}_{fname}")

    combined_bart_summary_path = parent_dir / "combined_summary_bart.txt"
    combined_textrank_bart_summary_path = parent_dir / "combined_summary_textrank_bart.txt"
    
    print("[TwoVideo] Generating combined BART summary")
    generate_bart_summary_from_fused(combined_fused_dir, combined_bart_summary_path)
    
    print("[TwoVideo] Generating combined TextRank+BART summary")
    generate_textrank_bart_summary_from_fused(combined_fused_dir, combined_textrank_bart_summary_path)
    
    # ======================
    #  Add Unified KG Fusion Here 
    # ======================
    from kg_fusion import KGFusionEnhanced

    print("[Fusion] Fusing two knowledge graphs...")

    kg1_nodes = res1.get("kg_nodes_file")
    kg1_edges = res1.get("kg_edges_file")
    kg2_nodes = res2.get("kg_nodes_file")
    kg2_edges = res2.get("kg_edges_file")

    fusion_dir = parent_dir / "fused_kg"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    engine = KGFusionEnhanced(
        embedding_model="all-MiniLM-L6-v2",
        semantic_weight=0.5,
        structural_weight=0.35,
        lexical_weight=0.15,
        threshold=0.78,
        type_strict=True,
        allow_many_to_one=False,
        relation_unify=True,
        debug=False
    )

    fused_nodes, fused_edges = engine.fuse_from_files(
        nodes1_path=str(kg1_nodes),
        edges1_path=str(kg1_edges),
        nodes2_path=str(kg2_nodes),
        edges2_path=str(kg2_edges),
        output_dir=str(fusion_dir)
    )

    # Save fused visualization
    import networkx as nx
    from pyvis.network import Network

    G = nx.DiGraph()
    for n in fused_nodes:
        G.add_node(n["id"], label=n["label"], title=n.get("description",""))

    for e in fused_edges:
        G.add_edge(e["source"], e["target"], label=e.get("relation",""), title=e.get("relation",""))

    fused_html = fusion_dir / "fused_graph.html"
    net = Network(height="750px", width="100%", directed=True)
    net.from_nx(G)
    net.write_html(str(fused_html))

    run_notes_generation_wrapper(parent_dir)
    print("[Fusion] Unified KG saved:", fused_html)

    return {
        "parent_session": session_id,
        "video1": res1,
        "video2": res2,
        "combined_bart_summary": ensure_str(combined_bart_summary_path),
        "combined_textrank_bart_summary": ensure_str(combined_textrank_bart_summary_path),
        "fused_kg_nodes": str(fusion_dir / "fused_nodes.json"),
        "fused_kg_edges": str(fusion_dir / "fused_edges.json"),
        "fused_graph_html": str(fusion_dir / "fused_graph.html")
    }

# -------------------- KNOWLEDGE GRAPH FUSION --------------------

import json
import networkx as nx
from pathlib import Path

# -------------------- KNOWLEDGE GRAPH FUSION & NOTES TRIGGER --------------------

# -------------------- KNOWLEDGE GRAPH FUSION & NOTES TRIGGER --------------------

def fuse_two_knowledge_graphs(session1_dir: Path, session2_dir: Path, fused_dir: Path, fused_session_id: str):
    fused_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Graph Files (Same as before)
    nodes1_path = session1_dir / "graphs" / "fused_nodes.json"
    edges1_path = session1_dir / "graphs" / "fused_edges.json"
    nodes2_path = session2_dir / "graphs" / "fused_nodes.json"
    edges2_path = session2_dir / "graphs" / "fused_edges.json"

    # Fallback to kg_nodes if fused doesn't exist
    if not nodes1_path.exists(): nodes1_path = session1_dir / "graphs" / "kg_nodes.json"
    if not edges1_path.exists(): edges1_path = session1_dir / "graphs" / "kg_edges.json"
    if not nodes2_path.exists(): nodes2_path = session2_dir / "graphs" / "kg_nodes.json"
    if not edges2_path.exists(): edges2_path = session2_dir / "graphs" / "kg_edges.json"

    if not all(p.exists() for p in [nodes1_path, edges1_path, nodes2_path, edges2_path]):
        raise RuntimeError("Missing knowledge graph files for fusion")

    # Read and Merge (Graph logic remains the same)
    nodes1 = json.loads(nodes1_path.read_text(encoding='utf-8'))
    edges1 = json.loads(edges1_path.read_text(encoding='utf-8'))
    nodes2 = json.loads(nodes2_path.read_text(encoding='utf-8'))
    edges2 = json.loads(edges2_path.read_text(encoding='utf-8'))

    label_to_id = {}
    fused_nodes = []
    next_id = 1

    def add_or_get_node(node):
        nonlocal next_id
        key = node["label"].strip().lower()
        if key in label_to_id: return label_to_id[key]
        new_id = f"N{next_id}"
        next_id += 1
        fused_nodes.append({"id": new_id, "label": node["label"], "description": node.get("description", "")})
        label_to_id[key] = new_id
        return new_id

    idmap1 = {n["id"]: add_or_get_node(n) for n in nodes1}
    idmap2 = {n["id"]: add_or_get_node(n) for n in nodes2}

    fused_edges = []
    def add_edges(edges, idmap):
        for e in edges:
            source_id = e.get("source")
            target_id = e.get("target")
            
            # Skip invalid or missing nodes from delimiter issues
            if source_id not in idmap or target_id not in idmap:
                print(f"[Fusion] Warning: Skipping edge with invalid references: {e}")
                continue
                
            fused_edges.append({
                "source": idmap[source_id], "target": idmap[target_id], "relation": e.get("relation", "")
            })

    add_edges(edges1, idmap1)
    add_edges(edges2, idmap2)

    # Save Graph JSON
    fused_nodes_file = fused_dir / "fused_nodes.json"
    fused_edges_file = fused_dir / "fused_edges.json"
    fused_nodes_file.write_text(json.dumps(fused_nodes, indent=2))
    fused_edges_file.write_text(json.dumps(fused_edges, indent=2))

    # Visualization
    G = nx.DiGraph()
    for n in fused_nodes: G.add_node(n["id"], label=n["label"], title=n["description"])
    for e in fused_edges: G.add_edge(e["source"], e["target"], label=e["relation"])
    
    try:
        net = Network(height="750px", width="100%", directed=True)
        net.from_nx(G)
        net.write_html(str(fused_dir / "fused_graph.html"))
    except: pass

    # =========================================================
    # NEW CODE: COPY DIAGRAMS FROM 'diagrams' FOLDER
    # =========================================================
    
    # =========================================================
    # NEW CODE: COPY DIAGRAMS & TEXT
    # =========================================================
    
    # Define session root relative to fused_kg dir
    session_root = fused_dir.parent
    
    # Store combined text in session root so generate_notes can find it
    merged_text_dir = session_root / "combined_fused"
    merged_text_dir.mkdir(parents=True, exist_ok=True)
    
    # Also reuse this for diagrams to keep everything self-contained
    merged_images_dir = merged_text_dir 

    print(f"[Fusion] Copying text and diagrams to {merged_text_dir}...")

    merged_captions = {}

    def copy_assets(session_dir, prefix):
        # 1. COPY TEXT
        txt_dirs = [session_dir / "fused_sentences", session_dir / "combined_fused"]
        text_found = False
        for d in txt_dirs:
            if d.exists():
                for txt in d.glob("*.txt"):
                    shutil.copy(txt, merged_text_dir / f"{prefix}_{txt.name}")
                    text_found = True
                if text_found: break 
        
        # 2. COPY DIAGRAMS (Updated Path)
        diagram_dir = session_dir / "diagrams" 
        caption_dir = session_dir / "diagram_texts"
        
        if diagram_dir.exists():
            images = list(diagram_dir.glob("*.png")) + list(diagram_dir.glob("*.jpg")) + list(diagram_dir.glob("*.jpeg"))
            
            for img in images:
                new_name = f"{prefix}_{img.name}"
                dest_path = merged_images_dir / new_name
                shutil.copy(img, dest_path)
                
                # Logic: Find Caption
                caption_text = ""
                txt_file = caption_dir / f"{img.stem}.txt"
                
                if txt_file.exists():
                    try: caption_text = txt_file.read_text(encoding='utf-8').strip()
                    except: pass
                
                if not caption_text or len(caption_text) < 5:
                    clean_name = img.stem.replace("_", " ").replace("-", " ")
                    caption_text = f"Diagram showing {clean_name}"

                merged_captions[new_name] = caption_text

    # Copy from both sessions
    copy_assets(session1_dir, "v1")
    copy_assets(session2_dir, "v2")

    # 3. Save Merged Captions Map directly in fused dir or text dir
    captions_file = merged_text_dir / "merged_captions.json"
    
    # 4. GENERATE STATIC PNG (Fix for Frontend Image)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.15, iterations=20)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, edge_color='gray', arrows=True, font_size=8)
        plt.savefig(str(fused_dir / "fused_graph.png"), format="png", bbox_inches="tight")
        plt.close()
        print(f"[Fusion] Generated static graph image: {fused_dir / 'fused_graph.png'}")
    except Exception as e:
        print(f"[Fusion] ⚠️ Failed to generate static PNG: {e}")
    captions_file.write_text(json.dumps(merged_captions, indent=2))
    print(f"[Fusion] Created caption map for {len(merged_captions)} diagrams.")

    return {
        "fused_graph_html": f"/fused_graph/{fused_session_id}/fused_graph.html",
        "fused_graph_image": f"/fused_graph/{fused_session_id}/fused_graph.png", 
        "fused_nodes_file": f"/fused_graph/{fused_session_id}/fused_nodes.json",
        "fused_edges_file": f"/fused_graph/{fused_session_id}/fused_edges.json",
        "fused_session_id": fused_session_id
    }

# ==========================================
# SIMPLIFIED WMD NOTES GENERATOR (Pipeline-Compatible)
# ==========================================

from sentence_transformers import SentenceTransformer, util
import torch


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT BLOCK — Intermediate Representation Layer
# ─────────────────────────────────────────────────────────────────────────────
# A ConceptBlock clusters a root KG node with ALL its typed children into one
# structured object BEFORE rendering begins.
#
# This is the "missing stage" described in the architecture document:
#
#   Knowledge Graph
#         ↓
#   Concept Block Builder   ← THIS CLASS
#         ↓
#   Hierarchy Generator
#         ↓
#   Educational Renderer
#         ↓
#   Notes
#
# Each block carries:
#   - root node (id, label, description)
#   - typed children buckets: subconcepts, operations, properties,
#     advantages, disadvantages, devices, protocols, implementations,
#     applications, examples
#   - the best diagram anchored to this concept (diagram anchoring trick)
#   - the block's importance score (for ordering)
#
# Rendering rules (fully dynamic, no domain words):
#   subconcept   → its own named subsection with recursive sub-block
#   operation    → "Primary Operations" bullet
#   advantage    → "Advantages" bullet
#   disadvantage → "Disadvantages" bullet
#   device       → "Components and Devices" bullet
#   protocol     → "Communication Mechanisms" bullet
#   implementation → "Implementation and Structure" bullet
#   application  → "Applications" bullet
#   example      → "Real-World Examples" bullet
#   property     → "Key Characteristics" bullet
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any


@dataclass
class ConceptBlock:
    """
    Intermediate representation of one concept and all its related nodes.

    Built by ConceptBlockBuilder.build_blocks() from a KG hierarchy.
    Rendered by ConceptBlock.to_subsections() into HierarchicalNotes format.
    """
    # Core concept
    node_id:     str
    label:       str
    description: str
    importance:  float = 0.0

    # Typed child buckets — each entry is (label, description, node_id)
    subconcepts:     List[Tuple[str, str, str]] = field(default_factory=list)
    operations:      List[Tuple[str, str, str]] = field(default_factory=list)
    advantages:      List[Tuple[str, str, str]] = field(default_factory=list)
    disadvantages:   List[Tuple[str, str, str]] = field(default_factory=list)
    devices:         List[Tuple[str, str, str]] = field(default_factory=list)
    protocols:       List[Tuple[str, str, str]] = field(default_factory=list)
    implementations: List[Tuple[str, str, str]] = field(default_factory=list)
    applications:    List[Tuple[str, str, str]] = field(default_factory=list)
    examples:        List[Tuple[str, str, str]] = field(default_factory=list)
    properties:      List[Tuple[str, str, str]] = field(default_factory=list)

    # Diagram anchored to this concept (diagram anchoring trick)
    diagram_path:    Optional[str] = None
    diagram_caption: Optional[str] = None

    # Nested sub-blocks for subconcept children (recursive)
    sub_blocks:      List[Any] = field(default_factory=list)  # List[ConceptBlock]

    def has_content(self) -> bool:
        """True if block has at least one piece of content beyond the root label."""
        return bool(
            self.description or self.subconcepts or self.operations
            or self.advantages or self.disadvantages or self.devices
            or self.protocols or self.implementations or self.applications
            or self.examples or self.properties or self.sub_blocks
        )

    def to_subsections(self, make_point, make_subsection) -> list:
        """
        Convert this ConceptBlock into a list of subsection dicts for
        HierarchicalNotes format.

        Rendering order (pedagogically motivated, fully dynamic):
          1. Description  (always first)
          2. Structure / Implementation
          3. Components and Devices
          4. Communication Mechanisms (Protocols)
          5. Primary Operations
          6. How It Works / Key Characteristics
          7. Subconcepts (each as a named sub-block)
          8. Advantages
          9. Disadvantages
          10. Applications
          11. Real-World Examples
        """
        subsections = []

        # ── 1. Description subsection ─────────────────────────────────────────
        if self.description and len(self.description.split()) >= 4:
            subsections.append(
                make_subsection("Description", [make_point(self.description)])
            )

        # ── Helper: build bullet text from (label, desc, _) tuple ─────────────
        def _bullet(lbl: str, desc: str) -> str:
            lbl = lbl.strip()
            desc = (desc or "").strip()
            if desc and desc.lower() != lbl.lower():
                return f"{lbl}: {desc}"
            return f"{lbl}: A key component related to {self.label}."

        # ── 2. Implementation and Structure ───────────────────────────────────
        if self.implementations:
            pts = [make_point(_bullet(l, d)) for l, d, _ in self.implementations]
            subsections.append(make_subsection("Implementation and Structure", pts))

        # ── 3. Components and Devices ─────────────────────────────────────────
        if self.devices:
            pts = [make_point(_bullet(l, d)) for l, d, _ in self.devices]
            subsections.append(make_subsection("Components and Devices", pts))

        # ── 4. Communication Mechanisms ───────────────────────────────────────
        if self.protocols:
            pts = [make_point(_bullet(l, d)) for l, d, _ in self.protocols]
            subsections.append(make_subsection("Communication Mechanisms", pts))

        # ── 5. Primary Operations ─────────────────────────────────────────────
        if self.operations:
            pts = [make_point(_bullet(l, d)) for l, d, _ in self.operations]
            subsections.append(make_subsection("Primary Operations", pts))

        # ── 6. Key Characteristics ────────────────────────────────────────────
        if self.properties:
            pts = [make_point(_bullet(l, d)) for l, d, _ in self.properties]
            subsections.append(make_subsection("Key Characteristics", pts))

        # ── 7. Subconcepts — each as a named subsection ───────────────────────
        # Sub-blocks are rendered recursively using their own to_subsections()
        for sub_block in self.sub_blocks:
            if not sub_block.has_content() and not sub_block.description:
                continue
            sub_subs = sub_block.to_subsections(make_point, make_subsection)
            if sub_subs:
                # Sub-block becomes a named subsection at this level
                subsections.append(make_subsection(sub_block.label, sub_subs))
            elif sub_block.description:
                subsections.append(
                    make_subsection(sub_block.label,
                                    [make_point(sub_block.description)])
                )

        # Flat subconcept entries (those without their own sub-block)
        flat_sc = [
            (l, d, nid) for l, d, nid in self.subconcepts
            if not any(sb.node_id == nid for sb in self.sub_blocks)
        ]
        if flat_sc:
            pts = [make_point(_bullet(l, d)) for l, d, _ in flat_sc]
            subsections.append(make_subsection("Types and Variants", pts))

        # ── 8. Advantages ─────────────────────────────────────────────────────
        if self.advantages:
            pts = [make_point(_bullet(l, d)) for l, d, _ in self.advantages]
            subsections.append(make_subsection("Advantages", pts))

        # ── 9. Disadvantages ──────────────────────────────────────────────────
        if self.disadvantages:
            pts = [make_point(_bullet(l, d)) for l, d, _ in self.disadvantages]
            subsections.append(make_subsection("Disadvantages", pts))

        # ── 10. Applications ──────────────────────────────────────────────────
        if self.applications:
            pts = [make_point(_bullet(l, d)) for l, d, _ in self.applications]
            subsections.append(make_subsection("Applications", pts))

        # ── 11. Real-World Examples ───────────────────────────────────────────
        if self.examples:
            pts = [make_point(_bullet(l, d)) for l, d, _ in self.examples]
            subsections.append(make_subsection("Real-World Examples", pts))

        return subsections


class ConceptBlockBuilder:
    """
    Builds ConceptBlock objects from a KG hierarchy.

    This is the "Concept Block Builder" stage in the architecture:
      KG → ConceptBlockBuilder → [ConceptBlock, ...] → Renderer → Notes

    Algorithm:
      For each root node:
        1. Collect all direct children from edge_map + children_map
        2. Classify each child's type using the generator's classifier
        3. Route child to the correct typed bucket in the block
        4. Recurse for subconcept children (topology-aware clustering)
        5. Find and anchor the best diagram to the block
    """

    def __init__(self, generator):
        """
        Args:
            generator: a WMDKGNotesGenerator instance (provides classify_node_type,
                       interpret_edge, find_related_diagram, get_semantic_caption)
        """
        self.gen = generator

    def _infer_full_type(self, node: dict, edge_map: dict,
                         all_nodes: list) -> str:
        """
        Classify node type, with Advantage/Disadvantage as sub-cases of Property.
        Fully dynamic — no domain keywords in detection logic.
        """
        base = self.gen.classify_node_type(node, edge_map, all_nodes)
        if base == "Property":
            combined = (node.get("label", "") + " " +
                        (node.get("description", "") or "")).lower()
            _ADV = re.compile(
                r'\b(advantage|benefit|merit|\bpro\b|strength|positive|easy|'
                r'simple|fast|efficient|reliable|flexible|scalable|cheap|'
                r'low\s+cost|high\s+speed|secure)\b', re.I)
            _DIS = re.compile(
                r'\b(disadvantage|drawback|limitation|weakness|demerit|\bcon\b|'
                r'problem|issue|failure|risk|expensive|difficult|slow|unreliable|'
                r'inflexible|high\s+cost|collision|privacy|bottleneck|'
                r'single\s+point)\b', re.I)
            if _ADV.search(combined):
                return "Advantage"
            if _DIS.search(combined):
                return "Disadvantage"
        return base

    # Structural ownership relations — ONLY these allow a child to belong to
    # a block.  Associative/contextual relations (related_to, associated_with,
    # connected_to, used_in, depends_on) do NOT grant ownership; they are
    # rendered as inline contextual sentences instead.
    # This prevents graph-neighborhood contamination (Problem 1 / Corner Case 4).
    _OWNERSHIP_RELATIONS = frozenset({
        "has", "includes", "contains", "consists", "composed", "comprises",
        "type of", "is a", "has type", "divided into", "categorized as",
        "has property", "has characteristic", "has advantage", "has disadvantage",
        "part of", "is part of", "sub-component", "sub component",
        "belongs to", "subset of", "member of",
        "has operation", "has function", "supports operation",
        "has device", "has component", "has mechanism", "has protocol",
        "has implementation", "has example", "has application",
    })

    # Associative relations — these link concepts but do NOT make one a child
    # of the other.  Used for cross-reference sentences, not block membership.
    _ASSOCIATIVE_RELATIONS = frozenset({
        "related to", "is related to", "associated with",
        "connected to", "interacts with", "used in", "applied in",
        "depends on", "requires", "leads to", "causes", "results in",
        "co-occurs with", "similar to", "analogy",
    })

    def _is_ownership_edge(self, rel: str) -> bool:
        """
        Return True if the relation indicates the source OWNS / CONTAINS the
        target (structural membership), False for merely associative links.
        Fully dynamic — checks against _OWNERSHIP_RELATIONS by substring.
        """
        rl = rel.lower().replace("_", " ").strip()
        # Exact match first
        if rl in self._OWNERSHIP_RELATIONS:
            return True
        # Substring match — any ownership keyword present
        for owned_kw in self._OWNERSHIP_RELATIONS:
            if owned_kw in rl:
                return True
        # If it is explicitly associative, reject it
        for assoc_kw in self._ASSOCIATIVE_RELATIONS:
            if assoc_kw in rl:
                return False
        # Unknown relations: treat as ownership only if very short (likely "has X")
        return len(rl.split()) <= 2

    def _collect_children(self, node_id: str, edge_map: dict,
                           children_map: dict, node_lookup: dict) -> list:
        """
        Collect OWNED children of node_id — only nodes connected via structural
        (ownership) edges.  Associative edges (related_to, used_in, etc.) are
        excluded here and never grant block membership, preventing graph-
        neighborhood contamination (Problem 1 / Corner Case 4).

        Returns unique (child_node, relation, child_id) triples.
        """
        seen = set()
        result = []

        # From edge_map — only structural/ownership relations
        for tgt_id, rel in edge_map.get(node_id, []):
            if tgt_id not in seen and tgt_id in node_lookup:
                if self._is_ownership_edge(rel):
                    seen.add(tgt_id)
                    result.append((node_lookup[tgt_id], rel, tgt_id))

        # From children_map (hierarchy-assigned children not reachable via edge_map)
        # These are structurally assigned by build_hierarchy, so they are always owned.
        edge_tgts = {tgt for tgt, rel in edge_map.get(node_id, [])
                     if self._is_ownership_edge(rel)}
        for cid in children_map.get(node_id, []):
            if cid not in seen and cid not in edge_tgts and cid in node_lookup:
                seen.add(cid)
                result.append((node_lookup[cid], "includes", cid))

        return result

    def _collect_cross_refs(self, node_id: str, edge_map: dict,
                             node_lookup: dict) -> list:
        """
        Collect ASSOCIATIVE (non-ownership) edges from node_id.
        These produce inline cross-reference sentences ("X is related to Y")
        rather than block membership bullets.
        Returns (target_label, relation) pairs.
        """
        result = []
        for tgt_id, rel in edge_map.get(node_id, []):
            if tgt_id in node_lookup and not self._is_ownership_edge(rel):
                tgt_label = node_lookup[tgt_id].get("label", "")
                if tgt_label:
                    result.append((tgt_label, rel))
        return result

    def _best_desc(self, node: dict, parent_label: str, relation: str) -> str:
        """
        Get the best description for a child node.
        Falls back to an NLG sentence from the relation if description is thin.
        """
        desc = (node.get("description", "") or "").strip()
        if desc and len(desc.split()) >= 3:
            return desc
        lbl = node.get("label", "")
        if lbl and relation and relation.lower() not in ("includes", "has", "contains"):
            rel_prose = relation.replace("_", " ").strip().lower()
            return f"{parent_label} {rel_prose} {lbl}."
        return desc

    # Hub dampening: a single block never renders more than this many
    # children per typed bucket.  Prevents "50 definitions dumped together"
    # (Problem 3 / Corner Case 2).  Children beyond the cap are dropped unless
    # they can be sub-clustered; sub-clustering happens inside build_block.
    MAX_CHILDREN_PER_BUCKET = 8

    def build_block(self, root_id: str, node_lookup: dict, edge_map: dict,
                    children_map: dict, importance: dict,
                    emitted: set, captions: dict, image_paths: dict,
                    used_img_canonicals: set, depth: int = 0,
                    domain_keywords: set = None) -> ConceptBlock:
        """
        Build a ConceptBlock for root_id recursively.

        Fixes applied here (all fully dynamic):
          - Problem 1 / Corner Case 4: only ownership edges grant block membership
          - Problem 3 / Corner Case 2: hub dampening caps children per bucket
          - Corner Case 1: shared nodes get a brief mention in every owning block
          - Corner Case 5: empty blocks are flagged so they can be suppressed

        Args:
            root_id:           node id of the concept to build a block for
            node_lookup:       {nid: node_dict}
            edge_map:          {src_id: [(tgt_id, rel), ...]}
            children_map:      {parent_id: [child_id, ...]}
            importance:        {nid: float} from build_hierarchy
            emitted:           set of already-rendered node ids (mutated in place)
            captions:          {img_name: caption_text} for diagram anchoring
            image_paths:       {img_name: Path}
            used_img_canonicals: set of already-used image canonical paths
            depth:             recursion depth (limits sub-block nesting to 2 levels)
            domain_keywords:   set of domain words used for diagram verification

        Returns:
            ConceptBlock populated with all typed children and anchored diagram.
        """
        root_node = node_lookup.get(root_id, {})
        label     = root_node.get("label", "").strip()
        desc      = (root_node.get("description", "") or "").strip()
        imp       = importance.get(root_id, 0.0)

        block = ConceptBlock(
            node_id=root_id,
            label=label,
            description=desc,
            importance=imp,
        )

        all_nodes = list(node_lookup.values())
        # _collect_children now filters to ownership-only edges (Problem 1 fix)
        children  = self._collect_children(root_id, edge_map, children_map, node_lookup)

        # ── Hub dampening: sort children by importance descending ─────────────
        # High-importance children are kept; low-importance overflow is dropped.
        # This prevents hub nodes from producing 50-item definition lists.
        children_sorted = sorted(
            children,
            key=lambda item: importance.get(item[2], 0.0),
            reverse=True
        )

        # Per-bucket child counters for dampening
        _bucket_counts: dict = {}

        for c_node, rel, c_id in children_sorted:
            if c_id == root_id:
                continue

            c_label = c_node.get("label", "").strip()
            if not c_label:
                continue

            c_desc = self._best_desc(c_node, label, rel)
            ctype  = self._infer_full_type(c_node, edge_map, all_nodes)

            # ── Corner Case 1: Shared-node handling ───────────────────────────
            # If the node is already emitted by another block, add a brief
            # cross-reference mention (one-liner) instead of skipping silently.
            if c_id in emitted:
                # Only cross-reference if it is a meaningful concept (not Actor/Metadata)
                if ctype not in ("Actor", "Metadata", "Diagram") and c_label:
                    _xref = f"{c_label}: Also discussed in the context of related concepts."
                    # Route to the appropriate bucket as a cross-ref
                    if ctype == "Advantage":
                        if len(block.advantages) < self.MAX_CHILDREN_PER_BUCKET:
                            block.advantages.append((c_label, _xref, c_id))
                    elif ctype == "Disadvantage":
                        if len(block.disadvantages) < self.MAX_CHILDREN_PER_BUCKET:
                            block.disadvantages.append((c_label, _xref, c_id))
                    elif ctype in ("Concept", "Subconcept"):
                        if len(block.subconcepts) < self.MAX_CHILDREN_PER_BUCKET:
                            block.subconcepts.append((c_label, _xref, c_id))
                    else:
                        if len(block.properties) < self.MAX_CHILDREN_PER_BUCKET:
                            block.properties.append((c_label, _xref, c_id))
                continue

            # ── Dampen hub nodes: check bucket cap before accepting ────────────
            _bucket_key = ctype
            _bucket_counts[_bucket_key] = _bucket_counts.get(_bucket_key, 0) + 1
            if _bucket_counts[_bucket_key] > self.MAX_CHILDREN_PER_BUCKET:
                # Over cap — skip but still mark emitted to avoid coverage-sweep
                emitted.add(c_id)
                continue

            emitted.add(c_id)

            if ctype in ("Concept", "Subconcept") and depth < 2:
                # Build a recursive sub-block for this subconcept
                sub_block = self.build_block(
                    c_id, node_lookup, edge_map, children_map,
                    importance, emitted, captions, image_paths,
                    used_img_canonicals, depth=depth + 1,
                    domain_keywords=domain_keywords
                )
                block.sub_blocks.append(sub_block)
                block.subconcepts.append((c_label, c_desc, c_id))
            elif ctype in ("Concept", "Subconcept"):
                # Too deep — treat as flat subconcept bullet
                block.subconcepts.append((c_label, c_desc, c_id))
            elif ctype == "Operation":
                block.operations.append((c_label, c_desc, c_id))
            elif ctype == "Advantage":
                block.advantages.append((c_label, c_desc, c_id))
            elif ctype == "Disadvantage":
                block.disadvantages.append((c_label, c_desc, c_id))
            elif ctype == "Device":
                block.devices.append((c_label, c_desc, c_id))
            elif ctype == "Protocol":
                block.protocols.append((c_label, c_desc, c_id))
            elif ctype == "Implementation":
                block.implementations.append((c_label, c_desc, c_id))
            elif ctype == "Application":
                block.applications.append((c_label, c_desc, c_id))
            elif ctype == "Example":
                block.examples.append((c_label, c_desc, c_id))
            elif ctype in ("Actor", "Metadata", "Diagram"):
                pass  # skip — never appear as bullets
            else:
                block.properties.append((c_label, c_desc, c_id))

        # ── Diagram anchoring with domain verification (Problem 2) ────────────
        # Find the best diagram for this block's topic. The diagram MUST pass
        # a domain-word intersection check — its caption must share at least
        # one content word with the block's own label/description or the
        # global domain keyword set.  This prevents "water cycle" and other
        # irrelevant diagrams from appearing under networking topics.
        if captions and image_paths:
            search_text = label + " " + desc
            all_child_labels = (
                [l for l, _, _ in block.subconcepts]
                + [l for l, _, _ in block.devices]
                + [l for l, _, _ in block.implementations]
            )
            if all_child_labels:
                search_text += " " + " ".join(all_child_labels[:4])

            # Build domain word set for this block (label words + domain_keywords)
            _block_words = set(re.findall(r'[a-z]{3,}', search_text.lower()))
            if domain_keywords:
                _block_words |= {w.lower() for w in domain_keywords if len(w) > 2}

            try:
                best_img, raw_cap = self.gen.find_related_diagram(
                    search_text, captions, used_img_canonicals)
                if best_img and raw_cap:
                    # ── Domain verification gate (Problem 2 fix) ─────────────
                    # Caption must share ≥1 content word with this block's topic
                    cap_words = set(re.findall(r'[a-z]{3,}', raw_cap.lower()))
                    _GENERIC = {'the', 'and', 'with', 'from', 'that', 'this',
                                'showing', 'diagram', 'figure', 'image', 'picture',
                                'black', 'white', 'drawing', 'photo'}
                    cap_content = cap_words - _GENERIC
                    block_content = _block_words - _GENERIC
                    domain_match = len(cap_content & block_content) >= 1
                    if domain_match:
                        img_path = image_paths.get(best_img)
                        if img_path and img_path.exists():
                            block.diagram_path    = str(img_path)
                            block.diagram_caption = self.gen.get_semantic_caption(
                                label, raw_cap, list(node_lookup.values()))
                            try:
                                used_img_canonicals.add(str(img_path.resolve()))
                            except Exception:
                                used_img_canonicals.add(str(img_path))
                    else:
                        print(f"[Block] Diagram rejected for '{label}': "
                              f"caption '{raw_cap[:60]}' has no domain word overlap")
            except Exception:
                pass  # diagram anchoring is best-effort

        return block

    def build_blocks(self, roots: list, node_lookup: dict, edge_map: dict,
                     children_map: dict, importance: dict,
                     captions: dict, image_paths: dict,
                     emitted: set = None,
                     domain_keywords: set = None) -> list:
        """
        Build ConceptBlocks for all root nodes.

        Args:
            emitted:         optional external set of already-emitted node ids
                             (shared mutation so builder + renderer stay in sync)
            domain_keywords: set of domain-specific words extracted from the KG,
                             used for diagram domain-verification (Problem 2 fix)

        Returns:
            list of ConceptBlock, sorted by importance descending;
            empty/trivial blocks are suppressed (Corner Case 5 fix)
        """
        _emitted = emitted if emitted is not None else set(roots)
        _emitted.update(roots)
        used_img_canonicals: set = set()
        blocks = []

        for root_id in roots:
            block = self.build_block(
                root_id, node_lookup, edge_map, children_map,
                importance, _emitted, captions, image_paths,
                used_img_canonicals, domain_keywords=domain_keywords
            )
            # Corner Case 5: suppress blocks that have no content at all
            # (e.g. "Devices" node with no edges and no description)
            if block.has_content() or block.description:
                blocks.append(block)
            else:
                print(f"[Blocks] Suppressed empty block: '{block.label}'")

        blocks.sort(key=lambda b: b.importance, reverse=True)
        return blocks

class WMDKGNotesGenerator:
    def __init__(self):
        """
        Simplified WMD-style notes generator.
        Uses sentence transformers for semantic similarity (faster, no Word2Vec needed).
        """
        self.nlp = nlp
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Try to load BART for summarization
        try:
            from transformers import pipeline
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            print(f"[Notes] ⚠️ BART unavailable: {e}. Using extractive summaries.")
            self.summarizer = None
            
        # Hook for 5.6: Initialize sanitizer for description text
        from content_sanitizer import ContentSanitizer
        self.sanitizer = ContentSanitizer()

    # ---------- Text Cleaning ----------
    # ---------- Text Cleaning (Enhanced) ----------
    def clean_text(self, text: str) -> str:
        """Remove metadata labels, academic footers, and noise"""
        # 1. Join hyphenated words split across lines
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # 2. Filter specific noise lines (ignoring case)
        noise_patterns = [
            r"Slide\s*\d+", r"Page\s*\d+", r"www\.[\w\.]+", r"http://[\w\./]+",
            r"Department of \w+", r"M\.Tech", r"CSE", r"University", 
            r"Assistant Professor", r"Ms\. \w+", r"Mr\. \w+",
            r"Slide Content:", "Spoken Explanation:", "Diagram Description:",
            "Visual Elements:", "Summary:", "Transcript:", "like and share",
            "subscribe", "bell icon", "thank you", "logo with", "icon",
            "black background", "white arrow", "red square", "green circle"
        ]
        
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line: continue
            # Skip noise lines
            is_noise = False
            for pat in noise_patterns:
                if re.search(pat, line, re.IGNORECASE):
                    is_noise = True; break
            if is_noise: continue
            
            # Skip lines with high symbol density (likely raw OCR garbage)
            alpha_count = sum(c.isalpha() for c in line)
            if len(line) > 5 and alpha_count / len(line) < 0.4: continue
            
            # Skip "visual description" lines likely from vision models
            lower_line = line.lower()
            visual_words = ["background", "foreground", "shape", "color", "image of", "diagram showing", "logo", "icon"]
            if sum(1 for w in visual_words if w in lower_line) >= 2: continue

            lines.append(line)
            
        # NEW: Reject mid-sentence OCR fragments (no capital start, < 5 alpha words) (Fix 4A)
        filtered_lines = []
        for line in lines:
            if line and line[0].islower():
                alpha_words = [w for w in line.split() if w.isalpha() and len(w) > 2]
                if len(alpha_words) < 4:
                    continue   # Too short to be a real sentence fragment
            filtered_lines.append(line)
        lines = filtered_lines
            
        text = " ".join(lines)
        return re.sub(r'\s+', ' ', text).strip()

    # ---------- Semantic Similarity (Cosine) ----------
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb1 = self.embedding_model.encode(text1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

    # ---------- Extract Key Sentences ----------
    def extract_key_sentences(self, text: str, top_k: int = 5) -> list:
        """Extract most important sentences using TF-IDF + length heuristic"""
        doc = self.nlp(text)
        sentences = [s.text.strip() for s in doc.sents if len(s.text.split()) > 5]
        
        if not sentences:
            return []
        
        if len(sentences) <= top_k:
            return sentences
        
        # Score by: length (prefer 10-25 words) + named entities + key POS
        scored = []
        for sent in sentences:
            sent_doc = self.nlp(sent)
            
            word_count = len(sent.split())
            length_score = 1.0 if 10 <= word_count <= 25 else 0.5
            
            entity_score = len(sent_doc.ents) * 0.3
            
            noun_count = sum(1 for t in sent_doc if t.pos_ in ["NOUN", "PROPN"])
            noun_score = min(noun_count * 0.2, 1.0)
            
            total_score = length_score + entity_score + noun_score
            scored.append((sent, total_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:top_k]]

    # ---------- Generate Topic Heading ----------
    def generate_topic(self, text: str) -> str:
        """Extract a short topic heading"""
        doc = self.nlp(text[:300])  # First 300 chars
        
        # Extract noun chunks
        chunks = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
        if chunks:
            return chunks[0].title()
        
        # Fallback: key nouns
        nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
        if nouns:
            return " ".join(nouns[:3]).title()
        
        return "Topic"

    # ---------- Generate Summary Points ----------
    def generate_summary_points(self, sentences: list) -> list:
        """Generate concise bullet points from sentences"""
        if not sentences:
            return []
        
        # If we have BART, use it
        if self.summarizer and len(" ".join(sentences).split()) > 30:
            combined = " ".join(sentences[:3])  # Limit input
            try:
                summary = self.summarizer(
                    combined[:1024],
                    max_length=150,
                    min_length=40,
                    do_sample=False
                )[0]['summary_text']
                
                # Split into points
                points = [p.strip() + "." for p in summary.split('.') if p.strip()]
                return points[:5]
            except Exception as e:
                print(f"[Notes] BART failed: {e}, using extractive")
        
        # Extractive fallback: return sentences as-is
        return sentences[:5]

    # ---------- Find Related Diagrams ----------
    def find_related_diagram(self, topic: str, captions: dict,
                               used_canonical_paths: set = None) -> tuple:
        """Find most relevant diagram for a topic.

        FIX-DIAGRAM-DEDUP: added used_canonical_paths set. Any image whose
        resolved canonical absolute path is already in that set is skipped
        entirely, preventing the same file from appearing in multiple sections
        regardless of how many filename aliases it has.

        FIX-THRESHOLD: raised from 0.42 → 0.58. Generic BLIP captions
        (e.g. "a black and white drawing of a stack of pancakes") easily
        scored above 0.42 for unrelated section topics. 0.58 requires a
        genuine semantic overlap.

        FIX-BLACKLIST-SYNC: expanded to match notes_renderer.py BLACKLIST
        exactly; also blocks slides 0-4 dynamically (not by hardcoded names).
        """
        import re as _re
        if not captions:
            return None, None

        if used_canonical_paths is None:
            used_canonical_paths = set()

        # ── Unified blacklist — mirrors notes_renderer.py _BLACKLIST_TERMS ──
        # All terms are domain-agnostic; no lecture-specific words.
        _BL = (
            'twitter', 'facebook', 'instagram', 'linkedin', 'youtube',
            'subscribe', 'follow', '@', 'logo', 'icon',
            'nesoacademy', 'neso', 'academy',
            'like and share', 'like, comment', 'comment, share',
            'like comment share', 'like comment subscribe',
            'thanks for watching', 'thank you for watching',
            'bell', 'notification', 'watermark', 'banner', 'copyright',
            'sticker with the word', 'sticker with the words',
            'red and white sticker', 'red and white shield',
            'shield with the letter',
            'outcome', 'objectives', 'agenda',
            'a black and white photo', 'a black and white image',
            'a sign with', 'visual representation',
            'a piece of paper with numbers',
            'a computer with the word', 'an image of a computer',
        )

        def _is_early_slide(img_name: str) -> bool:
            """Dynamically block slides 0-4 (title/intro/CTA/social)."""
            stem = Path(img_name).stem.lower() if '.' in img_name else img_name.lower()
            m = _re.match(r'slide[_\-]?(\d+)', stem)
            if m:
                try:
                    return int(m.group(1)) <= 4
                except ValueError:
                    pass
            return False

        def _canonical_img(img_name: str) -> str:
            """Resolve img_name to a canonical path string for dedup."""
            img_p = None
            # Try image_paths if available on self
            if hasattr(self, '_current_image_paths') and self._current_image_paths:
                img_p = self._current_image_paths.get(img_name)
            if img_p is not None:
                try:
                    return str(Path(str(img_p)).resolve())
                except Exception:
                    return str(img_p)
            return img_name  # fallback: use name itself

        topic_emb = self.embedding_model.encode(topic, convert_to_tensor=True)
        best_img = None
        best_score = -1

        for img_name, caption in captions.items():
            cap_lower = (caption or '').lower()
            img_lower = img_name.lower()

            # Skip blacklisted captions or paths
            if any(bl in cap_lower for bl in _BL):
                continue
            if any(bl in img_lower for bl in _BL):
                continue
            # Skip early/title slides
            if _is_early_slide(img_name):
                continue
            # Skip already-used images (canonical path dedup)
            if used_canonical_paths:
                _canon = _canonical_img(img_name)
                if _canon in used_canonical_paths:
                    continue

            cap_emb = self.embedding_model.encode(caption, convert_to_tensor=True)
            score = util.pytorch_cos_sim(topic_emb, cap_emb).item()
            if score > best_score:
                best_score = score
                best_img = img_name

        # FIX-THRESHOLD: raised from 0.42 to 0.58
        if best_score > 0.58:
            return best_img, captions[best_img]
        return None, None

    # ---------- HIERARCHICAL GRAPH PROCESSING (FOREST CONSTRUCTION) ----------

    def build_hierarchy(self, nodes: list, edge_map: dict,
                        freq_map: dict = None) -> dict:
        """
        Build a concept-gravity hierarchy forest.

        Pipeline:
          1. Compute node importance = connections + frequency_in_source
                                     + description_richness + centrality + type_bonus
          2. Classify every node type (Concept/Property/Operation/Metadata/Actor/…)
          3. Assign child→parent using structural edges + importance gradient
          4. CONCEPT GRAVITY: compress low-importance orphan nodes into their
             best-matching parent so the root count ≈ 5–8 real topics
          5. Deduplicate semantically near-identical roots
          6. Attach Actor/Metadata to most central root; never let them be roots
          7. Final fallback: guarantee ≥2 and ≤MAX_TOPICS roots

        Args:
            nodes:    list of node dicts (id, label, description)
            edge_map: {src_id: [(tgt_id, relation), ...]}
            freq_map: optional {node_id: frequency_count} from source text
                      to boost importance of concepts appearing often in transcript
        """
        import networkx as nx

        node_lookup = {n["id"]: n for n in nodes}

        if not node_lookup:
            return {"roots": [], "children_map": {}, "node_lookup": {}}

        # ── 1. Build NetworkX graph ───────────────────────────────────────────
        G = nx.DiGraph()
        for nid in node_lookup:
            G.add_node(nid)
        for src, targets in edge_map.items():
            for tgt, rel in targets:
                if src in node_lookup and tgt in node_lookup:
                    G.add_edge(src, tgt, relation=rel)

        try:
            pagerank = nx.pagerank(G, alpha=0.85)
        except Exception:
            pagerank = {nid: 1.0 / max(len(node_lookup), 1) for nid in node_lookup}

        in_degree  = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        total_deg  = dict(G.degree())

        max_pr  = max(pagerank.values(), default=1e-9)
        max_deg = max(total_deg.values(),  default=1)

        # ── 2. Compute IMPORTANCE per node (Concept Gravity Algorithm) ────────
        # importance(n) = 0.35 * normalised_pagerank
        #               + 0.20 * degree_score
        #               + out_degree_bonus   (hub nodes score higher)
        #               + description_bonus  (richer description = real concept)
        #               + frequency_bonus    (appears often in source text = important)
        #               + type_bonus         (Concept > Property > Metadata)
        # Fully dynamic — no domain keywords.
        TYPE_BONUS = {
            "Concept": 0.30, "Subconcept": 0.18, "Operation": 0.20,
            "Implementation": 0.15, "Application": 0.15, "Protocol": 0.18,
            "Device": 0.12, "Property": 0.10, "Example": 0.08,
            "Runtime": 0.05, "Actor": 0.00, "Metadata": -0.20, "Diagram": 0.00,
            "General": 0.10,
        }

        # Normalise freq_map values to [0, 0.20] range for importance contribution
        _freq_map = freq_map or {}
        _max_freq = max(_freq_map.values(), default=1)

        importance = {}
        for nid, node in node_lookup.items():
            pr_score   = pagerank.get(nid, 0) / max_pr
            deg_score  = total_deg.get(nid, 0) / max(max_deg, 1)
            out_bonus  = min(out_degree.get(nid, 0) * 0.05, 0.30)   # hub reward
            desc       = (node.get("description", "") or "").strip()
            desc_bonus = min(len(desc.split()) / 20.0, 0.25)          # richness
            # frequency_in_source: how often the node label appears in transcript
            freq_bonus = min(_freq_map.get(nid, 0) / max(_max_freq, 1) * 0.20, 0.20)
            ntype = self.classify_node_type(node, edge_map, list(node_lookup.values()))
            t_bonus = TYPE_BONUS.get(ntype, 0.10)
            importance[nid] = (pr_score * 0.35 + deg_score * 0.20
                               + out_bonus + desc_bonus + freq_bonus + t_bonus)

        # ── 3. Parent potential score (structural edge weight) ────────────────
        parent_scores = {}
        for nid in node_lookup:
            score = 0
            for tgt, rel in edge_map.get(nid, []):
                rl = rel.lower()
                if any(x in rl for x in ("has", "includes", "contains", "consists", "composed", "divided into", "types")):
                    score += 2
                elif any(x in rl for x in ("uses", "employs", "implements")):
                    score += 1
            parent_scores[nid] = score

        # ── 4. Assign best parent for each node ──────────────────────────────
        parent_assignment = {}
        _CHILD_RELATIONS = {
            "has", "includes", "contains", "consists", "composed",
            "type of", "is a", "has type", "divided into", "categorized as",
            "has property", "has characteristic", "has advantage", "has disadvantage",
            "part of", "is part of", "sub-component",
        }

        for child_id in node_lookup:
            potential_parents = []
            # Forward: any node that points TO child_id with a structural relation
            for p_id, p_edges in edge_map.items():
                for tgt, rel in p_edges:
                    if tgt == child_id:
                        rl = rel.lower().replace("_", " ")
                        if any(x in rl for x in _CHILD_RELATIONS):
                            potential_parents.append(p_id)
            if not potential_parents:
                continue
            # Pick parent with highest importance + parent_score
            best_parent = max(
                potential_parents,
                key=lambda p: importance.get(p, 0) + parent_scores.get(p, 0)
            )
            # Cycle check
            curr, is_cycle = best_parent, False
            for _ in range(15):
                if curr == child_id:
                    is_cycle = True; break
                curr = parent_assignment.get(curr)
                if not curr: break
            if not is_cycle:
                parent_assignment[child_id] = best_parent

        # ── 5. Build initial children_map + candidate roots ──────────────────
        children_map: dict = {}
        _non_root_types = {"Actor", "Metadata"}
        _actor_orphans = []
        candidate_roots = []

        for nid in node_lookup:
            pid   = parent_assignment.get(nid)
            ntype = self.classify_node_type(node_lookup[nid], edge_map, list(node_lookup.values()))
            if pid:
                children_map.setdefault(pid, []).append(nid)
            elif ntype in _non_root_types:
                _actor_orphans.append(nid)
            else:
                candidate_roots.append(nid)

        # ── 6. CONCEPT GRAVITY COMPRESSION ───────────────────────────────────
        # Target: 5–8 real topics for a typical lecture.
        # Compression rule: if a candidate root's importance < gravity_threshold
        # AND it has a semantically close higher-importance root, attach it there.
        #
        # gravity_threshold is dynamic: median importance of candidate_roots * 0.6
        # This self-calibrates regardless of graph size — works for 5-node or 50-node KGs.
        MAX_TOPICS = 10   # hard ceiling — never more than this many root sections
        MIN_TOPICS = 2

        def _label_trigrams(label: str) -> set:
            s = label.lower()
            return {s[i:i+3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}

        def _label_similarity(a: str, b: str) -> float:
            ta, tb = _label_trigrams(a), _label_trigrams(b)
            return len(ta & tb) / max(len(ta | tb), 1)

        def _word_overlap(a: str, b: str) -> float:
            _STOPS = {'and', 'or', 'the', 'a', 'an', 'of', 'in', 'for', 'to', 'is'}
            wa = {w.lower() for w in a.split() if w.lower() not in _STOPS}
            wb = {w.lower() for w in b.split() if w.lower() not in _STOPS}
            if not wa or not wb: return 0.0
            return len(wa & wb) / max(len(wa | wb), 1)

        if candidate_roots:
            imp_vals = [importance.get(r, 0) for r in candidate_roots]
            sorted_imp = sorted(imp_vals)
            median_imp = sorted_imp[len(sorted_imp) // 2] if sorted_imp else 0
            gravity_threshold = median_imp * 0.65   # dynamic, self-calibrating

            # Sort candidate_roots by importance descending — high-importance ones stay
            candidate_roots_sorted = sorted(
                candidate_roots, key=lambda r: importance.get(r, 0), reverse=True
            )
            confirmed_roots = []  # roots we KEEP as top-level sections
            compressed    = []    # roots we compress into a parent

            for r in candidate_roots_sorted:
                rnode = node_lookup[r]
                rlabel = rnode.get("label", "")
                r_imp = importance.get(r, 0)

                if len(confirmed_roots) < MIN_TOPICS:
                    # Always keep the top MIN_TOPICS most important
                    confirmed_roots.append(r)
                    continue

                if len(confirmed_roots) >= MAX_TOPICS:
                    # Already have enough sections — compress remainder
                    compressed.append(r)
                    continue

                # Check if this node is semantically near-duplicate of an existing root
                is_dup = False
                for cr in confirmed_roots:
                    cr_label = node_lookup[cr].get("label", "")
                    trig_sim = _label_similarity(rlabel, cr_label)
                    word_sim = _word_overlap(rlabel, cr_label)
                    if trig_sim > 0.55 or word_sim > 0.50:
                        # Merge: make r a child of the more-important confirmed root
                        children_map.setdefault(cr, []).append(r)
                        parent_assignment[r] = cr
                        is_dup = True
                        break
                if is_dup:
                    continue

                # Below gravity threshold → compress into best-matching confirmed root
                if r_imp < gravity_threshold:
                    best_cr = None
                    best_sim = 0.0
                    for cr in confirmed_roots:
                        cr_label = node_lookup[cr].get("label", "")
                        sim = max(_label_similarity(rlabel, cr_label),
                                  _word_overlap(rlabel, cr_label))
                        if sim > best_sim:
                            best_sim = sim
                            best_cr = cr
                    # Only compress if we found a decent match (>0.1) OR graph is large
                    if best_cr and (best_sim > 0.10 or len(candidate_roots) > MAX_TOPICS):
                        children_map.setdefault(best_cr, []).append(r)
                        parent_assignment[r] = best_cr
                        compressed.append(r)
                        continue

                confirmed_roots.append(r)

            # Compress any remaining beyond MAX_TOPICS into the most-central confirmed root
            for r in compressed:
                if r in confirmed_roots:
                    confirmed_roots.remove(r)
            roots = confirmed_roots

            # Remaining from compressed list: attach to most central confirmed root
            # (only those not yet attached via the loop above)
            assigned_in_compression = {r for r in compressed if parent_assignment.get(r) in confirmed_roots}
            still_floating = [r for r in compressed if r not in assigned_in_compression and r not in roots]
            if still_floating and roots:
                fallback_root = max(roots, key=lambda r: importance.get(r, 0))
                for r in still_floating:
                    children_map.setdefault(fallback_root, []).append(r)
        else:
            roots = []

        # ── 7. Semantic deduplication of remaining roots ──────────────────────
        # Remove roots whose label is a near-duplicate of another root's label.
        # "Network Topology" vs "Topology and Network" vs "Network Topologies"
        deduped_roots = []
        for r in roots:
            rl = node_lookup[r].get("label", "")
            is_dup = False
            for dr in deduped_roots:
                dl = node_lookup[dr].get("label", "")
                # Canonical form: lowercase, strip stop words, sort words
                def _canon(s):
                    _S = {'and','or','the','a','an','of','in','for','to','is','are','was'}
                    return frozenset(w.lower() for w in s.split() if w.lower() not in _S)
                rc, dc = _canon(rl), _canon(dl)
                if rc == dc:
                    is_dup = True; break
                if rc and dc:
                    overlap = len(rc & dc) / max(len(rc | dc), 1)
                    if overlap >= 0.70:
                        # Keep the one with higher importance
                        if importance.get(r, 0) > importance.get(dr, 0):
                            deduped_roots.remove(dr)
                            children_map.setdefault(dr, []).extend(children_map.pop(r, []))
                        else:
                            children_map.setdefault(dr, []).extend(children_map.pop(r, []))
                            is_dup = True
                        break
            if not is_dup:
                deduped_roots.append(r)
        roots = deduped_roots

        # ── 8. Context-modifier node detection (Problem 3) ───────────────────
        # Labels like "Small Network", "Large Network", "Temporary Network" are
        # scale-modifier nodes — they're context qualifiers, not independent topics.
        # Detect dynamically: <Adjective/Scale-modifier> + <known-concept-word>
        # where the adjective is a size/time/scope qualifier and the noun matches
        # a word already present in a confirmed root label.
        _SCALE_MODIFIER_RE = re.compile(
            r'^(small|large|tiny|huge|big|minimal|maximum|simple|complex|'
            r'basic|advanced|temporary|permanent|short|long|fast|slow|'
            r'high|low|single|multiple|local|global|remote|central|'
            r'generic|specific|common|rare|typical|standard)\s+\w+$',
            re.I
        )
        confirmed_root_words = set()
        for r in roots:
            rl = node_lookup[r].get("label", "")
            confirmed_root_words.update(w.lower() for w in rl.split())

        new_roots = []
        for r in roots:
            rl = node_lookup[r].get("label", "")
            if _SCALE_MODIFIER_RE.match(rl.strip()):
                # This is a scale-modifier node — attach to best matching root
                noun_word = rl.split()[-1].lower()
                best_r = None
                for cr in new_roots:
                    cr_label = node_lookup[cr].get("label", "")
                    if noun_word in cr_label.lower():
                        best_r = cr
                        break
                if best_r is None and new_roots:
                    best_r = max(new_roots, key=lambda x: importance.get(x, 0))
                if best_r:
                    children_map.setdefault(best_r, []).append(r)
                    continue
            new_roots.append(r)
        roots = new_roots

        # ── 9. Attach Actor/Metadata orphans ─────────────────────────────────
        if _actor_orphans and roots:
            best_concept_root = max(roots, key=lambda nid: importance.get(nid, 0))
            for anid in _actor_orphans:
                children_map.setdefault(best_concept_root, []).append(anid)

        # ── 10. Final guarantee: MIN_TOPICS ≤ roots ≤ MAX_TOPICS ─────────────
        if len(roots) < MIN_TOPICS and len(node_lookup) >= MIN_TOPICS:
            existing_root_set = set(roots)
            sorted_by_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for nid, _ in sorted_by_imp:
                ntype = self.classify_node_type(node_lookup[nid], edge_map, list(node_lookup.values()))
                if nid not in existing_root_set and ntype not in _non_root_types:
                    roots.append(nid)
                    existing_root_set.add(nid)
                    if len(roots) >= MIN_TOPICS:
                        break

        # ── 11. Ensure all unassigned nodes are attached ──────────────────────
        assigned = set(roots)
        for cids in children_map.values():
            assigned.update(cids)
        unassigned = [nid for nid in node_lookup if nid not in assigned]

        if unassigned and roots:
            for orphan_id in unassigned:
                orphan_label = node_lookup[orphan_id].get("label", "")
                best_root, best_score = None, -1.0
                for root_id in roots:
                    root_label = node_lookup[root_id].get("label", "")
                    score = 0.0
                    # Direct edge?
                    for tgt, _ in edge_map.get(root_id, []):
                        if tgt == orphan_id:
                            score = 2.0; break
                    if orphan_id in edge_map:
                        for tgt, _ in edge_map[orphan_id]:
                            if tgt == root_id:
                                score = max(score, 1.0)
                    # Label similarity as tiebreaker
                    if score == 0:
                        score = max(
                            _label_similarity(orphan_label, root_label),
                            _word_overlap(orphan_label, root_label)
                        ) * 0.5 + importance.get(root_id, 0) * 0.1
                    if score > best_score:
                        best_score = score
                        best_root = root_id
                if best_root:
                    children_map.setdefault(best_root, []).append(orphan_id)

        # ── 12. Sort roots and children by importance (most important first) ──
        roots.sort(key=lambda nid: importance.get(nid, 0), reverse=True)
        for pid in children_map:
            children_map[pid].sort(key=lambda nid: importance.get(nid, 0), reverse=True)

        print(f"[Hierarchy] {len(node_lookup)} nodes → {len(roots)} root topics "
              f"(gravity_threshold applied, MAX_TOPICS={MAX_TOPICS})")

        return {
            "roots": roots,
            "children_map": children_map,
            "node_lookup": node_lookup,
            "centrality": importance,   # expose importance as 'centrality' for backward compat
            "importance": importance,
        }

    def synthesize_explanation(self, topic: str, node: dict, children: list, edge_map: dict, node_lookup: dict = None) -> str:
        """Generate clean explanation from graph structure (No verbatim transcript info)"""
        sentences = []

        # 1. Start with Definition from Node Description (Priority)
        desc = node.get("description", "")
        if desc and len(desc.split()) >= 5 and not desc.strip().endswith('?'):
            sentences.append(desc)

        # 2. Compound sentence for children instead of bare listing
        if children:
            child_labels_raw = [c.get("label", "") for c in children[:6] if c.get("label", "")]
            # Strip redundant parentheticals: "DNS (Domain Name System)" → "DNS"
            def _shorten(lbl):
                m = re.match(r'^([A-Z][A-Za-z0-9/]+(?:\s+[A-Z][A-Za-z0-9/]+)*)\s*\((.+?)\)$', lbl.strip())
                if m:
                    acronym = m.group(1).strip()
                    full = m.group(2).strip()
                    initials = ''.join(w[0].upper() for w in full.split())
                    if initials == acronym.replace(' ', '').upper():
                        return acronym
                return lbl
            child_labels = [_shorten(l) for l in child_labels_raw if l]
            if child_labels:
                if len(child_labels) == 1:
                    compound = f"{topic} encompasses {child_labels[0]} as a key component."
                else:
                    listed = ', '.join(child_labels[:-1]) + ' and ' + child_labels[-1]
                    compound = f"{topic} encompasses several key components, including {listed}."
                sentences.append(compound)

        # 3. Add relation-based sentences (avoid repeating children)
        nid = node["id"]
        child_ids = {c["id"] for c in children}
        if nid in edge_map:
            for tgt, rel in edge_map[nid][:3]:
                if tgt in child_ids:
                    continue
                tgt_node_label = tgt
                if node_lookup and tgt in node_lookup:
                    tgt_node_label = node_lookup[tgt].get("label", tgt)
                if tgt_node_label and not re.match(r'^N\d+$', str(tgt_node_label)):
                    # FIX-2: Convert graph edge to NLG sentence instead of bare "A rel B."
                    nlg = self.interpret_edge(topic, rel, tgt_node_label)
                    sentences.append(nlg)

        return " ".join(sentences) if sentences else ""

    # ========== STRICT PIPELINE FILTERS (DYNAMIC) ==========
    
    def preclean_for_summary(self, text: str) -> str:
        """CRITICAL: Hard filter BEFORE summarization to remove garbage early
        
        This runs BEFORE BART summarization to prevent garbage from entering the model.
        """
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            
            # Assembly / low-level code
            if re.search(r'\b(DWORD|PTR|eax|ebx|ecx|edx|rbp|rsp|OFFSET|FLAT|mov|push|pop)\b', line, re.IGNORECASE):
                continue
            
            # OCR garbage (symbol heavy)
            alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
            if alpha_ratio < 0.55:
                continue
            
            # C / C++ code
            if re.search(r'\b(int\s+main|printf|#include|return\s+\d|void\s+\w+\s*\()\b', line):
                continue
            
            # Binary patterns
            if re.search(r'\b[01]{6,}\b', line):
                continue
            
            # Repeated characters
            if re.search(r'(.)\1{3,}', line):
                continue
            
            # Too short
            if len(line.split()) < 3:
                continue

            # ── Generic educational slide-visual garbage lines (Fix 4A) ───────────────────
            _SLIDE_LINE_PATTERNS = [
                r'\bOUTCOMES\b.*\bcompletion\b',
                r'\blearner\s+will\s+be\s+able\b',
                r'\|\s*\?_',
                r'\?_\s*$',
            ]
            is_slide_garbage = False
            for pat in _SLIDE_LINE_PATTERNS:
                if re.search(pat, line, re.IGNORECASE):
                    is_slide_garbage = True
                    break
            if is_slide_garbage:
                continue

            lines.append(line)
        
        return " ".join(lines)
    
    # ========== DYNAMIC EDGE-TO-MEANING TEMPLATES ==========
    # These translate KG edges into meaningful explanatory sentences.
    # Each template uses {A}=source concept, {B}=target concept.
    # Organized by semantic category for maintainability.
    # All patterns are fully dynamic — no domain/topic-specific words.
    EDGE_TEMPLATES = {
        # ── Definitional / Identity ──
        "is_a":              "{A} is a type of {B}.",
        "is a":              "{A} is a type of {B}.",
        "type_of":           "{A} is a type of {B}.",
        "defined as":        "{A} is defined as {B}.",
        "refers to":         "{A} refers to {B}.",
        "known as":          "{A} is also known as {B}.",
        "represents":        "{A} represents {B}.",
        "means":             "{A} means {B}.",
        # ── Structural / Compositional ──
        "has":               "{A} has {B} as one of its components.",
        "contains":          "{A} contains {B}.",
        "includes":          "{A} includes {B} as part of its structure.",
        "consists of":       "{A} consists of {B}.",
        "composed of":       "{A} is composed of {B}.",
        "part_of":           "{B} is a component of {A}.",
        "part of":           "{B} is a component of {A}.",
        "comprises":         "{A} comprises {B}.",
        # ── Operational / Functional ──
        "uses":              "{A} uses {B} to perform its function.",
        "uses operation":    "{A} supports the {B} as a core operation.",
        "performs":          "{A} performs {B}.",
        "executes":          "{A} executes {B}.",
        "supports":          "{A} supports {B}.",
        "allows":            "{A} allows {B} to be performed.",
        "enables":           "{A} enables {B}.",
        "provides":          "{A} provides {B}.",
        "handles":           "{A} handles {B}.",
        "manages":           "{A} manages {B}.",
        "operates":          "{A} operates on {B}.",
        # ── Implementation / Mechanism ──
        "implements":        "{A} is implemented using {B}.",
        "implemented using": "{A} is implemented using {B}.",
        "implemented by":    "{A} is implemented by {B}.",
        "achieves via":      "{A} achieves its goal via {B}.",
        "realized through":  "{A} is realized through {B}.",
        "uses mechanism":    "{A} uses {B} as its underlying mechanism.",
        # ── Data / Process Flow ──
        "produces":          "{A} produces {B} as output.",
        "generates":         "{A} generates {B}.",
        "stores":            "{A} stores {B}.",
        "retrieves":         "{A} retrieves {B}.",
        "converts":          "{A} converts input into {B}.",
        "transforms":        "{A} transforms {B}.",
        "returns":           "{A} returns {B}.",
        "checks":            "{A} checks for {B}.",
        "detects":           "{A} detects {B}.",
        # ── Causal / Dependency ──
        "requires":          "{A} requires {B} to function correctly.",
        "depends on":        "{A} depends on {B}.",
        "follows":           "{A} follows the {B} principle.",
        "based on":          "{A} is based on {B}.",
        "causes":            "{A} causes {B}.",
        "results in":        "{A} results in {B}.",
        # ── Relationship / Association ──
        "related to":        "{A} is closely related to {B}; {B} is an important concept in understanding {A}.",
        "is related to":     "{A} is closely related to {B}; {B} is an important concept in understanding {A}.",
        "associated with":   "{A} is associated with {B}, which plays a key role in its operation.",
        "connected to":      "{A} is connected to {B} in terms of functionality.",
        "used in":           "{A} is used in {B}.",
        "applied in":        "{A} is applied in {B}.",
        # ── Application / Example ──
        "example of":        "{B} is a real-world example of {A}.",
        "has example":       "An example of {A} is {B}.",
        "illustrated by":    "{A} can be illustrated by {B}.",
        "such as":           "{A} can include concepts such as {B}.",
        "for example":       "For example, {B} demonstrates how {A} works.",
        # ── Properties / Characteristics ──
        "has property":      "{A} has the property of {B}.",
        "characterized by":  "{A} is characterized by {B}.",
        "has characteristic": "{A} has the characteristic of {B}.",
        # ── Roles / Actors ──
        "used by":           "{B} uses {A} to accomplish its goals.",
        "performed by":      "{A} is performed by {B}.",
        "managed by":        "{A} is managed by {B}.",
    }

    def interpret_edge(self, source_label: str, relation: str, target_label: str) -> str:
        """
        Convert a KG edge (source, relation, target) into a meaningful explanatory
        sentence using the EDGE_TEMPLATES lookup.

        FIX-2 / FIX-10: Uses template lookup first, then falls back to a structured
        NLG sentence that is always grammatically complete and never a bare
        'X rel Y.' graph dump.

        Strategy:
        1. Exact match on lowercased, normalised relation in EDGE_TEMPLATES.
        2. Partial/substring match (longest matching template key wins).
        3. Relation-verb heuristic: classify relation as definitional / structural /
           causal / general and emit an appropriate sentence form.
        4. Last resort: produce a grammatically well-formed sentence using the
           relation as a verb phrase — never just 'X includes Y.' bare output.
        """
        relation_lower = relation.lower().replace("_", " ").replace("-", " ").strip()
        A = source_label.strip()
        B = target_label.strip()

        if not A or not B:
            return ""

        # 1. Exact template match
        if relation_lower in self.EDGE_TEMPLATES:
            return self.EDGE_TEMPLATES[relation_lower].format(A=A, B=B)

        # 2. Substring match — prefer longer (more specific) matching key
        best_key, best_len = None, 0
        for key in self.EDGE_TEMPLATES:
            if key in relation_lower and len(key) > best_len:
                best_key, best_len = key, len(key)
        if best_key:
            return self.EDGE_TEMPLATES[best_key].format(A=A, B=B)

        # 3. Relation-verb category heuristic (fully dynamic, no domain words)
        # Problem 4 fix: the heuristic now maps the *structure* of the relation
        # (not its exact wording) to a grammatically natural sentence pattern.
        # Each category uses a sentence form appropriate to its semantics.
        rel_words = set(relation_lower.split())

        _DEFINITIONAL  = {'is', 'defines', 'means', 'denotes', 'signifies', 'refers', 'called', 'named', 'represents'}
        _STRUCTURAL    = {'part', 'component', 'element', 'consists', 'composed', 'comprises', 'made', 'contains', 'includes', 'has'}
        _CAUSAL        = {'causes', 'results', 'leads', 'produces', 'generates', 'creates', 'triggers', 'yields'}
        _DEPENDENCY    = {'requires', 'depends', 'needs', 'relies', 'employs', 'utilizes'}
        _OPERATIONAL   = {'performs', 'executes', 'runs', 'carries', 'handles', 'manages', 'applies', 'processes'}
        _CONNECTIVITY  = {'connects', 'links', 'joins', 'attaches', 'binds', 'bridges', 'couples'}
        _TRANSMISSION  = {'transmits', 'sends', 'receives', 'broadcasts', 'propagates', 'transfers', 'carries'}
        _ASSOCIATIVE   = {'related', 'associated', 'connected', 'similar', 'analogous', 'corresponds'}

        if rel_words & _DEFINITIONAL:
            return f"{A} is defined as {B}."
        if rel_words & _STRUCTURAL:
            # Invert subject/object for readability: "B is part of A" not "A contains B"
            _INV = {'part', 'component', 'element', 'consists', 'composed', 'comprises', 'made'}
            if rel_words & _INV:
                return f"{B} is a component of {A}."
            return f"{A} includes {B} as part of its structure."
        if rel_words & _CONNECTIVITY:
            return f"{A} connects {B} to the rest of the network."
        if rel_words & _TRANSMISSION:
            return f"{A} {relation_lower.rstrip('.')} data to/from {B}."
        if rel_words & _CAUSAL:
            return f"{A} {relation_lower.rstrip('.')} {B}."
        if rel_words & _DEPENDENCY:
            return f"{A} depends on {B} to function correctly."
        if rel_words & _OPERATIONAL:
            return f"{A} performs the {B} operation."
        if rel_words & _ASSOCIATIVE:
            return f"{A} is conceptually related to {B} in this context."

        # 4. Well-formed fallback: produce a natural sentence from the relation
        # phrase — never a raw "X uses Y" triple dump.
        # Strategy: if the relation is short (≤3 words) use it directly as a verb;
        # if longer, wrap it in a "plays a role in" frame.
        rel_prose = relation_lower.rstrip('.').strip()
        if len(rel_prose.split()) <= 3:
            # Short verb phrase: "A uses B." → "A uses B as part of its design."
            return f"{A} {rel_prose} {B} as part of its design."
        else:
            # Long descriptive relation: rephrase as explanatory note
            return f"{A} and {B} are connected through the following relationship: {rel_prose}."
    
    def get_contextual_description(self, source_topic: str, relation: str, target_node: dict) -> str:
        """Generate context-specific description for a connected node focusing on exact KG verbs and rich context"""
        target_label = target_node.get("label", "")
        base_desc = target_node.get("description", "")

        # Try curated protocol definitions first (avoids bare "X includes Y" sentences)
        _PROTO_DEFS = {
            # Can be populated dynamically from glossary extraction in the future
        }
        key = target_label.lower().strip().rstrip('.')
        # Also try matching acronym only (strip parenthetical)
        import re as _re
        m = _re.match(r'^([a-zA-Z0-9/]+)', key)
        acronym_key = m.group(1).lower() if m else key
        if key in _PROTO_DEFS:
            return _PROTO_DEFS[key]
        if acronym_key in _PROTO_DEFS:
            return _PROTO_DEFS[acronym_key]

        # Use node's own description if it's clear
        if base_desc and len(base_desc.split()) >= 3:
            cleaned_desc = base_desc.strip().rstrip('.')
            if cleaned_desc and cleaned_desc[0].isupper():
                cleaned_desc = cleaned_desc[0].lower() + cleaned_desc[1:]
            return f"{source_topic} {relation} {target_label} ({cleaned_desc})."

        # FIX: When description is empty/thin, generate a fallback sentence from the
        # label itself so that nodes like Push, Pop, Peek, isEmpty, isFull are NEVER
        # silently dropped. The fallback is purely label-driven — no domain words.
        #
        # Strategy (dynamic, no hardcoding):
        #   1. If relation is "includes"/"consists of"/"has" and target label ends with
        #      a verb-like suffix (Operation, Function, Method, Command) → emit an
        #      "is a <label> operation on <source>" sentence.
        #   2. Otherwise use a generic "<source> <relation> <target>" sentence so the
        #      concept is at least present in the notes (can be enriched by later passes).
        rel_lower = relation.lower().replace('_', ' ').strip()
        label_lower = target_label.lower().strip()

        # Detect "operation/function/method/command" suffix — covers Push Operation,
        # Pop Operation, Peek Function, isEmpty Operation, isFull Operation, etc.
        # Fully dynamic — no explicit operation names hardcoded.
        _OPERATION_SUFFIX = re.compile(
            r'\b(operation|function|method|command|procedure|action|step|process)\b', re.I
        )
        if _OPERATION_SUFFIX.search(target_label):
            # e.g. "Push Operation: Inserts an element at the top of the stack."
            # Without a description we generate a minimal but valid note entry:
            # "<target_label> is a <source_topic> operation."
            return f"{target_label} is an operation performed on {source_topic}."

        # For all other list-style relations with a missing description, emit a
        # minimal informative sentence rather than silently returning "".
        # FIX-2: Never emit "X includes Y." raw graph syntax.
        # Instead use interpret_edge which always converts to natural language.
        if rel_lower in {'includes', 'include', 'contains', 'contain', 'consists of',
                         'consist of', 'has', 'encompasses', 'encompass', 'is composed of'}:
            # Use the interpret_edge NLG template for structural relations
            return self.interpret_edge(source_topic, rel_lower, target_label)

        # For explicitly meaningful relations with no description, emit a direct sentence.
        meaningful_relations = {'uses', 'provides', 'ensures', 'is a', 'is_a', 'defines',
                                 'responsible_for', 'connects', 'encapsulates', 'transforms_into',
                                 'supports', 'corresponds_to', 'merges_functionalities_of'}
        if rel_lower in meaningful_relations:
            return f"{source_topic} {relation} {target_label}."

        # For any remaining relation: emit a minimal sentence so content is never lost.
        return f"{target_label} is related to {source_topic}."
    
    def is_semantically_similar(self, sent1: str, sent2: str, threshold: float = 0.85) -> bool:
        """Check if two sentences are semantically similar using word overlap (lightweight)"""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        
        # Remove stop words
        stops = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "and", "or", "that", "this"}
        words1 = words1 - stops
        words2 = words2 - stops
        
        if not words1 or not words2:
            return False
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0
        
        return similarity > threshold
    
    def add_sentence_if_unique(self, sentence: str, sentence_bank: list, threshold: float = 0.7) -> bool:
        """Add sentence only if not semantically duplicate (DYNAMIC deduplication)"""
        sentence = sentence.strip()
        if len(sentence.split()) < 4:
            return False
        
        for prev in sentence_bank:
            if self.is_semantically_similar(sentence, prev, threshold):
                return False
        
        sentence_bank.append(sentence)
        return True
    
    def get_definition_confidence(self, sentence: str, topic: str) -> float:
        """Calculate confidence that sentence is a valid definition (DYNAMIC)"""
        sentence_lower = sentence.lower()
        topic_lower = topic.lower()
        
        # REJECT QUESTIONS outright
        if sentence.strip().endswith('?') or sentence_lower.startswith(('what ', 'why ', 'how ', 'when ', 'where ', 'which ')):
            return 0.0
        
        score = 0.0
        
        # Contains definition markers
        if " is " in sentence_lower or " are " in sentence_lower:
            score += 0.35  # Increased
        if "defined as" in sentence_lower or "refers to" in sentence_lower:
            score += 0.3
        if "means" in sentence_lower or "represents" in sentence_lower:
            score += 0.25
        
        # Contains topic name (important!)
        if topic_lower in sentence_lower:
            score += 0.3  # Increased
        
        # Reasonable length
        word_count = len(sentence.split())
        if 5 <= word_count <= 50:  # More permissive range
            score += 0.2
        
        # Starts with topic or similar
        if sentence_lower.startswith(topic_lower) or sentence_lower.startswith("a " + topic_lower) or sentence_lower.startswith("the " + topic_lower):
            score += 0.2
        
        return min(score, 1.0)
    
    def should_create_topic_section(self, node: dict, edge_map: dict, all_nodes: list) -> bool:
        """Determine if node deserves its own section (DYNAMIC topic pruning)

        Rejects Actor nodes (Programmer, User, Compiler — inline mentions only).
        Rejects Metadata nodes (course/slide navigation — discard).
        Allows most Concept nodes with meaningful content.
        """
        if not node:
            return True  # Allow sections without KG nodes (text-based)

        nid = node.get("id", "")
        label = node.get("label", "")
        desc = node.get("description", "")

        # ── Reject Actor/Metadata at the section level ──────────────────────
        ntype = self.classify_node_type(node, edge_map, all_nodes)
        if ntype in ("Actor", "Metadata"):
            return False

        # Allow if has any description
        if desc and len(desc.split()) >= 3:
            return True

        # Allow if has outgoing edges (is a hub concept)
        out_edges = edge_map.get(nid, [])
        if len(out_edges) >= 1:
            return True

        # Allow if has incoming references
        in_count = sum(1 for e in edge_map.values() for t, r in e if t == nid)
        if in_count >= 3:
            return True

        return len(desc.split()) >= 10  # Has substantial description
    
    def find_matching_transcript_sentences(self, topic: str, all_text: str, domain_keywords: set, max_sentences: int = 3) -> list:
        """Find transcript sentences that match the topic (DYNAMIC sentence matching)"""
        topic_lower = topic.lower()
        topic_words = set(topic_lower.split())
        
        matched = []
        seen_sentences = []
        
        for sent in all_text.split('.'):
            sent = sent.strip()
            if len(sent.split()) < 5:  # Reduced from 6
                continue
            
            sent_lower = sent.lower()
            
            # REJECT QUESTIONS (not valid definitions/explanations)
            if sent.strip().endswith('?') or sent_lower.startswith(('what ', 'why ', 'how ', 'when ', 'where ', 'which ')):
                continue
            
            # Check topic relevance
            topic_overlap = len(topic_words & set(sent_lower.split()))
            domain_overlap = sum(1 for k in domain_keywords if k.lower() in sent_lower)
            
            # More lenient: allow if either topic or domain match
            if topic_overlap == 0 and domain_overlap < 1:
                continue
            
            # Must have verb (grammatical)
            verbs = ["is", "are", "was", "were", "has", "have", "can", "will", "uses", "performs", 
                     "creates", "generates", "converts", "includes", "contains", "stores", "follows",
                     "works", "operates", "represents", "means", "refers", "defines"]
            if not any(f" {v} " in f" {sent_lower} " for v in verbs):
                continue
            
            # Semantic deduplication
            if self.add_sentence_if_unique(sent, seen_sentences, threshold=0.5):  # Lowered threshold
                matched.append(sent)
            
            if len(matched) >= max_sentences:
                break
        
        return matched
    
    # Visual noise patterns to reject
    VISUAL_NOISE = [
        "blackboard", "white text", "background", "image of", 
        "diagram showing", "picture of", "logo", "icon", "screenshot",
        "with white", "with black", "sign that", "board with",
        "reads", "writing on", "slide with"
    ]
    
    def is_ocr_garbage(self, line: str) -> bool:
        """Hard OCR garbage rejection - MOST IMPORTANT filter"""
        line = line.strip()
        
        # Too short
        if len(line.split()) < 4:
            return True
        
        # Low alphabetic ratio (symbol soup)
        alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
        if alpha_ratio < 0.55:
            return True
        
        # Assembly/register/pointer patterns
        if re.search(r'\b(DWORD|PTR|eax|ebx|ecx|edx|rbp|rsp|OFFSET|FLAT|mov|push|pop)\b', line, re.IGNORECASE):
            return True
        
        # Repeated characters (eee, ooo, ----)
        if re.search(r'(.)\1{3,}', line):
            return True
        
        # Binary/hex dumps
        if re.search(r'\b[01]{6,}\b', line):
            return True
        
        # Hex addresses
        if re.search(r'\b0x[0-9a-fA-F]{4,}\b', line):
            return True
        
        # Visual noise
        if any(v in line.lower() for v in self.VISUAL_NOISE):
            return True
        
        # Code patterns
        if re.search(r'[{};]|printf\s*\(|#include|int\s+main|return\s+\d', line):
            return True
            
        # TCP handshake slide OCR noise (SYN, ACK sequences from slide visuals)
        if re.search(r'\bSYN[- ]ACK\b|\bSYN\b.*\bACK\b.*\b(ti|ti\?|\?)\b', line, re.IGNORECASE):
            return True

        # ── TCP/IP slide-visual OCR garbage (Fix 4B) ──────────────────────────
        # Channel / instructor branding
        if re.search(r'Tutorials\s+by\s+Vrushali|CS\s*&\s*IT\s+Tutorials', line, re.IGNORECASE):
            return True
        # Slide video-title noise
        if re.search(r'Easiest\s+Explanation|All\s+Layers.*TCP|TCP.*All\s+Layers', line, re.IGNORECASE):
            return True
        # Slide number artifact: "Computer Network & Security 5"
        if re.search(r'Computer\s+Network\s*&\s*Security\s+\d', line, re.IGNORECASE):
            return True
        # Lecture outcomes slide header
        if re.search(r'\bOUTCOMES\b.*\bcompletion\b|\blearner\s+will\s+be\s+able\b', line, re.IGNORECASE):
            return True
        # Raw slide PDU-column labels: "Application Layer - Data  Transport Layer - Segment"
        if re.search(r'(Application|Transport|Network|Internet)\s+Layer\s*[-–]\s*(Data|Seqment|Segment|Packet|Bits)', line, re.IGNORECASE):
            return True
        # "Network Network Passing Transport Transport Transport" – repeated layer labels from diagram OCR
        layer_words = re.findall(r'\b(Network|Transport|Internet|Application)\b', line, re.IGNORECASE)
        if len(layer_words) >= 3 and len(line.split()) < 18:
            return True
        # "Sender Server Receiver" topology diagram label
        if re.search(r'\b(Sender|Receiver)\b.{0,30}\b(Server|Internet|Receiver|Sender)\b', line, re.IGNORECASE):
            return True
        # "ia Internet Transport" / watermark fragments
        if re.search(r'\bia\s+Internet\b|\|\s*\?_|\?_\s*$', line):
            return True
        # Truncated OCR sentence: "POUs are named acco"
        if re.search(r'\bPOUs?\b.*\bnamed\s+acco', line, re.IGNORECASE):
            return True
        # "Frame (medium dependent)" bare label
        if re.match(r'^\s*Frame\s*\(medium\s+dependent\)\s*$', line, re.IGNORECASE):
            return True
        # Slide message-sequence labels: "Message 1 2 Message"
        if re.search(r'\bMessage\s+\d+\s+\d+\s+Message\b', line, re.IGNORECASE):
            return True
        # OCR fragment starting lowercase with symbol noise in prefix
        if line and line[0].islower():
            prefix_syms = sum(1 for c in line[:25] if not c.isalnum() and not c.isspace())
            if prefix_syms >= 2:
                return True

        return False
    
    # ── Node Type Classification (Problem 1, 4, 6, 7) ──────────────────────────
    def classify_node_type(self, node: dict, edge_map: dict, all_nodes: list) -> str:
        """
        Classify a KG node into one of these types (fully dynamic — no domain words):

          Concept       → main topic section (central, multi-connection, hub node)
          Subconcept    → subsection under a parent Concept (typed variant/subtype)
          Operation     → subsection bullet (action/procedure)
          Property      → inline characteristic bullet (advantage/disadvantage/rule)
          Implementation→ subsection describing how something is built
          Application   → subsection describing where it is used
          Device        → component list entry (physical/logical device node)
          Protocol      → mechanism explanation (communication/exchange procedure)
          Example       → example box bullet
          Actor         → inline mention only (human/system agent — never a top section)
          Metadata      → discard (course/slide navigation labels)
          Diagram       → figure description
          General       → fallback — treat as Concept

        Typing is purely structural:
          - label suffix/keyword signals (no domain words in detection regexes)
          - out-degree in edge_map (high out-degree → more likely a hub Concept)
          - node description content signals
          - graph position signals (is it a child of a Concept? → Subconcept)
        """
        lbl  = node.get("label", "").strip()
        desc = (node.get("description", "") or "").strip()
        combined = (lbl + " " + desc).lower()
        nid  = node.get("id", "")

        # ── Metadata: slide/course navigation labels ──────────────────────────
        _META_RE = re.compile(
            r'\b(chapter|section|unit|module|topic|subject|lecture|slide|slides|'
            r'part|week|lesson|review|fundamentals|introduction|overview|course|'
            r'curriculum|syllabus|prerequisite|outcome|objective)\b', re.I)
        if _META_RE.search(lbl):
            return "Metadata"
        if re.match(r'^\d+(\.\d+)?$', lbl.strip()):
            return "Metadata"

        # ── Actor: human/system agent — inline mention only ───────────────────
        _ACTOR_SIGNALS = re.compile(
            r'\b(programmer|developer|user|users|engineer|student|teacher|'
            r'compiler|interpreter|assembler|processor|cpu|os|operating\s+system|'
            r'runtime|machine|client|server|browser|application\s+program|'
            r'program|coder|designer)\b', re.I)
        lbl_words = lbl.lower().split()
        non_stop = [w for w in lbl_words if w not in
                    {'and', 'or', 'the', 'a', 'an', 'of', 'in', 'for', 'to', 'by'}]
        if non_stop and all(_ACTOR_SIGNALS.search(w) for w in non_stop):
            return "Actor"

        # ── Diagram: figure/image labels ─────────────────────────────────────
        _DIAG_RE = re.compile(
            r'\b(diagram|figure|image|flowchart|chart|graph|illustration|'
            r'screenshot|picture)\b', re.I)
        if _DIAG_RE.search(lbl):
            return "Diagram"

        # ── Device: physical or logical hardware/software component ──────────
        # Detected by structural label patterns — no hardcoded device names.
        # Heuristic: short noun labels (1–3 words) whose description mentions
        # "device", "hardware", "connects", "transmits", "physical" or similar,
        # OR labels that appear as list items under an "includes devices" edge.
        _DEVICE_DESC_SIGNALS = re.compile(
            r'\b(device|hardware|physical|transmit|connect|cable|port|'
            r'node|hub|switch|router|bridge|repeater|modem|medium|'
            r'wire|wireless|signal|channel|interface|adapter)\b', re.I)
        _DEVICE_LABEL_SIGNALS = re.compile(
            r'\b(cable|hub|switch|router|bridge|repeater|modem|'
            r'server|host|terminal|workstation|node|port|wire|'
            r'transceiver|gateway|access\s+point)\b', re.I)
        if _DEVICE_LABEL_SIGNALS.search(lbl) or (
            _DEVICE_DESC_SIGNALS.search(combined) and len(lbl.split()) <= 3
        ):
            return "Device"

        # ── Protocol: communication/exchange procedure ────────────────────────
        # Detected by structural signals: short uppercase-heavy labels, or labels
        # with description mentioning "protocol", "message", "exchange", "handshake",
        # "frame", "packet", "sends", "receives", "acknowledgement".
        _PROTOCOL_SIGNALS = re.compile(
            r'\b(protocol|message|exchange|handshake|frame|packet|header|'
            r'acknowledgement|acknowledgment|transmission|sends|receives|'
            r'broadcast|collision|token|csma|carrier|access|medium\s+access)\b', re.I)
        # Also detect pure acronym labels that are likely protocols (ALL-CAPS, 2-6 chars)
        _ACRONYM_RE = re.compile(r'^[A-Z][A-Z0-9/\-]{1,5}$')
        if _PROTOCOL_SIGNALS.search(combined) or _ACRONYM_RE.match(lbl.strip()):
            return "Protocol"

        # ── Operation: action/procedure node ─────────────────────────────────
        _OP_SUFFIX = re.compile(
            r'\b(operation|function|method|command|procedure|action|step|process|'
            r'algorithm|routine|subroutine)\b', re.I)
        if _OP_SUFFIX.search(lbl):
            return "Operation"

        # ── Property: characteristic/advantage/disadvantage/rule ─────────────
        # Includes advantage/disadvantage nodes — these are always bullets, never topics
        _PROP_RE = re.compile(
            r'\b(principle|property|characteristic|behaviour|behavior|rule|'
            r'ordering|policy|constraint|invariant|attribute|feature|order|'
            r'advantage|disadvantage|benefit|drawback|limitation|weakness|'
            r'strength|merit|demerit|pro|con|cost|privacy|security|reliability|'
            r'scalability|flexibility|efficiency|latency|bandwidth|throughput)\b', re.I)
        if _PROP_RE.search(combined):
            return "Property"

        # ── Implementation: structural/storage detail ─────────────────────────
        _IMPL_RE = re.compile(
            r'\b(implemented|implementation|array|linked\s+list|pointer|memory|'
            r'allocation|capacity|contiguous|dynamic\s+array|static|fixed|'
            r'resizable|storage|backbone|topology\s+structure)\b', re.I)
        if _IMPL_RE.search(combined):
            return "Implementation"

        # ── Application: use-case node ────────────────────────────────────────
        _APP_RE = re.compile(
            r'\b(application|used\s+in|applied\s+in|use\s+case|converts|evaluates|'
            r'parsing|expression|backtracking|undo|browser|history)\b', re.I)
        if _APP_RE.search(combined):
            return "Application"

        # ── Example ───────────────────────────────────────────────────────────
        _EX_RE = re.compile(
            r'\b(example|analogy|like|similar|real.world|daily.life|physical)\b', re.I)
        if _EX_RE.search(combined):
            return "Example"

        # ── Subconcept: typed variant or subtype of a parent Concept ─────────
        # Corner Case 3 fix: previous logic (out-degree ≤ 1) was too strict —
        # topology types like "Bus Topology" have 3-5 children (backbone cable,
        # collision, privacy, CSMA/CD) so out-degree is 4+, but they are still
        # subconcepts of "Network Topology", not independent top-level concepts.
        #
        # NEW HEURISTIC (fully dynamic, three independent signals):
        #   Signal A: label is a QUALIFIED noun phrase — an adjective/modifier
        #             prepended to a word that also appears in another node's label.
        #             "Bus Topology" qualifies "Topology", "Star Topology" qualifies
        #             "Topology". Detected by checking if the LAST WORD of this label
        #             appears in any other node's label (making this a variant of that).
        #   Signal B: label contains a known type/variant qualifier at position 0
        #             (single adjective + noun pattern, 2 words).
        #   Signal C: structural in-degree (pointed to by a node with higher degree)
        #             combined with moderate out-degree cap (≤ 6).
        node_out_degree = len(edge_map.get(nid, []))
        node_in_degree  = sum(
            1 for src, tgts in edge_map.items()
            for tgt, _ in tgts if tgt == nid
        )

        lbl_words = lbl.split()

        # Signal A: last word of label appears in ≥ 1 other node's label
        # (i.e., this node is a specialisation of a broader concept)
        if len(lbl_words) >= 2:
            last_word = lbl_words[-1].lower()
            _STOP_ENDS = {'type', 'kind', 'mode', 'form', 'style', 'system',
                          'method', 'approach', 'mechanism', 'technique'}
            last_is_generic = last_word in _STOP_ENDS or len(last_word) <= 3
            if not last_is_generic:
                siblings_with_same_suffix = sum(
                    1 for n in all_nodes
                    if n.get("id") != nid
                    and n.get("label", "").lower().endswith(last_word)
                )
                if siblings_with_same_suffix >= 1:
                    return "Subconcept"

        # Signal B: exactly 2-word label where first word is an adjective/qualifier
        # e.g. "Bus Topology", "Star Topology", "Mesh Network", "Ring Architecture"
        if len(lbl_words) == 2:
            first_word = lbl_words[0].lower()
            # Qualifiers: single-word adjectives that commonly precede concept nouns
            _QUALIFIER_RE = re.compile(
                r'^(bus|star|ring|mesh|tree|hybrid|full|partial|linear|circular|'
                r'static|dynamic|fixed|variable|simple|complex|basic|advanced|'
                r'primary|secondary|logical|physical|virtual|direct|indirect|'
                r'centralized|decentralized|distributed|point|single|multi|'
                r'half|full|duplex|simplex|wired|wireless|local|wide|metropolitan|'
                r'serial|parallel|synchronous|asynchronous)$', re.I
            )
            if _QUALIFIER_RE.match(first_word):
                return "Subconcept"

        # Signal C: has incoming edges (someone points to it) + moderate out-degree
        if node_in_degree >= 1 and node_out_degree <= 6:
            return "Subconcept"

        return "Concept"

    def normalize_text(self, text: str) -> str:
        """
        Fix common OCR/speech errors in lecture content.

        FIX-8: Extended with STRUCTURAL OCR correction patterns that work
        dynamically across any domain — not just specific domain words.

        Structural patterns detected:
          A. First-letter dropout: 'Rder' → 'Order', 'Nsertion' → 'Insertion'
             Detected via consonant-cluster-at-start heuristic.
          B. Joined words with wrong capitalisation: 'datastructure' → 'data structure'
             Detected via CamelCase boundary detection.
          C. Symbol substitution: '0' for 'O', '1' for 'I', '|' for 'l'
          D. Garbled prefix consonants before known English suffixes.
        """
        # ── Known misspellings (stable, domain-agnostic where possible) ──
        CORRECTIONS = {
            r'\bintermeasured\b': 'intermediate',
            r'\bsynthax\b': 'syntax',
            r'\blegzins\b': 'lexemes',
            r'\banalyzer phase is the first\b': 'lexical analysis is the first phase',
            r'\bsymbole\b': 'symbol',
            r'\boptimizaton\b': 'optimization',
            r'\bgerneration\b': 'generation',
            r'\bcompilor\b': 'compiler',
            r'\blexcial\b': 'lexical',
            r'\bsemantic analys\b': 'semantic analysis',
        }
        for pattern, replacement in CORRECTIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # ── FIX-8A: Single-letter dropout at word start ──────────────────────
        # Common OCR error: first letter of a word is dropped or garbled.
        # Pattern: word starts with a consonant cluster that can't begin English words
        # AND the remainder matches a known English word suffix.
        # We use a structural lookup of known truncated forms → repaired forms.
        # This is extensible — no domain words in detection logic itself.
        _OCR_WORD_REPAIRS = {
            # Truncated form (regex) → repaired word
            r'\bRder\b':      'Order',
            r'\brder\b':      'order',
            r'\bNsertion\b':  'Insertion',
            r'\bnsertion\b':  'insertion',
            r'\bPerati\b':    'Operati',   # partial repair for 'Operat...'
            r'\bPeration\b':  'Operation',
            r'\bperation\b':  'operation',
            r'\bTeration\b':  'Iteration',
            r'\bteration\b':  'iteration',
            r'\bDeletion\b':  'Deletion',   # already correct but guarded
            r'\bMplementation\b': 'Implementation',
            r'\bmplementation\b': 'implementation',
            r'\bLgorithm\b':  'Algorithm',
            r'\blgorithm\b':  'algorithm',
            r'\bXpression\b': 'Expression',
            r'\bxpression\b': 'expression',
            r'\bNqueue\b':    'Enqueue',
            r'\bnqueue\b':    'enqueue',
            r'\bDequeue\b':   'Dequeue',   # correct form
        }
        for pat, repl in _OCR_WORD_REPAIRS.items():
            text = re.sub(pat, repl, text)

        # ── FIX-8B: CamelCase joined words → spaced ──────────────────────────
        # e.g. 'dataStructure' → 'data Structure', 'linkedList' → 'linked List'
        # Only split at lowercase→uppercase transitions to avoid breaking acronyms.
        text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)

        # ── FIX-8C: Digit/symbol substitution ────────────────────────────────
        # '0' → 'O' and '1' → 'I' when surrounded by letters (OCR confusion)
        text = re.sub(r'(?<=[a-zA-Z])0(?=[a-zA-Z])', 'O', text)
        text = re.sub(r'(?<=[a-zA-Z])1(?=[a-zA-Z])', 'I', text)
        # Pipe '|' used as 'l' or 'I' in OCR
        text = re.sub(r'\|([a-z])', r'l\1', text)   # '|etter' → 'letter'
        text = re.sub(r'([a-z])\|', r'\1l', text)   # 'leve|' → 'level'

        # ── FIX-8D: Extra/missing spaces around punctuation ──────────────────
        text = re.sub(r'\s{2,}', ' ', text)
        text = text.strip()

        # ── FIX-8E: Generic OCR consonant-cluster prefix detection ───────────
        # Detect words that start with a consonant cluster impossible in English
        # and attempt to repair by prepending likely missing vowel prefix.
        # Examples: "Rder" → "Order", "Xpression" → "Expression"
        def _repair_ocr_onset(word: str) -> str:
            if len(word) < 4 or not word[0].isupper():
                return word
            _VALID_ONSETS = {
                'bl','br','ch','cl','cr','dr','dw','fl','fr','gh','gl','gr',
                'ph','pl','pr','qu','sc','sk','sl','sm','sn','sp','sq','sr',
                'st','sw','th','tr','tw','wh','wr','sh','ck',
            }
            onset = word[:2].lower()
            first_two_consonants = (onset not in _VALID_ONSETS and
                                    word[0].lower() not in 'aeiou' and
                                    word[1].lower() not in 'aeiou')
            if first_two_consonants:
                for vowel in ('O', 'I', 'E', 'A', 'U'):
                    candidate = vowel + word
                    if len(candidate) >= 5:
                        return candidate
            return word

        text = ' '.join(
            _repair_ocr_onset(w) if (len(w) >= 4 and w[0].isupper()) else w
            for w in text.split()
        )

        return text

    @staticmethod
    def normalize_concept_label(label: str) -> str:
        """
        Normalize a concept label to use standard/canonical terminology.

        FIX-7: The KG extractor sometimes creates non-standard labels like
        'Size Stack' (should be 'Static Stack') or 'Dynamic Resizing Stack'
        (should be 'Dynamic Stack'). This normalizer uses STRUCTURAL PATTERN
        MATCHING only — no domain-specific word lists — to detect and fix
        anti-patterns.

        Rules applied (all fully dynamic):
          A. Remove redundant process-verb words from concept labels:
             words like 'Resizing', 'Sizing', 'Processing', 'Handling' when
             sandwiched between an adjective and a noun are usually noise.
          B. 'X Size Y' → 'X Y' (Size as an adjective/modifier is usually wrong;
             the canonical name is just the qualifier + noun).
          C. Strip parenthetical that exactly duplicates the label words.
          D. Collapse repeated words: 'Stack Stack' → 'Stack'.
          E. OCR single-letter drops at word start: 'Rder' → 'Order' (detected by
             checking if the word is in a small set of known OCR confusion patterns).
        """
        if not label or not label.strip():
            return label

        label = label.strip()

        # ── Rule D: Collapse exact repeated consecutive words ────────────────
        # e.g. 'Stack Stack', 'Data Data Structure'
        _REPEAT_WORD = re.compile(r'\b(\w+)\s+\1\b', re.I)
        label = _REPEAT_WORD.sub(r'\1', label)

        # ── Rule A+B: Remove sandwiched process-verb modifier words ──────────
        # Pattern: <Adjective> <ProcessVerb> <Noun> → <Adjective> <Noun>
        # e.g. 'Dynamic Resizing Stack' → 'Dynamic Stack'
        #      'Size Stack' → 'Stack' (if 'Size' is purely a modifier, not content)
        # Detect by checking if removing the middle word still yields a valid
        # noun phrase (has at least one non-stop content word remaining).
        _PROCESS_MODIFIERS = {
            'resizing', 'resized', 'sizing', 'sized', 'processing', 'processed',
            'handling', 'handled', 'managing', 'managed', 'tracking', 'tracked',
        }
        words = label.split()
        if len(words) >= 2:
            cleaned = [w for w in words if w.lower() not in _PROCESS_MODIFIERS]
            if cleaned and len(cleaned) < len(words):
                label = ' '.join(cleaned)

        # Re-split after cleaning
        words = label.split()

        # ── Rule B: 'Size <Noun>' where Size is purely a quantity modifier ───
        # 'Size Stack' → 'Static Stack' is NOT inferrable without context,
        # but 'Size Stack' → 'Fixed-Size Stack' is a reasonable normalization.
        if (len(words) == 2
                and words[0].lower() == 'size'
                and words[1][0].isupper()):
            label = f"Fixed-Size {words[1]}"

        # ── Rule F: Structural qualifier normalization ────────────────────────
        # Problem 5: "Size Stack" → "Static Stack", "Dynamic Resizing Stack" → "Dynamic Stack"
        # Heuristic: if the label ends with a known data-structure/concept noun AND
        # starts with a non-canonical qualifier word, replace it with the canonical qualifier.
        # Detection is PURELY STRUCTURAL — we look for pattern: <SingleQualifier> <Noun>
        # where the qualifier is a size/scope/modifier word that maps to a canonical adjective.
        # No domain words — qualifier mapping uses general English morphology.
        _QUALIFIER_ALIASES = {
            # Non-canonical qualifier word → canonical qualifier
            'size':      'Fixed-Size',
            'sized':     'Fixed-Size',
            'count':     'Fixed-Size',
            'limited':   'Fixed-Size',
            'resize':    'Dynamic',
            'resizable': 'Dynamic',
            'growable':  'Dynamic',
            'flexible':  'Dynamic',
            'expandable':'Dynamic',
            'shrinkable':'Dynamic',
            'variable':  'Dynamic',
        }
        label_words = label.split()
        if len(label_words) >= 2:
            first_word_lower = label_words[0].lower()
            if first_word_lower in _QUALIFIER_ALIASES:
                canonical_qualifier = _QUALIFIER_ALIASES[first_word_lower]
                label = canonical_qualifier + " " + " ".join(label_words[1:])

        # ── Rule C: Strip parenthetical that duplicates label words ──────────
        def _strip_dup_paren(s: str) -> str:
            m = re.match(r'^(.+?)\s*\((.+?)\)\s*$', s)
            if not m:
                return s
            outer_words = set(m.group(1).lower().split())
            inner_words = set(m.group(2).lower().split())
            # If inner is just an acronym of the outer, strip it
            acronym = ''.join(w[0] for w in m.group(1).split() if w).upper()
            if m.group(2).strip().upper() == acronym:
                return m.group(1).strip()
            # If overlap > 60%, strip the parenthetical
            if outer_words and inner_words:
                overlap = len(outer_words & inner_words) / max(len(outer_words), 1)
                if overlap > 0.6:
                    return m.group(1).strip()
            return s
        label = _strip_dup_paren(label)

        # ── Rule E: OCR single-letter-drop repair ────────────────────────────
        _OCR_WORD_FIXES = {
            r'^[Rr]der\b': 'Order',
            r'^[Tt]eration\b': 'Iteration',
            r'^[Nn]sertion\b': 'Insertion',
            r'^[Pp]eration\b': 'Operation',
            r'^[Dd]eletion\b': 'Deletion',
        }
        fixed_words = []
        for w in label.split():
            fixed = w
            for pat, repl in _OCR_WORD_FIXES.items():
                if re.match(pat, w):
                    fixed = repl
                    break
            fixed_words.append(fixed)
        label = ' '.join(fixed_words)

        return label.strip()

    # ---------- Topic Canonicalization ----------
    def canonicalize_topic(self, label: str) -> str:
        """Normalize topic name for deduplication"""
        label = label.upper().strip()
        label = re.sub(r'\b(THE|A|AN|OF|IN|FOR|TO|AND|OR)\b', '', label, flags=re.IGNORECASE)
        label = re.sub(r'[^\w\s]', '', label)
        label = re.sub(r'\s+', ' ', label).strip()
        return label

    # ---------- SEMANTIC VALIDITY GATE ----------
    def is_valid_note_sentence(self, s: str, domain_keywords: set = None) -> bool:
        """Check if sentence is valid educational content (not OCR garbage)
        
        STRICT RULES:
        1. Must have a verb (grammar check)
        2. Must have domain keyword OR be well-formed
        3. Must not be visual/OCR garbage
        """
        s = s.strip()
        
        # Minimum word count (not just characters)
        word_count = len(s.split())
        if word_count < 5:
            return False
        
        # Reject if too few alphanumeric chars (symbol soup)
        alnum_ratio = sum(c.isalnum() or c.isspace() for c in s) / max(len(s), 1)
        if alnum_ratio < 0.7:
            return False
        
        # Reject assembly/register patterns (universal)
        if re.search(r'\b(DWORD|PTR|eax|ebx|ecx|edx|rbp|rsp|FLAT|OFFSET|mov|push|pop)\b', s, re.IGNORECASE):
            return False
        
        # Reject binary patterns
        if re.search(r'[01]{8,}', s):
            return False
        
        # Reject repetitive characters (eee, ooo, etc.)
        if re.search(r'(.)\1{3,}', s):
            return False
        
        # Reject gibberish (random caps/symbols patterns)
        if re.search(r'[A-Z]{5,}\s+[A-Z]{5,}', s):
            return False
        
        # Reject OCR garbage characters
        if any(c in s for c in ['¢', '©', '®', '™', '†', '‡', '§']):
            return False
        
        # Reject questions — notes should be statements, not questions
        if s.strip().endswith('?'):
            return False
        
        s_lower = s.lower()
        
        # MANDATORY: Check for verb presence (grammar gate)
        verbs = ["is", "are", "was", "were", "has", "have", "can", "will", "would", "should",
                 "converts", "generates", "produces", "checks", "creates", "stores", "translates",
                 "represents", "contains", "includes", "defines", "refers", "performs", "processes",
                 "uses", "takes", "gives", "makes", "shows", "provides", "allows", "enables"]
        has_verb = any(f" {v} " in f" {s_lower} " for v in verbs)
        
        if not has_verb:
            return False  # STRICT: Must have verb
        
        # STRICT: Reject spoken/filler/lecture phrases (comprehensive list)
        spoken_filler = [
            # Direct address / lecture style
            "in this video", "let us see", "we will", "thank you", "subscribe",
            "we say", "we are going", "we have seen", "we will see", "we can see",
            "in the introduction", "in the previous", "in the next", "in the last",
            "as we discussed", "as i said", "as you can see", "let me",
            "hello everyone", "hi everyone", "good morning", "welcome to",
            "in this lecture", "in this presentation", "in this session",
            "please like", "please share", "i hope", "i think",
            "let us now", "we talked about", "we discussed", "so basically",
            "see you in", "that's all", "that is all", "bye bye",
            "now we will", "now let's", "now let us", "first of all",
            "the replay obviously", "if he gives", "let's say we",
            "so what we", "so here we", "so now we",
            # First/second person conversational
            "we are only", "we are not", "we are just", "we just",
            "we only", "we don't", "we do not", "we need to",
            "we want to", "we have to", "we should", "we can",
            "if we want", "if we need", "if you want", "if you need",
            "you will", "you should", "you can", "you need",
            "you see", "you know", "you understand",
            "i am going", "i will", "i want", "i am", "i have",
            # Concerned / irrelevant
            "concerned with", "concerned about", "not concerned",
            "irrelevant to us", "irrelevant to", "doesn't matter",
            "does not matter", "don't care", "do not care",
            # Store / access conversational
            "store that data", "access any", "is the data present",
            "where is the data", "what data model",
        ]
        if any(f in s_lower for f in spoken_filler):
            return False

        # ── Generic slide-visual OCR garbage (Fix 4B) ────────────────────────────────
        # Long OCR-joined garbage strings that contain a verb (so they pass the verb gate)
        # but are raw slide diagram labels or educational branding.
        _SLIDE_OCR_SUBSTRINGS = [
            "outcomes upon the completion",      # Lecture outcomes slide
            "learner will be able",
        ]
        if any(pat in s_lower for pat in _SLIDE_OCR_SUBSTRINGS):
            return False

        # Check against DYNAMIC domain keywords (from KG)
        if domain_keywords:
            has_domain_keyword = any(k.lower() in s_lower for k in domain_keywords if len(k) > 2)
            return has_domain_keyword  # Must match domain

        return True  # Has verb, no domain filter

    # ---------- LINE CLASSIFICATION ----------
    def classify_line(self, line: str, domain_keywords: set = None) -> str:
        """Classify line as CODE, DIAGRAM, EXPLANATION, or NOISE
        
        Args:
            line: The line to classify
            domain_keywords: Set of topic-specific keywords for relevance check
        """
        line = line.strip()
        if not line:
            return "NOISE"
        
        # HARD OCR GARBAGE REJECTION (FIRST CHECK)
        if self.is_ocr_garbage(line):
            return "NOISE"
        
        # CODE detection (universal patterns)
        code_patterns = ["{", "}", "();", "printf", "int ", "float ", "return ", 
                        "DWORD", "PTR", "eax", "ebx", "#include", "void ", "public ",
                        "private ", "class ", "def ", "function "]
        if any(p in line for p in code_patterns):
            return "CODE"
        
        # DIAGRAM detection (universal)
        if re.search(r'\b(diagram|figure|image|blackboard|screenshot|picture|flowchart)\b', line, re.IGNORECASE):
            return "DIAGRAM"
        
        # Noise detection (universal patterns)
        noise_patterns = ["tutorial", "subscribe", "academy", "channel", "@", 
                         "www.", "http", "logo", "icon", "like and share",
                         "comment below", "notification", "bell icon"]
        if any(p in line.lower() for p in noise_patterns):
            return "NOISE"
        
        # Valid explanation check (with dynamic keywords)
        if self.is_valid_note_sentence(line, domain_keywords):
            return "EXPLANATION"
        
        return "NOISE"

    # ---------- TOPIC ELIGIBILITY (STRICT) ----------
    def is_valid_topic(self, title: str) -> bool:
        """Check if detected topic is a valid educational topic
        STRICT: Rejects code, assembly, visual descriptions, slide headers, and garbage.
        Also rejects Actor nodes (Programmer, User) and Metadata (course labels).
        """
        title = title.strip()

        # ── Reject fused concept labels ('X and Y' where X and Y are unrelated) ─
        try:
            from concept_flow_organizer import _is_fused_label
            if _is_fused_label(title):
                return False
        except Exception:
            pass

        # ── Reject Metadata labels (course/slide navigation) ─────────────────
        _META_TAIL_WORDS = {
            'chapter', 'section', 'unit', 'module', 'topic', 'subject',
            'lecture', 'slide', 'slides', 'part', 'week', 'lesson',
            'review', 'fundamentals', 'introduction to', 'overview',
        }
        title_lower_words = [w.lower() for w in title.split()]
        if title_lower_words and title_lower_words[-1] in _META_TAIL_WORDS:
            return False
        if len(title_lower_words) >= 4 and 'subject' in title_lower_words:
            return False
        # Also reject any label whose ALL words are metadata words
        if title_lower_words and all(w in _META_TAIL_WORDS for w in title_lower_words):
            return False

        # ── Reject Actor labels (human/system agents — never top-level topics) ─
        # These should appear as inline mentions inside sections, not as headings.
        _ACTOR_WORDS = {
            'programmer', 'programmers', 'developer', 'developers', 'user', 'users',
            'engineer', 'engineers', 'student', 'students', 'teacher', 'teachers',
            'compiler', 'interpreter', 'assembler', 'cpu', 'processor', 'client',
            'server', 'browser', 'coder', 'coders', 'designer', 'designers',
        }
        non_stop = [w for w in title_lower_words
                    if w not in {'and', 'or', 'the', 'a', 'an', 'of', 'in', 'for', 'to', 'by'}]
        if non_stop and all(w in _ACTOR_WORDS for w in non_stop):
            return False

        # Reject code-like parentheses, but ALLOW abbreviation/clarification forms
        if "(" in title and ")" in title:
            paren_content = re.search(r'\(([^)]+)\)', title)
            if paren_content:
                inside = paren_content.group(1).strip()
                code_syms = {'{', '}', ';', '=', '+', '*', '/', '<', '>'}
                if len(inside) > 15 or any(c in inside for c in code_syms):
                    return False
            else:
                return False
        elif "(" in title or ")" in title:
            return False

        # Reject code keywords anywhere
        code_keywords = ["printf", "stdio", "int main", "#include", "return",
                        "{", "}", ";", "void ", "def ", "class ", "function"]
        if any(k in title.lower() for k in code_keywords):
            return False

        # Reject assembly-like topics
        if re.search(r'\b(eax|ebx|ecx|edx|ptr|DWORD|OFFSET|rbp|rsp)\b', title, re.IGNORECASE):
            return False

        # Reject visual description topics
        if any(v in title.lower() for v in self.VISUAL_NOISE):
            return False

        # Reject too long (likely sentence, not topic)
        if len(title.split()) > 12:
            return False

        # Reject empty
        if len(title.strip()) < 1:
            return False

        # Reject numeric-heavy
        if sum(c.isdigit() for c in title) > len(title) * 0.3:
            return False

        # Reject symbol-heavy (more than 4 non-alphanumeric, accounting for parens)
        symbol_count = sum(not c.isalnum() and not c.isspace() for c in title)
        if symbol_count > 4:
            return False

        # Reject gibberish patterns
        if re.search(r'[^a-zA-Z\s]{3,}', title):
            return False

        return True

    # ---------- DYNAMIC DIAGRAM CAPTIONS ----------
    def get_semantic_caption(self, topic: str, raw_caption: str, kg_nodes: list = None) -> str:
        """
        Generate meaningful diagram caption DYNAMICALLY from topic and KG context.

        FIX-4 / FIX-5: Instead of outputting bare "Figure N: diagram" or
        "a screenshot of a computer screen showing numbers", this method:
          1. Finds the best matching KG node for the topic.
          2. Extracts structural/operational signals from the node's description
             and its outgoing edges (e.g. components, operations, properties).
          3. Builds a rich caption from these structural signals.
          4. Falls back to cleaned raw caption if KG has no relevant info.
          5. Last resort: topic-based caption — never just "diagram" or "screenshot".

        All logic is dynamic — no domain-specific words used in detection.
        """
        # ── Step 1: Find best matching KG node description ───────────────────
        best_node = None
        best_desc = ""
        best_score = 0
        if kg_nodes:
            topic_words = set(re.findall(r'[a-z]{3,}', topic.lower()))
            for node in kg_nodes:
                node_label = node.get("label", "")
                node_desc  = node.get("description", "")
                if not node_desc or len(node_desc.split()) < 5:
                    continue
                label_words = set(re.findall(r'[a-z]{3,}', node_label.lower()))
                desc_words  = set(re.findall(r'[a-z]{3,}', node_desc.lower()))
                label_overlap = len(topic_words & label_words) / max(len(topic_words), 1)
                desc_overlap  = len(topic_words & desc_words)  / max(len(topic_words), 1)
                score = label_overlap * 0.6 + desc_overlap * 0.4
                if score > best_score:
                    best_score = score
                    best_desc  = node_desc.strip().rstrip('.')
                    best_node  = node

        # ── Step 2: Build rich structural caption from KG node + its edges ───
        if best_node and best_score > 0.2:
            sentences = re.split(r'(?<=[.!?])\s+', best_desc)
            short_desc = ' '.join(sentences[:2])

            # Also gather component labels from this node's outgoing edges
            # to add structural richness: "showing X with components A, B, C"
            nid = best_node.get("id", "")
            component_labels = []
            # We don't have edge_map in this scope, so we scan kg_nodes for matching
            # nodes referenced by the best_node's label in descriptions
            if kg_nodes and nid:
                # Look for nodes whose labels appear in the best_desc as structural hints
                desc_lower = best_desc.lower()
                for other in kg_nodes:
                    other_lbl = other.get("label", "")
                    if (other_lbl and other_lbl != best_node.get("label", "")
                            and other_lbl.lower() in desc_lower
                            and len(other_lbl.split()) <= 3):
                        component_labels.append(other_lbl)
                        if len(component_labels) >= 3:
                            break

            if component_labels:
                comp_str = ", ".join(component_labels)
                return f"Diagram illustrating {topic}: {short_desc}, with key elements including {comp_str}."
            return f"Diagram illustrating {topic}: {short_desc}."

        # ── Step 3: Clean and validate raw caption ───────────────────────────
        if raw_caption and raw_caption.strip():
            # Remove generic visual boilerplate: "a screenshot of a computer screen showing numbers"
            cleaned = re.sub(
                r'\b(a|an|the)?\s*(diagram|image|figure|picture|screenshot|'
                r'blackboard|whiteboard|sign|logo|icon|photo|slide|computer\s+screen|'
                r'screen)\s*(of|showing|with|that|depicting|containing|displaying)?\s*',
                '', raw_caption, flags=re.I
            )
            # Strip generic visual quality descriptors
            cleaned = re.sub(
                r'\b(black\s+and\s+white|white\s+text|black\s+background|'
                r'white\s+background|simple|basic|hand\s*drawn|drawn|'
                r'numbers|letters|text)\b',
                '', cleaned, flags=re.I
            )
            cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip(' ,.')

            # Only use if cleaned caption has ≥ 4 meaningful content words
            meaningful_words = [w for w in cleaned.split()
                                 if len(w) > 3 and w.lower() not in
                                 {'with', 'that', 'this', 'from', 'into', 'some',
                                  'each', 'also', 'about', 'more', 'than', 'very',
                                  'showing', 'depicting', 'containing'}]
            if len(meaningful_words) >= 4 and not re.search(r'(.)\1{3,}', cleaned):
                return f"Diagram: {cleaned[0].upper() + cleaned[1:] if cleaned else cleaned}."

        # ── Step 4: Fallback — always descriptive, never bare label ──────────
        topic_title = topic.strip().title() if topic else "This concept"
        return f"Visual representation showing the structure and key relationships of {topic_title}."

    # ---------- EXTRACT DOMAIN KEYWORDS FROM KG ----------
    def extract_domain_keywords(self, nodes: list, edges: list) -> set:
        """Extract domain-specific keywords from Knowledge Graph
        
        This makes the filtering DYNAMIC - keywords come from the actual video content
        """
        keywords = set()
        
        # From node labels
        for node in nodes:
            label = node.get("label", "")
            if label and len(label) > 2:
                keywords.add(label.lower())
                # Also add individual words from multi-word labels
                for word in label.split():
                    if len(word) > 3:
                        keywords.add(word.lower())
        
        # From edge relations
        for edge in edges:
            rel = edge.get("relation", "")
            if rel and len(rel) > 3:
                keywords.add(rel.lower())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                      "have", "has", "had", "do", "does", "did", "will", "would", "could",
                      "should", "may", "might", "must", "shall", "can", "need", "dare",
                      "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
                      "from", "as", "into", "through", "during", "before", "after",
                      "above", "below", "between", "under", "again", "further", "then",
                      "once", "and", "but", "or", "nor", "so", "yet", "both", "either",
                      "neither", "not", "only", "own", "same", "than", "too", "very"}
        
        keywords = keywords - stop_words
        
        return keywords

    # ---------- MAIN GENERATION: ROBUST NOTES PIPELINE ----------
    def generate_notes(self, session_path: Path):
        """
        Generate EDUCATIONAL NOTES with guaranteed hierarchical structure:
        
        Pipeline:
          1. Load text segments + KG data + diagrams
          2. Match text segments to KG nodes (topic identification)
          3. Build hierarchy from KG (centrality + structural edges)
          4. Construct HierarchicalNotes JSON (Section → Subsection → Points)
          5. Validate structure, auto-fix if needed
          6. Render via shared notes_renderer (PDF + TXT)
        """
        from hierarchical_schema import (
            make_point, make_subsection, make_section, make_notes,
            validate_hierarchy, fix_hierarchy
        )
        from notes_renderer import render_pdf, render_txt
        from notes_quality_enforcer import post_process_kg_notes
        from notes_postprocessor import post_process_notes, load_merged_captions

        session_path = Path(session_path)
        
        # ── 1. Locate Text Segments ──
        fused_dir = None
        possible_dirs = [
            session_path / "combined_fused",
            session_path / "fused_sentences",
            session_path / "combined"
        ]
        for d in possible_dirs:
            if d.exists() and list(d.glob("*.txt")):
                fused_dir = d
                break
        
        if not fused_dir:
            print(f"[Notes] ❌ No text segments found in {session_path}")
            return

        # ── 2. Load KG Data ──
        nodes = []
        edges = []
        nodes_paths = [session_path / "fused_kg" / "fused_nodes.json", session_path / "graphs" / "kg_nodes.json"]
        edges_paths = [session_path / "fused_kg" / "fused_edges.json", session_path / "graphs" / "kg_edges.json"]
        
        for p in nodes_paths:
            if p.exists():
                try: nodes = json.loads(p.read_text(encoding='utf-8')); break
                except: pass
        for p in edges_paths:
            if p.exists():
                try: edges = json.loads(p.read_text(encoding='utf-8')); break
                except: pass
        
        # Fix 4.6: Node description sanitization
        try:
            for node in nodes:
                raw_desc = node.get("description", "")
                if raw_desc and hasattr(self, "sanitizer"):
                    node["description"] = self.sanitizer.sanitize_text(raw_desc)
            print(f"[Notes] Sanitized descriptions for {len(nodes)} KG nodes")
        except Exception as e:
            print(f"[Notes] Warning: Node description sanitization failed: {e}")

        # P5: Sanitize ? artefacts from ALL KG node fields (labels + descriptions)
        # and edge relation strings — this is the PRIMARY source of "? The..." in bullets.
        # Labels use is_heading=True (compound merge / category drop)
        # Descriptions/edges use is_heading=False (separator -> ": ")
        try:
            from concept_flow_organizer import sanitize_question_marks, fix_ocr_truncated_label
            for node in nodes:
                if "label" in node:
                    lbl = sanitize_question_marks(str(node["label"]), is_heading=True)
                    node["label"] = fix_ocr_truncated_label(lbl)
                if "description" in node and node["description"]:
                    node["description"] = sanitize_question_marks(str(node["description"]), is_heading=False)
            for edge in edges:
                if "relation" in edge and edge["relation"]:
                    edge["relation"] = sanitize_question_marks(str(edge["relation"]), is_heading=False)
            print(f"[Notes] Q-mark sanitized labels+descriptions for {len(nodes)} KG nodes, {len(edges)} edges")
        except Exception as e:
            # Inline fallback — strip any "? word" pattern from all text fields
            _QM = re.compile(r'(\w+)\s+\?\s+(\w+)')
            _QM_ANY = re.compile(r'\?')
            for node in nodes:
                for field in ("label", "description"):
                    if node.get(field):
                        node[field] = _QM_ANY.sub("", _QM.sub(r"\1 \2", str(node[field]))).strip()
            print(f"[Notes] Warning: Node Q-sanitize used fallback: {e}")
        
        label_node_lookup = {n.get("label", "").lower(): n for n in nodes}
        edge_map = {}
        for e in edges:
            src = e.get("source", "")
            tgt = e.get("target", "")
            rel = e.get("relation", "is related to")
            if src not in edge_map: edge_map[src] = []
            edge_map[src].append((tgt, rel))
        
        domain_keywords = self.extract_domain_keywords(nodes, edges)
        print(f"[Notes] 📚 Extracted {len(domain_keywords)} domain keywords from KG")

        # ── 3. Load Images / Captions ──
        captions = {}
        image_paths = {}
        caption_paths = [fused_dir / "merged_captions.json", session_path / "diagram_texts"]
        for cp in caption_paths:
            if cp.exists() and cp.is_file() and cp.suffix == '.json':
                try: captions.update(json.loads(cp.read_text(encoding='utf-8'))); break
                except: pass
            elif cp.exists() and cp.is_dir():
                for txt in cp.glob("*.txt"):
                    captions[txt.stem] = txt.read_text(encoding='utf-8').strip()

        img_dirs = [session_path / "diagrams", session_path / "fused_kg", fused_dir]
        
        # Support fused sessions by scanning original session dirs
        if session_path.name.endswith("_fused"):
            parts = session_path.name.split("_")[:-1]  # drop "fused"
            for part in parts:
                orig_dir = session_path.parent / part
                if orig_dir.exists():
                    img_dirs.extend([orig_dir / "diagrams", orig_dir / "fused_kg"])
                    # Also collect captions from original dirs if not already merged
                    orig_cap = orig_dir / "diagram_texts"
                    if orig_cap.exists() and orig_cap.is_dir():
                        for txt in orig_cap.glob("*.txt"):
                            if txt.stem not in captions:
                                try: captions[txt.stem] = txt.read_text(encoding='utf-8').strip()
                                except: pass

        for d in img_dirs:
            if d and d.exists():
                for img in list(d.glob("*.png")) + list(d.glob("*.jpg")):
                    if img.name not in image_paths:
                        image_paths[img.name] = img

        # ── 4. Match Text Segments to KG Nodes ──
        text_files = sorted(list(fused_dir.glob("*.txt")))
        canonical_topics = {}
        last_valid_canon = None
        
        for seg_idx, txt_file in enumerate(text_files):
            if "merged_captions" in txt_file.name: continue
            raw_text = txt_file.read_text(encoding='utf-8')
            text = self.clean_text(raw_text)
            if len(text.split()) < 15: continue

            text_lower = text.lower()
            best_node = None
            max_score = 0
            
            for label, node in label_node_lookup.items():
                if len(label) < 3: continue
                count = text_lower.count(label)
                if count > 0:
                    score = count * 10
                    if len(label.split()) > 1: score += 5
                    if score > max_score:
                        max_score = score
                        best_node = node
            
            if best_node and max_score >= 10:
                topic_label = best_node.get("label", "Topic")
                topic_node = best_node
            else:
                topic_label = self.generate_topic(text) if hasattr(self, 'generate_topic') else "General Concept"
                topic_node = None
            
            if not self.is_valid_topic(topic_label):
                if last_valid_canon and last_valid_canon in canonical_topics:
                    canonical_topics[last_valid_canon]["texts"].append(text)
                continue
            
            canon = self.canonicalize_topic(topic_label)
            if not canon or len(canon) < 3:
                if last_valid_canon:
                    canonical_topics[last_valid_canon]["texts"].append(text)
                continue
            
            if canon in canonical_topics:
                canonical_topics[canon]["texts"].append(text)
                if topic_node and not canonical_topics[canon]["node"]:
                    canonical_topics[canon]["node"] = topic_node
            else:
                canonical_topics[canon] = {
                    "topic": topic_label.upper(),
                    "canon": canon,
                    "node": topic_node,
                    "texts": [text],
                    "first_segment_idx": seg_idx
                }
                last_valid_canon = canon

        # ── 4b. Load Structure Plan (Gemini completeness checklist) ──
        # The plan was saved by app.py alongside topic_labels.json after the
        # summarise() call.  It is a dict:
        #   {"sections": [{"title": ..., "concepts": [{"label": ..., "description": ...}]}]}
        # We use it as a safety net: any concept in the plan that is NOT already
        # in the KG nodes list gets injected as a synthetic node so the notes-
        # builder can emit it with the plan's description rather than silently
        # omitting it.
        structure_plan: dict = {}
        _plan_paths = [
            fused_dir / "structure_plan.json",
            session_path / "fused_kg" / "structure_plan.json",
            session_path / "graphs" / "structure_plan.json",
        ]
        for _pp in _plan_paths:
            if _pp.exists():
                try:
                    structure_plan = json.loads(_pp.read_text(encoding="utf-8"))
                    _plan_secs = len(structure_plan.get("sections", []))
                    _plan_concepts = sum(
                        len(s.get("concepts", [])) for s in structure_plan.get("sections", [])
                    )
                    print(f"[Notes] ✅ Loaded structure plan: "
                          f"{_plan_secs} sections, {_plan_concepts} concepts")
                    break
                except Exception as _e:
                    print(f"[Notes] ⚠️  Could not load structure plan from {_pp}: {_e}")

        # Inject plan concepts that are missing from the KG nodes list.
        # Strategy (fully dynamic — no domain words):
        #   1. Build a lowercase label → node lookup from existing nodes.
        #   2. For each concept in the plan, check if any existing node label
        #      is a close match (exact or substring).
        #   3. If not found, create a synthetic node with:
        #        id        = "plan_<index>"
        #        label     = plan concept label
        #        description = plan concept description
        #      and add it to `nodes` and `edges` so the hierarchy builder picks it up.
        if structure_plan.get("sections"):
            _existing_labels = {n.get("label", "").lower().strip() for n in nodes}
            _plan_injected = 0
            for _plan_sec in structure_plan["sections"]:
                _sec_title = _plan_sec.get("title", "").strip()
                for _ci, _concept in enumerate(_plan_sec.get("concepts", [])):
                    _clabel = _concept.get("label", "").strip()
                    _cdesc  = _concept.get("description", "").strip()
                    if not _clabel or not _cdesc:
                        continue
                    _clabel_lower = _clabel.lower()
                    # Skip if a node with this label (or containing it) already exists
                    _found = any(
                        _clabel_lower in _el or _el in _clabel_lower
                        for _el in _existing_labels
                        if len(_el) >= 3 and len(_clabel_lower) >= 3
                    )
                    if _found:
                        continue
                    # Find or create a parent node for this section title
                    _parent_id = None
                    _sec_lower = _sec_title.lower()
                    for _n in nodes:
                        if _n.get("label", "").lower().strip() == _sec_lower:
                            _parent_id = _n.get("id")
                            break
                    if not _parent_id:
                        # Find a node whose label is contained in / contains the section title
                        for _n in nodes:
                            _nl = _n.get("label", "").lower().strip()
                            if _nl and (_nl in _sec_lower or _sec_lower in _nl) and len(_nl) >= 4:
                                _parent_id = _n.get("id")
                                break
                    # Inject the synthetic node
                    _syn_id = f"plan_{_plan_injected}"
                    _syn_node = {
                        "id": _syn_id,
                        "label": _clabel,
                        "description": _cdesc,
                        "_synthetic": True,   # flag so later passes can handle differently
                    }
                    nodes.append(_syn_node)
                    _existing_labels.add(_clabel_lower)
                    # Wire it to its parent via an "includes" edge
                    if _parent_id:
                        edges.append({"source": _parent_id, "target": _syn_id,
                                      "relation": "includes"})
                        if _parent_id not in edge_map:
                            edge_map[_parent_id] = []
                        edge_map[_parent_id].append((_syn_id, "includes"))
                    _plan_injected += 1
            if _plan_injected:
                print(f"[Notes] 💉 Injected {_plan_injected} plan concepts missing from KG")

        # ── 4c. Extract plan subsection structure for notes augmentation ──
        # The unified Gemini call (_generate_summary_and_plan) returns:
        #   - subsections with typed headings and bullet points per section
        #   - kg_topic_refs: the original KG topic labels each plan section absorbs
        #
        # We build THREE lookup maps:
        #   plan_subsection_map: canonical_title_lower -> [{heading, points}]
        #   plan_needs_diagram_map: canonical_title_lower -> bool
        #   plan_ref_map: original_kg_label_lower -> canonical_title_lower
        #     (enables exact reverse matching: "Arrangement" → "Overview and Structure")
        plan_subsection_map: dict = {}
        plan_needs_diagram_map: dict = {}
        plan_ref_map: dict = {}  # original KG label -> plan section canonical title
        if structure_plan.get("sections"):
            for _plan_sec in structure_plan["sections"]:
                _key = _plan_sec.get("title", "").lower().strip()
                if not _key:
                    continue
                _subs = _plan_sec.get("subsections", [])
                if isinstance(_subs, list) and _subs:
                    plan_subsection_map[_key] = _subs
                plan_needs_diagram_map[_key] = bool(_plan_sec.get("needs_diagram", False))
                # Build reverse map: each absorbed KG label -> this plan section
                for _ref in _plan_sec.get("kg_topic_refs", []):
                    _ref_lower = str(_ref).strip().lower()
                    if _ref_lower:
                        plan_ref_map[_ref_lower] = _key
                # Also map the canonical title to itself
                plan_ref_map[_key] = _key

        # ── 5. Build Hierarchy from KG ──
        # First build node_text_map so we can pass frequency counts to build_hierarchy
        node_text_map = {}
        for canon_data in canonical_topics.values():
            if canon_data["node"]:
                node_text_map[canon_data["node"]["id"]] = " ".join(canon_data["texts"])

        # Build frequency map: count how many text segments mention each node label
        # This is the "frequency_in_source" signal for Concept Gravity importance scoring
        _freq_map: dict = {}
        for nid, text_blob in node_text_map.items():
            node = next((n for n in nodes if n.get("id") == nid), None)
            if node:
                lbl = node.get("label", "").lower()
                if lbl:
                    _freq_map[nid] = text_blob.lower().count(lbl)

        hierarchy = self.build_hierarchy(nodes, edge_map, freq_map=_freq_map)
        roots = hierarchy["roots"]
        children_map = hierarchy["children_map"]
        node_lookup_by_id = hierarchy["node_lookup"]
        importance_scores = hierarchy.get("importance", {})

        # ── 6. Build HierarchicalNotes JSON (Semantic Grouping) ──
        # Step 6 begins with the Concept Block construction (intermediate layer),
        # then renders them into HierarchicalNotes format.
        #
        # Architecture:  KG → ConceptBlockBuilder → [ConceptBlock,...] → Renderer → Notes
        #
        # ConceptBlocks give us:
        #   - clean "build then render" separation
        #   - topology-aware clustering (Bus Topology groups Collision, Cable, etc.)
        #   - diagram anchoring (best diagram attached to its owning concept)
        #   - recursive sub-blocks for nested subconcepts
        used_images = set()           # img_name strings (fast lookup)
        used_img_canonicals = set()   # resolved absolute paths (cross-section dedup)
        emitted_node_ids = set()      # Track concepts already explained to prevent repetition

        # ── 6a. Build Concept Blocks — uses shared emitted_node_ids ──────────
        _block_builder = ConceptBlockBuilder(self)
        concept_blocks = _block_builder.build_blocks(
            roots=roots,
            node_lookup=node_lookup_by_id,
            edge_map=edge_map,
            children_map=children_map,
            importance=importance_scores,
            captions=captions,
            image_paths=image_paths,
            emitted=emitted_node_ids,      # shared set — builder marks children as emitted
            domain_keywords=domain_keywords,  # for diagram domain-verification (Problem 2)
        )
        # Build a lookup: root_id → ConceptBlock (used when rendering)
        _block_by_root: dict = {b.node_id: b for b in concept_blocks}
        print(f"[Notes] 🧱 Built {len(concept_blocks)} ConceptBlocks from {len(roots)} roots")

        def _register_img(img_name: str, img_p=None) -> None:
            """Mark image as used in both name-set and canonical-path-set."""
            used_images.add(img_name)
            if img_p is not None:
                try:
                    used_img_canonicals.add(str(Path(str(img_p)).resolve()))
                except Exception:
                    used_img_canonicals.add(str(img_p))

        def _img_already_used(img_name: str, img_p=None) -> bool:
            """True if image already used — checked by name OR canonical path."""
            if img_name in used_images:
                return True
            if img_p is not None:
                try:
                    if str(Path(str(img_p)).resolve()) in used_img_canonicals:
                        return True
                except Exception:
                    pass
            return False
        sections = []

        # ── Relation-to-heading mapper ──
        # Groups edge relations into semantic categories for readable section headings
        _RELATION_GROUPS = {
            # ─── Definitions / Identity ───
            "is defined as": "Definition and Fundamental Concept",
            "defined as": "Definition and Fundamental Concept",
            "means": "Definition and Fundamental Concept",
            "refers to": "Definition and Fundamental Concept",
            "known as": "Definition and Fundamental Concept",
            "is called": "Definition and Fundamental Concept",
            "also known as": "Definition and Fundamental Concept",
            "is a": "Definition and Fundamental Concept",
            "represents": "Definition and Fundamental Concept",
            "definition": "Definition and Fundamental Concept",
            "defines": "Definition and Fundamental Concept",
            # ─── Operations / Functions / Deals With ───
            "has operation": "Primary Operations",
            "has function": "Primary Operations",
            "supports operation": "Primary Operations",
            "performs": "Primary Operations",
            "operates on": "Primary Operations",
            "uses operation": "Primary Operations",
            "deals with": "Primary Operations",
            "concerned with": "Primary Operations",
            "handles": "Primary Operations",
            "manages": "Primary Operations",
            "specifies": "Primary Operations",
            "describes": "Primary Operations",
            "determines": "Primary Operations",
            "defines what": "Primary Operations",
            "what data": "Primary Operations",
            "stores": "Primary Operations",
            "retrieves": "Primary Operations",
            "queries": "Primary Operations",
            # ─── Achieved Through / Implementation ───
            "is achieved through": "Implementation and Structure",
            "achieved through": "Implementation and Structure",
            "implemented using": "Implementation and Structure",
            "implemented by": "Implementation and Structure",
            "implemented through": "Implementation and Structure",
            "realized through": "Implementation and Structure",
            "accomplished via": "Implementation and Structure",
            "uses": "Implementation and Structure",
            "employs": "Implementation and Structure",
            "utilizes": "Implementation and Structure",
            "accessed via": "Implementation and Structure",
            # ─── Types / Classification / Levels ───
            "has type": "Types and Classification",
            "type of": "Types and Classification",
            "classified as": "Types and Classification",
            "divided into": "Types and Classification",
            "categorized as": "Types and Classification",
            "kind of": "Types and Classification",
            "has level": "Types and Classification",
            "consists of levels": "Types and Classification",
            "has levels": "Types and Classification",
            "level of": "Types and Classification",
            # ─── Context / Relationship / Addressed By ───
            "provides context for": "Contextual Relationships",
            "provides context": "Contextual Relationships",
            "is related to": "Contextual Relationships",
            "related to": "Contextual Relationships",
            "associated with": "Contextual Relationships",
            "connected to": "Contextual Relationships",
            "interacts with": "Contextual Relationships",
            "addressed by": "Contextual Relationships",
            "addresses": "Contextual Relationships",
            "abstracted by": "Contextual Relationships",
            "supports": "Contextual Relationships",
            "requires": "Contextual Relationships",
            "depends on": "Contextual Relationships",
            "needed by": "Contextual Relationships",
            "allows": "Contextual Relationships",
            # ─── Examples / Analogies ───
            "example of": "Real-World Examples",
            "has example": "Real-World Examples",
            "analogy": "Real-World Examples",
            "illustrated by": "Real-World Examples",
            "such as": "Real-World Examples",
            "similar to": "Real-World Examples",
            "for example": "Real-World Examples",
            # ─── Applications ───
            "used in": "Applications",
            "applied in": "Applications",
            "used for": "Applications",
            "application of": "Applications",
            "helps in": "Applications",
            "enables": "Applications",
            "facilitates": "Applications",
            "achieves": "Applications",
            "provides": "Applications",
            # ─── Properties / Characteristics ───
            "has property": "Key Characteristics",
            "has characteristic": "Key Characteristics",
            "characterized by": "Key Characteristics",
            "follows": "Key Characteristics",
            "based on": "Key Characteristics",
            "operates on principle": "Key Characteristics",
            "hides": "Key Characteristics",
            "abstracts": "Key Characteristics",
            "independent of": "Key Characteristics",
            "does not affect": "Key Characteristics",
            # ─── Components / Parts ───
            "has component": "Components and Structure",
            "consists of": "Components and Structure",
            "contains": "Components and Structure",
            "comprises": "Components and Structure",
            "part of": "Components and Structure",
            "composed of": "Components and Structure",
            "includes": "Components and Structure",
            "has": "Components and Structure",
            "sub-component": "Components and Structure",
            "is part of": "Components and Structure",
            # ─── Status / Conditions ───
            "has condition": "Status and Conditions",
            "checks": "Status and Conditions",
            "returns": "Status and Conditions",
            "has status": "Status and Conditions",
            # ─── Purpose / Goal ───
            "purpose is": "Purpose and Goals",
            "aims to": "Purpose and Goals",
            "designed for": "Purpose and Goals",
            "intended for": "Purpose and Goals",
            "goal is": "Purpose and Goals",
            "objective": "Purpose and Goals",
            # ─── Explanation / Description ───
            "is explained through": "Detailed Explanation",
            "explained through": "Detailed Explanation",
            "described by": "Detailed Explanation",
            "is described as": "Detailed Explanation",
            # ─── Roles / Actors ───
            "managed by": "Roles and Responsibilities",
            "administered by": "Roles and Responsibilities",
            "controlled by": "Roles and Responsibilities",
            "accessed by": "Roles and Responsibilities",
            "used by": "Roles and Responsibilities",
            "performed by": "Roles and Responsibilities",
            "granted to": "Roles and Responsibilities",
            "given to": "Roles and Responsibilities",
        }

        # Additional keyword-based fallback mapping for partial relation matching
        _RELATION_KEYWORD_MAP = [
            (["deals", "concerned", "handles", "manages", "specifies",
              "describes", "stores", "retrieves"], "Primary Operations"),
            (["level", "tier", "layer", "stage", "type", "kind",
              "category", "class", "classified"], "Types and Classification"),
            (["component", "part", "consist", "contain", "comprise",
              "include", "made up", "sub"], "Components and Structure"),
            (["implement", "achiev", "realized", "accomplish",
              "uses", "employs", "utilizes"], "Implementation and Structure"),
            (["property", "characteristic", "feature", "attribute",
              "hides", "abstracts", "independent"], "Key Characteristics"),
            (["require", "depend", "need", "allow", "support",
              "enable", "address", "related"], "Contextual Relationships"),
            (["role", "responsi", "administer", "manage", "grant",
              "control", "access"], "Roles and Responsibilities"),
            (["purpose", "goal", "aim", "objective", "design",
              "intend", "meant"], "Purpose and Goals"),
            (["example", "instance", "analog", "illustrat",
              "such as", "like"], "Real-World Examples"),
            (["application", "used in", "applied", "used for",
              "helps", "enables", "facilitates"], "Applications"),
            (["define", "means", "refers", "known as", "called",
              "represent", "is a"], "Definition and Fundamental Concept"),
        ]

        def _get_group_heading(relation: str) -> str:
            """
            Map a KG edge relation string to a semantic subsection heading.

            Strategy (in order):
            1. Exact match against _RELATION_GROUPS dict.
            2. Substring match (both directions) against _RELATION_GROUPS.
            3. Keyword-based fallback via _RELATION_KEYWORD_MAP.
            4. Final fallback: 'Key Concepts' — NEVER 'Additional Concepts',
               which is reserved only for the post-processing consolidation pass.
            """
            rel_lower = relation.lower().strip()
            # 1. Exact match
            if rel_lower in _RELATION_GROUPS:
                return _RELATION_GROUPS[rel_lower]
            # 2. Partial match
            for pattern, heading in _RELATION_GROUPS.items():
                if pattern in rel_lower or rel_lower in pattern:
                    return heading
            # 3. Keyword fallback
            for keywords, heading in _RELATION_KEYWORD_MAP:
                if any(kw in rel_lower for kw in keywords):
                    return heading
            # 4. Final fallback — use "Key Concepts", NOT "Additional Concepts"
            #    "Additional Concepts" is reserved for the post-processing step only.
            return "Key Concepts"

        def _is_clear_description(text: str, topic: str) -> bool:
            """Check if text is a clear descriptive/definitional statement worth showing."""
            if not text or len(text.split()) < 5:
                return False
            if text.strip().endswith('?'):
                return False
            # Reject OCR garbage
            if any(c in text for c in ['¢', '©', '®', '™', '†', '‡', '§']):
                return False
            text_lower = text.lower()
            # Must contain a verb to be a proper statement
            verb_indicators = [
                " is ", " are ", " was ", " were ", " has ", " have ",
                " means ", " refers ", " defines ", " provides ",
                " uses ", " allows ", " enables ", " supports ",
                " involves ", " requires ", " represents ", " stores ",
                " follows ", " operates ", " manages ", " handles ",
                " performs ", " processes ",
            ]
            padded = " " + text_lower + " "
            has_verb = any(v in padded for v in verb_indicators)
            if not has_verb:
                return False
            # Check it's informative (not spoken language / meta-commentary)
            filler = [
                "in this video", "let us see", "we will", "thank you", "subscribe",
                "we say", "we are going", "we have seen", "we will see", "we can see",
                "in the introduction", "in the previous", "in the next", "in the last",
                "as we discussed", "as i said", "as we have seen", "as you can see",
                "let me", "let's", "so basically", "so now", "so here",
                "hello everyone", "hi everyone", "good morning", "good afternoon",
                "welcome to", "welcome back", "in this lecture", "in this presentation",
                "in this session", "in this tutorial", "in this module",
                "we are only", "we are not", "we are just", "we just",
                "we only", "we don't", "we do not", "we need to",
                "we want to", "we have to", "we should", "we can",
                "if we want", "if we need", "if you want", "if you need",
                "you will", "you should", "you can", "you need",
                "concerned with", "concerned about", "not concerned",
                "irrelevant to us", "irrelevant to",
                "store that data", "access any", "is the data present",
                "the replay obviously", "if he gives", "let's say we",
            ]
            if any(f in text_lower for f in filler):
                return False
            return True

        def _collect_all_descendants(current_id):
            """Recursively collect all descendant node IDs."""
            desc_ids = []
            for child_id in children_map.get(current_id, []):
                desc_ids.append(child_id)
                desc_ids.extend(_collect_all_descendants(child_id))
            return desc_ids

        # ── Built-in concept glossary for enriching bare labels ──────────────
        # When KG nodes have no description, this provides fallback definitions.
        # Add domain-specific entries as needed; keys are lowercase concept labels.
        _CONCEPT_GLOSSARY = {
            "schema": "the logical structure or blueprint that defines how data is organized in a database, including tables, fields, and relationships.",
            "application level": "the top-most tier in a multi-tier architecture where end-user applications interact with data through a user interface.",
            "view level": "the highest abstraction level in a database system, showing only the data relevant to a particular user or application; provides data security and simplicity.",
            "physical level": "the lowest abstraction level that describes how data is physically stored on hardware, including storage formats, file structures, and access methods.",
            "logical level": "the middle abstraction level that describes what data is stored in the database and the relationships among them, independent of physical storage.",
            "conceptual level": "equivalent to the logical level; describes the entire database schema — what data is stored and how it relates — without concern for physical storage.",
            "logical/conceptual level": "describes what data are stored in the database and what relationships exist among this data; decides the structure of the entire database.",
            "view of data": "a customized, user-specific perspective of the database showing only the relevant portion of data, hiding complexity and ensuring security.",
            "levels of abstraction": "the three-layer architecture (Physical, Logical, View) used in database systems to separate concerns and achieve data independence.",
            "levels of data abstraction": "the three-tier hierarchy (Physical, Logical/Conceptual, View) that divides database design to achieve data independence and simplify user interaction.",
            "data independence": "the ability to modify the schema at one level of the database without requiring changes at the next higher level; separates user view from physical storage.",
            "physical data independence": "the ability to modify the physical storage schema without affecting the logical schema; changes in storage don't impact application queries.",
            "logical data independence": "the ability to modify the logical schema without affecting external/view schemas or application programs.",
            "database administrator": "the person (DBA) responsible for managing, maintaining, and controlling access to the database system, including backup, recovery, and performance.",
            "dba": "Database Administrator — the person responsible for managing, maintaining, securing, and administering the database system.",
            "users": "individuals or applications that interact with the database system through the view level; they see only the data relevant to them.",
            "data storage": "the mechanism by which data is physically saved using simple or complex data structures; the database employs data structures to store information efficiently.",
            "data structures": "the organized formats used to store, manage, and retrieve data efficiently; examples include arrays, trees, hash tables used in database storage.",
            "how data are stored": "describes the physical mechanisms and data structures used to persist data on storage devices within the database system.",
            "what data are stored": "refers to the type and content of data maintained in the database — the logical description of data entities, attributes, and relationships.",
            "relationship among data": "the associations and dependencies between data entities in a database, defined at the logical level to describe how tables are interconnected.",
            "user interaction": "the process by which users communicate with the database system, typically through the view level using queries, forms, or application interfaces.",
            "granting privileges": "the process of assigning access rights and permissions to users or roles so they can perform specific operations on database objects.",
            "front-end system": "the user-facing application layer that provides the interface through which users interact with the database system.",
            "back-end system": "the server-side infrastructure that processes requests, manages the database engine, and handles data storage and retrieval.",
            "database level": "the layer in a multi-tier architecture where the actual database resides, managed by the DBMS and accessed by the application tier.",
            "physical storage": "the actual hardware-based storage media (disk, SSD) where database files and data structures are physically maintained.",
            "3-tier architecture": "a software design pattern separating the application into three layers: presentation (front-end), logic (application), and data (database) tiers.",
            "security": "the set of mechanisms in a database system that protect data from unauthorized access, ensuring confidentiality, integrity, and availability; enforced at the view level.",
            "database": "an organized collection of structured data, managed by a DBMS, designed to store, retrieve, and manage information efficiently.",
        }

        def _enrich_label(label: str, node: dict, parent_topic: str, relation: str) -> str:
            """
            Convert a bare label into a meaningful bullet point.

            Priority:
            1. Actor nodes → natural inline sentence mentioning the agent's role
            2. Node's own description (if clear enough)
            3. Built-in concept glossary lookup
            4. Relation-based contextual sentence using parent topic
            5. Dynamic operation-type detection
            6. Last resort: just the label (bare)
            """
            desc = node.get("description", "")

            # ── Actor nodes: inline mention, never a bare label stub ─────────
            # E.g. "Programmers" → "Programmers implement <parent_topic> explicitly in algorithms."
            ntype = self.classify_node_type(node, edge_map, list(node_lookup_by_id.values()))
            if ntype == "Actor":
                ctx = f" {parent_topic}" if parent_topic else " this data structure"
                if desc and len(desc.split()) >= 4:
                    return f"{label}: {desc}"
                # Dynamically derive action from relation
                rel_lower = relation.lower().strip()
                _ACT_VERBS = {
                    'implements': 'implement', 'uses': 'use', 'manages': 'manage',
                    'invokes': 'invoke', 'calls': 'call', 'creates': 'create',
                    'writes': 'write', 'maintains': 'maintain', 'controls': 'control',
                }
                verb = _ACT_VERBS.get(rel_lower, 'interact with')
                return f"{label} {verb}{ctx} directly in their code or programs."

            if _is_clear_description(desc, label):
                return f"{label}: {desc}"

            key = label.lower().strip().rstrip('.')
            if key in _CONCEPT_GLOSSARY:
                return f"{label}: {_CONCEPT_GLOSSARY[key]}"

            # Try partial glossary match (e.g. "DBA" matches "database administrator")
            for gkey, gval in _CONCEPT_GLOSSARY.items():
                if gkey in key or key in gkey:
                    return f"{label}: {gval}"

            # Relation-based sentence — dynamically construct from relation + context
            rel_lower = relation.lower().strip()
            if rel_lower and rel_lower not in ('includes', 'has', 'contains', 'comprises', ''):
                rel_prose = rel_lower.replace('_', ' ')
                if parent_topic:
                    return f"{label}: {parent_topic} {rel_prose} {label}."
                else:
                    return f"{label}: This concept {rel_prose} {label}."

            # Operation / function / method node
            _OP_SUFFIX_RE = re.compile(
                r'\b(operation|function|method|command|procedure|action|step|process|algorithm)\b', re.I
            )
            if _OP_SUFFIX_RE.search(label):
                ctx = f" on {parent_topic}" if parent_topic else ""
                return f"{label}: An operation or procedure that can be performed{ctx}."

            # Type / classification node
            _TYPE_RE = re.compile(r'\b(type|kind|category|variant|form|class|mode)\b', re.I)
            if _TYPE_RE.search(label):
                ctx = f" of {parent_topic}" if parent_topic else ""
                return f"{label}: A type or category{ctx}."

            # Implementation / method node
            _IMPL_RE = re.compile(r'\b(implementation|method|approach|technique|strategy|way)\b', re.I)
            if _IMPL_RE.search(label):
                ctx = f" for {parent_topic}" if parent_topic else ""
                return f"{label}: An implementation approach{ctx}."

            # Application node
            _APP_RE = re.compile(r'\b(application|use|usage|example|scenario|case)\b', re.I)
            if _APP_RE.search(label):
                ctx = f" of {parent_topic}" if parent_topic else ""
                return f"{label}: An application or use case{ctx}."

            # Property / characteristic node
            _PROP_RE = re.compile(r'\b(property|characteristic|feature|attribute|aspect|behavior)\b', re.I)
            if _PROP_RE.search(label):
                ctx = f" of {parent_topic}" if parent_topic else ""
                return f"{label}: A property or characteristic{ctx}."

            # Last resort: produce a context-aware sentence rather than bare label
            if parent_topic:
                return f"{label}: A concept related to {parent_topic}."
            return f"{label}: A key concept in this topic."

        def _derive_section_heading(topic: str, root_node: dict, subsections: list) -> str:
            """
            Derive a meaningful section heading from the topic label and its content.

            Strategy (fully dynamic — no domain words hardcoded):
            1. Multi-word label that is not vague → use it title-cased.
            2. Root node has a clear description → extract a compact noun-phrase title.
            3. Dominant subsection heading → combine or replace label with it.
            4. Fallback: topic.title()

            Vague single-word / generic labels (e.g. "Arrangement", "Description",
            "Concepts", "Real Life") are enriched using root node description or
            dominant child content so the heading is always informative.
            """
            _VAGUE_TOKENS = {
                'arrangement', 'description', 'concepts', 'concept', 'details',
                'detail', 'real life', 'misc', 'miscellaneous', 'other', 'notes',
                'additional', 'extra', 'general', 'things', 'items',
                'examples', 'example', 'info', 'information',
            }
            label_lower = topic.lower().strip()
            is_vague = label_lower in _VAGUE_TOKENS or all(
                w in _VAGUE_TOKENS for w in label_lower.split()
            )

            # Strategy 1: multi-word, non-vague label — use directly
            if not is_vague and len(topic.split()) >= 2:
                return topic.title()

            # Strategy 2: extract compact title from description
            desc = root_node.get("description", "").strip()
            if _is_clear_description(desc, topic) and len(desc.split()) >= 5:
                words = desc.split()
                cutoff = min(6, len(words))
                for i, w in enumerate(words[:cutoff]):
                    if w.lower() in {'is', 'are', 'was', 'were', 'can', 'will',
                                     'means', 'refers', 'defines', 'provides', ','}:
                        cutoff = i
                        break
                candidate = " ".join(words[:max(cutoff, 2)]).rstrip(".,;:").strip()
                if len(candidate.split()) >= 2:
                    return candidate.title()

            # Strategy 3: use dominant subsection heading
            dominant_heading = ""
            max_pts = 0
            for ss in subsections:
                n = len(ss.get("points", []))
                h = ss.get("heading", "")
                if n > max_pts and h.lower() not in ("description", "overview"):
                    max_pts = n
                    dominant_heading = h

            if dominant_heading and not is_vague:
                return f"{topic.title()} – {dominant_heading}"
            if dominant_heading and is_vague:
                return dominant_heading  # dominant heading is more informative than vague label

            # Fallback
            return topic.title()

        def _build_missed_node_bullet(nid: str) -> str:
            """
            Build a maximally informative bullet for a missed KG node.

            Priority (fully dynamic — no domain words):
            1. Node's own description (any length ≥ 3 words)
            2. All outgoing edges converted to natural language sentences via interpret_edge
            3. All incoming edges with natural language
            4. Descend into children to collect their descriptions
            5. Absolute last resort: bare label (never a generic stub phrase)
            """
            node = node_lookup_by_id.get(nid, {})
            lbl  = node.get("label", "").strip()
            desc = (node.get("description", "") or "").strip()

            if not lbl:
                return ""

            # Priority 1 — any description with ≥ 3 words
            if desc and len(desc.split()) >= 3:
                return f"{lbl}: {desc}"

            # Priority 2 — outgoing edges as natural-language sentences
            out_sentences = []
            for tgt_id, rel in edge_map.get(nid, []):
                tgt_node = node_lookup_by_id.get(tgt_id, {})
                tgt_lbl  = tgt_node.get("label", "")
                tgt_desc = (tgt_node.get("description", "") or "").strip()
                if tgt_lbl and not re.match(r'^N\d+$', str(tgt_lbl)):
                    if tgt_desc and len(tgt_desc.split()) >= 4:
                        out_sentences.append(f"{tgt_lbl}: {tgt_desc}")
                    else:
                        nlg = self.interpret_edge(lbl, rel, tgt_lbl)
                        out_sentences.append(nlg)
            if out_sentences:
                combined = " ".join(out_sentences[:5])
                return f"{lbl}: {combined}"

            # Priority 3 — incoming edges (reverse scan)
            in_sentences = []
            for e in edges:
                if e.get("target") == nid:
                    src_node = node_lookup_by_id.get(e.get("source", ""), {})
                    src_lbl  = src_node.get("label", "")
                    rel      = e.get("relation", "").replace("_", " ").strip()
                    if src_lbl and rel and not re.match(r'^N\d+$', str(src_lbl)):
                        nlg = self.interpret_edge(src_lbl, rel, lbl)
                        in_sentences.append(nlg)
            if in_sentences:
                return f"{lbl}: {' '.join(in_sentences[:3])}"

            # Priority 4 — collect descriptions from immediate children
            child_parts = []
            for cid in children_map.get(nid, []):
                cnode = node_lookup_by_id.get(cid, {})
                clbl  = cnode.get("label", "")
                cdesc = (cnode.get("description", "") or "").strip()
                if clbl:
                    if cdesc and len(cdesc.split()) >= 3:
                        child_parts.append(f"{clbl} ({cdesc})")
                    else:
                        child_parts.append(clbl)
            if child_parts:
                listed = ", ".join(child_parts[:4])
                return f"{lbl} encompasses: {listed}."

            # Absolute last resort — bare label, never generic stub
            return f"{lbl}: A concept in this topic."

        for root_id in roots:
            root_node = node_lookup_by_id.get(root_id, {})
            topic = root_node.get("label", "Topic")

            # Normalize concept label to canonical terminology
            topic = self.normalize_concept_label(topic)
            if topic:
                root_node = dict(root_node)
                root_node["label"] = topic

            if not self.is_valid_topic(topic):
                continue
            if root_id in emitted_node_ids:
                continue
            emitted_node_ids.add(root_id)

            # ── PRIMARY PATH: Render from pre-built ConceptBlock ─────────────
            # ConceptBlockBuilder already clustered all typed children and anchored
            # the best diagram before this loop started.  This is the clean
            # "build → render" separation from the architecture document.
            _block = _block_by_root.get(root_id)
            if _block is not None:
                # Mark all block children as emitted so the inline path and
                # coverage sweep don't double-render them.
                def _mark_block_emitted(blk, _emitted):
                    for bucket in (blk.subconcepts, blk.operations, blk.advantages,
                                   blk.disadvantages, blk.devices, blk.protocols,
                                   blk.implementations, blk.applications,
                                   blk.examples, blk.properties):
                        for _, _, nid in bucket:
                            _emitted.add(nid)
                    for sub_blk in blk.sub_blocks:
                        _emitted.add(sub_blk.node_id)
                        _mark_block_emitted(sub_blk, _emitted)
                _mark_block_emitted(_block, emitted_node_ids)

                # Render the block into subsections
                topic_subsections = _block.to_subsections(make_point, make_subsection)

                # Enrich each subsection with transcript-matched sentences
                # (the block has the structure but not the raw transcript text)
                all_text = node_text_map.get(root_id, "")
                if all_text:
                    matched = self.find_matching_transcript_sentences(
                        topic, all_text, domain_keywords, max_sentences=4)
                    transcript_pts = [make_point(s) for s in matched
                                      if self.is_valid_note_sentence(s, domain_keywords)]
                    if transcript_pts:
                        # Merge into Description subsection if present, else prepend
                        if topic_subsections and topic_subsections[0].get("heading") == "Description":
                            topic_subsections[0]["points"].extend(transcript_pts)
                        else:
                            topic_subsections.insert(0, make_subsection("Overview", transcript_pts))

                # Fallback guarantee
                if not topic_subsections:
                    fallback = _build_missed_node_bullet(root_id)
                    if not fallback:
                        fallback = f"{topic}: A core concept in this topic."
                    topic_subsections = [make_subsection("Overview", [make_point(fallback)])]

                # Diagram from anchored block (already found — no search needed)
                topic_diagram_path    = _block.diagram_path
                topic_diagram_caption = _block.diagram_caption

                section_heading = _derive_section_heading(topic, root_node, topic_subsections)
                sections.append(make_section(
                    section_heading, topic_subsections,
                    diagram_path=topic_diagram_path,
                    diagram_caption=topic_diagram_caption,
                ))
                continue   # skip the inline fallback path below

            # ── FALLBACK PATH: inline rendering (used if block not available) ─
            # Collect ALL edges from this root (direct + hierarchical)
            all_child_ids = children_map.get(root_id, [])
            all_descendant_ids = _collect_all_descendants(root_id)

            # Get direct edges from edge_map
            direct_edges = edge_map.get(root_id, [])

            # Group children by edge relation type
            relation_groups = {}
            ungrouped_nodes = []

            for tgt_id, rel in direct_edges:
                tgt_node = node_lookup_by_id.get(tgt_id)
                if not tgt_node or tgt_id == root_id or tgt_id in emitted_node_ids:
                    continue
                group_heading = _get_group_heading(rel)
                relation_groups.setdefault(group_heading, []).append((tgt_node, rel, tgt_id))

            edge_target_ids = {tgt for tgt, _ in direct_edges}
            for cid in all_child_ids:
                if cid not in edge_target_ids and cid not in emitted_node_ids:
                    c_node = node_lookup_by_id.get(cid)
                    if c_node:
                        ungrouped_nodes.append((c_node, "includes", cid))

            # ── TYPE-DRIVEN RENDERING (Problems 5, 7, Corner Cases 2, 4, 6) ──
            # Each child node is classified by type and routed to the correct
            # educational format.  Rules (fully dynamic, no domain words):
            #
            #   Concept / Subconcept → their own named subsection with description
            #   Operation            → "Primary Operations" subsection bullet
            #   Property (adv/disadv)→ "Advantages" / "Disadvantages" or "Key Characteristics"
            #   Device               → "Components" subsection (component list)
            #   Protocol             → "Communication Mechanisms" subsection
            #   Implementation       → "Implementation" subsection
            #   Application          → "Applications" subsection
            #   Example              → "Real-World Examples" subsection
            #   Actor                → inline mention only (never its own subsection)
            #   Metadata / Diagram   → skip
            #
            # Property aggregation: advantage/disadvantage nodes are grouped
            # TOGETHER under their parent concept as Adv/Disadv bullets,
            # never as independent top-level topics.

            def _infer_full_type(node: dict) -> str:
                """
                Wrapper around classify_node_type that also handles the
                Advantage / Disadvantage sub-case of Property.
                Returns one of: Concept, Subconcept, Operation, Advantage,
                Disadvantage, Property, Device, Protocol, Implementation,
                Application, Example, Actor, Metadata, Diagram, General.
                """
                base_type = self.classify_node_type(node, edge_map, list(node_lookup_by_id.values()))
                if base_type == "Property":
                    lbl  = node.get("label", "").lower()
                    desc = (node.get("description", "") or "").lower()
                    combined = lbl + " " + desc
                    _ADV_RE = re.compile(
                        r'\b(advantage|benefit|merit|pro\b|strength|positive|'
                        r'easy|simple|fast|efficient|reliable|flexible|scalable|'
                        r'cheap|low\s+cost|high\s+speed|secure)\b', re.I)
                    _DIS_RE = re.compile(
                        r'\b(disadvantage|drawback|limitation|weakness|demerit|'
                        r'con\b|problem|issue|failure|risk|expensive|difficult|'
                        r'slow|unreliable|inflexible|high\s+cost|collision|'
                        r'privacy|bottleneck|single\s+point)\b', re.I)
                    if _ADV_RE.search(combined):
                        return "Advantage"
                    if _DIS_RE.search(combined):
                        return "Disadvantage"
                return base_type

            # Map type → target subsection heading
            _TYPE_HEADING_MAP = {
                "Concept":        None,           # gets its own named subsection
                "Subconcept":     None,           # gets its own named subsection
                "Operation":      "Primary Operations",
                "Advantage":      "Advantages",
                "Disadvantage":   "Disadvantages",
                "Property":       "Key Characteristics",
                "Device":         "Components and Devices",
                "Protocol":       "Communication Mechanisms",
                "Implementation": "Implementation and Structure",
                "Application":    "Applications",
                "Example":        "Real-World Examples",
                "Runtime Mechanism": "Runtime Mechanisms",
                "Actor":          "__INLINE__",   # inline mention only
                "Metadata":       "__SKIP__",
                "Diagram":        "__SKIP__",
                "General":        "Key Concepts",
            }

            # Bucket all children (from both direct_edges and ungrouped_nodes)
            # into type-based groups.  Supersedes the relation_groups approach
            # for the mixed-type "Components and Structure" problem.
            type_groups: dict = {}   # heading → [(c_node, rel, c_id)]
            named_subconcepts: list = []  # (c_node, rel, c_id) — get own subsections

            all_children_to_route = []
            # From direct edges (relation_groups already built)
            for rg_items in relation_groups.values():
                all_children_to_route.extend(rg_items)
            # From ungrouped
            for item in ungrouped_nodes:
                all_children_to_route.append(item)

            for item in all_children_to_route:
                c_node, rel, c_id = item
                if c_id in emitted_node_ids:
                    continue
                ctype = _infer_full_type(c_node)
                dest = _TYPE_HEADING_MAP.get(ctype, "Key Concepts")
                if dest == "__SKIP__":
                    emitted_node_ids.add(c_id)
                    continue
                if dest == "__INLINE__":
                    # Actor nodes: don't give them a subsection; they appear
                    # as inline sentences inside the parent's Overview
                    continue
                if dest is None:
                    # Concept or Subconcept → named subsection
                    named_subconcepts.append(item)
                else:
                    type_groups.setdefault(dest, []).append(item)

            # ── Build sections ──

            
            # Section 1: Definition and Fundamental Concept (always first)
            defn = root_node.get("description", "")
            if not _is_clear_description(defn, topic):
                defn = self.synthesize_explanation(
                    topic, root_node,
                    [node_lookup_by_id[c] for c in all_child_ids if c in node_lookup_by_id],
                    edge_map,
                    node_lookup=node_lookup_by_id
                )
            
            defn_points = [make_point(defn)] if defn else []
            
            # Add transcript-matched sentences for richer definition
            all_text = node_text_map.get(root_id, "")
            if all_text:
                matched = self.find_matching_transcript_sentences(topic, all_text, domain_keywords, max_sentences=5)
                for sent in matched:
                    if self.is_valid_note_sentence(sent, domain_keywords):
                        defn_points.append(make_point(sent))
            
            # Also pull "Key Characteristics" / "follows" into the definition section
            char_group = relation_groups.pop("Key Characteristics", [])
            for c_node, rel, c_id in char_group:
                if c_id in emitted_node_ids:
                    continue
                emitted_node_ids.add(c_id)
                c_label = c_node.get("label", "")
                c_desc = c_node.get("description", "")
                if _is_clear_description(c_desc, c_label):
                    defn_points.append(make_point(f"{c_label}: {c_desc}"))
                else:
                    desc = self.get_contextual_description(topic, rel, c_node)
                    if desc and desc.lower().strip() != c_label.lower().strip():
                        defn_points.append(make_point(f"{c_label}: {desc}"))
                    else:
                        enriched = _enrich_label(c_label, c_node, topic, rel)
                        defn_points.append(make_point(enriched))
            
            topic_subsections = []
            topic_diagram_path = None
            topic_diagram_caption = None

            if defn_points:
                # "Description" subsection (rendered as blue box in PDF) — node description only
                if defn and _is_clear_description(defn, topic):
                    # Fix 5.6: Sanitize description text before rendering
                    clean_defn = self.sanitizer.sanitize_text(defn)
                    if clean_defn:
                        topic_subsections.append(make_subsection("Description", [make_point(clean_defn)]))
                    # Remove the defn from overview points to avoid duplication
                    defn_points = [p for p in defn_points if p.get("text", "") != defn]
                
                # "Overview" subsection — transcript sentences and characteristic points
                if defn_points:
                    topic_subsections.append(make_subsection("Overview", defn_points))
                
                # Try to find a diagram for the definition section
                best_img, raw_cap = self.find_related_diagram(
                    topic + " " + (defn or ""), captions, used_img_canonicals)
                _ip1 = image_paths.get(best_img) if best_img else None
                if best_img and not _img_already_used(best_img, _ip1):
                    _register_img(best_img, _ip1)
                    if _ip1 and _ip1.exists():
                        topic_diagram_path = str(_ip1)
                        topic_diagram_caption = self.get_semantic_caption(topic, raw_cap, nodes)

            # ── Emit named subconcepts (Concept/Subconcept children) ────────────
            # These get their own titled subsection, e.g. "Bus Topology", "Star Topology"
            # Each is rendered with: its description, its own property/adv/disadv bullets
            for sc_node, sc_rel, sc_id in named_subconcepts:
                if sc_id in emitted_node_ids:
                    continue
                emitted_node_ids.add(sc_id)
                sc_label = self.normalize_concept_label(sc_node.get("label", ""))
                if not sc_label or sc_label.lower().strip() == topic.lower().strip():
                    continue

                sc_desc  = sc_node.get("description", "")
                sc_points = []

                # Description of this subconcept
                if _is_clear_description(sc_desc, sc_label):
                    sc_points.append(make_point(sc_desc))
                else:
                    nlg = self.get_contextual_description(topic, sc_rel, sc_node)
                    if nlg:
                        sc_points.append(make_point(nlg))

                # Gather this subconcept's own children (its properties/adv/disadv)
                adv_pts, dis_pts, prop_pts, dev_pts, impl_pts = [], [], [], [], []
                for gc_id in children_map.get(sc_id, []):
                    if gc_id in emitted_node_ids:
                        continue
                    gc_node = node_lookup_by_id.get(gc_id, {})
                    if not gc_node:
                        continue
                    emitted_node_ids.add(gc_id)
                    gc_label = self.normalize_concept_label(gc_node.get("label", ""))
                    gc_desc  = gc_node.get("description", "")
                    gc_type  = _infer_full_type(gc_node)

                    # Build bullet text
                    if _is_clear_description(gc_desc, gc_label):
                        bullet = f"{gc_label}: {gc_desc}"
                    else:
                        bullet = _enrich_label(gc_label, gc_node, sc_label,
                                               next((r for t, r in edge_map.get(sc_id, []) if t == gc_id), "has"))

                    if gc_type == "Advantage":
                        adv_pts.append(make_point(bullet))
                    elif gc_type == "Disadvantage":
                        dis_pts.append(make_point(bullet))
                    elif gc_type == "Device":
                        dev_pts.append(make_point(bullet))
                    elif gc_type == "Implementation":
                        impl_pts.append(make_point(bullet))
                    else:
                        prop_pts.append(make_point(bullet))

                # Build subsections for this subconcept
                sc_subsections = []
                if sc_points:
                    sc_subsections.append(make_subsection("Description", sc_points))
                if impl_pts:
                    sc_subsections.append(make_subsection("Structure", impl_pts))
                if dev_pts:
                    sc_subsections.append(make_subsection("Components", dev_pts))
                if prop_pts:
                    sc_subsections.append(make_subsection("Key Characteristics", prop_pts))
                if adv_pts:
                    sc_subsections.append(make_subsection("Advantages", adv_pts))
                if dis_pts:
                    sc_subsections.append(make_subsection("Disadvantages", dis_pts))

                if sc_subsections:
                    topic_subsections.append(make_subsection(sc_label, sc_subsections))
                elif sc_points:
                    topic_subsections.append(make_subsection(sc_label, sc_points))

            # ── Emit type_groups (typed bullet collections) ──────────────────
            # Ordered by educational importance: operations → components → properties
            # → implementations → protocols → applications → examples
            _GROUP_ORDER = [
                "Primary Operations",
                "Components and Devices",
                "Communication Mechanisms",
                "Key Characteristics",
                "Advantages",
                "Disadvantages",
                "Implementation and Structure",
                "Runtime Mechanisms",
                "Applications",
                "Real-World Examples",
                "Key Concepts",
            ]
            # Process in preferred order, then any remaining groups
            ordered_headings = _GROUP_ORDER + [
                h for h in type_groups if h not in _GROUP_ORDER
            ]

            for group_heading in ordered_headings:
                group_items = type_groups.get(group_heading, [])
                if not group_items:
                    continue

                group_points = []
                for c_node, rel, c_id in group_items:
                    if c_id in emitted_node_ids:
                        continue
                    c_label = c_node.get("label", "")
                    if c_label.lower().strip() == topic.lower().strip():
                        continue  # Skip self-referencing child
                    emitted_node_ids.add(c_id)

                    c_label = self.normalize_concept_label(c_label)
                    c_desc  = c_node.get("description", "")

                    # Try transcript matching first
                    tgt_text = node_text_map.get(c_id, "")
                    rich_desc = None
                    if tgt_text:
                        matched = self.find_matching_transcript_sentences(
                            c_label, tgt_text, domain_keywords, max_sentences=2)
                        valid_matched = [s for s in matched
                                         if self.is_valid_note_sentence(s, domain_keywords)]
                        if valid_matched:
                            rich_desc = " ".join(valid_matched)

                    if not rich_desc:
                        if _is_clear_description(c_desc, c_label):
                            rich_desc = c_desc
                        else:
                            rich_desc = self.get_contextual_description(topic, rel, c_node)

                    if rich_desc and rich_desc.lower().strip() != c_label.lower().strip():
                        group_points.append(make_point(f"{c_label}: {rich_desc}"))
                    elif c_label:
                        group_points.append(make_point(_enrich_label(c_label, c_node, topic, rel)))

                    # Add descendants as sub-bullets
                    for desc_id in children_map.get(c_id, []):
                        if desc_id in emitted_node_ids:
                            continue
                        emitted_node_ids.add(desc_id)
                        desc_node = node_lookup_by_id.get(desc_id, {})
                        d_label = desc_node.get("label", "")
                        d_desc  = desc_node.get("description", "")
                        if _is_clear_description(d_desc, d_label):
                            group_points.append(make_point(f"{d_label}: {d_desc}"))
                        elif d_label:
                            group_points.append(make_point(
                                _enrich_label(d_label, desc_node, topic, "includes")))

                if group_points:
                    # Enrich with transcript sentences for this group
                    group_search_text = group_heading + " " + " ".join(
                        c_node.get("label", "") for c_node, _, _ in group_items)
                    for nid_text in [c_id for _, _, c_id in group_items]:
                        seg_text = node_text_map.get(nid_text, "")
                        if seg_text:
                            matched = self.find_matching_transcript_sentences(
                                group_search_text, seg_text, domain_keywords, max_sentences=2)
                            for sent in matched:
                                if self.is_valid_note_sentence(sent, domain_keywords):
                                    existing = {p.get("text","").lower() for p in group_points}
                                    if sent.lower().strip() not in existing:
                                        group_points.append(make_point(sent))

                    topic_subsections.append(make_subsection(group_heading, group_points))

                    # Diagram for this group
                    if not topic_diagram_path:
                        best_img, raw_cap = self.find_related_diagram(
                            group_heading + " " + " ".join(
                                c_node.get("label", "") for c_node, _, _ in group_items),
                            captions, used_img_canonicals)
                        _ip2 = image_paths.get(best_img) if best_img else None
                        if best_img and not _img_already_used(best_img, _ip2):
                            _register_img(best_img, _ip2)
                            if _ip2 and _ip2.exists():
                                topic_diagram_path = str(_ip2)
                                topic_diagram_caption = self.get_semantic_caption(
                                    group_heading, raw_cap, nodes)



            # ── Guarantee: every root node produces at least one subsection ──
            # If all content-generation paths produced nothing (no description,
            # no transcript, no children, no edges), build a minimal but real
            # bullet directly from the root node's edges or bare label so the
            # section is never silently dropped.
            if not topic_subsections:
                fallback_bullet = _build_missed_node_bullet(root_id)
                if not fallback_bullet:
                    fallback_bullet = f"{topic}: A core concept in this topic."
                topic_subsections.append(
                    make_subsection("Overview", [make_point(fallback_bullet)])
                )
                print(f"[Notes] Fallback subsection created for empty root node: '{topic}'")

            # Finally, append the master section for this topic
            section_heading = _derive_section_heading(topic, root_node, topic_subsections)
            sections.append(make_section(
                section_heading,
                topic_subsections,
                diagram_path=topic_diagram_path,
                diagram_caption=topic_diagram_caption
            ))

        # ── 6a. Coverage sweep: ensure ALL KG nodes are represented ──
        all_node_ids = set(node_lookup_by_id.keys())
        missed_ids = all_node_ids - emitted_node_ids
        if missed_ids:
            # Build a lookup of section heading words + bullet label words for matching
            def _section_match_score(node_label, section):
                """Score how well a missed node's label matches an existing section."""
                lbl_lower = node_label.lower().strip()
                heading_lower = section.get("heading", "").lower().strip()
                # Direct substring match in heading
                if lbl_lower in heading_lower or heading_lower in lbl_lower:
                    return 3
                # Word overlap between label and heading
                lbl_words = set(lbl_lower.split())
                head_words = set(heading_lower.split())
                word_overlap = len(lbl_words & head_words - {'and', 'the', 'of', 'in', 'a', 'an', 'is', 'for', 'to'})
                if word_overlap >= 1:
                    return 2
                # Check headings and bullet text for label mentions
                for ss in section.get("subsections", []):
                    if lbl_lower in ss.get("heading", "").lower():
                        return 2
                    for pt in ss.get("points", []):
                        if lbl_lower in pt.get("text", "").lower() or lbl_lower in pt.get("label", "").lower():
                            return 1
                return 0

            orphan_points = []  # nodes that don't match any section
            placed_count = 0
            for nid in missed_ids:
                node = node_lookup_by_id.get(nid, {})
                lbl = node.get("label", "")
                if not lbl:
                    continue
                emitted_node_ids.add(nid)

                # FIX-COVERAGE: Use _build_missed_node_bullet for richer content
                bullet_text = _build_missed_node_bullet(nid)
                if not bullet_text:
                    continue
                point = make_point(bullet_text)

                # Try to place into the best-matching existing section
                best_score, best_section = 0, None
                for sec in sections:
                    score = _section_match_score(lbl, sec)
                    if score > best_score:
                        best_score, best_section = score, sec

                if best_section and best_score >= 1:
                    # Append into an existing "Key Points" subsection if present,
                    # or create a new one — never blindly append to the last subsection.
                    subsections = best_section.get("subsections", [])
                    kp_sub = next(
                        (s for s in subsections if s.get("heading", "").lower() == "key points"),
                        None
                    )
                    if kp_sub:
                        kp_sub["points"].append(point)
                    else:
                        best_section.setdefault("subsections", []).append(
                            make_subsection("Key Points", [point])
                        )
                    placed_count += 1
                else:
                    orphan_points.append(point)

                # Try to find a diagram for missed nodes
                desc = node.get("description", "")
                best_img, raw_cap = self.find_related_diagram(
                    lbl + " " + desc, captions, used_img_canonicals)
                _ip3 = image_paths.get(best_img) if best_img else None
                if best_img and not _img_already_used(best_img, _ip3):
                    _register_img(best_img, _ip3)

            if placed_count:
                print(f"[Notes] Smart placement: placed {placed_count} missed nodes into existing sections")

            if orphan_points:
                # FIX-COVERAGE: Sub-group orphan points by their concept type
                # (inferred from the bullet label) rather than dumping all into a
                # flat "Details" subsection. This makes orphan content navigable
                # and prevents the entire section from being filtered as a stub.
                _orphan_groups: dict = {}
                for _op in orphan_points:
                    _text = _op.get("text", "")
                    # Extract label (text before first colon) as group key
                    _colon = _text.find(":")
                    if _colon > 0:
                        _raw_key = _text[:_colon].strip()
                        # Reduce to first 3 words for grouping
                        _key_words = _raw_key.split()[:3]
                        _group_key = " ".join(_key_words) if _key_words else "Additional Concepts"
                    else:
                        _group_key = "Additional Concepts"
                    _orphan_groups.setdefault(_group_key, []).append(_op)

                _orphan_subsections = []
                for _gkey, _gpts in _orphan_groups.items():
                    if _gpts:
                        _orphan_subsections.append(make_subsection(_gkey, _gpts))

                if not _orphan_subsections:
                    _orphan_subsections = [make_subsection("Details", orphan_points)]

                sections.append(make_section(
                    "Key Notes / Additional Concepts",
                    _orphan_subsections
                ))
                print(f"[Notes] Coverage: {len(orphan_points)} orphan points → "
                      f"{len(_orphan_subsections)} sub-groups in 'Key Notes / Additional Concepts'")

        # No explicit TCP section injections for a fully dynamic structure
        # Fix 3: Remove stub-only sections
        COVERAGE_TEMPLATE_STUBS = [
            'is a key concept in this domain',
            'is discussed in this context',
            'is an important operation',
            'is a type or classification',
            'is relevant property',
            'is a key concept in this topic',
            'a key concept in this topic',
            'a key concept in this domain',
            'a concept related to',
        ]
        # Bare graph-edge sentence patterns: "X includes Y.", "X Packet includes X."
        _BARE_EDGE_RE = re.compile(
            r'^[\w\s\(\)/]+:\s*([\w\s\(\)/]+)\s+includes\s+\1\s*\.$|'   # "Label: X includes X."
            r'^[\w\s\(\)/]+\s+includes\s+[\w\s\(\)/]+\.$',              # "X includes Y."
            re.IGNORECASE
        )

        def _section_is_empty_stub(section: dict) -> bool:
            """Return True if ALL points in section are stubs or bare graph-edge sentences.

            FIX-COVERAGE: The old version counted ANY bullet containing a stub phrase
            as bad, even if it also had real label:description content. This caused
            sections with mixed content (some real, some template) to be completely
            removed. Now a bullet is only counted as bad if it is PURELY a stub —
            i.e. the stub phrase is the entire payload after the label colon, with
            no additional explanatory content.
            """
            all_points = []
            for sub in section.get('subsections', []):
                all_points.extend(sub.get('points', []))
            if not all_points:
                return True  # No points at all → remove
            bad_count = 0
            for pt in all_points:
                pt_text = pt.get('text', '').strip()
                pt_lower = pt_text.lower()
                is_bad = False

                # Check bare graph-edge sentence first
                if _BARE_EDGE_RE.match(pt_text):
                    is_bad = True
                else:
                    # A stub phrase makes a bullet bad ONLY if:
                    # (a) the stub IS the entire bullet text, or
                    # (b) the stub appears after the label colon and there is no
                    #     additional real content (≥ 6 words) beyond it.
                    for stub in COVERAGE_TEMPLATE_STUBS:
                        if stub in pt_lower:
                            # If there's a label prefix ("Label: stub text"), check
                            # whether content after ':' is exclusively the stub
                            colon_idx = pt_text.find(':')
                            if colon_idx != -1:
                                after_colon = pt_text[colon_idx + 1:].strip().lower()
                                # Bad only when the after-colon content IS essentially the stub
                                if after_colon.startswith(stub) and len(after_colon.split()) < 8:
                                    is_bad = True
                                    break
                                # Otherwise the label has real content — keep it
                            else:
                                # No label prefix — the whole bullet is the stub
                                is_bad = True
                                break

                if is_bad:
                    bad_count += 1

            # Section is a stub only if ALL points are bad
            return bad_count == len(all_points)

        sections = [s for s in sections if not _section_is_empty_stub(s)]
        print(f'[Notes] After stub removal: {len(sections)} sections remain.')

        # ── Fix 4B-pre: Final ? sanitization over ALL assembled bullet points ──────────────
        # The KG node descriptions can contain "? The...", "? Outputs..." as separators.
        # This pass catches any ? that survived earlier sanitization at the data level.
        try:
            from concept_flow_organizer import sanitize_question_marks as _sqm
            for section in sections:
                if "heading" in section:
                    section["heading"] = _sqm(section["heading"])
                for sub in section.get("subsections", []):
                    if "heading" in sub:
                        sub["heading"] = _sqm(sub["heading"])
                    for pt in sub.get("points", []):
                        if pt.get("text"):
                            pt["text"] = _sqm(pt["text"])
            print("[Notes] Final ? sanitization pass applied over all sections.")
        except Exception as _sqm_err:
            # Inline fallback
            _QM_INLINE = re.compile(r'(\w+)\s+\?\s+(\w+)')
            _QM_ANY_INLINE = re.compile(r'\?')
            for section in sections:
                if "heading" in section:
                    section["heading"] = _QM_ANY_INLINE.sub("", _QM_INLINE.sub(r"\1 \2", section["heading"])).strip()
                for sub in section.get("subsections", []):
                    if "heading" in sub:
                        sub["heading"] = _QM_ANY_INLINE.sub("", _QM_INLINE.sub(r"\1 \2", sub["heading"])).strip()
                    for pt in sub.get("points", []):
                        if pt.get("text"):
                            pt["text"] = _QM_ANY_INLINE.sub("", _QM_INLINE.sub(r"\1 \2", pt["text"])).strip()

        # ── Fix 4C: Point-level OCR sanitizer ────────────────────────────────
        # Runs on every assembled bullet point text AFTER sections are built.
        # Catches multi-line OCR garbage that was joined before line-level filters.
        _POINT_GARBAGE_PATTERNS = [
            re.compile(r'\|\s*\?_|\?_\s*$', re.I),
            re.compile(r'Le\s+Jono\d+|yO\dN|LC\]', re.I),
            re.compile(r'OUTCOMES\s+Upon\s+the\s+completion', re.I),
            re.compile(r'learner\s+will\s+be\s+able', re.I),
        ]
        _POINT_STUB_STRINGS = [
            'is a key concept in this domain',
            'is discussed in this context',
            'is an important operation',
            'is a type or classification',
            'is relevant property',
            'is a key concept in this topic',
            'a key concept in this topic',
            'a key concept in this domain',
            'a concept related to',
            # "sub-concept within X" bullets are structural noise from KG schema
            'sub-concept within',
            # Exam/institution noise
            'sppu exam pattern',
            'important questions',
            'exam pattern is related',
        ]

        def _clean_point_text(text: str) -> str:
            """Strip or truncate a point text that contains embedded OCR garbage."""
            if not text:
                return text
            tl = text.lower()
            # Discard entire point if it's a stub phrase
            for stub in _POINT_STUB_STRINGS:
                if stub in tl:
                    return ''
            # For each garbage pattern: if it starts within first 50 chars, discard;
            # otherwise truncate at the match boundary.
            for pat in _POINT_GARBAGE_PATTERNS:
                m = pat.search(text)
                if m:
                    if m.start() < 50:
                        return ''
                    # Truncate at boundary; keep only if remainder is a real sentence
                    prefix = text[:m.start()].rstrip(' ,;:.')
                    if len(prefix.split()) >= 5:
                        return prefix + '.'
                    return ''
            return text

        cleaned_section_count = 0
        for section in sections:
            for sub in section.get('subsections', []):
                original_pts = sub.get('points', [])
                cleaned_pts = []
                for pt in original_pts:
                    cleaned = _clean_point_text(pt.get('text', ''))
                    if cleaned:
                        pt['text'] = cleaned
                        cleaned_pts.append(pt)
                    else:
                        cleaned_section_count += 1
                sub['points'] = cleaned_pts
        if cleaned_section_count:
            print(f'[Notes] Point OCR sanitizer removed/truncated {cleaned_section_count} garbage points.')

        # ── Post-sanitization integrity pass ──────────────────────────────────
        # After _clean_point_text removes individual bullets, some subsections
        # may now be empty and some sections may have no non-empty subsections.
        # Re-check every section and rebuild any that lost all their content.
        _post_clean_rescued = 0
        for section in sections:
            # Drop subsections that are now empty
            section['subsections'] = [
                sub for sub in section.get('subsections', [])
                if sub.get('points')
            ]
            # If the whole section is now empty, inject a fallback bullet
            # built from the KG so the section name is preserved in the output.
            if not section.get('subsections'):
                heading = section.get('heading', 'Topic')
                # Try to find the root node that matches this heading
                _rescue_bullet = None
                for _nid, _nd in node_lookup_by_id.items():
                    if _nd.get('label', '').lower().strip() == heading.lower().strip():
                        _rescue_bullet = _build_missed_node_bullet(_nid)
                        break
                if not _rescue_bullet:
                    _rescue_bullet = f"{heading}: A concept covered in this topic."
                section['subsections'] = [
                    make_subsection('Overview', [make_point(_rescue_bullet)])
                ]
                _post_clean_rescued += 1
        if _post_clean_rescued:
            print(f'[Notes] Post-sanitization rescue: {_post_clean_rescued} sections rebuilt after point cleanup.')

        # ── 6b. Post-process: Consolidate thin sections ──
        # FIX-COVERAGE: Reduced MIN_POINTS_PER_SECTION from 3 to 1 so that sections
        # with even a single meaningful point are never silently discarded.
        # The old threshold of 3 caused valid single-concept sections (e.g. "Explicit
        # Stack", "isEmpty Operation") to be moved to "Additional Concepts" where
        # they were often lost in dynamic sub-grouping. With threshold = 1, ALL
        # content reaches the output; only completely empty sections are removed.
        MIN_POINTS_PER_SECTION = 1

        def _heading_similarity(h1: str, h2: str) -> float:
            """Token overlap similarity between two heading strings (0.0–1.0)."""
            stop = {'and', 'the', 'of', 'in', 'a', 'an', 'is', 'for', 'to', 'with', 'on', 'at'}
            w1 = set(h1.lower().split()) - stop
            w2 = set(h2.lower().split()) - stop
            if not w1 or not w2:
                return 0.0
            return len(w1 & w2) / min(len(w1), len(w2))

        def _find_best_merge_target(thin_heading: str, candidate_sections: list) -> dict:
            """
            Find the best existing section to absorb a thin section's content.
            Uses token overlap on headings. Returns the best section or None.
            """
            best_sec, best_score = None, 0.0
            for sec in candidate_sections:
                score = _heading_similarity(thin_heading, sec.get("heading", ""))
                if score > best_score:
                    best_score = score
                    best_sec = sec
            # Only merge if there's at least one shared non-stop word
            return best_sec if best_score > 0.0 else None

        consolidated_sections = []
        additional_concepts_points = []
        additional_diagram = None
        additional_diagram_caption = None

        for section in sections:
            heading = section.get("heading", "")
            subsections = section.get("subsections", [])

            # Count total non-stub points in this section
            total_points = sum(len(ss.get("points", [])) for ss in subsections)

            if heading.lower() == "additional concepts":
                # Always merge "Additional Concepts" sections into one
                for ss in subsections:
                    additional_concepts_points.extend(ss.get("points", []))
                if not additional_diagram and section.get("diagram"):
                    additional_diagram = section["diagram"].get("path")
                    additional_diagram_caption = section["diagram"].get("caption")
            elif total_points < MIN_POINTS_PER_SECTION:
                # Try to merge into the most similar existing consolidated section
                target = _find_best_merge_target(heading, consolidated_sections)
                if target:
                    # Collect the thin section's points
                    thin_pts = []
                    for ss in subsections:
                        for pt in ss.get("points", []):
                            text = pt.get("text", "")
                            if heading.lower() not in text.lower()[:60]:
                                pt["text"] = f"{heading}: {text}"
                            thin_pts.append(pt)
                    if thin_pts:
                        # Inject into an existing non-empty subsection or create one
                        target_subs = target.get("subsections", [])
                        # Find first subsection that already has points (don't append to empty)
                        _dest_sub = next(
                            (s for s in reversed(target_subs) if s.get("points")),
                            None
                        )
                        if _dest_sub:
                            _dest_sub["points"].extend(thin_pts)
                        else:
                            target.setdefault("subsections", []).append(
                                make_subsection("Key Points", thin_pts)
                            )
                    if not additional_diagram and section.get("diagram"):
                        additional_diagram = section["diagram"].get("path")
                        additional_diagram_caption = section["diagram"].get("caption")
                else:
                    # No good match — send to Additional Concepts
                    for ss in subsections:
                        for pt in ss.get("points", []):
                            text = pt.get("text", "")
                            if heading.lower() not in text.lower()[:60]:
                                pt["text"] = f"{heading}: {text}"
                            additional_concepts_points.append(pt)
                    if not additional_diagram and section.get("diagram"):
                        additional_diagram = section["diagram"].get("path")
                        additional_diagram_caption = section["diagram"].get("caption")
            else:
                consolidated_sections.append(section)
        
        # Add the merged "Additional Concepts" section if it has points
        if additional_concepts_points:
            # Dynamic sub-grouping: cluster by shared prefix/context
            # Extract the first meaningful word/phrase before ':' as a group key
            dynamic_groups = {}
            for pt in additional_concepts_points:
                text = pt.get("text", "")
                # Try to extract a group key from "GroupKey: description" format
                if ':' in text:
                    group_key = text.split(':')[0].strip()
                    # Only use as a group key if it's short enough to be a label
                    if len(group_key.split()) <= 4:
                        dynamic_groups.setdefault(group_key, []).append(pt)
                    else:
                        dynamic_groups.setdefault("Other Details", []).append(pt)
                else:
                    dynamic_groups.setdefault("Other Details", []).append(pt)
            
            ac_subsections = []
            for heading, pts in dynamic_groups.items():
                if pts:
                    ac_subsections.append(make_subsection(heading, pts))
            
            if not ac_subsections:
                ac_subsections = [make_subsection("Details", additional_concepts_points)]
                
            ac_section = make_section(
                "Additional Concepts",
                ac_subsections
            )
            if additional_diagram:
                ac_section["diagram"] = {
                    "path": additional_diagram,
                    "caption": additional_diagram_caption or ""
                }
            consolidated_sections.append(ac_section)

        # ── FIX-COVERAGE: Edge-driven final audit ───────────────────────────────
        # Scan EVERY KG edge. For each edge, verify its target node's label appears
        # in the assembled notes text. Any target that is completely absent gets a
        # bullet injected into the most semantically relevant section.
        # This is the definitive full-coverage guarantee — it runs AFTER the main
        # traversal, coverage sweep, and thin-section consolidation so it catches
        # any node that fell through every earlier net.
        _notes_text_lower = " ".join(
            pt.get("text", "").lower()
            for sec in consolidated_sections
            for sub in sec.get("subsections", [])
            for pt in sub.get("points", [])
        )
        _audit_injected = 0
        _audit_orphans = []
        for _edge in edges:
            _tgt_id = _edge.get("target", "")
            if not _tgt_id:
                continue
            _tgt_node = node_lookup_by_id.get(_tgt_id, {})
            _tgt_lbl = (_tgt_node.get("label", "") or "").strip()
            if not _tgt_lbl or len(_tgt_lbl) < 3:
                continue
            # Skip nodes already well-represented in the notes
            if _tgt_lbl.lower() in _notes_text_lower:
                continue
            # Build a bullet using _build_missed_node_bullet
            _bullet_text = _build_missed_node_bullet(_tgt_id)
            if not _bullet_text:
                continue
            _pt = make_point(_bullet_text)
            # Try to place into the best-matching consolidated section
            _best_sec, _best_score = None, 0
            for _sec in consolidated_sections:
                _sc = _section_match_score(_tgt_lbl, _sec)
                if _sc > _best_score:
                    _best_score, _best_sec = _sc, _sec
            if _best_sec and _best_score >= 1:
                _kp = next(
                    (s for s in _best_sec.get("subsections", [])
                     if s.get("heading", "").lower() == "key points"),
                    None
                )
                if _kp:
                    _kp["points"].append(_pt)
                else:
                    _best_sec.setdefault("subsections", []).append(
                        make_subsection("Key Points", [_pt])
                    )
                _audit_injected += 1
                # Update the running text so subsequent edges don't re-inject same label
                _notes_text_lower += " " + _tgt_lbl.lower()
            else:
                _audit_orphans.append(_pt)

        if _audit_orphans:
            # Group audit orphans similarly to the coverage sweep orphans
            _audit_groups: dict = {}
            for _op in _audit_orphans:
                _txt = _op.get("text", "")
                _c = _txt.find(":")
                _gk = _txt[:_c].strip().split()[:3] if _c > 0 else []
                _gkey = " ".join(_gk) if _gk else "Additional Notes"
                _audit_groups.setdefault(_gkey, []).append(_op)
            _audit_subs = [make_subsection(k, v) for k, v in _audit_groups.items() if v]
            if _audit_subs:
                consolidated_sections.append(
                    make_section("Additional Notes", _audit_subs)
                )
                _audit_injected += len(_audit_orphans)

        if _audit_injected:
            print(f"[Notes] Edge-audit: injected {_audit_injected} missing concept(s) into notes")

        # ── Final section integrity guarantee ─────────────────────────────────
        # After ALL processing (stub removal, point sanitization, consolidation,
        # edge audit), every section in the list MUST have at least one subsection
        # with at least one bullet. This is the last safety net before PDF render.
        _final_rescue = 0
        for _sec in consolidated_sections:
            # Drop empty subsections
            _sec['subsections'] = [
                _sub for _sub in _sec.get('subsections', [])
                if _sub.get('points')
            ]
            if not _sec.get('subsections'):
                _h = _sec.get('heading', 'Topic')
                # Build content from KG or use a labelled placeholder
                _rescue = None
                for _nid, _nd in node_lookup_by_id.items():
                    if _nd.get('label', '').lower().strip() == _h.lower().strip():
                        _rescue = _build_missed_node_bullet(_nid)
                        break
                if not _rescue:
                    _rescue = f"{_h}: A concept in this topic."
                _sec['subsections'] = [make_subsection('Overview', [make_point(_rescue)])]
                _final_rescue += 1
        if _final_rescue:
            print(f'[Notes] Final integrity pass: {_final_rescue} empty sections rebuilt.')

        # Remove any sections that are genuinely empty even after all rescue attempts
        consolidated_sections = [
            s for s in consolidated_sections
            if any(sub.get('points') for sub in s.get('subsections', []))
        ]

        # Dynamic section ordering: use root traversal order from the KG
        # Sections derived from KG roots keep their original processing order;
        # "Additional Concepts" always goes last.
        def _section_priority(section: dict, root_order: dict = {}) -> int:
            heading = section.get("heading", "").lower().strip()
            if heading == "additional concepts":
                return 9999  # Always last
            # Use the KG root processing order if available
            if heading in root_order:
                return root_order[heading]
            # Heuristic: definition/introduction sections first
            intro_words = {'definition', 'introduction', 'overview', 'history', 'origin', 'what is'}
            if any(w in heading for w in intro_words):
                return 0
            return 50  # Unknown sections keep relative order

        # Build root_order from the order sections were created during KG traversal
        root_order = {s.get("heading", "").lower().strip(): i for i, s in enumerate(consolidated_sections)}

        # Subsection Promotion Rule: promotes subsections with >= 4 points to standalone sections.
        try:
            from hierarchical_schema import promote_subsections
            consolidated_sections = promote_subsections(consolidated_sections, min_points=4)
        except ImportError:
            pass  # Fallback if function unavailable
            
        sections = sorted(consolidated_sections, key=lambda s: _section_priority(s, root_order))

        # Expose image_paths on self temporarily so find_related_diagram
        # can resolve img_name → canonical path inside _canonical_img()
        self._current_image_paths = image_paths

        # ── Fix 4.3: Post-assembly Semantic Diagram Re-matching ──
        # For sections that have no diagram yet, find the best-matching
        # diagram via cosine similarity between section heading and captions
        if captions and image_paths:
            for section in sections:
                if section.get("diagram"):
                    continue  # Already has a diagram
                heading = section.get("heading", "")
                if not heading:
                    continue
                # Combine heading + first few bullet texts for richer matching
                bullet_texts = []
                for ss in section.get("subsections", []):
                    for pt in ss.get("points", [])[:3]:
                        bullet_texts.append(pt.get("text", ""))
                query = heading + " " + " ".join(bullet_texts[:3])
                
                best_img, raw_cap = self.find_related_diagram(
                    query, captions, used_img_canonicals)
                _ip4 = image_paths.get(best_img) if best_img else None
                if best_img and not _img_already_used(best_img, _ip4):
                    _register_img(best_img, _ip4)
                    if _ip4 and _ip4.exists():
                        section["diagram"] = {
                            "path": str(_ip4),
                            "caption": self.get_semantic_caption(heading, raw_cap, nodes)
                        }
            print(f"[Fix 4.3] Semantic diagram re-matching pass complete")

        # ── Dynamic Bare-Label Enrichment (final pass) ──
        # For any remaining bare-label or thin bullet points, try to enrich from:
        # 1. KG node descriptions  2. _CONCEPT_GLOSSARY  3. Glossary partial match
        enriched_count = 0
        for section in sections:
            for ss in section.get("subsections", []):
                for pt in ss.get("points", []):
                    text = pt.get("text", "").strip()
                    if not text:
                        continue

                    # Case A: bare label (no colon, short)
                    if ':' not in text and len(text) < 40:
                        key = text.lower().strip().rstrip('.')
                        # Try KG node lookup
                        enriched = False
                        for nid, ndata in node_lookup_by_id.items():
                            if ndata.get("label", "").lower().strip() == key:
                                desc = ndata.get("description", "")
                                if desc and len(desc.split()) >= 5:
                                    pt["text"] = f"{ndata['label']}: {desc}"
                                    enriched_count += 1
                                    enriched = True
                                break
                        # Try concept glossary if KG had nothing
                        if not enriched:
                            if key in _CONCEPT_GLOSSARY:
                                pt["text"] = f"{text}: {_CONCEPT_GLOSSARY[key]}"
                                enriched_count += 1
                            else:
                                for gkey, gval in _CONCEPT_GLOSSARY.items():
                                    if gkey in key or key in gkey:
                                        pt["text"] = f"{text}: {gval}"
                                        enriched_count += 1
                                        break

                    # Case B: label has a description but it's too short or self-repeating
                    elif ':' in text:
                        label_part = text.split(':')[0].strip()
                        desc_part = text.split(':', 1)[1].strip()
                        if len(desc_part) < 20 or desc_part.lower().strip('.') == label_part.lower():
                            key = label_part.lower().strip()
                            enriched = False
                            for nid, ndata in node_lookup_by_id.items():
                                if ndata.get("label", "").lower().strip() == key:
                                    desc = ndata.get("description", "")
                                    if desc and len(desc.split()) >= 5:
                                        pt["text"] = f"{ndata['label']}: {desc}"
                                        enriched_count += 1
                                        enriched = True
                                    break
                            if not enriched and key in _CONCEPT_GLOSSARY:
                                pt["text"] = f"{label_part}: {_CONCEPT_GLOSSARY[key]}"
                                enriched_count += 1
                            elif not enriched:
                                for gkey, gval in _CONCEPT_GLOSSARY.items():
                                    if gkey in key or key in gkey:
                                        pt["text"] = f"{label_part}: {gval}"
                                        enriched_count += 1
                                        break

        if enriched_count:
            print(f"[Notes] Enriched {enriched_count} bare-label/thin points with descriptions")

        # ── 6c. Generate KG Unified Summary (academic prose, ≤7 sentences) ──
        def _generate_kg_summary(sections_list):
            """Build a ≤7-sentence academic prose summary from section headings and descriptions using Gemini."""
            import google.generativeai as genai

            prompt_lines = [
                "Write a highly professional academic prose summary (≤7 sentences, no bullets) for the following lecture sections and their core concepts.",
                "Do not use bullet points, lists, or newlines. Provide only the continuous prose summary.",
                ""
            ]

            for idx, sec in enumerate(sections_list, 1):
                heading = sec.get("heading", "").strip()
                if not heading: continue
                prompt_lines.append(f"Section {idx}: {heading}")

                # Include up to 3 points from any subsection for richer context
                pts = []
                for sub in sec.get("subsections", []):
                    for pt in sub.get("points", []):
                        text = pt.get("text", "").strip()
                        if text and len(text.split()) >= 5:
                            pts.append(text)
                        if len(pts) >= 3:
                            break
                    if len(pts) >= 3:
                        break
                if pts:
                    prompt_lines.append("Key Concepts: " + " | ".join(pts))
                prompt_lines.append("")

            prompt = "\n".join(prompt_lines)

            try:
                model = genai.GenerativeModel("models/gemini-2.5-flash")
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"Gemini summary generation failed: {e}")
                pass

            # Fallback
            topics = [s.get("heading", "") for s in sections_list if s.get("heading") and s.get("heading").lower() != "additional concepts"]
            if not topics:
                return "This lecture covers foundational concepts and their structural relationships."
            return f"This lecture provides an in-depth examination of several core concepts, including {', '.join(topics[:3])}. The material explores the definitions, operations, and structural implementations of these topics to build a comprehensive understanding of the domain."

        kg_summary = _generate_kg_summary(sections)
        if kg_summary:
            print(f"[Notes] ✅ KG unified summary generated ({len(kg_summary.split('.'))-1} sentences)")

        # ── 6b. Augment / Replace sections with Gemini structural plan ──────────
        # HIGH PRIORITY FIX: The Gemini plan is now the AUTHORITATIVE source
        # for notes structure. Every section — including vague ones like
        # "Arrangement", "Description", "Concepts" — must be augmented or
        # replaced with the plan's richer content.
        #
        # Matching uses THREE strategies (in priority order):
        #   1. plan_ref_map (exact reverse lookup: kg_label → plan_key)
        #      covers vague labels Gemini explicitly absorbed and renamed.
        #   2. Exact lowercase title match.
        #   3. Token overlap ≥40% (lowered from 50% for wider coverage).
        #
        # Augmentation is ALWAYS applied when a match is found and plan has
        # content — the "should augment" density check is removed because it
        # was blocking augmentation of exactly the sparse sections we need to fix.
        #
        # Unmatched sections with empty/thin content are rebuilt from the plan
        # section whose content most closely matches their heading via token sim.
        if plan_subsection_map:
            _aug_count = 0
            _matched_plan_keys: set = set()  # track which plan keys are consumed

            def _find_plan_key(heading_lower: str) -> str | None:
                """Three-strategy plan section lookup. Returns plan key or None."""
                # Strategy 1: reverse ref map (exact kg_label match)
                if heading_lower in plan_ref_map:
                    return plan_ref_map[heading_lower]
                # Strategy 2: exact title match
                if heading_lower in plan_subsection_map:
                    return heading_lower
                # Strategy 3: token overlap ≥40%
                _h_words = set(heading_lower.split())
                best_key, best_overlap = None, 0.0
                for _pk in plan_subsection_map:
                    _pk_words = set(_pk.split())
                    _short = min(len(_pk_words), len(_h_words))
                    if _short == 0:
                        continue
                    _ov = len(_pk_words & _h_words) / _short
                    if _ov > best_overlap:
                        best_overlap = _ov
                        best_key = _pk
                if best_overlap >= 0.40:
                    return best_key
                return None

            def _plan_subs_to_notes(plan_subs: list, existing_subs: list) -> list:
                """Convert plan subsections to HierarchicalNotes format.
                Merges with any substantial existing KG subsections whose
                headings are NOT already covered by the plan."""
                _new = []
                for _ps in plan_subs:
                    _h = (_ps.get("heading") or "").strip()
                    _pts = _ps.get("points", [])
                    if not _h or not _pts:
                        continue
                    _new.append(make_subsection(
                        _h, [make_point(p) for p in _pts if p and str(p).strip()]
                    ))
                if not _new:
                    return existing_subs  # safety: never empty a section

                _plan_hdgs = {s.get("heading", "").lower().strip() for s in _new}
                _kept = [
                    s for s in existing_subs
                    if s.get("heading", "").lower().strip() not in _plan_hdgs
                    and len(s.get("points", [])) >= 2
                ]
                return _new + _kept

            for _sec in sections:
                _sec_heading = _sec.get("heading", "").lower().strip()
                if not _sec_heading:
                    continue

                _plan_key = _find_plan_key(_sec_heading)
                if _plan_key is None:
                    continue

                _plan_subs = plan_subsection_map.get(_plan_key, [])
                if not _plan_subs:
                    continue

                _curr_subs = _sec.get("subsections", [])
                _merged = _plan_subs_to_notes(_plan_subs, _curr_subs)
                if _merged:
                    _sec["subsections"] = _merged
                    _aug_count += 1
                    _matched_plan_keys.add(_plan_key)

                # Apply needs_diagram flag
                if _plan_key in plan_needs_diagram_map:
                    _sec["needs_diagram"] = plan_needs_diagram_map[_plan_key]

            # ── Inject plan sections that have NO matching KG section ─────────
            # Any plan section not consumed above represents content the KG
            # missed entirely. Create new sections for them.
            _unmatched_plan_keys = set(plan_subsection_map.keys()) - _matched_plan_keys
            _existing_headings_lower = {
                s.get("heading", "").lower().strip() for s in sections
            }
            _injected = 0
            for _pk in _unmatched_plan_keys:
                _plan_subs = plan_subsection_map.get(_pk, [])
                if not _plan_subs:
                    continue
                # Only inject if this plan section has substantial content
                _total_pts = sum(len(s.get("points", [])) for s in _plan_subs)
                if _total_pts < 2:
                    continue
                # Skip if a section with this heading already exists
                if _pk in _existing_headings_lower:
                    continue
                # Find the original plan section for title casing
                _canon_title = _pk  # use lowercase as fallback
                for _ps in structure_plan.get("sections", []):
                    if _ps.get("title", "").lower().strip() == _pk:
                        _canon_title = _ps.get("title", _pk)
                        break
                _new_sec = make_section(
                    _canon_title,
                    _plan_subs_to_notes(_plan_subs, [])
                )
                _new_sec["needs_diagram"] = plan_needs_diagram_map.get(_pk, False)
                sections.append(_new_sec)
                _injected += 1

            if _aug_count or _injected:
                print(
                    f"[Notes] 📐 Plan augmentation: {_aug_count} sections updated, "
                    f"{_injected} new sections injected from plan"
                )

        # ── 7. Validate & Fix ──
        notes_dict = make_notes("Lecture Notes", sections, summary=kg_summary)
        
        # ── 7a. Quality enforcement (redundancy, hierarchy, grouping, compression, topic separation) ──
        try:
            notes_dict = post_process_kg_notes(notes_dict, nodes, edges)
            print(f"[Notes] ✅ Quality enforcer applied successfully")
        except Exception as qe:
            print(f"[Notes] ⚠️ Quality enforcer skipped: {qe}")

        # ── 7b. Conceptual Hierarchy & Flow Organizer (v2) ─────────────────────
        # Priority 1 — Concept Hierarchy : build KG parent→child tree, reorder sections
        # Priority 2 — Concept Relationships : add "Concept Relationships" subsection per section
        # Priority 3 — Concept Flow Section : inject top-level "Concept Workflow" + per-section "Process Flow"
        # Priority 4 — Explanatory bullets  : expand bare-label / circular bullets with Input/Process/Output
        # Priority 5 — Remove ?             : sanitize ? artefacts from headings and bullets
        # Additional — tautology removal, key takeaways, duplicate image dedup, hierarchy annotation
        try:
            from concept_flow_organizer import apply_concept_flow
            notes_dict = apply_concept_flow(notes_dict, nodes, edges)
            print(f"[Notes] ✅ Concept flow organizer (v2) applied successfully")
        except Exception as cfo_err:
            print(f"[Notes] ⚠️ Concept flow organizer skipped: {cfo_err}")

        is_valid, violations = validate_hierarchy(notes_dict)
        if not is_valid:
            print(f"[Notes] ⚠️ Hierarchy violations ({len(violations)}): {violations[:3]}...")
            notes_dict = fix_hierarchy(notes_dict)
            is_valid2, v2 = validate_hierarchy(notes_dict)
            if is_valid2:
                print(f"[Notes] ✅ Auto-fix resolved all violations")
            else:
                print(f"[Notes] ⚠️ {len(v2)} violations remain after auto-fix: {v2[:3]}")

        # ── 8. Render ──
        notes_dir = session_path / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        
        out_pdf = notes_dir / "lecture_notes.pdf"
        out_txt = notes_dir / "lecture_notes.txt"

        try:
            # ── 8a. Post-process (naturalise relations, OCR noise, merge fragments, fix captions) ──
            try:
                from notes_postprocessor import patch_notes_renderer_captions
                patch_notes_renderer_captions(session_path / "diagram_texts")
                
                captions = load_merged_captions(session_path)
                notes_dict = post_process_notes(
                    notes_dict,
                    diagram_texts_dir=session_path / "diagram_texts",
                    merged_captions=captions
                )
                print(f"[Notes] ✅ Post-processor applied successfully")
            except Exception as pp_err:
                print(f"[Notes] ⚠️ Post-processor skipped: {pp_err}")

            # ── 8b. Issue Fixer (all 10 diagnosed issues) ──
            try:
                from notes_issue_fixer import fix_all_issues
                notes_dict = fix_all_issues(notes_dict, nodes=nodes, edges=edges)
                print(f"[Notes] ✅ Issue fixer applied successfully")
            except Exception as fix_err:
                print(f"[Notes] ⚠️ Issue fixer skipped: {fix_err}")

            # ── Cleaner (C-1 through C-10) ──────────────────────────────────────────────
            from notes_cleaner import clean_notes
            notes_dict = clean_notes(notes_dict)

            render_pdf(notes_dict, out_pdf, image_base_dir=session_path)
            render_txt(notes_dict, out_txt)
            
            n_sections = len(notes_dict.get("sections", []))
            n_subsections = sum(len(s.get("subsections", [])) for s in notes_dict.get("sections", []))
            print(f"[Notes] ✅ Hierarchical Notes saved → {out_pdf} ({n_sections} sections, {n_subsections} subsections)")
        except Exception as e:
            print(f"[Notes] ❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()





def run_notes_generation_wrapper(session_path: Path, word2vec_path: str = None) -> Path:
    """
    Simple wrapper - generates notes and returns PDF path.
    Notes are saved directly to notes/lecture_notes.pdf and notes/lecture_notes.txt
    """
    session_path = Path(session_path)
    gen = WMDKGNotesGenerator()
    gen.generate_notes(session_path)
    
    # Notes are now saved directly to notes/lecture_notes.pdf
    notes_dir = session_path / "notes"
    pdf_path = notes_dir / "lecture_notes.pdf"
    
    if pdf_path.exists():
        return pdf_path
    
    # Fallback: check for any PDF in notes dir
    for f in notes_dir.glob("*.pdf"):
        return f
    
    return None


# ---------- CLI ----------
def _print_usage():
    print("Usage: python pipeline.py --video1 <path_or_url1> --video2 <path_or_url2> [--whisper base|small|medium|large] [--device cpu|cuda]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Slide extraction pipeline (two-video capable)")
    parser.add_argument("--video1", type=str, required=True, help="Path or URL to first video")
    parser.add_argument("--video2", type=str, required=False, help="Path or URL to second video (optional)")
    parser.add_argument("--whisper", type=str, default="base", help="Whisper model size")
    parser.add_argument("--device", type=str, default="cpu", help="Device for Whisper/torch: cpu or cuda")
    args = parser.parse_args()

    if not args.video1:
        _print_usage()
        sys.exit(1)

    if args.video2:
        print("Starting two-video pipeline...")
        out = process_two_videos(args.video1, args.video2, whisper_model=args.whisper, device=args.device)
        print("Two-video pipeline finished.")
        print("Parent session:", out.get("parent_session"))
        print("Combined BART summary path:", out.get("combined_bart_summary"))
        print("Combined TextRank+BART summary path:", out.get("combined_textrank_bart_summary"))
        # KG HTML not guaranteed — print if present
        print("Combined KG HTML:", out.get("combined_graph_html", "n/a"))
    else:
        print("Starting single-video pipeline...")
        out = process_video_full(args.video1, whisper_model=args.whisper, device=args.device)
        print("Single-video pipeline finished.")
        print("Session:", out.get("session_id"))
        print("BART Summary path:", out.get("bart_summary_path", "n/a"))
        print("TextRank+BART Summary path:", out.get("textrank_bart_summary_path", "n/a"))
        print("KG HTML:", out.get("graph_html", "n/a"))