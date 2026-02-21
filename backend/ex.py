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

def _get_bart_model():
    """Get cached BART tokenizer and model. Loads on first call only."""
    if "tokenizer" not in _BART_CACHE:
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
            
    # Raise error if no audio extracted - DO NOT create silent file
    raise RuntimeError(f"Could not extract any audio from {video_file}. The video might be silent or corrupted.")





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


def transcribe_full_audio(audio_file: Path, transcripts_dir: Path,
                          model_size: str = "base", device: str = "cpu") -> Path:
    """
    Transcribe the FULL video audio as a single transcript.
    Returns path to the full_transcript.txt file.
    """
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / "full_transcript.txt"
    
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
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_file = combined_dir / "all_fused_text.txt"
    
    # Collect slide fused files
    files = []
    for pattern in ["*fused*", "*slide*"]:
        files.extend(list(fused_dir.glob(pattern)))
    
    files = list(set([f for f in files if f.is_file() and f.suffix == '.txt']))
    files.sort()
    
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
                    all_texts.append(transcript)
                    print(f" Added full transcript ({len(transcript)} chars)")
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
                        all_texts.append(f"--- {file_path.stem} ---\n{text}\n")
                        print(f" Added {file_path.name} ({len(text)} chars)")
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

def generate_global_summary(text: str, model, tokenizer, device, max_len=150, min_len=80) -> str:
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
        length_penalty=1.5,
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
    bart_max_output_tokens: int = 120,   
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
        summary = " ".join(sentences[:5])
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
        summary = " ".join(sentences[:min(7, len(sentences))])  
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
                max_length=250,      # Context-rich (user spec)
                min_length=60,       
                num_beams=4,         
                length_penalty=1.2,
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
            min_sentences=7,
            max_sentences=10,
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
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY

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

    combined_text = "\n\n".join(all_texts)
    print(f"[NonKG Notes] Combined text: {len(combined_text)} chars")

    # 2. Split into sentences
    sentences = sent_tokenize(combined_text)
    # Filter short/noisy sentences
    sentences = [s for s in sentences if len(s.split()) >= 5 and sum(c.isalpha() for c in s) / max(len(s), 1) > 0.5]

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
    topics = []
    for topic_idx in range(n_topics):
        top_word_indices = H[topic_idx].argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_word_indices]
        # Create a readable topic label from top 2-3 words
        label_words = [w for w in top_words[:3] if len(w) > 2]
        topic_label = " & ".join(label_words).title() if label_words else f"Topic {topic_idx + 1}"

        # Get sentences assigned to this topic (highest weight)
        topic_sentences = []
        for sent_idx, sent in enumerate(sentences):
            if W[sent_idx, topic_idx] > 0.1:  # threshold
                topic_sentences.append((sent, W[sent_idx, topic_idx]))

        # Sort by weight (most relevant first)
        topic_sentences.sort(key=lambda x: x[1], reverse=True)

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

    print(f"[NonKG Notes] {len(topics)} topics extracted: {[t['label'] for t in topics]}")

    # 6. Find relevant diagrams for each topic
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    topic_diagrams = {}
    if diagram_paths and diagram_captions:
        for topic in topics:
            topic_emb = embedding_model.encode(topic["label"] + " " + " ".join(topic["top_words"]), convert_to_tensor=True)
            best_img = None
            best_score = 0.3  # minimum threshold
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

    # 7. Generate text notes
    txt_lines = []
    txt_lines.append("=" * 60)
    txt_lines.append("LECTURE NOTES (Non-KG Based - Topic Modeling)")
    txt_lines.append("=" * 60)
    txt_lines.append("")

    for i, topic in enumerate(topics, 1):
        txt_lines.append(f"\n{'─' * 50}")
        txt_lines.append(f"  {i}. {topic['label']}")
        txt_lines.append(f"{'─' * 50}")
        txt_lines.append(f"  Keywords: {', '.join(topic['top_words'])}")
        txt_lines.append("")
        for sent in topic["sentences"]:
            txt_lines.append(f"  • {sent}")
        if topic["label"] in topic_diagrams:
            txt_lines.append(f"\n  [Diagram: {topic_diagrams[topic['label']]}]")
        txt_lines.append("")

    txt_content = "\n".join(txt_lines)
    txt_path = notes_dir / "non_kg_notes.txt"
    safe_write_text(txt_path, txt_content)
    print(f"[NonKG Notes] Text notes saved to {txt_path}")

    # 8. Generate PDF
    pdf_path = notes_dir / "non_kg_notes.pdf"
    try:
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                                leftMargin=50, rightMargin=50,
                                topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('NotesTitle', parent=styles['Title'],
                                     fontSize=18, spaceAfter=20)
        heading_style = ParagraphStyle('TopicHeading', parent=styles['Heading2'],
                                       fontSize=14, spaceAfter=8, spaceBefore=16,
                                       textColor='#2c3e50')
        keyword_style = ParagraphStyle('Keywords', parent=styles['Italic'],
                                       fontSize=9, textColor='#7f8c8d', spaceAfter=6)
        bullet_style = ParagraphStyle('BulletPoint', parent=styles['Normal'],
                                      fontSize=10, leftIndent=20, spaceAfter=4,
                                      alignment=TA_JUSTIFY)
        caption_style = ParagraphStyle('DiagramCaption', parent=styles['Italic'],
                                       fontSize=9, textColor='#555555',
                                       alignment=TA_LEFT, spaceAfter=8)

        elements = []
        elements.append(Paragraph("Lecture Notes (Non-KG Based)", title_style))
        elements.append(Paragraph("Generated via NMF Topic Modeling", styles['Italic']))
        elements.append(Spacer(1, 20))

        for i, topic in enumerate(topics, 1):
            elements.append(Paragraph(f"{i}. {topic['label']}", heading_style))
            elements.append(Paragraph(f"Keywords: {', '.join(topic['top_words'])}", keyword_style))

            for sent in topic["sentences"]:
                # Escape XML chars
                safe_sent = sent.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                elements.append(Paragraph(f"• {safe_sent}", bullet_style))

            # Add diagram if available
            if topic["label"] in topic_diagrams:
                img_name = topic_diagrams[topic["label"]]
                img_path = diagram_paths.get(img_name)
                if img_path and img_path.exists():
                    try:
                        elements.append(Spacer(1, 8))
                        elements.append(RLImage(str(img_path), width=4*inch, height=3*inch,
                                               kind='proportional'))
                        cap_text = diagram_captions.get(img_name, f"Diagram: {img_name}")
                        safe_cap = cap_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        elements.append(Paragraph(safe_cap, caption_style))
                    except Exception as e:
                        print(f"[NonKG Notes] Failed to embed image {img_name}: {e}")

            elements.append(Spacer(1, 12))

        doc.build(elements)
        print(f"[NonKG Notes] PDF saved to {pdf_path}")
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
            "You are helping a computer science student understand the ideas in a topic.\n"
            "Your job is to organize the ideas into a simple concept map.\n"
            "A concept map has items (concepts) and connections (how they relate with each other).\n\n"
            "Please represent the concept map using this JSON format:\n"
            "{\n"
            '  \"nodes\": [ {\"id\": \"N1\", \"label\": \"Concept\", \"description\": \"A concise description of the concept derived directly from the video content.\"} ],\n'
            '  \"edges\": [ {\"source\": \"N1\", \"target\": \"N2\", \"relation\": \"how they connect\"} ]\n'
            "}\n\n"
            "Important Rules:\n"
            "1. Extract meaningful concepts.\n"
            "2. Provide a description for every node based on the video content explicitly.\n"
            "3. Do NOT include explanations — only return the JSON.\n\n"
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

        node_objs = [Node(**{k: str(v) for k, v in n.items()}) for n in normalized_nodes]
        edge_objs = [Edge(**{k: str(v) for k, v in e.items()}) for e in normalized_edges]

        return KnowledgeGraph(nodes=node_objs, edges=edge_objs)

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
                max_length=200,           # Rich per-segment output
                min_length=40,            # Ensure substantial content
                num_beams=4,
                early_stopping=True,
                length_penalty=1.5,       # Encourage longer, richer output
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
                max_length=400,
                min_length=150,
                num_beams=4,
                early_stopping=True,
                length_penalty=1.2,
                no_repeat_ngram_size=3,
            )
            full_summary = tokenizer.decode(merge_ids[0], skip_special_tokens=True)
        else:
            # Fits in one pass — final coherence pass
            merge_inputs = tokenizer([merged_text], max_length=1024, return_tensors="pt", truncation=True)
            merge_ids = model.generate(
                merge_inputs["input_ids"],
                max_length=400,
                min_length=150,
                num_beams=4,
                early_stopping=True,
                length_penalty=1.2,
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
            fused_edges.append({
                "source": idmap[e["source"]], "target": idmap[e["target"]], "relation": e.get("relation", "")
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
    def find_related_diagram(self, topic: str, captions: dict) -> tuple:
        """Find most relevant diagram for a topic"""
        if not captions:
            return None, None
        
        topic_emb = self.embedding_model.encode(topic, convert_to_tensor=True)
        
        best_img = None
        best_score = -1
        
        for img_name, caption in captions.items():
            cap_emb = self.embedding_model.encode(caption, convert_to_tensor=True)
            score = util.pytorch_cos_sim(topic_emb, cap_emb).item()
            
            if score > best_score:
                best_score = score
                best_img = img_name
        
        # Only return if similarity is reasonable
        if best_score > 0.3:
            return best_img, captions[best_img]
        
        return None, None

    # ---------- HIERARCHICAL GRAPH PROCESSING (FOREST CONSTRUCTION) ----------

    def build_hierarchy(self, nodes: list, edge_map: dict) -> dict:
        """
        Build a strict Parent-Child Forest (Tree structure).
        Ensures every node appears fully ONLY ONCE in the most relevant place.
        """
        # Node lookup
        node_lookup = {n["id"]: n for n in nodes}
        
        # 1. Calculate "Parent Potential" based on OUTgoing structure edges
        # Nodes that "have" or "contain" things are likely parents
        parent_scores = {}
        for nid in node_lookup:
            score = 0
            if nid in edge_map:
                for tgt, rel in edge_map[nid]:
                    rel_lower = rel.lower()
                    if any(x in rel_lower for x in ["has", "includes", "contains", "consists", "composed"]):
                        score += 2  # Strong structural parent
                    elif any(x in rel_lower for x in ["uses", "employs"]):
                        score += 1
            parent_scores[nid] = score

        # 2. Assign "Best Parent" for every node
        # A node is a child of the neighbor that has the Highest Parent Score + Structural Link
        parent_assignment = {}  # child_id -> parent_id
        
        for child_id in node_lookup:
            best_parent = None
            max_score = -1
            
            # Check all INCOMING edges to find potential parents
            # (Who points to me with a structural edge?)
            potential_parents = []
            for potential_p_id, edges in edge_map.items():
                for tgt, rel in edges:
                    if tgt == child_id:
                        rel_lower = rel.lower()
                        # Only structural edges create hierarchy
                        if any(x in rel_lower for x in ["has", "includes", "contains", "consists", "composed", "type of", "is a"]):
                            potential_parents.append(potential_p_id)
            
            # Select the "strongest" parent
            for p_id in potential_parents:
                p_score = parent_scores.get(p_id, 0)
                if p_score > max_score:
                    max_score = p_score
                    best_parent = p_id
            
            # Additional check: Don't assign if it creates a cycle
            if best_parent:
                # Simple cycle check: is child an ancestor of parent?
                curr = best_parent
                is_cycle = False
                for _ in range(10): # Limit depth
                    if curr == child_id:
                        is_cycle = True
                        break
                    curr = parent_assignment.get(curr)
                    if not curr: break
                
                if not is_cycle:
                    parent_assignment[child_id] = best_parent

        # 3. separate Roots (Main Sections) from Children
        roots = []
        children_map = {} # parent_id -> list of child_ids
        
        for nid in node_lookup:
            pid = parent_assignment.get(nid)
            if pid:
                if pid not in children_map: children_map[pid] = []
                children_map[pid].append(nid)
            else:
                # It's a root (Main Topic) - only if it has content or children
                if parent_scores.get(nid, 0) > 0 or len(node_lookup[nid].get("description", "").split()) > 5:
                    roots.append(nid)
        
        # Sort roots by video appearance if possible (using temporary order for now)
        # In full implementation, mapping back to timestamps would be ideal
        
        return {"roots": roots, "children_map": children_map, "node_lookup": node_lookup}

    def synthesize_explanation(self, topic: str, node: dict, children: list, edge_map: dict) -> str:
        """Generate clean explanation from graph structure (No verbatim transcript info)"""
        sentences = []
        
        # 1. Start with Definition from Node Description (Priority)
        desc = node.get("description", "")
        if desc and len(desc.split()) >= 5 and not desc.strip().endswith('?'):
            sentences.append(desc)
        else:
            sentences.append(f"{topic} is a key concept in this domain.")
            
        # 2. Add structural sentence if has children
        if children:
            child_labels = [c.get("label", "") for c in children[:3]]
            if child_labels:
                sentences.append(f"It includes components such as {', '.join(child_labels)}.")
                
        # 3. Add relation-based sentences
        nid = node["id"]
        if nid in edge_map:
            for tgt, rel in edge_map[nid][:3]:
                # Skip if tgt is already a child (avoid repetition)
                is_child = False
                for c in children:
                    if c["id"] == tgt: is_child = True; break
                if is_child: continue
                
                # Interpret edge
                tgt_node = node # Placeholder, need lookup passed ideally
                # Use simple interpretation here as fallback
                sentences.append(f"It {rel.replace('_', ' ')} {tgt}.")
        
        return " ".join(sentences)

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
            
            lines.append(line)
        
        return " ".join(lines)
    
    # ========== DYNAMIC EDGE-TO-MEANING TEMPLATES ==========
    # These translate KG edges into meaningful sentences (relation-driven, not topic-driven)
    EDGE_TEMPLATES = {
        "uses": "{A} uses {B} during its operation.",
        "has": "{A} has {B} as a component.",
        "contains": "{A} contains {B}.",
        "produces": "{A} produces {B} as output.",
        "part_of": "{B} is a phase or component of {A}.",
        "detects": "{A} detects {B}.",
        "is_a": "{A} is a type of {B}.",
        "related_to": "{A} is related to {B}.",
        "converts": "{A} converts input into {B}.",
        "checks": "{A} checks for {B}.",
        "stores": "{A} stores {B}.",
        "generates": "{A} generates {B}.",
        "requires": "{A} requires {B} to function.",
        "performs": "{A} performs {B}.",
        "implements": "{A} implements {B}.",
        "includes": "{A} includes {B}.",
    }
    
    def interpret_edge(self, source_label: str, relation: str, target_label: str) -> str:
        """Convert KG edge into meaningful sentence (DYNAMIC edge interpretation)"""
        relation_lower = relation.lower().replace("_", " ").replace("-", " ")
        
        # Find matching template
        for key, template in self.EDGE_TEMPLATES.items():
            if key in relation_lower:
                return template.format(A=source_label.title(), B=target_label.title())
        
        # Default: construct grammatical sentence
        return f"{source_label.title()} {relation_lower} {target_label.title()}."
    
    def get_contextual_description(self, source_topic: str, relation: str, target_node: dict) -> str:
        """Generate context-specific description for a connected node
        
        PRIORITY:
        1. Use target node's own description (from KG)
        2. Fall back to contextual explanation based on relation type
        """
        target_label = target_node.get("label", "")
        base_desc = target_node.get("description", "")
        
        # PRIORITY 1: Use the node's actual KG description if available
        if base_desc and len(base_desc.split()) >= 3:
            return base_desc
        
        # PRIORITY 2: Create contextual explanation based on relation
        relation_lower = relation.lower().replace("_", " ").replace("-", " ")
        
        if "uses" in relation_lower:
            return f"Used by {source_topic.title()} during its operation."
        elif "has" in relation_lower or "contains" in relation_lower:
            return f"A component or part of {source_topic.title()}."
        elif "is_a" in relation_lower or "type" in relation_lower:
            return f"A category or type classification."
        elif "produces" in relation_lower or "generates" in relation_lower:
            return f"Output produced by {source_topic.title()}."
        elif "detects" in relation_lower or "checks" in relation_lower:
            return f"Detected or verified by {source_topic.title()}."
        elif "stores" in relation_lower:
            return f"Stored or managed by {source_topic.title()}."
        else:
            return f"Related to {source_topic.title()} through {relation_lower}."
    
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
        
        RELAXED: Allow most nodes with any meaningful content
        """
        if not node:
            return True  # Allow sections without KG nodes (text-based)
        
        nid = node.get("id", "")
        label = node.get("label", "")
        desc = node.get("description", "")
        
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
        
        return False
    
    def normalize_text(self, text: str) -> str:
        """Fix common OCR/speech errors in lecture content (DYNAMIC)"""
        # Common misspellings (can be extended per domain)
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
        
        return text

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
        
        s_lower = s.lower()
        
        # MANDATORY: Check for verb presence (grammar gate)
        verbs = ["is", "are", "was", "were", "has", "have", "can", "will", "would", "should",
                 "converts", "generates", "produces", "checks", "creates", "stores", "translates",
                 "represents", "contains", "includes", "defines", "refers", "performs", "processes",
                 "uses", "takes", "gives", "makes", "shows", "provides", "allows", "enables"]
        has_verb = any(f" {v} " in f" {s_lower} " for v in verbs)
        
        if not has_verb:
            return False  # STRICT: Must have verb
        
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
        STRICT: Rejects code, assembly, visual descriptions, and garbage
        """
        title = title.strip()
        
        # Reject code-like topics
        if "(" in title or ")" in title:
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
        if len(title.split()) > 5:
            return False
        
        # Reject too short
        if len(title) < 3:
            return False
        
        # Reject numeric-heavy
        if sum(c.isdigit() for c in title) > len(title) * 0.3:
            return False
        
        # Reject symbol-heavy (more than 2 non-alphanumeric)
        symbol_count = sum(not c.isalnum() and not c.isspace() for c in title)
        if symbol_count > 2:
            return False
        
        # Reject gibberish patterns
        if re.search(r'[^a-zA-Z\s]{3,}', title):
            return False
        
        return True

    # ---------- DYNAMIC DIAGRAM CAPTIONS ----------
    def get_semantic_caption(self, topic: str, raw_caption: str, kg_nodes: list = None) -> str:
        """Generate meaningful diagram caption DYNAMICALLY from topic and KG
        
        Args:
            topic: The topic this diagram relates to
            raw_caption: Raw OCR/vision caption (may be noisy)
            kg_nodes: List of KG nodes to find related descriptions
        """
        # Try to find description from KG nodes for this topic
        if kg_nodes:
            topic_lower = topic.lower()
            for node in kg_nodes:
                node_label = node.get("label", "").lower()
                if node_label and (node_label in topic_lower or topic_lower in node_label):
                    desc = node.get("description", "")
                    if desc and len(desc) > 20:
                        return f"Diagram illustrating: {desc}"
        
        # Clean raw caption if available
        if raw_caption:
            # Remove visual noise
            cleaned = re.sub(r'(a |an |the )?(diagram|image|figure|picture|screenshot|blackboard|sign|logo|icon)\s*(of|showing|with|that)?\s*', '', raw_caption, flags=re.IGNORECASE)
            cleaned = re.sub(r'black and white|white text|black background|white background', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Validate cleaned caption
            if len(cleaned) > 15 and not re.search(r'(.)\1{3,}', cleaned):
                return f"Diagram: {cleaned}"
        
        # Fallback: simple topic-based caption
        return f"Visual representation of {topic.title()}"

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
        Generate EDUCATIONAL NOTES with STRICT VALIDATION:
        1. Topic Eligibility Check (no code/garbage topics)
        2. Line Classification (CODE/DIAGRAM/EXPLANATION/NOISE)
        3. Semantic Validity Gate (only meaningful content)
        4. Last-Mile Safety Net (drop weak sections)
        """
        session_path = Path(session_path)
        
        # 1. Locate Text Segments
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

        # 2. Load KG Data
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
        
        node_lookup = {n.get("label", "").lower(): n for n in nodes}
        edge_map = {}
        for e in edges:
            src = e.get("source", "")
            tgt = e.get("target", "")
            rel = e.get("relation", "is related to")
            if src not in edge_map: edge_map[src] = []
            edge_map[src].append((tgt, rel))
        
        # DYNAMIC: Extract domain keywords from this video's KG
        domain_keywords = self.extract_domain_keywords(nodes, edges)
        print(f"[Notes] 📚 Extracted {len(domain_keywords)} domain keywords from KG")

        # 3. Load Images
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
        for d in img_dirs:
            if d.exists():
                for img in list(d.glob("*.png")) + list(d.glob("*.jpg")):
                    if img.name not in image_paths:
                        image_paths[img.name] = img

        # 4. Process Segments with STRICT Topic Validation
        text_files = sorted(list(fused_dir.glob("*.txt")))
        canonical_topics = {}
        last_valid_canon = None  # For merging invalid topics into previous
        
        for seg_idx, txt_file in enumerate(text_files):
            if "merged_captions" in txt_file.name: continue
            raw_text = txt_file.read_text(encoding='utf-8')
            text = self.clean_text(raw_text)
            if len(text.split()) < 15: continue

            # Identify Topic from KG
            text_lower = text.lower()
            best_node = None
            max_score = 0
            
            for label, node in node_lookup.items():
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
            
            # TOPIC ELIGIBILITY CHECK
            if not self.is_valid_topic(topic_label):
                # Merge into previous valid topic
                if last_valid_canon and last_valid_canon in canonical_topics:
                    canonical_topics[last_valid_canon]["texts"].append(text)
                continue
            
            canon = self.canonicalize_topic(topic_label)
            if not canon or len(canon) < 3:
                if last_valid_canon:
                    canonical_topics[last_valid_canon]["texts"].append(text)
                continue
            
            # Merge or create new
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

        # 5. Build Strict Hierarchy (Forest)
        hierarchy = self.build_hierarchy(nodes, edge_map)
        roots = hierarchy["roots"]
        children_map = hierarchy["children_map"]
        node_lookup = hierarchy["node_lookup"]
        
        # Map text segments to nodes for context enrichment
        node_text_map = {}
        for canon_data in canonical_topics.values():
            if canon_data["node"]:
                node_text_map[canon_data["node"]["id"]] = " ".join(canon_data["texts"])

        # 6. Generate PDF with HIERARCHICAL STRUCTURE
        # Save notes directly to notes/ folder
        notes_dir = session_path / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = notes_dir / "lecture_notes.pdf"
        out_txt = notes_dir / "lecture_notes.txt"
        
        def safe(s): return s.encode("latin-1", "replace").decode("latin-1")
        txt_output = ["LECTURE NOTES\n=============\n\n"]

        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            pdf.set_font("Arial", "B", 20)
            pdf.cell(0, 12, "Lecture Notes", ln=True, align='C')
            pdf.set_font("Arial", "I", 10)
            pdf.cell(0, 6, "Knowledge Graph Enhanced Educational Notes", ln=True, align='C')
            pdf.ln(8)
            
            used_images = set()
            section_num = 0
            
            # Helper for creating bullet points from children
            def add_child_bullets(parent_id, current_pdf):
                if parent_id in children_map:
                    current_pdf.ln(2)
                    for child_id in children_map[parent_id]:
                        child = node_lookup[child_id]
                        lbl = child.get("label", "Unknown")
                        desc = child.get("description", "")
                        # Short explanation
                        if not desc:
                            # Try synthesize from edge
                            msg = self.interpret_edge(node_lookup[parent_id]["label"], "related", lbl)
                            bullet = f"  - {safe(lbl)}"
                        else:
                            bullet = f"  - {safe(lbl)}: {safe(desc)}"
                        
                        current_pdf.set_font("Arial", "", 10)
                        current_pdf.multi_cell(0, 5, bullet)
                        txt_output.append(f"  - {lbl}: {desc}\n")
            
            # MAIN LOOP - Iterate Roots
            for root_id in roots:
                root_node = node_lookup[root_id]
                topic = root_node.get("label", "Topic")
                
                section_num += 1
                
                # LEVEL 1: Main Section Header
                pdf.set_font("Arial", "B", 14)
                pdf.set_fill_color(230, 240, 255)
                pdf.cell(0, 8, f"{section_num}. {safe(topic.title())}", ln=True, fill=True)
                pdf.ln(2)
                txt_output.append(f"\n{section_num}. {topic.title()}\n")
                
                # LEVEL 1: Definition (Priority: Node Description)
                defn = root_node.get("description", "")
                if defn and len(defn.split()) >= 5 and not defn.strip().endswith('?'):
                    pass # Use this
                else:
                    # Fallback to synthesis
                    defn = self.synthesize_explanation(topic, root_node, 
                                                     [node_lookup[c] for c in children_map.get(root_id, [])], 
                                                     edge_map)
                
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 6, "Definition:", ln=True)
                pdf.set_font("Arial", "", 10)
                pdf.multi_cell(0, 5, safe(defn))
                pdf.ln(2)
                txt_output.append(f"Definition: {defn}\n")
                
                # LEVEL 2: Subsections (Children)
                if root_id in children_map:
                    child_ids = children_map[root_id]
                    
                    # Group children: if they have their OWN children, they get a subsection (1.1)
                    # If they are leaf nodes, they go into a "Components" list
                    
                    complex_children = []
                    simple_children = []
                    
                    for cid in child_ids:
                        if cid in children_map: # Has grandchildren
                            complex_children.append(cid)
                        else:
                            simple_children.append(cid)
                    
                    # 2A. Handle Complex Children (Subsections 1.X)
                    sub_idx = 0
                    for cid in complex_children:
                        sub_idx += 1
                        child_node = node_lookup[cid]
                        sub_label = child_node.get("label", "Subtopic")
                        
                        pdf.set_font("Arial", "B", 11)
                        pdf.cell(0, 6, f"{section_num}.{sub_idx} {safe(sub_label)}", ln=True)
                        txt_output.append(f"{section_num}.{sub_idx} {sub_label}\n")
                        
                        # Child Description
                        c_desc = child_node.get("description", "")
                        if c_desc and len(c_desc.split()) >= 4:
                            pdf.set_font("Arial", "", 10)
                            pdf.multi_cell(0, 5, safe(c_desc))
                            txt_output.append(f"{c_desc}\n")
                        
                        # LEVEL 3: Grandchildren (Bullets)
                        add_child_bullets(cid, pdf)
                        pdf.ln(2)
                        
                    # 2B. Handle Simple Children (Bullet List under Main)
                    if simple_children:
                        pdf.set_font("Arial", "B", 11)
                        if complex_children:
                            pdf.cell(0, 6, "Other Components:", ln=True)
                        else:
                            pdf.cell(0, 6, "- Components and Structure", ln=True)
                        
                        pdf.set_font("Arial", "", 10)
                        for scid in simple_children:
                            child = node_lookup[scid]
                            lbl = child.get("label", "")
                            desc = child.get("description", "")
                            
                            # Contextual description from edge if needed
                            if not desc:
                                desc = self.get_contextual_description(topic, "includes", child)
                                
                            bullet = f"  - {safe(lbl)}: {safe(desc)}"
                            pdf.multi_cell(0, 5, bullet)
                            txt_output.append(f"  - {lbl}: {desc}\n")
                        pdf.ln(2)

                # LEVEL 1: Related Concepts (Non-hierarchical edges)
                # Find edges that are NOT parent/child relationships
                related_items = []
                if root_id in edge_map:
                    for tgt, rel in edge_map[root_id]:
                        # Skip if target is my child or parent (already handled)
                        if tgt in children_map.get(root_id, []): continue
                        if hierarchy["node_lookup"][tgt]["id"] == root_id: continue # Self
                        
                        # Add to related
                        tgt_node = node_lookup.get(tgt)
                        if tgt_node:
                            lbl = tgt_node.get("label", tgt)
                            desc = self.get_contextual_description(topic, rel, tgt_node)
                            related_items.append(f"{lbl}: {desc}")
                            
                if related_items:
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 6, "- Related Concepts", ln=True)
                    pdf.set_font("Arial", "", 10)
                    for item in related_items[:5]:
                        pdf.multi_cell(0, 5, f"  - {safe(item)}")
                        txt_output.append(f"  - {item}\n")
                    pdf.ln(2)
                
                # Visual Representation
                best_img, raw_cap = self.find_related_diagram(topic + " " + defn, captions)
                if best_img and best_img not in used_images:
                    used_images.add(best_img)
                    img_path = image_paths.get(best_img)
                    if img_path and img_path.exists():
                        semantic_cap = self.get_semantic_caption(topic, raw_cap, nodes)
                        
                        pdf.set_font("Arial", "B", 11)
                        pdf.cell(0, 6, "Visual Representation:", ln=True)
                        txt_output.append(f"Visual Representation:\n")
                        txt_output.append(f"Figure {section_num}: {semantic_cap}\n")
                        
                        try:
                            from PIL import Image as PILImage
                            with PILImage.open(str(img_path)) as img:
                                w_px, h_px = img.size
                            max_w, max_h = 150, 90
                            aspect = h_px / w_px if w_px > 0 else 1
                            if w_px > h_px:
                                w_mm = min(max_w, w_px * 0.264583)
                                h_mm = w_mm * aspect
                            else:
                                h_mm = min(max_h, h_px * 0.264583)
                                w_mm = h_mm / aspect
                            
                            pdf.image(str(img_path), x=30, w=w_mm)
                            pdf.ln(2)
                            pdf.set_font("Arial", "I", 9)
                            pdf.multi_cell(0, 4, f"Figure {section_num}: {safe(semantic_cap)}", align='C')
                        except: pass
                
                pdf.ln(6)

            pdf.output(out_pdf)
            print(f"[Notes] ✅ Validated Notes saved → {out_pdf} ({section_num} sections)")
            Path(out_txt).write_text("".join(txt_output), encoding='utf-8')

        except Exception as e:
            print(f"[Notes] ❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()


# ---------- Simple Wrapper ----------
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
