# ---- CPU Codespaces: stubs to avoid heavy optional deps on import ----
import os, sys, types
os.environ.setdefault("GRADIO_DISABLE_API_INFO", "1")
# xformers: not needed on CPU; audiocraft may import it unconditionally
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # force CPU
os.environ.setdefault("AUDIOCRAFT_DISABLE_XFORMERS", "1")
# ---- CPU-only Codespaces: stub xformers so audiocraft 0.0.2 runs without it ----
#if "xformers" not in sys.modules:
#    _xf = types.ModuleType("xformers")
#    _xf.ops = types.SimpleNamespace()      # audiocraft will fall back without special ops
#    sys.modules["xformers"] = _xf
# provide a tiny xformers.ops shim that Audiocraft can import
if "xformers" not in sys.modules:
    xf_mod  = types.ModuleType("xformers")
    ops_mod = types.ModuleType("xformers.ops")

    # minimal SDPA fallback that matches xformers.ops API
    def _sdpa(q, k, v, attn_bias=None, p: float = 0.0, scale=None, **kwargs):
        import torch
        import torch.nn.functional as F
        if scale is None:
            # inverse sqrt(d_k)
            scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_bias is not None:
            scores = scores + attn_bias
        if p and p > 0:
            scores = F.dropout(scores, p, training=False)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    # audiocraft also imports this symbol; a dummy class is fine here
    class LowerTriangularMask:
        def __init__(self, *a, **k): pass

    # publish the symbols on the ops module
    ops_mod.memory_efficient_attention = _sdpa
    ops_mod.scaled_dot_product_efficient_attention = _sdpa
    ops_mod.LowerTriangularMask = LowerTriangularMask

    # wire up the module hierarchy
    xf_mod.ops = ops_mod
    sys.modules["xformers"] = xf_mod
    sys.modules["xformers.ops"] = ops_mod

# spaCy: imported by audiocraft.conditioners, but not needed unless you use the spaCy tokenizer.
# Provide a tiny placeholder so import succeeds. If code actually calls into spaCy, we raise.
if "spacy" not in sys.modules:
    _sp = types.ModuleType("spacy")
    def _not_available(*a, **k):
        raise ImportError("spaCy is not available in this environment (stub).")
    _sp.load = _not_available
    _sp.__version__ = "0.0-stub"
    sys.modules["spacy"] = _sp
# -------------------------------------------------------------------------------
# app.py
# Global Fusion Music – Pro UI with long-form generation (chunking + crossfade)
# Requires: gradio >= 4.44.1; audiocraft installed; torch/torchaudio installed.

import os, sys, socket
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "0"
os.environ["GRADIO_MIXPANEL_ENABLED"] = "0"
os.environ["GRADIO_STATS_ENABLED"] = "0"
import math
import time
import datetime as dt
import torch
import torchaudio
import numpy as np
import gradio as gr
import json, hashlib
from pathlib import Path
import soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
import audiocraft.modules.transformer as _ac_t

# ---- Model (Meta MusicGen via audiocraft) ----------------------------------
from pathlib import Path
from typing import List, Tuple, Optional
from audiocraft.models import MusicGen
from dataclasses import dataclass, asdict
from PIL import Image
from types import SimpleNamespace

# --- ensure xformers shim is present early (works on Win + Python 3.12 CPU) ---
try:
    # import xformers  # real package or our site-packages shim
    x = 4
except Exception:
    import sys, types
    import torch
    import torch.nn.functional as F

    def _sdpa(q, k, v, attn_bias=None, p: float = 0.0, scale=None):
        if scale is None:
            scale = (q.shape[-1]) ** -0.5
        scores = q @ k.transpose(-2, -1) * scale
        if attn_bias is not None:
            scores = scores + attn_bias
        if p and p > 0:
            scores = F.dropout(scores, p, training=False)
        weights = scores.softmax(dim=-1)
        return weights @ v

    # Names Audiocraft may look for:
    def memory_efficient_attention(q, k, v, attn_bias=None, p: float = 0.0, scale=None, **kwargs):
        return _sdpa(q, k, v, attn_bias=attn_bias, p=p, scale=scale)

    def scaled_dot_product_efficient_attention(q, k, v, attn_bias=None, p: float = 0.0, scale=None, **kwargs):
        return _sdpa(q, k, v, attn_bias=attn_bias, p=p, scale=scale)

    ops = SimpleNamespace(
        memory_efficient_attention=memory_efficient_attention,
        scaled_dot_product_efficient_attention=scaled_dot_product_efficient_attention,
    )

    # only if something elsewhere refers to `mod.ops`:
    mod = SimpleNamespace(ops=ops)
# -------------------------------------------------------------------------------
    
# --- neutralize CUDA-only xformers checks in Audiocraft (safe for CPU) ---

def _noop(*a, **k):  # no-op verifier
    return None
# Some builds use one or both of these gates:
# Only disable xformers on AMD (DirectML) or when you force CPU
if os.getenv("USE_DML") == "1" or os.getenv("FORCE_CPU") == "1":
    # Tell Audiocraft there is NO xformers on these backends
    setattr(_ac_t, "_verify_xformers_memory_efficient_compat", _noop)
    setattr(_ac_t, "_is_xformers_available", lambda: False)
# On NVIDIA: do NOT override; real xformers can be used if installed
# ---------------------------------------------------------------------------

# UTILITIES

APP_NAME = "Global Fusion Music"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
THUMBS  = OUT_DIR / "_thumbs"
THUMBS.mkdir(exist_ok=True, parents=True)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Keep a stable mapping so Gallery index -> audio path
RECENT_ITEMS: list[tuple[str, str]] = []  # [(audio_path, thumb_path), ...]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_NAME = "facebook/musicgen-small"
# MELODY_MODEL = "facebook/musicgen-melody"
MODEL_NAME   = "small"     # was "facebook/musicgen-small"
MELODY_MODEL = "melody"    # was "facebook/musicgen-melody"

_model_small: Optional[MusicGen] = None
_model_melody: Optional[MusicGen] = None

def _move_musicgen_to_device(m):
    """Back-compat: some Audiocraft builds accept device=..., others expose set_device."""
    try:
        # most recent builds: handled at load time; nothing to do
        m.eval()
        return m
    except TypeError:
        pass
    if hasattr(m, "set_device"):
        m.set_device(DEVICE)
    try:
        m.eval()
    except Exception:
        pass
    return m

def _load_small() -> MusicGen:
    global _model_small
    if _model_small is None:
        try:
            _model_small = MusicGen.get_pretrained(MODEL_NAME, device=DEVICE)
        except TypeError:
            _model_small = MusicGen.get_pretrained(MODEL_NAME)
            _move_musicgen_to_device(_model_small)
    return _model_small

def _load_melody() -> MusicGen:
    global _model_melody
    if _model_melody is None:
        try:
            _model_melody = MusicGen.get_pretrained(MELODY_MODEL, device=DEVICE)
        except TypeError:
            _model_melody = MusicGen.get_pretrained(MELODY_MODEL)
            _move_musicgen_to_device(_model_melody)
    return _model_melody

def _resolve_path(file_obj):
    # gr.File can be str, tempfile, or dict in 4.x
    if isinstance(file_obj, dict):
        return file_obj.get("name") or file_obj.get("path") or ""
    if hasattr(file_obj, "name"):
        return str(file_obj.name)
    return str(file_obj or "")

def _append_history(hist, meta, file_obj):
    """hist: list[dict], meta: dict, file_obj: gr.File return."""
    hist = hist or []
    path = _resolve_path(file_obj)
    row = {
        "time": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "prompt": (meta or {}).get("prompt", ""),
        "duration": (meta or {}).get("duration", ""),
        "seed": (meta or {}).get("seed", ""),
        "file": path,
    }
    return hist + [row]   # returns the *new state* only

def _sync_history(hist):
    """Return values for UI widgets fed from history_state."""
    # Return for the dataframe:
    table_rows = [[r.get("time",""), r.get("prompt",""), r.get("duration",""),
                   r.get("seed",""), r.get("file","")] for r in (hist or [])]

    # If you also have a gallery, convert to (path, caption)
    # gallery = [(r.get("file",""), r.get("prompt","")) for r in (hist or [])]
    # return table_rows, gallery

    return table_rows


 # ------------------------------------------------------------------------------- 

JOBS_DIR = Path("jobs"); JOBS_DIR.mkdir(exist_ok=True, parents=True)

def _job_id_from_params(prompt, duration_sec, top_k, temperature, cfg, bpm, style, rhythm, key_root, scale, seed, use_melody):
    h = hashlib.sha256()
    key = json.dumps({
        "p": prompt, "dur": duration_sec, "top_k": int(top_k), "temp": float(temperature),
        "cfg": float(cfg), "bpm": int(bpm), "style": style, "rhythm": rhythm,
        "key": key_root, "scale": scale, "seed": int(seed), "mel": bool(use_melody)
    }, sort_keys=True).encode()
    h.update(key)
    return h.hexdigest()[:12]  # short id

@dataclass
class JobState:
    job_id: str
    total_chunks: int
    next_chunk: int            # 0..total_chunks
    sr: int
    chunk_sec: float
    out_dir: str               # directory for this job
    prompt: str

    def save(self):
        p = Path(self.out_dir) / "state.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(job_dir: Path):
        p = job_dir / "state.json"
        if not p.exists(): 
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JobState(**data)
  
  
# ---- Audio utils ------------------------------------------------------------
SR = 32000  # MusicGen outputs 32k

def to_mono(x: torch.Tensor) -> torch.Tensor:
    # x: (channels, samples)
    if x.dim() != 2:
        raise ValueError("Audio tensor must be (channels, samples)")
    return x.mean(0, keepdim=True) if x.size(0) > 1 else x

def crossfade_concat(chunks: List[np.ndarray], sr: int = SR, xf_sec: float = 0.25) -> np.ndarray:
    """Concatenate with a short linear crossfade."""
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    xf = int(sr * max(0.0, xf_sec))
    out = chunks[0].copy()
    for i in range(1, len(chunks)):
        cur = chunks[i]
        if xf <= 0:
            out = np.concatenate([out, cur], axis=-1)
            continue
        n1, n2 = out.shape[-1], cur.shape[-1]
        ov = min(xf, n1, n2)
        if ov > 0:
            fade_out = np.linspace(1.0, 0.0, ov, dtype=np.float32)
            fade_in = 1.0 - fade_out
            out[..., -ov:] = out[..., -ov:] * fade_out + cur[..., :ov] * fade_in
            out = np.concatenate([out, cur[..., ov:]], axis=-1)
        else:
            out = np.concatenate([out, cur], axis=-1)
    return out

def save_wav(y: np.ndarray, sr: int, path: Path) -> Path:
    wav = torch.from_numpy(y).unsqueeze(0)  # (1, T)
    torchaudio.save(str(path), wav, sr)
    return path


# ---- Mixtape helpers (once) -------------------------------------------------

import numpy as np, soundfile as sf, datetime as dt, time

def load_audio(path: str, sr: int = SR) -> np.ndarray:
    y, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:  # stereo -> mono
        y = (y[:,0] + y[:,1]) * 0.5
    if file_sr != sr:
        import torchaudio, torch
        t = torch.from_numpy(y).unsqueeze(0)
        t = torchaudio.functional.resample(t, file_sr, SR)
        y = t.squeeze(0).numpy()
    return y

def normalize_peak(y: np.ndarray, peak_target: float = 0.95) -> np.ndarray:
    p = float(np.max(np.abs(y)) + 1e-9)
    return y if p == 0 else (y * (peak_target / p)).astype(np.float32)

def beat_aligned_xfade(a: np.ndarray, b: np.ndarray, sr: int, bpm: float, beats: float = 4.0) -> np.ndarray:
    xfade_samples = int((60.0 / bpm) * beats * sr)
    if xfade_samples <= 0:
        return np.concatenate([a, b], axis=0)
    f = min(xfade_samples, len(a), len(b))
    if f <= 0:
        return np.concatenate([a, b], axis=0)
    ramp_out = np.linspace(1.0, 0.0, f, dtype=np.float32)
    ramp_in  = np.linspace(0.0, 1.0, f, dtype=np.float32)
    head = a[:-f] if len(a) > f else np.zeros(0, dtype=np.float32)
    tail = a[-f:] * ramp_out + b[:f] * ramp_in
    return np.concatenate([head, tail, b[f:]], axis=0)

def render_segment_to_path(**kwargs) -> str:
    """Runs your existing generate_music() and returns the final .wav path."""
    last = None
    for triple in generate_music(**kwargs):  # streams status, audio, file
        last = triple
    if last is None:
        raise RuntimeError("No output from generate_music")
    _status, _audio_tuple, path = last
    return str(path)

def stitch_paths_to_wav(paths: list[str], bpm_for_xfade: float, xfade_beats: float, out_path: str | None = None) -> str:
    mix = None
    for p in paths:
        y = load_audio(p, sr=SR)
        y = normalize_peak(y, 0.9)
        if mix is None:
            mix = y
        else:
            mix = beat_aligned_xfade(mix, y, sr=SR, bpm=float(bpm_for_xfade), beats=float(xfade_beats))
    mix = normalize_peak(mix, 0.95)
    if out_path is None:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(OUT_DIR / f"mixtape_{ts}.wav")
    save_wav(mix, SR, out_path)
    return out_path


# ---------- Last-prompt + recent-thumbnails helpers (ADD ONCE) ---------------

# Reuse your OUT_DIR; only define THUMBS here
THUMBS = (OUT_DIR / "thumbs"); THUMBS.mkdir(parents=True, exist_ok=True)
LAST_PROMPT_FILE = OUT_DIR / "last_prompt.txt"

def load_last_prompt(default_text: str = "") -> str:
    try:
        txt = LAST_PROMPT_FILE.read_text(encoding="utf-8").strip()
        return txt or default_text
    except Exception:
        return default_text

def save_last_prompt(text: str) -> None:
    try:
        LAST_PROMPT_FILE.write_text(text or "", encoding="utf-8")
    except Exception:
        pass

# Make sure these point to your existing locations
def make_waveform_thumb(wav_path: Path, png_path: Path, width=640, height=160):
    """Create a small waveform thumbnail PNG for the gallery."""
    import numpy as np, matplotlib.pyplot as plt, soundfile as sf
    try:
        wav_path = Path(wav_path)
        png_path = Path(png_path)
        y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if getattr(y, "ndim", 1) == 2:
            y = y.mean(axis=1)
        # Downsample for speed if very long
        if len(y) > 200_000:
            step = max(1, len(y) // 200_000)
            y = y[::step]
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.plot(y)
        ax.axis("off")
        fig.savefig(str(png_path), dpi=100)
        plt.close(fig)
    except Exception as e:
        print("thumb make error:", e)

def recent_dataset_data(max_items=20):
    """
    Build samples for gr.Dataset: one row per recent item with:
      [ <wav-filepath>, <thumbnail-png-filepath> ]
    Dataset will render a draggable file chip above the image.
    """
    items = []
    THUMBS.mkdir(parents=True, exist_ok=True)
    waves = sorted(OUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_items]
    for wav in waves:
        thumb = THUMBS / (wav.stem + ".png")
        if not thumb.exists():
            make_waveform_thumb(wav, thumb)
        # Use absolute paths so Gradio serves them
        items.append([str(wav.resolve()), str(thumb.resolve())])
    return items

def recent_gallery_data(max_items: int = 20):
    """
    Return a list of dicts that Gradio Gallery accepts:
    [{"image": PIL.Image, "label": "file.wav", "metadata": {"path": "/abs/file.wav"}}, ...]
    """
    items = []
    waves = sorted(OUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_items]
    for wav in waves:
        thumb = THUMBS / (wav.stem + ".png")
        if not thumb.exists():
            make_waveform_thumb(wav, thumb)
        try:
            img = Image.open(thumb).convert("RGB")
        except Exception as e:
            print("thumb open error:", thumb, e)
            continue
        items.append([img, wav.name])
    return items

def _coerce_path(x):
    try:
        return Path(x) if isinstance(x, (str, os.PathLike)) else None
    except Exception:
        return None

def _guess_audio_from_label_or_thumb(label_or_thumb: str) -> Path | None:
    """
    Try to resolve the clicked gallery item's label/thumb to a real audio file.
    - If it's already an audio path, return it.
    - If it looks like a PNG/JPG thumb, try same-stem .wav in OUT_DIR.
    - If it's just a label like 'gfm_YYYYMMDD_HHMMSS.wav', join with OUT_DIR.
    """
    p = Path(label_or_thumb)
    if p.suffix.lower() in AUDIO_EXTS and p.exists():
        return p

    # If it's a thumbnail path, try same stem with .wav in OUT_DIR
    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        stem = p.stem
        wav = OUT_DIR / f"{stem}.wav"
        if wav.exists():
            return wav

    # If it's a bare label/filename, try OUT_DIR / label
    candidate = OUT_DIR / label_or_thumb
    if candidate.suffix.lower() in AUDIO_EXTS and candidate.exists():
        return candidate

    # Also try stem.wav if no suffix
    if candidate.suffix == "" and (OUT_DIR / f"{candidate.name}.wav").exists():
        return OUT_DIR / f"{candidate.name}.wav"

    return None

def _resolve_audio_path(label_or_path: str, out_dir: Path) -> Path:
    """Accept a full path or just the filename; return an existing .wav Path."""
    p = Path(label_or_path)
    if p.suffix.lower() != ".wav":
        p = p.with_suffix(".wav")
    # Case 1: already a full path
    if p.exists():
        return p
    # Case 2: try OUT_DIR / filename
    cand = out_dir / p.name
    if cand.exists():
        return cand
    # Case 3: search OUT_DIR (non-recursive + recursive)
    for q in [*out_dir.glob(p.name), *out_dir.rglob(p.name)]:
        if q.is_file():
            return q
    raise FileNotFoundError(f"Audio not found for: {label_or_path}")

def _resolve_audio_by_label(label: str, out_dir: Path) -> Path:
    # Ensure it ends with .wav
    name = label if label.lower().endswith(".wav") else (label + ".wav")
    # Try direct join
    p = (out_dir / name)
    if p.exists():
        return p
    # Try non-recursive + recursive search
    for q in [*out_dir.glob(name), *out_dir.rglob(name)]:
        if q.is_file():
            return q
    raise FileNotFoundError(name)

def _play_recent_from_gallery(evt: gr.SelectData):
    val = evt.value
    # Gallery gives [image, label]
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        label = str(val[1])
    else:
        label = str(val)
    try:
        audio_path = _resolve_audio_by_label(label, OUT_DIR)  # OUT_DIR = your outputs dir
    except FileNotFoundError:
        raise gr.Error("Selected item is not an audio file.")
    return str(audio_path)  # or (sr, y) if your Audio expects waveform

def _resolve_wav_from_label(label: str) -> Path:
    name = Path(label).name
    p = OUT_DIR / (name if name.lower().endswith(".wav") else f"{Path(name).stem}.wav")
    if p.exists():
        return p
    # fallback search
    for q in OUT_DIR.rglob(f"{Path(name).stem}.wav"):
        if q.is_file():
            return q
    raise gr.Error("Selected item is not an audio file.")

def _play_recent_from_gallery_numpy(val):
    # val is [image, label]
    label = str(val[1]) if isinstance(val, (list, tuple)) and len(val) >= 2 else str(val)
    p = _resolve_wav_from_label(label)
    sr, y = sf.read(p, dtype="float32", always_2d=False)
    if hasattr(y, "ndim") and y.ndim == 2:  # optional mono mixdown
        y = y.mean(axis=1)
    return (sr, y)

def _last_generated_wav(out_dir: Path) -> str | None:
    waves = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(waves[0]) if waves else None

def _last_generated_numpy():
    waves = sorted(OUT_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not waves:
        raise gr.Error("No generated audio found.")
    p = waves[0]
    sr, y = sf.read(p, dtype="float32", always_2d=False)
    if hasattr(y, "ndim") and y.ndim == 2:
        y = y.mean(axis=1)
    return (sr, y)

# Add this utility once (near other helpers)
def _play_any_audio_file(path_str: str):
    """
    Load an audio path and return (sr, wav) for gr.Audio(type='numpy').
    Used for file_out.change(...) and for playing from the gallery.
    """
    if not path_str:
        return gr.update()
    p = Path(path_str)
    if not p.exists() or p.suffix.lower() not in AUDIO_EXTS:
        raise gr.Error("Please drop/select a valid audio file.")
    wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if getattr(wav, "ndim", 1) == 2:
        wav = wav.mean(axis=1)
    return (sr, wav)


def _make_waveform_thumb(audio_path: str) -> str | None:
    """
    Create (or reuse) a waveform PNG thumbnail for the given audio file.
    Return the PNG path on success, or None if we had to skip it.
    """
    try:
        p = Path(audio_path)
        if not p.exists() or p.suffix.lower() not in AUDIO_EXTS:
            return None

        thumb = THUMBS / (p.stem + ".png")
        if thumb.exists() and thumb.stat().st_mtime > p.stat().st_mtime:
            return str(thumb)

        wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) == 2:
            wav = wav.mean(axis=1)

        # draw a compact waveform
        plt.figure(figsize=(3.2, 1.2), dpi=100)
        n = len(wav)
        if n == 0:
            return None
        # downsample for drawing speed
        step = max(1, n // 1200)
        xs = np.arange(0, n, step) / sr
        ys = wav[::step]
        plt.plot(xs, ys, linewidth=0.8)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(thumb, bbox_inches="tight", pad_inches=0)
        plt.close()
        return str(thumb)
    except Exception as e:
        print("thumb error:", audio_path, e)
        try:
            # last-ditch fallback (still a valid PNG)
            thumb = THUMBS / (Path(audio_path).stem + "_fallback.png")
            plt.figure(figsize=(3.2, 1.2), dpi=100)
            plt.text(0.5, 0.5, "Audio", ha="center", va="center")
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(thumb, bbox_inches="tight", pad_inches=0)
            plt.close()
            return str(thumb)
        except Exception as e2:
            print("fallback thumb error:", e2)
            return None

def _scan_recent(limit: int = 20):
    """Build RECENT_ITEMS and the Gallery payload."""
    RECENT_ITEMS.clear()
    items = []
    # newest first
    files = sorted(OUT_DIR.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files:
        if p.suffix.lower() not in AUDIO_EXTS:
            continue

        # thumbnail path
        thumb = THUMBS / (p.stem + ".png")
        if not thumb.exists():
            _make_waveform_thumb(str(p))   # creates THUMBS/<stem>.png

        # open the PNG and convert to RGB
        try:
            with Image.open(thumb) as im:
                img = im.convert("RGB").copy()  # copy() so file handle is closed
        except Exception as e:
            print("thumb open error:", thumb, e)
            continue

        # optional: keep a recent index for quick lookup
        RECENT_ITEMS.append((str(p), thumb))

        # >>> Gradio Gallery wants [image, label]
        items.append([img, p.name])

        if len(items) >= limit:
            break

    return items

def refresh_recent_gallery():
    """Callable for .load() / button / .then() to fill the Gallery."""
    return _scan_recent()

def _play_any_audio_file(path_str: str):
    """Load any audio path and return (sr, wav) for gr.Audio(type='numpy')."""
    if not path_str:
        return gr.update()
    p = Path(path_str)
    if not p.exists() or p.suffix.lower() not in AUDIO_EXTS:
        raise gr.Error("Please drop/select a valid audio file.")
    wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if getattr(wav, "ndim", 1) == 2:
        wav = wav.mean(axis=1)
    return (sr, wav)

def _on_recent_select(evt: gr.SelectData):
    """Gallery select → play the corresponding audio in Output."""
    idx = evt.index
    if idx is None or idx < 0 or idx >= len(RECENT_ITEMS):
        return gr.update()
    audio_path, _ = RECENT_ITEMS[idx]
    return _play_any_audio_file(audio_path)

def _as_str(x):
    # Coerce Path/str to str; ignore everything else
    try:
        return str(x) if isinstance(x, (str, Path)) else None
    except:
        return None

def _safe_gallery(items):
    """
    Accepts either:
      - [(thumb_path, caption), ...]  OR
      - [thumb_path, ...]
    Returns a clean list **only** containing (existing_png_path, caption).
    Drops anything invalid. Never returns audio paths.
    """
    cleaned = []
    for it in (items or []):
        if isinstance(it, (list, tuple)) and len(it) >= 1:
            thumb = _as_str(it[0])
            cap   = str(it[1]) if len(it) > 1 else ""
        else:
            thumb = _as_str(it)
            cap   = ""
        if not thumb:
            continue
        p = Path(thumb)
        # Only allow real image files (png/jpg/jpeg)
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if not p.exists():
            continue
        cleaned.append((str(p), cap))
    return cleaned

# ============================================================================```

# ---- Prompt builder ---------------------------------------------------------
STYLE_OPTIONS = [
    "Afrobeat", "Afro-fusion", "Highlife", "Calypso", "Soca", "Reggae",
    "Hip-hop", "Trap", "Lo-fi", "EDM", "House", "Techno", "Trance",
    "Jazz", "Funk", "Soul", "R&B", "Pop", "Rock",
    "Orchestral", "Cinematic", "Ambient", "World"
]

RHYTHM_OPTIONS = [
    "4/4 straight", "3/4 waltz", "6/8 Afro-groove", "Syncopated",
    "Polyrhythm (3:2)", "Shuffle/Swing", "Clave (son)", "Clave (rumba)"
]

KEY_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALES = ["Major", "Minor", "Dorian", "Mixolydian", "Lydian", "Phrygian", "Aeolian", "Pentatonic"]

PRESETS = {
    "Lo-Fi Chill": dict(style="Lo-fi", rhythm="4/4 straight", bpm=82, temperature=0.9, cfg=3.0, top_k=250),
    "Dance Floor": dict(style="EDM", rhythm="4/4 straight", bpm=128, temperature=1.1, cfg=2.5, top_k=350),
    "Epic Orchestral": dict(style="Cinematic", rhythm="4/4 straight", bpm=100, temperature=0.8, cfg=4.5, top_k=250),
    "Afro Highlife": dict(style="Highlife", rhythm="6/8 Afro-groove", bpm=118, temperature=1.0, cfg=3.0, top_k=250),
    "Reggae Roots": dict(style="Reggae", rhythm="4/4 straight", bpm=75, temperature=1.0, cfg=3.5, top_k=250),
}

def build_prompt(user_prompt: str,
                 style: str, rhythm: str,
                 key_root: str, scale: str,
                 bpm: int) -> str:
    parts = []
    base = (user_prompt or "").strip()
    if base:
        parts.append(base)
    if style:
        parts.append(f"Style: {style}.")
    if rhythm:
        parts.append(f"Rhythm: {rhythm}.")
    if key_root and scale:
        parts.append(f"Key: {key_root} {scale}.")
    if bpm:
        parts.append(f"{bpm} BPM.")
    parts.append("High quality, rich arrangement, coherent structure.")
    return " ".join(parts)


# ---- Generation (chunked) ---------------------------------------------------
MAX_SEC_PER_CALL = 24.0
XF_SEC = 0.25


# --- Backend: generate_music (streams progress) ------------------------------
@torch.no_grad()
def generate_music(
    user_prompt: str,
    melody_file: Optional[Tuple[int, np.ndarray]],
    duration_sec: int,
    top_k: int,
    temperature: float,
    cfg: float,
    bpm: int,
    style: str,
    rhythm: str,
    key_root: str,
    scale: str,
    seed: int,
):
    # 1) Build the prompt & select model
    prompt = build_prompt(user_prompt, style, rhythm, key_root, scale, bpm)
    if not prompt:
        raise gr.Error("Please enter a prompt or pick a preset.")

    # NEW: persist the last prompt (no UI change)
    save_last_prompt(user_prompt)

    duration_sec = int(max(1, min(duration_sec, 1800)))
    use_melody = melody_file is not None and melody_file[1] is not None
    model = _load_melody() if use_melody else _load_small()

    if isinstance(seed, (int, float)) and seed >= 0:
        torch.manual_seed(int(seed))

    # 2) Prep melody for FIRST chunk
    melody_wav = None
    if use_melody:
        melody_sr, wav = melody_file
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        melody_wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        if melody_sr != SR:
            melody_wav = torchaudio.functional.resample(melody_wav, melody_sr, SR)

    # 3) Job state (resume if present)
    n_chunks = int(math.ceil(duration_sec / MAX_SEC_PER_CALL))
    job_id = _job_id_from_params(
        prompt, duration_sec, top_k, temperature, cfg, bpm,
        style, rhythm, key_root, scale, seed, use_melody
    )
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(exist_ok=True, parents=True)

    state = JobState.load(job_dir) or JobState(
        job_id=job_id,
        total_chunks=n_chunks,
        next_chunk=0,
        sr=SR,
        chunk_sec=MAX_SEC_PER_CALL,
        out_dir=str(job_dir),
        prompt=prompt,
    )
    state.total_chunks = n_chunks
    state.save()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4) Main loop (resume-aware) — stream progress with 3 outputs every time
    for i in range(state.next_chunk, n_chunks):
        remaining = max(0.0, duration_sec - i * MAX_SEC_PER_CALL)
        this_len = float(min(MAX_SEC_PER_CALL, max(1.0, remaining)))

        gen_kwargs = dict(
            use_sampling=True,
            temperature=float(temperature),
            cfg_coef=float(cfg),
            duration=this_len,
        )
        tk = int(top_k)
        if tk > 0:
            gen_kwargs["top_k"] = tk
        model.set_generation_params(**gen_kwargs)

        # Melody on first chunk
        if use_melody and i == 0 and melody_wav is not None:
            mel = melody_wav
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).float()
            if mel.dim() == 1:
                mel = mel.unsqueeze(0)
            mel = mel.to(device, non_blocking=True)

            out = model.generate_with_chroma(
                descriptions=[prompt],
                melody_wavs=[mel],
                melody_sample_rate=SR,
            )[0]
        else:
            out = model.generate([prompt])[0]

        # Save chunk immediately (checkpoint)
        out = to_mono(out.detach().cpu()).squeeze(0).numpy().astype(np.float32)
        np.save(job_dir / f"chunk_{i:03}.npy", out)

        # Update state
        state.next_chunk = i + 1
        state.save()

        # ---- STREAM PROGRESS (exactly 3 outputs) ----
        status = f"Chunk {i+1}/{n_chunks} done"
        yield gr.update(value=status), gr.update(), gr.update()

    # 5) Stitch all chunks we have (resume-safe)
    chunks = [np.load(job_dir / f"chunk_{i:03}.npy") for i in range(n_chunks)]
    audio = chunks[0] if len(chunks) == 1 else crossfade_concat(chunks, sr=SR, xfade_sec=XF_SEC)

    # Normalize lightly
    peak = float(np.max(np.abs(audio)) + 1e-8)
    if peak > 1.0:
        audio = (audio / peak).astype(np.float32)

    # Save final & build status
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"gfm_{ts}.wav"
    path = OUT_DIR / fname
    save_wav(audio, SR, path)

    # NEW: make/update a waveform thumbnail for gallery
    make_waveform_thumb(Path(path), THUMBS / (Path(path).stem + ".png"))

    final_status = f"Done ✅ Saved {fname}"

    # ---- FINAL YIELD (same 3 outputs) ----
    yield gr.update(value=final_status), (SR, audio), str(path)


# ---- Gradio UI --------------------------------------------------------------
with gr.Blocks(
    title=APP_NAME,
    theme=gr.themes.Soft(),
) as demo:
    # hide the API docs UI
    demo.show_api = False

    # --- Workaround: bypass API schema building (gradio-client 1.3.0 bug) ---
    def _no_api_info():
        # minimal structure expected by Gradio; prevents schema conversion
        return {"named_endpoints": {}, "unnamed_endpoints": []}
    demo.get_api_info = _no_api_info
    # -----------------------------------------------------------------------
    gr.Markdown(f"## {APP_NAME}")
    gr.Markdown(
        "Prompt-to-music using **Meta MusicGen (small)** with optional **melody conditioning**. "
        "This app supports **long form generation** by stitching multiple segments with crossfades."
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Prompt",
                value=load_last_prompt("West African highlife with guitars and shakers, cheerful, 120 BPM"),
                placeholder="West African highlife with guitars and shakers, cheerful, 120 BPM",
                lines=3,
            )

            melody = gr.Audio(label="🎵 Optional melody (mono)", type="numpy")

            with gr.Row():
                duration = gr.Slider(1, 1800, value=30, step=1, label="Duration (seconds)")
                top_k    = gr.Slider(0, 500, value=250, step=1, label="top_k")

            with gr.Row():
                temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.01, label="Temperature")
                cfg         = gr.Slider(0.0, 6.0,  value=3.0, step=0.1,  label="CFG (text strength)")

            with gr.Row():
                bpm  = gr.Slider(60, 200, value=120, step=1, label="BPM")
                seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")

            with gr.Row():
                style  = gr.Dropdown(STYLE_OPTIONS,  value="Afro-fusion",     label="Style")
                rhythm = gr.Dropdown(RHYTHM_OPTIONS, value="6/8 Afro-groove", label="Rhythm")

            with gr.Row():
                key_root = gr.Dropdown(KEY_ROOTS, value="C",     label="Key root")
                scale    = gr.Dropdown(SCALES,    value="Major", label="Scale")

            gr.Markdown("**Presets** (click to load settings):")
            with gr.Row():
                preset_btns = [gr.Button(name, size="sm") for name in PRESETS.keys()]

            generate_btn = gr.Button("🚀 Generate", variant="primary", scale=2)
            clear_btn    = gr.Button("Clear History", variant="secondary")

        # ========================== RIGHT COLUMN ==========================
        with gr.Column(scale=2):
            status_box = gr.Textbox(label="Status", interactive=False)

            # Output player: numpy; allow drag & drop of audio files
            audio_out = gr.Audio(label="🔊 Output", type="numpy", interactive=False)

            # Optional: user can drop/select any audio file here to preview in Output
            file_out = gr.File(label="Download", type="file")

            # --- Column-local callbacks ----------------------------------
            def _play_any_audio_file(path_str: str):
                if not path_str:
                    return gr.update()
                p = Path(path_str)
                if not p.exists() or p.suffix.lower() not in AUDIO_EXTS:
                    raise gr.Error("Please drop a valid audio file.")
                y, sr = sf.read(str(p), dtype="float32", always_2d=False)
                if getattr(y, "ndim", 1) == 2:
                    y = y.mean(axis=1)
                return (sr, y)

            file_out.change(_play_any_audio_file, inputs=file_out, outputs=audio_out)

            # History table
            hist = gr.Dataframe(
                headers=["time", "prompt", "file", "duration_s", "style", "rhythm", "key", "bpm", "top_k", "temperature", "cfg", "seed"],
                datatype=["str", "str", "str", "number", "str", "str", "str", "number", "number", "number", "number", "number"],
                label="Generation History",
                interactive=False,
                wrap=True,
            )

            # Recent thumbnails
            recent = gr.Gallery(label="Recent (last 20)", columns=5, rows=4, height=600, preview=True)

            # Buttons (define BEFORE binding)
            last_btn = gr.Button("Last generated file")
            refresh_recent_btn = gr.Button("↻ Refresh recent", size="sm")

            # Load gallery after UI mounts
            demo.load(lambda: recent_gallery_data(), outputs=recent)

            # Find newest .wav and return (sr, y)
            def _last_generated_numpy():
                waves = sorted(OUT_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
                if not waves:
                    raise gr.Error("No generated audio found.")
                p = waves[0]
                y, sr = sf.read(str(p), dtype="float32", always_2d=False)
                if getattr(y, "ndim", 1) == 2:
                    y = y.mean(axis=1)
                return (sr, y)

            # Resolve outputs/<label>.wav → read and play (sr, y)
            DOUBLE_CLICK_WINDOW = 0.8  # seconds
            last_click_index = gr.State(-1)
            last_click_time  = gr.State(0.0)

            def _resolve_wav_from_label(label: str) -> Path:
                name = Path(label).name
                p = OUT_DIR / (name if name.lower().endswith(".wav") else f"{Path(name).stem}.wav")
                if p.exists():
                    return p
                for q in OUT_DIR.rglob(f"{Path(name).stem}.wav"):
                    if q.is_file():
                        return q
                raise gr.Error("Selected item is not an audio file.")

            def _coerce_label(val):
                # Gallery item should be [image, label]; accept plain string too
                if val is None:
                    return None
                if isinstance(val, (list, tuple)):
                    if len(val) >= 2:
                        return str(val[1])
                    if len(val) == 1:
                        return str(val[0])
                if isinstance(val, str):
                    return val
                return None

            def _play_recent_from_gallery_numpy(
                val,
                last_idx: int,
                last_ts: float,
                evt: gr.SelectData | None = None,   # evt can be None on some builds
            ):
                import time as _t
                # On load (evt is None) -> do nothing
                if evt is None:
                    return gr.update(), last_idx, last_ts

                idx = getattr(evt, "index", None)
                if idx is None:
                    return gr.update(), last_idx, last_ts

                now = _t.time()

                # Only act on double-click: same tile twice within 0.5s
                if last_idx == idx and (now - last_ts) <= 0.5:
                    label = _coerce_label(val)
                    if not label:
                        return gr.update(), -1, 0.0
                    p = _resolve_wav_from_label(label)
                    wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
                    if getattr(wav, "ndim", 1) == 2:
                        wav = wav.mean(axis=1)
                    return (sr, wav), -1, 0.0   # play + reset click state

                # Single click -> do nothing (just remember click for the next one)
                return gr.update(), idx, now

            # Binding: ONLY double-click will produce audio; load & single click do nothing.
            recent.select(
                _play_recent_from_gallery_numpy,
                inputs=[last_click_index, last_click_time],
                outputs=[audio_out, last_click_index, last_click_time],
            )
    

    # ---- State + Callbacks ----
    state_hist = gr.State([])  # keep as a Python list (not a DataFrame)

    # meta_out   = gr.JSON(label="Meta", visible=False)   # or: gr.State()
    meta_out   = gr.State({})               # or: gr.JSON(visible=False)
    history_state = gr.State([])               # list of dict rows
    history_df = gr.Dataframe(headers=["time","prompt","duration","seed","file"],
                                interactive=False)

    def on_preset(name: str):
        conf = PRESETS.get(name, {})
        return (
            conf.get("style", gr.update()),
            conf.get("rhythm", gr.update()),
            conf.get("bpm", gr.update()),
            conf.get("temperature", gr.update()),
            conf.get("cfg", gr.update()),
            conf.get("top_k", gr.update()),
        )

    for btn, name in zip(preset_btns, PRESETS.keys()):
        btn.click(
            fn=on_preset,                              # on_preset(name) -> returns the 6 outputs
            inputs=gr.State(name),                     # ✅ give on_preset its one argument
            outputs=[style, rhythm, bpm, temperature, cfg, top_k],
        )

    # NOTE: We call generate_music directly. It streams THREE outputs:
    #   1) status_box (text)  2) audio_out (numpy tuple)  3) file_out (path)
    generate_btn.click(
        fn=generate_music,
        inputs=[prompt, melody, duration, top_k, temperature, cfg, bpm, style, rhythm, key_root, scale, seed],
        outputs=[status_box, audio_out, file_out, meta_out],
    ).then(
        fn=_append_history,
        inputs=[history_state, meta_out, file_out],
        outputs=history_state,                      # returns only the new state
    ).then(
        fn=_sync_history,
        inputs=[history_state],
        outputs=[history_df],                       # or [history_df, history_gallery]
    ).then(
        # After generation finishes, append a history row (no streaming here).
        fn=lambda prompt_val, duration_val, topk_val, temp_val, cfg_val, bpm_val,
               style_val, rhythm_val, keyroot_val, scale_val, seed_val, hist_list, fpath: (
                   # return (new_state_hist, hist_table_update)
                   (hist_list or []) + [{
                       "time": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
                       "prompt": str(prompt_val or ""),
                       "file": str(fpath) if fpath else "",
                       "duration_s": int(duration_val),
                       "style": str(style_val),
                       "rhythm": str(rhythm_val),
                       "key": f"{keyroot_val} {scale_val}",
                       "bpm": int(bpm_val),
                       "top_k": int(topk_val),
                       "temperature": float(temp_val),
                       "cfg": float(cfg_val),
                       "seed": int(seed_val) if isinstance(seed_val, (int, float)) else 0,
                   }],
                   gr.update(value=(hist_list or []) + [{
                       "time": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
                       "prompt": str(prompt_val or ""),
                       "file": str(fpath) if fpath else "",
                       "duration_s": int(duration_val),
                       "style": str(style_val),
                       "rhythm": str(rhythm_val),
                       "key": f"{keyroot_val} {scale_val}",
                       "bpm": int(bpm_val),
                       "top_k": int(topk_val),
                       "temperature": float(temp_val),
                       "cfg": float(cfg_val),
                       "seed": int(seed_val) if isinstance(seed_val, (int, float)) else 0,
                   }])
               ),
        inputs=[prompt, duration, top_k, temperature, cfg, bpm, style, rhythm, key_root, scale, seed, state_hist, file_out],
        outputs=[state_hist, hist],
    ).then(
        # Now refresh the bottom-right Recent gallery
        fn=lambda: recent_gallery_data(),
        inputs=None,
        outputs=recent,
    )

    def do_clear():
        return [], gr.update(value=[])

    clear_btn.click(do_clear, inputs=[], outputs=[state_hist, hist])

    # ---- Mixtape Builder UI (added) -----------------------------------------
    with gr.Accordion("🎛️ Mixtape Builder (sequence clips with crossfades)", open=False):
        gr.Markdown(
            "Edit the table (one row per segment) and click **Build Mixtape**. "
            "Each row is rendered with your current generator and stitched using beat-aligned crossfades."
        )

        mix_headers  = ["prompt","duration_sec","bpm","style","rhythm","key_root","scale","seed","top_k","temperature","cfg"]
        mix_defaults = [
            ["Highlife intro, bright guitars, gentle shakers", 120, 120, "Afro-fusion", "6/8 Afro-groove", "C", "Major", 1001, 90, 1.0, 3.0],
            ["Afrobeat groove, horns enter, call-and-response", 180, 120, "Afro-fusion", "4/4 Syncopated",  "F", "Major", 1002, 90, 1.0, 3.0],
            ["Soukous flair, fast picking guitars, uplifting",   180, 122, "Afro-fusion", "4/4 Driving",     "G", "Major", 1003, 90, 1.0, 3.0],
        ]

        mix_df = gr.Dataframe(
            headers=mix_headers,
            value=mix_defaults,
            datatype=["str","number","number","str","str","str","str","number","number","number","number"],
            label="Mixtape Segments (edit/add rows)",
            interactive=True,
            wrap=True,
            row_count=(3, "dynamic"),
        )

        with gr.Row():
            bpm_xfade   = gr.Slider(60, 200, value=120, step=1, label="Crossfade BPM grid")
            xfade_beats = gr.Slider(1, 16,  value=8,   step=1, label="Crossfade length (beats)")

        build_mix_btn = gr.Button("🎚️ Build Mixtape", variant="primary")

        # Separate outputs (use filepath for audio to avoid streaming big arrays)
        mix_status = gr.Textbox(label="Mix status", interactive=False)
        mix_audio  = gr.Audio(label="Mixtape (preview/download)", type="filepath", interactive=False)
        mix_file   = gr.File(label="Final mixtape file (.wav)")

        @torch.no_grad()
        def build_mixtape_ui(rows, bpm_grid, beats):
            import os  # local import to avoid relying on top-level
            # Parse rows -> segments for your generator
            segments = []
            for r in (rows or []):
                if not r or not str(r[0]).strip():
                    continue
                segments.append(dict(
                    user_prompt=str(r[0]),
                    melody_file=None,                        # extend later if you add per-segment melody
                    duration_sec=int(r[1]),
                    top_k=int(r[8]),
                    temperature=float(r[9]),
                    cfg=float(r[10]),
                    bpm=int(r[2]),
                    style=str(r[3]),
                    rhythm=str(r[4]),
                    key_root=str(r[5]),
                    scale=str(r[6]),
                    seed=int(r[7]),
                ))
            if not segments:
                raise gr.Error("Please add at least one segment (prompt + duration).")

            # Render each segment using your existing generate_music()
            paths = []
            total = len(segments)
            for i, seg in enumerate(segments, 1):
                yield gr.update(value=f"Rendering segment {i}/{total}…"), gr.update(), gr.update()
                p = render_segment_to_path(**seg)
                paths.append(p)

            # Stitch with beat-aligned crossfades
            yield gr.update(value="Stitching segments…"), gr.update(), gr.update()
            out_path = stitch_paths_to_wav(paths, bpm_for_xfade=float(bpm_grid), xfade_beats=float(beats))

            # Final: status + audio filepath + downloadable file
            return gr.update(value=f"Done ✅ Saved {os.path.basename(out_path)}"), out_path, out_path

        build_mix_btn.click(
            fn=build_mixtape_ui,
            inputs=[mix_df, bpm_xfade, xfade_beats],
            outputs=[mix_status, mix_audio, mix_file],
            queue=True,
        ).then(
            fn=lambda: recent_gallery_data(),
            inputs=None,
            outputs=recent,
        )
    # ---- end Mixtape Builder UI ---------------------------------------------
    

# Stable queue: one worker; frequent status updates to keep WS alive
# Enable global queue (Gradio ≥ 4.x)
demo.queue(max_size=8, status_update_rate=1)

def _is_cloud_env() -> bool:
    """
    Heuristics to detect 'remote' environments where we must bind to 0.0.0.0
    (GitHub Codespaces, HF Spaces, many PaaS).
    """
    return any([
        os.getenv("CODESPACES"),
        os.getenv("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN"),
        os.getenv("HF_SPACE_ID") or os.getenv("SPACE_ID"),
        os.getenv("RENDER"),
        os.getenv("RAILWAY_STATIC_URL"),
        os.getenv("FLY_APP_NAME"),
        os.getenv("PORT")  # many hosts set PORT
    ])

def _pick_port(start=7860, end=7890) -> int:
    # 1) Respect env override if provided
    env_port = os.getenv("GRADIO_SERVER_PORT") or os.getenv("PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            pass

    # 2) Otherwise find the first free port in [start, end]
    bind_host = "0.0.0.0" if _is_cloud_env() else "127.0.0.1"
    for p in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((bind_host, p))
                return p
            except OSError:
                continue
    raise OSError(f"No free port found in range {start}-{end}")

if __name__ == "__main__":
    PORT = _pick_port()
    HOST = "0.0.0.0" if _is_cloud_env() else "127.0.0.1"

    demo.launch(
        share=True,            # <-- add this in Codespaces/remote
        inbrowser=False,       # don't try to open a local browser tab
        debug=True,
        show_error=True,
        show_api=False,        # see #2 below; avoids schema crash path
        server_name=HOST,
        server_port=PORT,
    )