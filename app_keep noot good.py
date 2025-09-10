# app.py
# Global Fusion Music – Pro UI with long-form generation (chunking + crossfade)
# Requires: gradio >= 4.44.1; audiocraft installed; torch/torchaudio installed.

import os, sys
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



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

    ops.memory_efficient_attention = memory_efficient_attention
    ops.scaled_dot_product_efficient_attention = scaled_dot_product_efficient_attention
    mod.ops = ops
# -------------------------------------------------------------------------------
    

# ---- Model (Meta MusicGen via audiocraft) ----------------------------------
from pathlib import Path
from typing import List, Tuple, Optional
from audiocraft.models import MusicGen
from dataclasses import dataclass, asdict

# --- neutralize CUDA-only xformers checks in Audiocraft (safe for CPU) ---
import audiocraft.modules.transformer as _ac_t
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/musicgen-small"
MELODY_MODEL = "facebook/musicgen-melody"

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


# --- persistence + thumbs config -------------------------------------------------
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
THUMBS = OUT_DIR / "thumbs"
THUMBS.mkdir(parents=True, exist_ok=True)

LAST_PROMPT = OUT_DIR / "last_prompt.txt"
DEFAULT_PROMPT = "West African highlife with guitars and shakers and Afro-Cuban music, 120 BPM"

def load_last_prompt() -> str:
    try:
        txt = LAST_PROMPT.read_text(encoding="utf-8").strip()
        return txt or DEFAULT_PROMPT
    except Exception:
        return DEFAULT_PROMPT

def save_last_prompt(p: str) -> None:
    try:
        LAST_PROMPT.write_text(p or "", encoding="utf-8")
    except Exception:
        pass

# ---------- thumbnails ----------
def make_waveform_thumb(wav_path: Path, thumb_path: Path, width=640, height=120):
    """Render a lightweight waveform PNG from a WAV/MP3/FLAC file."""
    try:
        audio, sr = sf.read(str(wav_path), always_2d=True)
        y = audio.mean(axis=1)  # mono for the plot
        # downsample for speed
        target_pts = width * 2
        step = max(1, int(len(y) / target_pts))
        y_ds = y[::step]

        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.axis("off"); plt.margins(0, 0)
        plt.plot(y_ds, linewidth=0.8)
        plt.tight_layout(pad=0)
        plt.savefig(str(thumb_path), bbox_inches="tight", pad_inches=0)
        plt.close()
    except Exception:
        # fallback blank
        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.axis("off"); plt.tight_layout(pad=0)
        plt.savefig(str(thumb_path), bbox_inches="tight", pad_inches=0)
        plt.close()

def recent_items(limit=10):
    """
    Returns newest-first list of dicts:
    [{ 'thumb': 'outputs/thumbs/foo.png', 'audio': 'outputs/foo.wav', 'caption': 'foo.wav' }, ...]
    """
    pairs = []
    # newest audio first
    for wav in sorted(OUT_DIR.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if wav.suffix.lower() not in {".wav", ".mp3", ".flac"}:
            continue
        thumb = THUMBS / (wav.stem + ".png")
        if thumb.exists():
            pairs.append({"thumb": thumb.as_posix(), "audio": wav.as_posix(), "caption": wav.name})
            if len(pairs) >= limit:
                break
    return pairs

def gallery_data(limit=10):
    """What the Gradio Gallery consumes: list of (image_path, caption)."""
    items = recent_items(limit)
    return [(it["thumb"], it["caption"]) for it in items]

def play_from_gallery(evt: gr.SelectData):
    """On click in Gallery, load the matching audio into the player."""
    items = recent_items(10)
    idx = evt.index
    if 0 <= idx < len(items):
        return items[idx]["audio"]
    return None


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

    # NEW: persist the raw user prompt as the “last prompt”
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

    # 4) Main loop (resume-aware) — stream progress with 5 outputs every time
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

        # ---- STREAM PROGRESS (exactly 5 outputs) ----
        # 1) status text update
        # 2) audio: no change yet    -> gr.update()
        # 3) last file: no change    -> gr.update()
        # 4) preview player: no change -> gr.update()
        # 5) recent gallery: no change -> gr.update()
        status = f"Chunk {i+1}/{n_chunks} done"
        yield (
            gr.update(value=status),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

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

    # NEW: make/update thumbnail for the gallery
    make_waveform_thumb(path, THUMBS / (path.stem + ".png"))

    final_status = f"Done ✅ Saved {fname}"

    # ---- FINAL YIELD (5 outputs) ----
    # 1) status text with success message
    # 2) audio tuple for gr.Audio(type="numpy")
    # 3) path string for "Last generated file"
    # 4) file path for preview player (type="filepath")
    # 5) refreshed gallery data
    yield (
        gr.update(value=final_status),
        (SR, audio),
        str(path),
        str(path),
        gallery_data(),
    )


# ---- Gradio UI --------------------------------------------------------------
with gr.Blocks(title=APP_NAME, theme=gr.themes.Soft(), fill_height=True) as demo:
    gr.Markdown(f"## {APP_NAME}")
    gr.Markdown(
        "Prompt-to-music using **Meta MusicGen (small)** with optional **melody conditioning**. "
        "This app supports **long form generation** by stitching multiple segments with crossfades."
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="e.g. West African highlife with guitars and shakers, cheerful, 120 BPM",
                lines=3,
            )

            melody = gr.Audio(
                label="🎵 Optional melody (mono)",
                type="numpy",
            )

            with gr.Row():
                duration = gr.Slider(1, 1800, value=30, step=1, label="Duration (seconds)")
                top_k = gr.Slider(0, 500, value=250, step=1, label="top_k")

            with gr.Row():
                temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.01, label="Temperature")
                cfg = gr.Slider(0.0, 6.0, value=3.0, step=0.1, label="CFG (text strength)")

            with gr.Row():
                bpm = gr.Slider(60, 200, value=120, step=1, label="BPM")
                seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")

            with gr.Row():
                style = gr.Dropdown(STYLE_OPTIONS, value="Afro-fusion", label="Style")
                rhythm = gr.Dropdown(RHYTHM_OPTIONS, value="6/8 Afro-groove", label="Rhythm")

            with gr.Row():
                key_root = gr.Dropdown(KEY_ROOTS, value="C", label="Key root")
                scale = gr.Dropdown(SCALES, value="Major", label="Scale")

            gr.Markdown("**Presets** (click to load settings):")
            with gr.Row():
                preset_btns = [gr.Button(name, size="sm") for name in PRESETS.keys()]

            generate_btn = gr.Button("🚀 Generate", variant="primary", scale=2)
            clear_btn = gr.Button("Clear History", variant="secondary")

        with gr.Column(scale=2):
            # ── Option A outputs: Status, Audio, Last file ─────────────────────
            status_box = gr.Textbox(label="Status", interactive=False)

            # IMPORTANT: audio_out must be type="numpy" since generate_music returns (sr, ndarray)
            audio_out = gr.Audio(label="🔊 Output", interactive=False, type="numpy")
            file_out = gr.File(label="Last generated file")

            # New: a separate file-backed player used when clicking a thumbnail
            audio_preview = gr.Audio(label="▶ Play selection", interactive=False, type="filepath")

            # New: bottom-right gallery of the last 10 generated files (waveform thumbnails)
            recent = gr.Gallery(
                label="Recent (last 10)",
                columns=2,
                rows=5,
                height=300,
                preview=True,
            )

        # History table remains, updated after completion via .then()
        hist = gr.Dataframe(
            headers=["time", "prompt", "file", "duration_s", "style", "rhythm", "key", "bpm", "top_k", "temperature", "cfg", "seed"],
            datatype=["str", "str", "str", "number", "str", "str", "str", "number", "number", "number", "number", "number"],
            label="Generation History",
            interactive=False,
            wrap=True,
        )
            
    # ---- State + Callbacks ----
    state_hist = gr.State([])  # keep as a Python list (not a DataFrame)

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
            on_preset,
            inputs=[],
            outputs=[style, rhythm, bpm, temperature, cfg, top_k],
        )

    # NOTE: We call generate_music directly. It streams THREE outputs:
    #   1) status_box (text)  2) audio_out (numpy tuple)  3) file_out (path)
    generate_btn.click(
        fn=generate_music,
        inputs=[prompt, melody, duration, top_k, temperature, cfg, bpm, style, rhythm, key_root, scale, seed],
        outputs=[status_box, audio_out, file_out],
        concurrency_limit=1,
        show_api=False,
        queue=True,
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
            ["Reggae flair, fast picking guitars, uplifting",   180, 122, "Afro-fusion", "4/4 Driving",     "A", "Major", 1002, 100, 1.0, 3.0],
            ["Afrobeat groove, horns enter, call-and-response", 180, 120, "Afro-fusion", "4/4 Syncopated",  "F", "Major", 1003, 90, 1.0, 3.0],
            ["Soukous flair, fast picking guitars, uplifting",   180, 122, "Afro-fusion", "4/4 Driving",     "G", "Major", 1004, 90, 1.0, 3.0],
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
            concurrency_limit=1,
            queue=True,
            show_api=False,
        )
             
        # ⬇️ Put these two lines here (still inside the Blocks context)
        recent.value = gallery_data()                       # initial fill when the app loads
        recent.select(fn=play_from_gallery, outputs=audio_preview)

    # ---- end Mixtape Builder UI ---------------------------------------------
    

# Stable queue: one worker; frequent status updates to keep WS alive
try:
    demo.queue(concurrency_count=1, max_size=8, status_update_rate=1)
except TypeError:
    demo.queue(max_size=8)

if __name__ == "__main__":
    demo.launch(
        share=False,
        inbrowser=True,
        debug=True,
        show_error=True,
        server_name="127.0.0.1",
        server_port=7860,
    )