# Professional demo generator + FFmpeg exporter
# Usage:
#   python scripts/generate.py --album meditation --seconds 10
# Outputs:
#   assets/outputs/<album>_<timestamp>.wav/.mp3/.flac

from pathlib import Path
import argparse
import datetime
import subprocess
import shutil
import numpy as np

try:
    import soundfile as sf  # pip install soundfile
except ImportError:
    raise SystemExit("Missing dependency: soundfile. Run: pip install soundfile")

SR = 44100  # sample rate


def envelope(sig, fade=0.02):
    """Apply short fade-in/out to avoid clicks."""
    n = sig.shape[0]
    nf = max(1, int(fade * SR))
    env = np.ones(n, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, nf, dtype=np.float32)
    env[:nf] = ramp
    env[-nf:] = ramp[::-1]
    return sig * env


def gen_meditation(seconds: int) -> np.ndarray:
    """Soft drones + slow LFO, stereo."""
    t = np.arange(int(seconds * SR)) / SR
    f1, f2 = 110.0, 220.0
    lfo = 0.3 * np.sin(2 * np.pi * 0.15 * t)  # slow vibrato
    left = 0.25 * np.sin(2 * np.pi * (f1 + lfo) * t)
    right = 0.25 * np.sin(2 * np.pi * (f2 + lfo) * t)
    stereo = np.stack([envelope(left), envelope(right)], axis=-1).astype(np.float32)
    return stereo


def gen_roadtrip(seconds: int) -> np.ndarray:
    """Upbeat arps + noise hiss, stereo."""
    t = np.arange(int(seconds * SR)) / SR
    freqs = np.array([220, 277, 330, 440, 554, 660], dtype=np.float32)
    step = int(SR * 0.25)  # change note every 250 ms
    note_idx = (np.arange(t.size) // step) % freqs.size
    f = freqs[note_idx]
    base = 0.18 * np.sin(2 * np.pi * f * t)
    hiss = 0.02 * np.random.randn(t.size)
    sig = envelope(base + hiss)
    stereo = np.stack([sig, sig], axis=-1).astype(np.float32)
    return stereo


def gen_dancefusion(seconds: int) -> np.ndarray:
    """Kick + bass pulse, stereo."""
    n = int(seconds * SR)
    sig = np.zeros(n, dtype=np.float32)
    kick_period = int(0.5 * SR)
    kick_len = int(0.08 * SR)
    for i in range(0, n, kick_period):
        k = np.exp(-np.linspace(0, 6, kick_len)).astype(np.float32)
        sig[i:i + kick_len] += 0.6 * k * np.sin(2 * np.pi * 60 * np.arange(kick_len) / SR).astype(np.float32)
    t = np.arange(n) / SR
    bass = 0.15 * np.sin(2 * np.pi * 90 * t)
    mix = envelope(sig + bass)
    stereo = np.stack([mix, mix], axis=-1).astype(np.float32)
    return stereo


ALBUMS = {
    "meditation": gen_meditation,
    "roadtrip": gen_roadtrip,
    "dancefusion": gen_dancefusion,
}


def export_with_ffmpeg(wav_path: Path, exports=("mp3", "flac")):
    """Convert WAV → requested formats using system ffmpeg."""
    if shutil.which("ffmpeg") is None:
        print("⚠️  FFmpeg not found in PATH – skipping MP3/FLAC export.")
        return []

    made = []
    stem = wav_path.with_suffix("")
    for fmt in exports:
        out_path = stem.with_suffix("." + fmt)
        if fmt.lower() == "mp3":
            cmd = ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "192k", str(out_path)]
        elif fmt.lower() == "flac":
            cmd = ["ffmpeg", "-y", "-i", str(wav_path), str(out_path)]
        else:
            print(f"⚠️  Unsupported export format: {fmt}")
            continue

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"🎧 Exported: {out_path}")
        made.append(out_path)
    return made


def main():
    p = argparse.ArgumentParser(description="Generate demo audio clip + export to MP3/FLAC (no model required)")
    p.add_argument("--album", choices=ALBUMS.keys(), default="meditation")
    p.add_argument("--seconds", type=int, default=10)
    p.add_argument("--no-export", action="store_true", help="Do not export mp3/flac (WAV only)")
    args = p.parse_args()

    out_dir = Path("assets/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.album}_{ts}"
    wav_path = out_dir / (base + ".wav")

    # 1) Generate real audio
    wave = ALBUMS[args.album](args.seconds)
    sf.write(wav_path.as_posix(), wave, SR, subtype="PCM_16")
    print(f"✅ WAV saved: {wav_path}  ({args.seconds}s, {SR} Hz, stereo)")

    # 2) Export via FFmpeg
    if not args.no_export:
        export_with_ffmpeg(wav_path, exports=("mp3", "flac"))


if __name__ == "__main__":
    main()