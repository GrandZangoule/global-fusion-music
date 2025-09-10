#!/usr/bin/env bash
# Usage:
#   ./run_app.sh                    # auto
#   ./run_app.sh amd                # force cpu | amd | nvidia
#   ./run_app.sh auto app.py --arg1 # backend entry other-args...

set -euo pipefail

BACKEND="${1:-auto}"
ENTRY="${2:-app.py}"
shift || true; [[ $# -gt 0 ]] && shift || true  # consume up to two args safely

detect_backend() {
  if command -v nvidia-smi >/dev/null 2>&1; then echo "nvidia"; return; fi
  if command -v lspci >/dev/null 2>&1 && lspci | grep -qi "AMD/ATI"; then echo "amd"; return; fi
  echo "cpu"
}

if [[ "$BACKEND" == "auto" ]]; then BACKEND="$(detect_backend)"; fi

case "$BACKEND" in
  nvidia) source ./venvNVIDIA/bin/activate ;;
  amd)    source ./venvAMD/bin/activate    ;;   # (ROCm wheels expected on Linux)
  cpu)    source ./venv/bin/activate       ;;
  *)      echo "Unknown backend: $BACKEND"; exit 1 ;;
esac

echo "▶ Backend: $BACKEND"
python "$ENTRY" "$@"