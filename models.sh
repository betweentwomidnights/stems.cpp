#!/usr/bin/env bash
# Download stems.cpp GGUFs from Hugging Face (public repos) with curl. No Python needed.
#
# Usage: ./models.sh [--encoding f32|f16] [--namespace <hf-org>] [--out DIR] [--dry-run] [MODEL ...|all]
#   MODEL: htdemucs (default), htdemucs_6s, htdemucs_ft, mel_band_roformer_kim, bs_roformer_viperx_317
#   defaults: htdemucs, F32, into ./models
#
# Files follow docs/DISTRIBUTION.md, e.g. htdemucs-42M-v1.0-F32.gguf; stems-server finds them by
# model name. F32 is what every parity number in docs/PARITY.md was measured on. F16 is
# published only where it was measured to hold up; asking for it on a model without one fetches
# F32 and says so. bs_roformer_viperx_317 has no stated upstream license: read its model card
# before redistributing it. To convert the checkpoints yourself instead, see tools/convert_*.py.
# Windows: models.cmd, or run this from git-bash.
set -eu

ENCODING="f32"
NAMESPACE="thepatch"
OUT="models"
DRY_RUN=0
MODELS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --encoding)  ENCODING="$2"; shift ;;
    --namespace) NAMESPACE="$2"; shift ;;
    --out)       OUT="$2"; shift ;;
    --dry-run)   DRY_RUN=1 ;;
    -h|--help)   sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)          echo "unknown option: $1" >&2; exit 1 ;;
    *)           MODELS+=("$1") ;;
  esac
  shift
done
[ ${#MODELS[@]} -gt 0 ] || MODELS=(htdemucs)
if [ "${MODELS[0]}" = all ]; then
  MODELS=(htdemucs htdemucs_6s htdemucs_ft mel_band_roformer_kim bs_roformer_viperx_317)
fi

case "$ENCODING" in
  f32|F32) ENC=F32 ;;
  f16|F16) ENC=F16 ;;
  *) echo "unknown --encoding '$ENCODING' (expected f32|f16)" >&2; exit 2 ;;
esac

# PowerShell may resolve `bash` to WSL. Normalize a Windows drive path so
# `--out C:/models` does not become a literal relative `C:` directory there.
case "$OUT" in
  [A-Za-z]:/*) if command -v wslpath >/dev/null 2>&1; then OUT=$(wslpath -u "$OUT"); fi ;;
esac

# Models with a published F16, space-delimited.
F16_PUBLISHED=" htdemucs htdemucs_6s htdemucs_ft mel_band_roformer_kim "

spec() {   # model -> "<repo> <size label>"
  case "$1" in
    htdemucs)               echo "htdemucs-GGUF 42M" ;;
    htdemucs_6s)            echo "htdemucs-GGUF 27M" ;;
    htdemucs_ft)            echo "htdemucs-GGUF 4x42M" ;;
    mel_band_roformer_kim)  echo "mel-band-roformer-kim-GGUF 0.2B" ;;
    bs_roformer_viperx_317) echo "bs-roformer-viperx-317-GGUF 0.2B" ;;
    *) return 1 ;;
  esac
}

dl() {   # dl <repo> <filename>
  local repo="$1" file="$2" dst="$OUT/$2" part="$OUT/$2.part"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[plan] https://huggingface.co/$repo/resolve/main/$file -> $dst"
    return
  fi
  if [ -s "$dst" ]; then echo "[skip] $file"; return; fi
  if [ -f "$part" ]; then echo "[resume] $file"; else echo "[download] $repo/$file"; fi
  curl -fL --retry 3 --continue-at - -o "$part" "https://huggingface.co/$repo/resolve/main/$file"
  mv "$part" "$dst"
}

mkdir -p "$OUT"
for m in "${MODELS[@]}"; do
  s=$(spec "$m") || { echo "unknown model: $m (try --help)" >&2; exit 1; }
  repo=${s% *}; label=${s#* }
  enc="$ENC"
  if [ "$enc" = F16 ] && [ "${F16_PUBLISHED#* $m }" = "$F16_PUBLISHED" ]; then
    echo "[note] $m has no published F16; fetching F32"
    enc=F32
  fi
  dl "$NAMESPACE/$repo" "$m-$label-v1.0-$enc.gguf"
done
echo "[done] ${MODELS[*]} -> $OUT"
