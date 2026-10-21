#!/usr/bin/env bash
# Session 0 of the BEATX continual speaker benchmark (scott), trained from scratch.
# No HDF5 cache: each run builds its speakers' chunks in memory from the raw
# BEATX files (beatx_source: raw in configs/cl_beatx/*.yaml).
# Run one stage at a time from anywhere inside the repo:
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/cl/session0_scott.sh codec upperhands
#   CUDA_VISIBLE_DEVICES=1 bash scripts/cl/session0_scott.sh codec lowertransvel
#   CUDA_VISIBLE_DEVICES=2 bash scripts/cl/session0_scott.sh codec faceexp
#   bash scripts/cl/session0_scott.sh link                 # experiments/cl_codec_*_pool -> latest codec runs
#   CUDA_VISIBLE_DEVICES=0 bash scripts/cl/session0_scott.sh glm
#   CUDA_VISIBLE_DEVICES=0 bash scripts/cl/session0_scott.sh eval experiments/<run>/last_<N>.safetensors
#
# Extra arguments after a stage are forwarded to train.py / test.py (e.g. --batch_size 64).
set -euo pipefail
cd "$(dirname "$0")/../.."

CODECS="upperhands lowertransvel faceexp"

stage=${1:-}
case "$stage" in
  codec)
    codec=${2:?usage: codec <upperhands|lowertransvel|faceexp> [train.py args]}
    python scripts/train.py --config "configs/cl_beatx/codec_${codec}_pool.yaml" "${@:3}"
    ;;
  link)
    for codec in $CODECS; do
      run=$(ls -d experiments/*_cl_codec_${codec}_pool 2>/dev/null | sort | tail -1 || true)
      if [ -z "$run" ]; then
        echo "no run dir matching experiments/*_cl_codec_${codec}_pool" >&2
        exit 1
      fi
      if [ ! -f "$run/last_2000.safetensors" ]; then
        echo "warning: $run has no last_2000.safetensors (codec not finished?)" >&2
      fi
      ln -sfn "$(basename "$run")" "experiments/cl_codec_${codec}_pool"
      echo "experiments/cl_codec_${codec}_pool -> $run"
    done
    ;;
  glm)
    python scripts/train.py --config configs/cl_beatx/gtdm3_s0_scott.yaml "${@:2}"
    ;;
  eval)
    ckpt=${2:?usage: eval <experiments/<run>/last_<N>.safetensors> [test.py args]}
    python scripts/test.py --config "$(dirname "$ckpt")/config.yaml" --test_ckpt "$ckpt" \
        --dataset_ratio speakers_beatx_fulllength \
        --beatx_speakers scott --batch_size 1 --cfg_scale 2.3 "${@:3}"
    ;;
  *)
    sed -n '2,14p' "$0"
    exit 1
    ;;
esac
