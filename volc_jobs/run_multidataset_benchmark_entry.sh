#!/usr/bin/env bash
set -euo pipefail

echo "[bench-entry] hostname=$(hostname)"
echo "[bench-entry] date=$(date -Is)"
echo "[bench-entry] nvidia-smi"
nvidia-smi || true

cd /home/vepfs/MIXjyx/DataFlex-main

python -V
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
PY

python -m pip install -U pip
LLAMAFACTORY_VERSION="$(python - <<'PY'
import sys
print("0.9.4" if sys.version_info >= (3, 11) else "0.9.3")
PY
)"
echo "[bench-entry] installing llamafactory==${LLAMAFACTORY_VERSION}"

python -m pip install \
  "llamafactory==${LLAMAFACTORY_VERSION}" \
  "traker==0.3.2" \
  "datasets==3.6.0" \
  "huggingface_hub>=0.24.0" \
  "pandas<3.0.0" \
  "pyarrow>=15.0.0" \
  "pyyaml" \
  "omegaconf" \
  "swanlab" \
  "gradio>=4.38.0,<=5.12.0" \
  "tyro==0.8.14" \
  "fire" \
  "tiktoken" \
  "sentencepiece" \
  "zstandard" \
  "matplotlib"
python -m pip install --no-build-isolation --no-deps -e .

export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"

python -m py_compile src/dataflex/train/selector/cluster_less_selector.py volc_jobs/run_multidataset_benchmark.py
python volc_jobs/run_multidataset_benchmark.py
