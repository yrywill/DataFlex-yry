#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /root/.volc/.profile

TASK_NAME="${TASK_NAME:-dataflex-selector-batch-$(date +%m%d%H%M%S)}"
QUEUE_ID="${QUEUE_ID:-q-20260318205115-lvfng}"
FLAVOR="${FLAVOR:-ml.pni2.3xlarge}"
IMAGE_URL="${IMAGE_URL:-vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:2.7.1-cu12.6.3-py3.10-ubuntu20.04}"
VEPFS_ID="${VEPFS_ID:-vepfs-cnbj2c98dea54433}"
VEPFS_SUB_PATH="${VEPFS_SUB_PATH:-queue010/20252203250}"
ACTIVE_DEADLINE="${ACTIVE_DEADLINE:-12h}"

cat > dataflex_selector_batch.yaml <<YAML
TaskName: "${TASK_NAME}"
Description: "DataFlex selector batch: random vs cluster_less vs less."
Tags:
  - "dataflex"
  - "selector"
  - "cluster_less"
UserCodePath: "./"
RemoteMountCodePath: "/root/code"
Entrypoint: "bash /home/vepfs/MIXjyx/DataFlex-main/volc_jobs/run_selector_batch_entry.sh"
Envs:
  - Name: "DATAFLEX_BATCH_SELECTORS"
    Value: "${DATAFLEX_BATCH_SELECTORS:-random,cluster_less,less}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_SEEDS"
    Value: "${DATAFLEX_BATCH_SEEDS:-42,43,44}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_TRAIN_SIZE"
    Value: "${DATAFLEX_BATCH_TRAIN_SIZE:-2048}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_EVAL_SIZE"
    Value: "${DATAFLEX_BATCH_EVAL_SIZE:-256}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_MAX_STEPS"
    Value: "${DATAFLEX_BATCH_MAX_STEPS:-80}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_UPDATE_STEP"
    Value: "${DATAFLEX_BATCH_UPDATE_STEP:-20}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_WARMUP_STEP"
    Value: "${DATAFLEX_BATCH_WARMUP_STEP:-20}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_UPDATE_TIMES"
    Value: "${DATAFLEX_BATCH_UPDATE_TIMES:-3}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_MODEL"
    Value: "${DATAFLEX_BATCH_MODEL:-/home/vepfs/models/qwen/Qwen2___5-0___5B-Instruct}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_DATASET"
    Value: "${DATAFLEX_BATCH_DATASET:-alpaca}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_STAGE"
    Value: "${DATAFLEX_BATCH_STAGE:-}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_CUTOFF_LEN"
    Value: "${DATAFLEX_BATCH_CUTOFF_LEN:-1024}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_SLIMPAJAMA_DIR"
    Value: "${DATAFLEX_BATCH_SLIMPAJAMA_DIR:-/home/vepfs/data/SlimPajama-6B-jsonl}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_CLUSTERING_METHOD"
    Value: "${DATAFLEX_BATCH_CLUSTERING_METHOD:-kmeans}"
    IsPrivate: false
  - Name: "DATAFLEX_BATCH_LSH_BITS"
    Value: "${DATAFLEX_BATCH_LSH_BITS:-}"
    IsPrivate: false
ImageUrl: "${IMAGE_URL}"
ResourceQueueID: "${QUEUE_ID}"
Framework: "Custom"
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 1
    Flavor: "${FLAVOR}"
ActiveDeadlineSeconds: "${ACTIVE_DEADLINE}"
DelayExitTimeSeconds: "10m"
AccessType: "Queue"
Storages:
  - Type: "Vepfs"
    MountPath: "/home/vepfs"
    ReadOnly: false
    VepfsId: "${VEPFS_ID}"
    SubPath: "${VEPFS_SUB_PATH}"
YAML

echo "Submitting ${TASK_NAME}"
echo "Queue=${QUEUE_ID} Flavor=${FLAVOR} Image=${IMAGE_URL}"
volc ml_task submit -c dataflex_selector_batch.yaml -o json
