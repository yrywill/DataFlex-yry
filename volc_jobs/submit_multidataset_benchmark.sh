#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /root/.volc/.profile

TASK_NAME="${TASK_NAME:-dataflex-multidata-benchmark-$(date +%m%d%H%M%S)}"
QUEUE_ID="${QUEUE_ID:-q-20260318205115-lvfng}"
FLAVOR="${FLAVOR:-ml.pni2.28xlarge}"
IMAGE_URL="${IMAGE_URL:-vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:2.7.1-cu12.6.3-py3.10-ubuntu20.04}"
VEPFS_ID="${VEPFS_ID:-vepfs-cnbj2c98dea54433}"
VEPFS_SUB_PATH="${VEPFS_SUB_PATH:-queue010/20252203250}"
ACTIVE_DEADLINE="${ACTIVE_DEADLINE:-36h}"

cat > dataflex_multidataset_benchmark.yaml <<YAML
TaskName: "${TASK_NAME}"
Description: "DataFlex multidataset benchmark: LESS vs ClusterLess(kmeans), metrics, speed, cache, plots."
Tags:
  - "dataflex"
  - "selector"
  - "cluster_less"
  - "benchmark"
UserCodePath: "./"
RemoteMountCodePath: "/root/code"
Entrypoint: "bash /home/vepfs/MIXjyx/DataFlex-main/volc_jobs/run_multidataset_benchmark_entry.sh"
Envs:
  - Name: "DATAFLEX_BENCH_DATASETS"
    Value: "${DATAFLEX_BENCH_DATASETS:-redpajama_arxiv,redpajama_c4,redpajama_common_crawl,redpajama_github,redpajama_stackexchange,redpajama_wikipedia,fineweb,fineweb_edu}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_SELECTORS"
    Value: "${DATAFLEX_BENCH_SELECTORS:-less,cluster_less}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_SEEDS"
    Value: "${DATAFLEX_BENCH_SEEDS:-42,43,44}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_TRAIN_SIZE"
    Value: "${DATAFLEX_BENCH_TRAIN_SIZE:-2048}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_EVAL_SIZE"
    Value: "${DATAFLEX_BENCH_EVAL_SIZE:-256}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_MAX_STEPS"
    Value: "${DATAFLEX_BENCH_MAX_STEPS:-80}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_WARMUP_STEP"
    Value: "${DATAFLEX_BENCH_WARMUP_STEP:-20}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_UPDATE_STEP"
    Value: "${DATAFLEX_BENCH_UPDATE_STEP:-20}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_UPDATE_TIMES"
    Value: "${DATAFLEX_BENCH_UPDATE_TIMES:-3}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_MODEL"
    Value: "${DATAFLEX_BENCH_MODEL:-/home/vepfs/models/qwen/Qwen2___5-0___5B-Instruct}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_CUTOFF_LEN"
    Value: "${DATAFLEX_BENCH_CUTOFF_LEN:-1024}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_PROJ_DIM"
    Value: "${DATAFLEX_BENCH_PROJ_DIM:-1024}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_CLUSTER_SIZE"
    Value: "${DATAFLEX_BENCH_CLUSTER_SIZE:-64}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_SAMPLES_PER_CLUSTER"
    Value: "${DATAFLEX_BENCH_SAMPLES_PER_CLUSTER:-3}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_SAMPLES_PER_CLUSTER_VALUES"
    Value: "${DATAFLEX_BENCH_SAMPLES_PER_CLUSTER_VALUES:-}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_REPRESENTATIVE_STRATEGY"
    Value: "${DATAFLEX_BENCH_REPRESENTATIVE_STRATEGY:-random}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_REPRESENTATIVE_STRATEGY_VALUES"
    Value: "${DATAFLEX_BENCH_REPRESENTATIVE_STRATEGY_VALUES:-}"
    IsPrivate: false
  - Name: "DATAFLEX_BENCH_GRAD_ACCUM"
    Value: "${DATAFLEX_BENCH_GRAD_ACCUM:-4}"
    IsPrivate: false
  - Name: "FORCE_TORCHRUN"
    Value: "${FORCE_TORCHRUN:-1}"
    IsPrivate: false
  - Name: "NPROC_PER_NODE"
    Value: "${NPROC_PER_NODE:-8}"
    IsPrivate: false
  - Name: "NNODES"
    Value: "${NNODES:-1}"
    IsPrivate: false
  - Name: "NODE_RANK"
    Value: "${NODE_RANK:-0}"
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
volc ml_task submit -c dataflex_multidataset_benchmark.yaml -o json
