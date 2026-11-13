#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
STEPS="${STEPS:-30000}"
SEEDS=(20260826 20260827 20260828)
SIGMAS=(0.02 0.01 0.005 0.002)
MODELS=(direct energy)
read -r -a GPUS <<<"${GPU_IDS:-0}"
pending_pids=()

wait_for_jobs() {
  local failed=0
  local pid
  for pid in "${pending_pids[@]-}"; do
    [[ -n "${pid}" ]] || continue
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  pending_pids=()
  if (( failed )); then
    printf 'Training failed; inspect the logs in %s/logs.\n' "${OUTPUT_ROOT}" >&2
  fi
  return "${failed}"
}

mkdir -p "${OUTPUT_ROOT}/logs"

job_index=0
for model in "${MODELS[@]}"; do
  for sigma in "${SIGMAS[@]}"; do
    sigma_tag="${sigma/./p}"
    for seed in "${SEEDS[@]}"; do
      gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
      save="${OUTPUT_ROOT}/checkpoints/${model}/sigma_${sigma_tag}/seed_${seed}.pt"
      log="${OUTPUT_ROOT}/logs/train_${model}_sigma_${sigma_tag}_seed_${seed}.log"
      mkdir -p "$(dirname "${save}")"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" experiment.py train \
        --model "${model}" \
        --sigma "${sigma}" \
        --seed "${seed}" \
        --steps "${STEPS}" \
        --output "${save}" >"${log}" 2>&1 &
      pending_pids+=("$!")
      job_index=$((job_index + 1))
      if (( ${#pending_pids[@]} >= ${#GPUS[@]} )); then
        wait_for_jobs
      fi
    done
  done
done
wait_for_jobs

"${PYTHON_BIN}" experiment.py make-support \
  --output "${OUTPUT_ROOT}/support/sphere_uniform_40m.npy" \
  --n-points 40000000

"${PYTHON_BIN}" experiment.py evaluate \
  --checkpoint-root "${OUTPUT_ROOT}/checkpoints" \
  --support "${OUTPUT_ROOT}/support/sphere_uniform_40m.npy" \
  --sigmas "${SIGMAS[@]}" \
  --output-dir "${OUTPUT_ROOT}/evaluation"
