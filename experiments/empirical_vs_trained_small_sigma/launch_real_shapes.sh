#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ASSET_ROOT="$(cd -- "${ASSET_ROOT:-${REPOSITORY_ROOT}}" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/real_shapes_30k}"
GARGOYLE_POINT_CLOUD="${GARGOYLE_POINT_CLOUD:-${SCRIPT_DIR}/assets/gargoyle_pointcloud_100k.csv}"
GARGOYLE_SURFACE="${GARGOYLE_SURFACE:-${SCRIPT_DIR}/assets/gargoyle_with_normals.ply}"
FLOWER_SURFACE="${FLOWER_SURFACE:-${SCRIPT_DIR}/assets/flower_reference_mesh.npz}"
STEPS="${STEPS:-30000}"
N_SUPPORT="${N_SUPPORT:-40000000}"
N_REFERENCE="${N_REFERENCE:-2000000}"
SEEDS=(20260828 20260829 20260830)
SIGMAS=(0.01 0.005 0.002)
read -r -a MODELS <<<"${MODEL_TYPES:-direct energy}"
read -r -a DATASET_NAMES <<<"${DATASETS:-bunny gargoyle flower}"
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

input_paths() {
  case "$1" in
    bunny)
      point_cloud="${ASSET_ROOT}/datasets/Bunny/bunny_surface_points_250k_20260526.csv"
      surface_kind=mesh
      surface="${ASSET_ROOT}/datasets/source_meshes/Bunny/BunnyMesh.ply"
      ;;
    gargoyle)
      point_cloud="${GARGOYLE_POINT_CLOUD}"
      surface_kind=oriented-points
      surface="${GARGOYLE_SURFACE}"
      ;;
    flower)
      point_cloud="${ASSET_ROOT}/datasets/FlowerPetal/flower_petal_closed_surface_8lobes_250k.csv"
      surface_kind=mesh
      surface="${FLOWER_SURFACE}"
      ;;
    *)
      printf 'Unknown dataset: %s. Choose bunny, gargoyle or flower.\n' "$1" >&2
      return 1
      ;;
  esac
}

cd "${SCRIPT_DIR}"
for dataset in "${DATASET_NAMES[@]}"; do
  input_paths "${dataset}"
  for input in "${point_cloud}" "${surface}"; do
    if [[ ! -f "${input}" ]]; then
      printf 'Missing %s input: %s\nSee %s/assets/README.md for input requirements.\n' \
        "${dataset}" "${input}" "${SCRIPT_DIR}" >&2
      exit 1
    fi
  done
done
mkdir -p "${OUTPUT_ROOT}/logs"

for dataset in "${DATASET_NAMES[@]}"; do
  input_paths "${dataset}"
  "${PYTHON_BIN}" real_shapes.py prepare \
    --dataset "${dataset}" \
    --point-cloud "${point_cloud}" \
    --surface-kind "${surface_kind}" \
    --surface "${surface}" \
    --n-support "${N_SUPPORT}" \
    --n-reference "${N_REFERENCE}" \
    --output-dir "${OUTPUT_ROOT}/data/${dataset}"
done

job_index=0
for dataset in "${DATASET_NAMES[@]}"; do
  for model in "${MODELS[@]}"; do
    for sigma in "${SIGMAS[@]}"; do
      sigma_tag="${sigma/./p}"
      for seed in "${SEEDS[@]}"; do
        save="${OUTPUT_ROOT}/checkpoints/${dataset}/${model}/sigma_${sigma_tag}/seed_${seed}.pt"
        log="${OUTPUT_ROOT}/logs/train_${dataset}_${model}_sigma_${sigma_tag}_seed_${seed}.log"
        if [[ -f "${save}" ]]; then
          continue
        fi
        gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
        mkdir -p "$(dirname "${save}")"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" real_shapes.py train \
          --dataset "${dataset}" \
          --support "${OUTPUT_ROOT}/data/${dataset}/support_40m.npy" \
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
done
wait_for_jobs

"${PYTHON_BIN}" real_shapes.py evaluate \
  --data-root "${OUTPUT_ROOT}/data" \
  --checkpoint-root "${OUTPUT_ROOT}/checkpoints" \
  --datasets "${DATASET_NAMES[@]}" \
  --sigmas "${SIGMAS[@]}" \
  --output-dir "${OUTPUT_ROOT}/evaluation"
