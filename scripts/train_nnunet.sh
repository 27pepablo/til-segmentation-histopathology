#!/usr/bin/env bash
# Plan, preprocess, train and predict with nnU-Net v2 for one dataset.
#
# Usage:   scripts/train_nnunet.sh <dataset_id> [gpu_id]
# Example: scripts/train_nnunet.sh 111 0
#
# Requires the nnU-Net environment variables (see docs/reproduce.md):
#   nnUNet_raw, nnUNet_preprocessed, nnUNet_results
set -euo pipefail

DATASET_ID="${1:?dataset id required (e.g. 111)}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${2:-0}"
TRAINER="nnUNetTrainer_500ep"

for var in nnUNet_raw nnUNet_preprocessed nnUNet_results; do
  [[ -n "${!var:-}" ]] || { echo "Environment variable $var is not set" >&2; exit 1; }
done

# Make the 500-epoch trainer visible to nnU-Net (idempotent).
NNUNET_DIR="$(python -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')"
cp "$(dirname "$0")/nnUNetTrainer_500ep.py" \
   "$NNUNET_DIR/training/nnUNetTrainer/variants/training_length/"

DATASET_DIR="$(find "$nnUNet_raw" -maxdepth 1 -type d -name "Dataset$(printf '%03d' "$DATASET_ID")_*" | head -n 1)"
[[ -n "$DATASET_DIR" ]] || { echo "Dataset $DATASET_ID not found in $nnUNet_raw" >&2; exit 1; }
PRED_DIR="$(dirname "$nnUNet_raw")/nnUNet_predicted/$(basename "$DATASET_DIR")"

# 1) Dataset fingerprint + experiment planning + preprocessing.
#    For 2-D RGB PNGs only the "2d" configuration is generated.
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity

# 2) Train a single model on the whole training set ("all" fold), as in the thesis.
nnUNetv2_train "$DATASET_ID" 2d all -tr "$TRAINER"

# 3) Predict the held-out test images.
nnUNetv2_predict -i "$DATASET_DIR/imagesTs" -o "$PRED_DIR" \
  -d "$DATASET_ID" -c 2d -f all -tr "$TRAINER"

# 4) Score against the held-out labels.
tilseg evaluate --pred "$PRED_DIR" --ref "$DATASET_DIR/labelsTs" --out "$PRED_DIR/metrics.csv"
