#!/usr/bin/env bash
# Detect and classify nuclei with CellViT++ on a folder of PNG tiles.
#
# Usage: scripts/run_cellvit.sh <png_dir> <work_dir> <cellvit_repo> <checkpoint>
#
#   png_dir       folder with the H&E tiles (PNG, 0.25 um/px, 40x)
#   work_dir      output folder; creates tiff/, filelist.csv and segmented/
#   cellvit_repo  clone of https://github.com/TIO-IKIM/CellViT-plus-plus
#   checkpoint    CellViT-SAM-H-x40-AMP.pth (PanNuke classes)
#
# Run inside the CellViT++ conda environment, with libvips (`vips`) on PATH and
# this package installed (for `tilseg cellvit-filelist`).
set -euo pipefail

PNG_DIR="$(realpath "${1:?png_dir required}")"
WORK_DIR="$(realpath -m "${2:?work_dir required}")"
CELLVIT_REPO="$(realpath "${3:?cellvit_repo required}")"
CHECKPOINT="$(realpath "${4:?checkpoint required}")"

TIFF_DIR="$WORK_DIR/tiff"
OUT_DIR="$WORK_DIR/segmented"
mkdir -p "$TIFF_DIR" "$OUT_DIR"

# 1) CellViT++ reads whole-slide formats: convert every tile to a tiled, pyramidal TIFF.
for src in "$PNG_DIR"/*.png; do
  name="$(basename "${src%.png}")"
  name="${name%_0000}"                       # drop the nnU-Net channel suffix, if present
  vips tiffsave "$src" "$TIFF_DIR/$name.tif" \
    --compression jpeg --Q 95 --tile --tile-width 256 --tile-height 256 --pyramid
done

# 2) File list with resolution metadata (the TIFFs carry none).
tilseg cellvit-filelist --images "$TIFF_DIR" --out "$WORK_DIR/filelist.csv" --mpp 0.25 --magnification 40

# 3) Inference. One <slide>_cell_detection.geojson per tile is written to $OUT_DIR.
cd "$CELLVIT_REPO"
python3 ./cellvit/detect_cells.py \
  --model "$CHECKPOINT" \
  --outdir "$OUT_DIR" \
  --geojson \
  process_dataset \
  --filelist "$WORK_DIR/filelist.csv"

echo "Detections written to $OUT_DIR"
