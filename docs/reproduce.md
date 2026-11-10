# Reproducing the thesis experiments

The thesis ran on the BIT-UPM compute server (Ubuntu 20.04, one 40 GB NVIDIA GPU,
512 GB RAM). Each nnU-Net training took about 6–7 hours.

This repository is a cleaned-up, tested rewrite of the scripts and notebooks used
there. The numbers in the README are the ones obtained with the original scripts; the
experiments have not been re-run from this code base, because the data and trained
models stay on the group's server.

## 0. Environments

Two separate conda environments are needed because nnU-Net and CellViT++ pin
incompatible dependencies.

**nnU-Net** (Python 3.11):

```bash
conda create -n nnunet python=3.11 && conda activate nnunet
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install nnunetv2==2.3.1
pip install -e .            # this repository
```

**CellViT++** (Python 3.10): follow the instructions in
[TIO-IKIM/CellViT-plus-plus](https://github.com/TIO-IKIM/CellViT-plus-plus) and download
the `CellViT-SAM-H-x40-AMP.pth` checkpoint (PanNuke classes). Also install
[libvips](https://www.libvips.org/) and this repository (`pip install -e .`).

nnU-Net locates its folders through three environment variables:

```bash
export nnUNet_raw=/data/nnunet/nnUNet_raw
export nnUNet_preprocessed=/data/nnunet/nnUNet_preprocessed
export nnUNet_results=/data/nnunet/nnUNet_results
```

## 1. Binary TIL masks

Both datasets store one class index per pixel. The TIL class becomes 1, the rest 0.

```bash
# CrowdsourcingDataset-Amgadetal2019: class 3 = lymphocytic infiltrate
tilseg binarize --dataset crowdsourcing \
    --src data/crowd/masks --dst data/crowd/til_masks

# BCSS-WSSS: palette index 2 = lymphocyte. Most 224x224 patches contain no TIL
# region at all (3,874 of the 4,986 patches used), so those are dropped.
tilseg binarize --dataset bcss-wsss --skip-empty \
    --src data/bcss_wsss/mask --dst data/bcss_wsss/til_masks
```

For BCSS-WSSS, keep only the images whose mask survived before the next step.

## 2. nnU-Net datasets

```bash
tilseg build-dataset --images data/crowd/images --labels data/crowd/til_masks \
    --nnunet-raw "$nnUNet_raw" --id 111 --name TILsCrowd       # 121 train / 30 test

tilseg build-dataset --images data/bcss_wsss/img_til --labels data/bcss_wsss/til_masks \
    --nnunet-raw "$nnUNet_raw" --id 333 --name TILsBCSS
```

This copies the files into the nnU-Net layout (`imagesTr`, `labelsTr`, `imagesTs`,
`labelsTs`), adds the `_0000` channel suffix to the images, makes a seeded random
80/20 split and writes `dataset.json`. The split is not stratified: the public
datasets come without the clinical variables that would be needed for it.

## 3. Experiment 1: nnU-Net on the original annotations

```bash
scripts/train_nnunet.sh 111     # Crowdsourcing  -> experiment 1A
scripts/train_nnunet.sh 333     # BCSS-WSSS      -> experiment 1B
```

The script runs `nnUNetv2_plan_and_preprocess`, trains the `2d` configuration on the
`all` fold for 500 epochs, predicts `imagesTs` and scores the predictions with
`tilseg evaluate`.

What nnU-Net configured automatically:

| | Crowdsourcing | BCSS-WSSS |
|---|---|---|
| Median image size | 3176 × 4035 px | 224 × 224 px |
| Patch size | 1024 × 1536 | 224 × 224 |
| Batch size | 2 | 2 |
| Loss | Dice + cross-entropy | Dice + cross-entropy |
| Epochs | 500 (default: 1000) | 500 |

Training on the `all` fold means a single model sees every training image and there is
no cross-validation estimate; the held-out 20 % is the only test.

## 4. CellViT++ inference

```bash
conda activate cellvit
scripts/run_cellvit.sh data/crowd/images work/cellvit \
    /path/to/CellViT-plus-plus /path/to/CellViT-SAM-H-x40-AMP.pth
```

CellViT++ only reads whole-slide formats, so the script first converts every PNG tile
to a tiled pyramidal TIFF with libvips, writes the file list with the resolution
metadata (0.25 µm/px, 40×) and then runs `detect_cells.py process_dataset`.

The output is one `<slide>_cell_detection.geojson` per tile: a `MultiPoint` feature per
cell class with the centroid of every nucleus. Note that CellViT++ truncates the slide
name at the first dot (`..._MPP-0.2500.tif` → `..._MPP-0_cell_detection.geojson`);
`tilseg til-masks` matches outputs to tiles by that prefix.

## 5. Cell detections → region masks

```bash
tilseg til-masks --geojson work/cellvit/segmented \
    --images data/crowd/images --out work/cellvit_masks
```

For each tile ([`til_regions.py`](../src/tilseg/til_regions.py)):

1. Take the centroids of the cells classified as *Inflammatory*.
2. For each one, count the TILs within `R = 100 px` (25 µm) and divide by the circle
   area. Keep the cell if the density is at least `0.0003 TILs/px²`, i.e. about 9 TILs
   in the neighbourhood including itself.
3. Stamp a disc of radius 2 px on an empty mask at every kept centroid.
4. Dilate with a 100 px elliptical kernel, so that cells closer than ~100 px merge.
5. Close with a 120 px elliptical kernel to fill holes and smooth the outline.

The thresholds were set empirically by inspecting the resulting regions over the tiles.
All of them are command-line options (`tilseg til-masks --help`).

An earlier variant, DBSCAN clustering followed by the convex hull of each cluster, is
kept as `til_regions.hull_region_mask`. It was dropped because hulls fill in the
concavities of band- and crescent-shaped infiltrates.

## 6. Experiment 2: nnU-Net on CellViT-derived labels

```bash
cp -r "$nnUNet_raw/Dataset111_TILsCrowd" "$nnUNet_raw/Dataset444_TILsCellViT"
sed -i 's/Dataset111_TILsCrowd/Dataset444_TILsCellViT/' "$nnUNet_raw/Dataset444_TILsCellViT/dataset.json"
tilseg replace-labels --dataset "$nnUNet_raw/Dataset444_TILsCellViT" --labels work/cellvit_masks
scripts/train_nnunet.sh 444
```

Copying the baseline dataset and replacing only the labels keeps the images and the
train/test split identical, so the two models differ in nothing but the label
definition. Tiles without a CellViT-derived mask (no dense TIL aggregate found) keep
their original label; the command reports how many.

## 7. External cohort

ROIs of 2048 × 2048 px were exported from 15 whole-slide images with QuPath and
processed exactly like the public tiles (steps 4 and 5) to obtain reference masks.
Both trained models then predict the same ROIs:

```bash
for id in 111 444; do
  name=$(basename "$(ls -d "$nnUNet_raw"/Dataset${id}_*)")
  nnUNetv2_predict -i data/external/images -o work/external/pred_$id \
      -d $id -c 2d -f all -tr nnUNetTrainer_500ep
  tilseg evaluate --pred work/external/pred_$id --ref data/external/til_masks \
      --out work/external/metrics_$id.csv
done
```

The images in `data/external/images` need the `_0000` suffix.

## 8. Figures

```bash
tilseg overlay --images "$nnUNet_raw/Dataset111_TILsCrowd/imagesTs" \
    --masks work/pred_111 --out work/overlays_111
```

Box plots of per-image Dice for several models: `tilseg.viz.dice_boxplot`.

## Differences from the original scripts

- The original 80/20 split was made with an unseeded random sample; here it is seeded
  and reproducible.
- 500 epochs were originally set by editing `num_epochs` in the installed
  `nnUNetTrainer`; here a subclass (`nnUNetTrainer_500ep`) does it without touching the
  library.
- Metrics were originally computed with nnU-Net's `compute_metrics_on_folder2` and
  MONAI's `DiceMetric`; `tilseg.metrics` computes the same per-image foreground Dice
  with NumPy only, so evaluation needs neither library.
- Region masks were originally saved as black-on-white RGB images and binarised in a
  second step; here the 0/1 label map is written directly.
