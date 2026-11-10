"""Command-line interface: ``tilseg <command> --help``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import cellvit_io, masks, metrics, nnunet_dataset, til_regions


def _cmd_binarize(args: argparse.Namespace) -> None:
    labels = args.labels or {
        "crowdsourcing": masks.CROWDSOURCING_TIL_LABELS,
        "bcss-wsss": masks.BCSS_WSSS_TIL_LABELS,
    }[args.dataset]
    report = masks.convert_folder(args.src, args.dst, labels, skip_empty=args.skip_empty)
    print(f"TIL labels: {tuple(labels)}")
    print(f"Masks written: {len(report['written'])}  |  without TIL pixels: {len(report['empty'])}")


def _cmd_build_dataset(args: argparse.Namespace) -> None:
    root = nnunet_dataset.build_dataset(
        args.images, args.labels, args.nnunet_raw, args.id, args.name, args.test_fraction, args.seed
    )
    n_tr = len(list((root / "labelsTr").glob("*.png")))
    n_ts = len(list((root / "labelsTs").glob("*.png")))
    print(f"{root}: {n_tr} training / {n_ts} test cases")


def _cmd_replace_labels(args: argparse.Namespace) -> None:
    counts = nnunet_dataset.replace_labels(args.dataset, args.labels)
    print(f"Labels replaced: {counts['replaced']}  |  kept (no new mask): {counts['kept']}")


def _cmd_filelist(args: argparse.Namespace) -> None:
    n = cellvit_io.write_filelist(args.images, args.out, args.mpp, args.magnification)
    print(f"{n} slides listed in {args.out}")


def _find_reference(image_dir: Path, slide: str) -> Path | None:
    """Image whose name starts with the slide name CellViT++ used for its output."""
    matches = sorted(p for p in image_dir.glob("*.png") if p.name.startswith(slide))
    return matches[0] if matches else None


def _cmd_til_masks(args: argparse.Namespace) -> None:
    params = til_regions.RegionParams(
        args.radius, args.min_density, args.seed_radius, args.dilate_kernel, args.close_kernel
    )
    geojson_dir, image_dir, out_dir = Path(args.geojson), Path(args.images), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(geojson_dir.glob(f"*{cellvit_io.DETECTION_SUFFIX}"))
    written = 0
    for path in files:
        slide = path.name[: -len(cellvit_io.DETECTION_SUFFIX)]
        ref = _find_reference(image_dir, slide)
        if ref is None:
            print(f"[skip] no reference image for {slide}")
            continue
        with Image.open(ref) as img:
            width, height = img.size
        points = cellvit_io.load_centroids(path)
        mask = til_regions.region_mask(points, (height, width), params)
        case = nnunet_dataset.case_id(ref)
        Image.fromarray(mask).save(out_dir / f"{case}.png")
        written += 1
        print(f"{case}: {len(points)} TILs, {100 * til_regions.til_area_fraction(mask):.1f}% area")
    print(f"{written}/{len(files)} masks written to {out_dir}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    rows = metrics.evaluate_folder(args.pred, args.ref)
    summary = {m: metrics.summarize(rows, m) for m in ("dice", "iou")}
    for name, s in summary.items():
        if s["n"]:
            print(
                f"{name:>4}: {s['mean']:.3f} ± {s['std']:.3f}  "
                f"(median {s['median']:.3f}, min {s['min']:.3f}, max {s['max']:.3f}, n={s['n']})"
            )
    if args.out:
        metrics.save_csv(rows, args.out)
        Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=2))
        print(f"Per-case metrics saved to {args.out}")


def _cmd_overlay(args: argparse.Namespace) -> None:
    from . import viz

    image_dir, mask_dir, out_dir = Path(args.images), Path(args.masks), Path(args.out)
    done = 0
    for image in sorted(image_dir.glob("*.png")):
        mask = mask_dir / f"{nnunet_dataset.case_id(image)}.png"
        if mask.exists():
            viz.overlay_file(image, mask, out_dir / f"{nnunet_dataset.case_id(image)}_overlay.png")
            done += 1
    print(f"{done} overlays written to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tilseg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("binarize", help="multi-class dataset masks -> binary TIL masks")
    p.add_argument("--dataset", choices=["crowdsourcing", "bcss-wsss"], required=True)
    p.add_argument("--src", required=True, help="folder with the original masks")
    p.add_argument("--dst", required=True, help="output folder")
    p.add_argument("--labels", type=int, nargs="+", help="override the TIL class indices")
    p.add_argument("--skip-empty", action="store_true", help="drop masks without TIL pixels")
    p.set_defaults(func=_cmd_binarize)

    p = sub.add_parser("build-dataset", help="create an nnU-Net v2 raw dataset (80/20 split)")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True, help="binary masks (same stems as the images)")
    p.add_argument("--nnunet-raw", required=True, help="nnUNet_raw folder")
    p.add_argument("--id", type=int, required=True, help="dataset id (1-999)")
    p.add_argument("--name", required=True, help="dataset name, e.g. TILsCrowd")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_cmd_build_dataset)

    p = sub.add_parser("replace-labels", help="swap dataset labels for CellViT-derived masks")
    p.add_argument("--dataset", required=True, help="DatasetXXX_NAME folder")
    p.add_argument("--labels", required=True, help="folder with the new masks")
    p.set_defaults(func=_cmd_replace_labels)

    p = sub.add_parser("cellvit-filelist", help="CSV file list for CellViT++ process_dataset")
    p.add_argument("--images", required=True, help="folder with pyramidal TIFFs")
    p.add_argument("--out", required=True, help="output CSV")
    p.add_argument("--mpp", type=float, default=0.25)
    p.add_argument("--magnification", type=int, default=40)
    p.set_defaults(func=_cmd_filelist)

    d = til_regions.RegionParams()
    p = sub.add_parser("til-masks", help="CellViT++ detections -> TIL-region masks")
    p.add_argument("--geojson", required=True, help="folder with *_cell_detection.geojson")
    p.add_argument("--images", required=True, help="folder with the PNG tiles (for the size)")
    p.add_argument("--out", required=True)
    p.add_argument("--radius", type=float, default=d.radius)
    p.add_argument("--min-density", type=float, default=d.min_density)
    p.add_argument("--seed-radius", type=int, default=d.seed_radius)
    p.add_argument("--dilate-kernel", type=int, default=d.dilate_kernel)
    p.add_argument("--close-kernel", type=int, default=d.close_kernel)
    p.set_defaults(func=_cmd_til_masks)

    p = sub.add_parser("evaluate", help="Dice / IoU between predicted and reference masks")
    p.add_argument("--pred", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out", help="CSV for per-case metrics (a .json summary is saved next to it)")
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("overlay", help="draw mask contours on the images")
    p.add_argument("--images", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_overlay)

    return parser


def main(argv: list[str] | None = None) -> None:
    np.seterr(all="ignore")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
