"""
segment.py

Extract building crops from street-level images using
Mask2Former (Swin-L, Mapillary Vistas panoptic).

Writes:
  <out_dir>/crops/<objectid_or_stem>/<stem>_<N>.jpg   — cropped building images
  <out_dir>/manifest.jsonl                             — one record per image (buildings found only)

Each manifest record:
  image_id       — Mapillary image ID (filename stem), links back to source image
  objectid       — address ID, links to DC parcel data
  primary_crop   — largest building crop (most likely the target address)
  other_crops    — remaining crops sorted by area descending

Images with no buildings detected are silently skipped.

Usage:
    python segment.py data/images/ --out-dir dc_crops
    python segment.py data/images/ --out-dir dc_crops --sample 1000
    python segment.py --full-run   --out-dir dc_crops
"""

import argparse
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

MODEL_ID      = "facebook/mask2former-swin-large-mapillary-vistas-panoptic"
SEG_BATCH     = 4      # images per forward pass (tuned for 24 GB VRAM)
IMAGE_BATCH   = 4
PREFETCH      = 8
MIN_AREA_FRAC = 0.06   # crop must cover >= this fraction of image pixels
IOU_THRESH    = 0.4    # dedup threshold


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input",       nargs="?", default=None,
                   help="Image file or directory (omit with --full-run)")
    p.add_argument("--full-run",  action="store_true")
    p.add_argument("--db",        default="data/image_status.db")
    p.add_argument("--min-area",  type=float, default=MIN_AREA_FRAC)
    p.add_argument("--out-dir",   default="dc_crops")
    p.add_argument("--sample",    type=int, default=None)
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


# =============================================================================
# DB / CHECKPOINT
# =============================================================================

def load_images_from_db(db_path):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT objectid, image_path, residential_type, address "
        "FROM status WHERE state='done' AND image_path IS NOT NULL"
    ).fetchall()
    conn.close()
    return [
        {"objectid": r[0], "image_path": str(Path(r[1].replace("\\", "/"))),
         "residential_type": r[2], "address": r[3]}
        for r in rows if Path(r[1].replace("\\", "/")).exists()
    ]


def load_checkpoint(path):
    if not Path(path).exists():
        return set()
    with open(path) as f:
        return set(json.load(f))


def save_checkpoint(path, done_ids):
    with open(path, "w") as f:
        json.dump(list(done_ids), f)


# =============================================================================
# SEGMENTATION
# =============================================================================

def iou(a, b):
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def segment_batch(images, model, processor, device, building_label_ids, min_area_frac):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=torch.float16 if v.is_floating_point() else v.dtype)
              for k, v in inputs.items()}

    target_sizes = [(img.height, img.width) for img in images]

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=target_sizes
    )

    per_image = []
    for pil, result in zip(images, results):
        W, H     = pil.size
        total_px = W * H
        seg_map  = result["segmentation"].cpu().numpy()   # H x W, value = segment id

        crops = []
        for seg_info in result["segments_info"]:
            if seg_info["label_id"] not in building_label_ids:
                continue
            mask      = seg_map == seg_info["id"]
            area_frac = mask.sum() / total_px
            if area_frac < min_area_frac:
                continue
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            y1, y2 = int(rows[0]), int(rows[-1])
            x1, x2 = int(cols[0]), int(cols[-1])
            bbox   = [x1, y1, x2 + 1, y2 + 1]
            crops.append({
                "bbox":          bbox,
                "area_fraction": float(area_frac),
                "score":         float(seg_info.get("score", 1.0)),
            })

        crops.sort(key=lambda c: -c["area_fraction"])
        kept = []
        for cand in crops:
            if all(iou(cand["bbox"], k["bbox"]) < IOU_THRESH for k in kept):
                kept.append(cand)

        per_image.append(kept)

    return per_image


# =============================================================================
# MAIN
# =============================================================================

def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {MODEL_ID} ...")
    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_ID)
    model     = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device).eval()

    building_label_ids = {
        lid for lid, name in model.config.id2label.items()
        if "building" in name.lower()
    }
    building_names = {model.config.id2label[lid] for lid in building_label_ids}
    print(f"Building classes ({len(building_label_ids)}): {building_names}")

    # ── Gather images ──────────────────────────────────────────────────────────
    out_dir         = Path(args.out_dir)
    crops_dir       = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.json"
    manifest_path   = out_dir / "manifest.jsonl"
    full_run        = args.full_run or not args.input

    if full_run:
        db_rows  = load_images_from_db(args.db)
        done_ids = load_checkpoint(checkpoint_path)
        metas    = [r for r in db_rows if str(r["objectid"]) not in done_ids]
        print(f"DB: {len(db_rows)} buildings  |  done: {len(done_ids)}  |  remaining: {len(metas)}")
    else:
        input_path = Path(args.input)
        paths = (sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
                 if input_path.is_dir() else [input_path])
        if args.sample and args.sample < len(paths):
            rng   = np.random.default_rng(args.seed)
            paths = list(rng.choice(paths, size=args.sample, replace=False))
            print(f"Sampled {args.sample} images")
        metas = [{"objectid": None, "image_path": str(p)} for p in paths]

    print(f"Processing {len(metas)} images  |  seg_batch={SEG_BATCH}\n")

    done_ids      = load_checkpoint(checkpoint_path) if full_run else set()
    n_crops_total = 0

    manifest_file = open(manifest_path, "a")

    with ThreadPoolExecutor(max_workers=PREFETCH) as loader:
        batches    = [metas[i:i+IMAGE_BATCH] for i in range(0, len(metas), IMAGE_BATCH)]
        prefetch_q = []

        def submit_batch(batch):
            return [(m, loader.submit(load_image, m["image_path"])) for m in batch]

        if batches:
            prefetch_q.append(submit_batch(batches[0]))

        for batch_idx in tqdm(range(len(batches)), unit="batch",
                               desc=f"batches of {IMAGE_BATCH}"):
            if batch_idx + 1 < len(batches):
                prefetch_q.append(submit_batch(batches[batch_idx + 1]))

            current = prefetch_q.pop(0)
            pil_images, valid_metas = [], []
            for meta, fut in current:
                img = fut.result()
                if img is not None:
                    pil_images.append(img)
                    valid_metas.append(meta)

            if not pil_images:
                continue

            # Run in SEG_BATCH-sized sub-batches
            all_crops = []
            for i in range(0, len(pil_images), SEG_BATCH):
                chunk = pil_images[i:i + SEG_BATCH]
                all_crops.extend(segment_batch(chunk, model, processor, device,
                                               building_label_ids, args.min_area))

            for meta, pil, crops in zip(valid_metas, pil_images, all_crops):
                if full_run:
                    done_ids.add(str(meta.get("objectid")))

                if not crops:
                    continue   # no buildings detected — skip

                img_path = Path(meta["image_path"])
                objectid = meta.get("objectid")
                image_id = img_path.stem   # Mapillary image ID encoded in filename
                folder   = str(objectid) if objectid is not None else image_id
                crop_dir = crops_dir / folder
                crop_dir.mkdir(parents=True, exist_ok=True)

                # crops already sorted largest-first; [0] is the primary building
                saved_crops = []
                for n, c in enumerate(crops):
                    x1, y1, x2, y2 = c["bbox"]
                    crop_img  = pil.crop((x1, y1, x2, y2))
                    crop_name = f"{image_id}_{n}.jpg"
                    crop_path = crop_dir / crop_name
                    crop_img.save(crop_path, quality=92)
                    saved_crops.append({
                        "crop_path":     str(crop_path),
                        "bbox":          c["bbox"],
                        "area_fraction": c["area_fraction"],
                        "score":         c["score"],
                    })
                    n_crops_total += 1

                record = {
                    "image_id":    image_id,
                    "image":       str(img_path),
                    "objectid":    objectid,
                    "primary_crop": saved_crops[0],
                    "other_crops":  saved_crops[1:],
                }
                if meta.get("residential_type"):
                    record["residential_type"] = meta["residential_type"]

                manifest_file.write(json.dumps(record) + "\n")

        manifest_file.flush()

        if full_run and len(done_ids) % 500 == 0:
            save_checkpoint(checkpoint_path, done_ids)
            tqdm.write(f"  checkpoint: {len(done_ids)} done, {n_crops_total} crops saved")

    manifest_file.close()
    if full_run:
        save_checkpoint(checkpoint_path, done_ids)

    print(f"\nManifest → {manifest_path}")
    print(f"Total crops saved: {n_crops_total}")


if __name__ == "__main__":
    main()
