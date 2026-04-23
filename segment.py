"""
segment.py

Extract building crops from street-level images using
SegFormer-b0 (ADE20K) with 3-strip tiling and per-pixel confidence filtering.
Connected components separate individual buildings within each strip mask.

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
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

MODEL_ID      = "nvidia/segformer-b0-finetuned-ade-512-512"
# ADE20K building class IDs (0-indexed): building, house, skyscraper, tower
BUILDING_IDS  = {1, 25, 48, 84}
SEG_BATCH     = 12     # strips per SegFormer forward pass
SEG_CONF      = 0.5    # per-pixel confidence threshold
SEG_SCORE_MIN = 0.65   # min mean confidence across component pixels
IMAGE_BATCH   = 6
PREFETCH      = 8
MIN_AREA_FRAC = 0.06   # crop must cover >= this fraction of strip pixels
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
    p.add_argument("--min-score", type=float, default=SEG_SCORE_MIN)
    p.add_argument("--out-dir",   default="dc_crops")
    p.add_argument("--sample",    type=int, default=None)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--clear",     action="store_true",
                   help="Delete existing crops and manifest before starting")
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


def segment_batch(images, model, processor, device, min_area_frac, min_score):
    # Build 3 overlapping strips per image (60% width, offset at 0/20/40%)
    strip_meta, strip_pils = [], []
    for img_idx, pil in enumerate(images):
        W, H = pil.size
        sw   = int(W * 0.6)
        for x0 in [0, int(W * 0.2), int(W * 0.4)]:
            x1 = min(x0 + sw, W)
            strip_meta.append((img_idx, x0, W, H, x1 - x0))
            strip_pils.append(pil.crop((x0, 0, x1, H)))

    per_image = [[] for _ in images]

    for i in range(0, len(strip_pils), SEG_BATCH):
        batch_pils = strip_pils[i:i + SEG_BATCH]
        batch_meta = strip_meta[i:i + SEG_BATCH]

        inputs = processor(images=batch_pils, return_tensors="pt")
        inputs = {k: v.to(device=device, dtype=torch.float16 if v.is_floating_point() else v.dtype)
                  for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits   # [B x C x h x w]

        for j, (img_idx, x_off, full_W, full_H, sW) in enumerate(batch_meta):
            up       = F.interpolate(logits[j:j+1], size=(full_H, sW),
                                     mode="bilinear", align_corners=False)
            probs    = F.softmax(up, dim=1)
            seg_map  = probs.argmax(dim=1).squeeze(0).cpu().numpy()
            max_prob = probs.max(dim=1).values.squeeze(0).cpu().numpy()

            # Building mask with per-pixel confidence gate
            mask = np.isin(seg_map, list(BUILDING_IDS)) & (max_prob >= SEG_CONF)
            if not mask.any():
                continue

            strip_px = sW * full_H
            full_px  = full_W * full_H

            # Connected components → individual buildings
            labeled, n_comp = ndimage.label(mask)
            for comp_id in range(1, n_comp + 1):
                comp = labeled == comp_id
                if comp.sum() / strip_px < min_area_frac:
                    continue
                mean_conf = float(max_prob[comp].mean())
                if mean_conf < min_score:
                    continue
                rows = np.where(comp.any(axis=1))[0]
                cols = np.where(comp.any(axis=0))[0]
                y1, y2 = int(rows[0]), int(rows[-1])
                x1 = int(cols[0]) + x_off
                x2 = int(cols[-1]) + x_off
                # Drop crops wider than half the full image — likely multi-building span
                if (x2 - x1) > full_W * 0.5:
                    continue
                per_image[img_idx].append({
                    "bbox":          [x1, y1, x2 + 1, y2 + 1],
                    "area_fraction": float(comp.sum() / full_px),
                    "score":         mean_conf,
                })

    # Dedup and sort per image
    out = []
    for crops in per_image:
        crops.sort(key=lambda c: -c["area_fraction"])
        kept = []
        for cand in crops:
            if all(iou(cand["bbox"], k["bbox"]) < IOU_THRESH for k in kept):
                kept.append(cand)
        out.append(kept)

    return out


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
    processor  = SegformerImageProcessor.from_pretrained(MODEL_ID)
    dtype      = torch.float16 if device == "cuda" else torch.float32
    model      = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device).eval()
    print(f"Building class IDs (ADE20K): {BUILDING_IDS}  |  min_score={args.min_score}  |  min_area={args.min_area}")

    # ── Gather images ──────────────────────────────────────────────────────────
    out_dir         = Path(args.out_dir)
    crops_dir       = out_dir / "crops"
    checkpoint_path = out_dir / "checkpoint.json"
    manifest_path   = out_dir / "manifest.jsonl"

    if args.clear:
        if crops_dir.exists():
            shutil.rmtree(crops_dir)
            print(f"Cleared {crops_dir}")
        if manifest_path.exists():
            manifest_path.unlink()
            print(f"Cleared {manifest_path}")
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    crops_dir.mkdir(parents=True, exist_ok=True)
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

    print(f"Processing {len(metas)} images  |  seg_batch={SEG_BATCH} strips\n")

    done_ids      = load_checkpoint(checkpoint_path) if full_run else set()
    n_crops_total = 0

    # Track already-written image_ids to avoid duplicates on re-run
    written_ids = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    written_ids.add(json.loads(line).get("image_id"))

    # In directory mode use written_ids as the resume set; skip already-done images
    if not full_run:
        before = len(metas)
        metas = [m for m in metas if Path(m["image_path"]).stem not in written_ids]
        if before != len(metas):
            print(f"Resuming: skipping {before - len(metas)} already-processed images")

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

            all_crops = segment_batch(pil_images, model, processor, device, args.min_area, args.min_score)

            for meta, pil, crops in zip(valid_metas, pil_images, all_crops):
                if full_run:
                    done_ids.add(str(meta.get("objectid")))

                if not crops:
                    continue

                img_path = Path(meta["image_path"])
                objectid = meta.get("objectid")
                image_id = img_path.stem
                folder   = str(objectid) if objectid is not None else image_id
                crop_dir = crops_dir / folder
                crop_dir.mkdir(parents=True, exist_ok=True)

                saved_crops = []
                for n, c in enumerate(crops):
                    x1, y1, x2, y2 = c["bbox"]
                    crop_img  = pil.crop((x1, y1, x2, y2))
                    crop_path = crop_dir / f"{image_id}_{n}.jpg"
                    crop_img.save(crop_path, quality=92)
                    saved_crops.append({
                        "crop_path":     str(crop_path),
                        "bbox":          c["bbox"],
                        "area_fraction": c["area_fraction"],
                        "score":         c["score"],
                    })
                    n_crops_total += 1

                if image_id not in written_ids:
                    record = {
                        "image_id":     image_id,
                        "image":        str(img_path),
                        "objectid":     objectid,
                        "primary_crop": saved_crops[0],
                        "other_crops":  saved_crops[1:],
                    }
                    if meta.get("residential_type"):
                        record["residential_type"] = meta["residential_type"]
                    manifest_file.write(json.dumps(record) + "\n")
                    written_ids.add(image_id)

        manifest_file.flush()

        if full_run and len(done_ids) % 500 == 0:
            save_checkpoint(checkpoint_path, done_ids)
            tqdm.write(f"  checkpoint: {len(done_ids)} done, {n_crops_total} crops saved")
        elif not full_run and batch_idx % 500 == 0 and batch_idx > 0:
            tqdm.write(f"  batch {batch_idx}/{len(batches)}  crops so far: {n_crops_total}")

    manifest_file.close()
    if full_run:
        save_checkpoint(checkpoint_path, done_ids)

    print(f"\nManifest → {manifest_path}")
    print(f"Total crops saved: {n_crops_total}")


if __name__ == "__main__":
    main()
