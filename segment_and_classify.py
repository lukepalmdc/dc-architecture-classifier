"""
segment_and_classify.py

Segment buildings from street-level images using SegFormer (ADE20K),
then classify each building crop with the trained CLIP architecture classifier.

Usage:
    python segment_and_classify.py data/images/ --sample 50 --save-viz
    python segment_and_classify.py --full-run --out-dir dc_results
"""

import argparse
import json
import jsonlines
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
import clip
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from scipy import ndimage

# ── ADE20K building class IDs (0-indexed) ───────────────────────────────────
BUILDING_IDS  = {1, 25, 48, 84}   # building/edifice, house, skyscraper, tower
BBOX_PAD_FRAC = 0.05
SEG_BATCH     = 12    # strips per SegFormer forward pass
CLIP_BATCH    = 64    # crops per CLIP forward pass
IMAGE_BATCH   = 4     # images loaded and processed together
PREFETCH      = 8     # images prefetched while GPU works
SEG_CONF      = 0.5   # per-pixel SegFormer confidence threshold

LABEL_DISPLAY = {
    "art_deco":           "Art Deco",
    "art_nouveau":        "Art Nouveau",
    "beaux_arts":         "Beaux-Arts",
    "brutalist":          "Brutalist",
    "colonial_revival":   "Colonial Revival",
    "craftsman":          "Craftsman",
    "federal":            "Federal",
    "gothic_revival":     "Gothic Revival",
    "greek_revival":      "Greek Revival",
    "midcentury_modern":  "Mid-Century Modern",
    "modernist":          "Modernist",
    "neoclassical":       "Neoclassical",
    "postmodern":         "Postmodern",
    "romanesque_revival": "Romanesque Revival",
    "tudor_revival":      "Tudor Revival",
    "victorian":          "Victorian",
}

COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input",          nargs="?", default=None,
                   help="Image file or directory (omit with --full-run)")
    p.add_argument("--full-run",     action="store_true",
                   help="Run on all downloaded buildings from data/image_status.db")
    p.add_argument("--db",           default="data/image_status.db")
    p.add_argument("--exp",          default=None,
                   help="Experiment name under outputs/ (default: best by accuracy)")
    p.add_argument("--top",          type=int,   default=3)
    p.add_argument("--min-conf",     type=float, default=0.7)
    p.add_argument("--min-area",     type=float, default=0.06)
    p.add_argument("--temperature",  type=float, default=0.3)
    p.add_argument("--out-dir",      default="dctest")
    p.add_argument("--save-viz",     action="store_true")
    p.add_argument("--sample",       type=int,   default=None)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# =============================================================================
# EXPERIMENT ARTIFACTS
# =============================================================================

def find_best_exp():
    best_exp, best_acc = None, -1.0
    for d in Path("outputs").iterdir():
        p = d / "metrics.json"
        if not p.exists():
            continue
        with open(p) as f:
            m = json.load(f)
        if m.get("accuracy", 0) > best_acc:
            best_acc = m["accuracy"]
            best_exp = d.name
    return best_exp


def load_experiment(exp_name):
    exp_dir = Path("outputs") / exp_name
    with open(exp_dir / "class_names.json") as f:
        class_names = json.load(f)

    prompts_path = exp_dir / "prompts.json"
    if prompts_path.exists():
        with open(prompts_path) as f:
            style_prompts = json.load(f)
    else:
        style_prompts = {
            cls: [
                f"a {LABEL_DISPLAY.get(cls, cls.replace('_',' ').title())} building",
                f"{LABEL_DISPLAY.get(cls, cls.replace('_',' ').title())} architecture",
                f"a building in {LABEL_DISPLAY.get(cls, cls.replace('_',' ').title())} style",
                f"photo of {LABEL_DISPLAY.get(cls, cls.replace('_',' ').title())} architecture",
            ]
            for cls in class_names
        }

    prototypes     = np.load(exp_dir / "prototypes.npy")     if (exp_dir / "prototypes.npy").exists()     else None
    prompt_weights = np.load(exp_dir / "prompt_weights.npy") if (exp_dir / "prompt_weights.npy").exists() else None
    return class_names, style_prompts, prototypes, prompt_weights


def encode_text(style_prompts, class_names, clip_model, device):
    all_prompts, class_indices = [], []
    cls_to_idx = {cls: i for i, cls in enumerate(class_names)}
    for cls, prompts in style_prompts.items():
        if cls not in cls_to_idx:
            continue
        for prompt in prompts:
            all_prompts.append(prompt)
            class_indices.append(cls_to_idx[cls])
    tokens = clip.tokenize(all_prompts).to(device)
    with torch.no_grad():
        tf = clip_model.encode_text(tokens)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    return tf, class_indices


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
# SEGMENTATION  (batched across strips)
# =============================================================================

def iou(a, b):
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _extract_crops_from_mask(building_mask, x_offset, full_W, full_H, min_area_frac):
    total_px  = full_W * full_H
    labeled, n = ndimage.label(building_mask)
    crops = []
    for comp_id in range(1, n + 1):
        comp      = labeled == comp_id
        area_frac = comp.sum() / total_px
        if area_frac < min_area_frac:
            continue
        rows = np.where(comp.any(axis=1))[0]
        cols = np.where(comp.any(axis=0))[0]
        y1, y2 = int(rows[0]),  int(rows[-1])
        x1, x2 = int(cols[0]) + x_offset, int(cols[-1]) + x_offset
        pad_y = max(1, int((y2 - y1) * BBOX_PAD_FRAC))
        pad_x = max(1, int((x2 - x1) * BBOX_PAD_FRAC))
        crops.append({
            "bbox":          [max(0,x1-pad_x), max(0,y1-pad_y),
                               min(full_W,x2+pad_x), min(full_H,y2+pad_y)],
            "area_fraction": float(area_frac),
        })
    return crops


def segment_batch(images, seg_model, seg_processor, device, min_area_frac):
    """
    Run SegFormer on a batch of PIL images using 3-strip tiling.
    Returns per-image list of crop dicts (bbox, area_fraction).
    All SegFormer forward passes are batched together.
    """
    # Build strip list: (img_idx, x_offset, full_W, full_H, strip_W, strip_H, strip_pil)
    strip_meta, strip_pils = [], []
    for img_idx, pil in enumerate(images):
        W, H    = pil.size
        sw      = int(W * 0.6)
        for x0 in [0, int(W * 0.2), int(W * 0.4)]:
            x1 = min(x0 + sw, W)
            s  = pil.crop((x0, 0, x1, H))
            strip_meta.append((img_idx, x0, W, H, x1 - x0, H))
            strip_pils.append(s)

    per_image = [[] for _ in images]

    # Run SegFormer in SEG_BATCH-sized chunks
    for i in range(0, len(strip_pils), SEG_BATCH):
        batch_pils = strip_pils[i:i + SEG_BATCH]
        batch_meta = strip_meta[i:i + SEG_BATCH]

        inputs = seg_processor(images=batch_pils, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = seg_model(**inputs).logits       # [B x C x h x w]

        for j, (img_idx, x_off, full_W, full_H, sW, sH) in enumerate(batch_meta):
            up       = F.interpolate(logits[j:j+1], size=(sH, sW),
                                     mode="bilinear", align_corners=False)
            probs    = F.softmax(up, dim=1)
            seg_map  = probs.argmax(dim=1).squeeze(0).cpu().numpy()
            max_prob = probs.max(dim=1).values.squeeze(0).cpu().numpy()
            mask     = np.isin(seg_map, list(BUILDING_IDS)) & (max_prob >= SEG_CONF)
            if mask.any():
                per_image[img_idx].extend(
                    _extract_crops_from_mask(mask, x_off, full_W, full_H, min_area_frac)
                )

    # Dedup per image, attach PIL crops
    results = []
    for img_idx, pil in enumerate(images):
        crops = per_image[img_idx]
        if not crops:
            results.append([])
            continue
        crops.sort(key=lambda c: -c["area_fraction"])
        kept = []
        for cand in crops:
            if all(iou(cand["bbox"], k["bbox"]) < 0.4 for k in kept):
                kept.append(cand)
        for c in kept:
            x1, y1, x2, y2 = c["bbox"]
            c["crop"] = pil.crop((x1, y1, x2, y2))
        results.append(kept)

    return results


# =============================================================================
# CLASSIFICATION  (batched across all crops from a batch of images)
# =============================================================================

def classify_batch(all_crops, class_names, prototypes, text_features,
                   class_indices, prompt_weights, temperature, top_k,
                   clip_model, clip_preprocess, device):
    """
    Classify all crops from a batch of images in one CLIP forward pass.
    Returns per-image list of prediction dicts.
    """
    num_classes = len(class_names)

    # Flatten crops
    flat = []   # (img_idx, crop_idx, pil_crop)
    for i, crops in enumerate(all_crops):
        for j, c in enumerate(crops):
            if c.get("crop") is not None:
                flat.append((i, j, c["crop"]))

    scores_map = {}   # (img_idx, crop_idx) -> scores_np [num_classes]

    for b in range(0, len(flat), CLIP_BATCH):
        chunk = flat[b:b + CLIP_BATCH]
        imgs  = torch.stack([clip_preprocess(x[2]) for x in chunk]).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
            feats = clip_model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.to(dtype=text_features.dtype)   # match fp16

        # Weighted scatter-add on GPU  [B x num_classes]
        raw  = feats @ text_features.T / temperature
        sims = torch.sigmoid(raw)

        w    = (torch.tensor(prompt_weights, dtype=feats.dtype, device=device)
                if prompt_weights is not None
                else torch.ones(len(class_indices), dtype=feats.dtype, device=device))
        idx  = torch.tensor(class_indices, device=device)

        weighted     = sims * w.unsqueeze(0)
        text_scores  = torch.zeros(len(chunk), num_classes, dtype=feats.dtype, device=device)
        text_scores.scatter_add_(1, idx.unsqueeze(0).expand(len(chunk), -1), weighted)
        weight_sums  = torch.zeros(num_classes, dtype=feats.dtype, device=device)
        weight_sums.scatter_add_(0, idx, w)
        text_scores /= weight_sums.clamp(min=1e-8).unsqueeze(0)

        if prototypes is not None:
            proto_t      = torch.tensor(prototypes, dtype=feats.dtype, device=device)
            proto_scores = torch.sigmoid(feats @ proto_t.T / temperature)
            scores       = (0.5 * proto_scores + 0.5 * text_scores).cpu().float().numpy()
        else:
            scores = text_scores.cpu().float().numpy()

        for k, (i, j, _) in enumerate(chunk):
            scores_map[(i, j)] = scores[k]

    # Rebuild per-image results
    per_image = [[] for _ in all_crops]
    for i, crops in enumerate(all_crops):
        for j, c in enumerate(crops):
            s = scores_map.get((i, j))
            if s is None:
                continue
            top_idx  = np.argsort(-s)[:top_k]
            top_conf = float(s[top_idx[0]])
            per_image[i].append({
                "bbox":         c["bbox"],
                "area_fraction":c["area_fraction"],
                "top":          [{"label": class_names[k], "confidence": round(float(s[k]),4)}
                                 for k in top_idx],
                "all_scores":   {class_names[k]: round(float(s[k]),4) for k in range(num_classes)},
                "other_p":      round(1.0 - top_conf, 4),
            })

    return per_image


# =============================================================================
# VISUALIZATION
# =============================================================================

def save_viz(pil_image, buildings, out_path):
    viz  = pil_image.copy()
    draw = ImageDraw.Draw(viz)
    for i, b in enumerate(buildings):
        if "top" not in b:
            continue
        x1, y1, x2, y2 = b["bbox"]
        color = COLORS[i % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        top = b["top"][0]
        draw.text((x1+4, y1+4),
                  f"{top['label']} {top['confidence']:.2f}  other={b['other_p']:.2f}",
                  fill=color)
    viz.save(out_path)


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_building_predictions(jsonl_path):
    by_building = defaultdict(list)
    with jsonlines.open(jsonl_path) as reader:
        for r in reader:
            oid = r.get("objectid")
            if oid is None:
                continue
            for b in r.get("buildings", []):
                if "all_scores" in b:
                    by_building[oid].append(b["all_scores"])

    aggregated = {}
    for oid, score_dicts in by_building.items():
        avg = defaultdict(float)
        for sd in score_dicts:
            for lbl, val in sd.items():
                avg[lbl] += val / len(score_dicts)
        top_label = max(avg, key=avg.get)
        aggregated[oid] = {
            "label":      top_label,
            "confidence": round(avg[top_label], 4),
            "other_p":    round(1.0 - avg[top_label], 4),
            "all_scores": {k: round(v,4) for k,v in sorted(avg.items(), key=lambda x:-x[1])},
            "n_crops":    len(score_dicts),
        }
    return aggregated


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

    # ── Models ───────────────────────────────────────────────────────────────
    print("Loading SegFormer (ADE20K b0 — optimised for throughput)...")
    seg_processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512")
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    ).to(device).eval().half()   # fp16

    print("Loading CLIP ViT-B/32...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    if device == "cuda":
        try:
            clip_model = torch.compile(clip_model, mode="reduce-overhead")
            print("torch.compile enabled for CLIP")
        except Exception:
            print("torch.compile not available, continuing without")

    # ── Experiment ───────────────────────────────────────────────────────────
    exp_name = args.exp or find_best_exp()
    if not exp_name:
        print("No experiment outputs found.")
        return
    print(f"Experiment: {exp_name}")

    class_names, style_prompts, prototypes, prompt_weights = load_experiment(exp_name)
    text_features, class_indices = encode_text(style_prompts, class_names, clip_model, device)

    # ── Gather images ─────────────────────────────────────────────────────────
    out_dir         = Path(args.out_dir);  out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.json"
    results_path    = out_dir / "results.jsonl"
    full_run        = args.full_run or not args.input

    if full_run:
        db_rows   = load_images_from_db(args.db)
        done_ids  = load_checkpoint(checkpoint_path)
        metas     = [r for r in db_rows if str(r["objectid"]) not in done_ids]
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

    print(f"Processing {len(metas)} images  |  seg_batch={SEG_BATCH}  clip_batch={CLIP_BATCH}\n")

    done_ids = load_checkpoint(checkpoint_path) if full_run else set()
    n_buildings_found = 0

    # Open JSONL in append mode so we resume without losing prior results
    jsonl_file = open(results_path, "a")

    with ThreadPoolExecutor(max_workers=PREFETCH) as loader:
        # Pre-submit first IMAGE_BATCH
        batches     = [metas[i:i+IMAGE_BATCH] for i in range(0, len(metas), IMAGE_BATCH)]
        prefetch_q  = []

        def submit_batch(batch):
            return [(m, loader.submit(load_image, m["image_path"])) for m in batch]

        if batches:
            prefetch_q.append(submit_batch(batches[0]))

        for batch_idx in tqdm(range(len(batches)), unit="batch",
                               desc=f"batches of {IMAGE_BATCH}"):
            # Pre-submit next batch while GPU works
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

            # SegFormer: segment all images in this batch
            all_crops = segment_batch(pil_images, seg_model, seg_processor,
                                      device, args.min_area)

            # CLIP: classify all crops across all images together
            all_preds = classify_batch(
                all_crops, class_names, prototypes, text_features,
                class_indices, prompt_weights, args.temperature, args.top,
                clip_model, clip_preprocess, device,
            )

            # Write results + checkpoint
            for meta, pil, preds in zip(valid_metas, pil_images, all_preds):
                if full_run:
                    pil.close()   # free memory, no viz needed
                img_path = Path(meta["image_path"])
                objectid = meta.get("objectid")

                # Filter by min confidence
                buildings = []
                for b in preds:
                    top_conf = b["top"][0]["confidence"] if b["top"] else 0.0
                    if top_conf >= args.min_conf:
                        out_b = {k: v for k, v in b.items() if k != "crop"}
                        buildings.append(out_b)
                        n_buildings_found += 1
                    else:
                        buildings.append({
                            "bbox":          b["bbox"],
                            "area_fraction": b["area_fraction"],
                            "filtered":      "low_confidence",
                            "top_confidence":top_conf,
                            "other_p":       b["other_p"],
                        })

                r = {"image": str(img_path), "objectid": objectid, "buildings": buildings}
                if meta.get("residential_type"):
                    r["residential_type"] = meta["residential_type"]

                if args.save_viz and not full_run and buildings:
                    viz_path = out_dir / (img_path.stem + ".viz.jpg")
                    save_viz(pil, [b for b in buildings if "top" in b], viz_path)

                jsonl_file.write(json.dumps(r) + "\n")

                if full_run:
                    done_ids.add(str(objectid))

            jsonl_file.flush()

            if full_run and len(done_ids) % 500 == 0:
                save_checkpoint(checkpoint_path, done_ids)
                tqdm.write(f"  checkpoint: {len(done_ids)} done, {n_buildings_found} buildings found")

    jsonl_file.close()
    if full_run:
        save_checkpoint(checkpoint_path, done_ids)

    print(f"\nResults → {results_path}")
    print(f"Buildings with predictions: {n_buildings_found}")

    # ── Aggregate per building ─────────────────────────────────────────────
    if full_run:
        print("Aggregating per-building labels...")
        aggregated = aggregate_building_predictions(results_path)
        agg_path   = out_dir / "buildings_classified.json"
        with open(agg_path, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"Aggregated {len(aggregated)} buildings → {agg_path}")

        label_counts = defaultdict(int)
        for v in aggregated.values():
            label_counts[v["label"]] += 1
        print("\nLabel distribution:")
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {lbl:30s} {cnt}")


if __name__ == "__main__":
    main()
