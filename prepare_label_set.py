"""
prepare_label_set.py

Sample N high-confidence classified buildings, segment one crop per building,
save to label_set/.

Usage:
    python prepare_label_set.py
    python prepare_label_set.py --csv dc_results/buildings_classified.csv \
                                 --db data/image_status.db \
                                 --out label_set --n 500 --seed 42 \
                                 --min-conf 0.85
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

BUILDING_IDS  = {1, 25, 48, 84}   # ADE20K: building, house, skyscraper, tower
SEG_CONF      = 0.5
MIN_AREA_FRAC = 0.06
BBOX_PAD_FRAC = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",      default="dc_results/buildings_classified.csv")
    p.add_argument("--db",       default="data/image_status.db")
    p.add_argument("--out",      default="label_set")
    p.add_argument("--n",        type=int,   default=500)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--min-conf", type=float, default=0.85,
                   help="Minimum CLIP confidence to include a building")
    return p.parse_args()


def sample_buildings(csv_path, n, seed, min_conf):
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "classified"].copy()
    df = df[pd.to_numeric(df["confidence"], errors="coerce") >= min_conf]
    print(f"  {len(df):,} buildings with confidence >= {min_conf}")
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    print(f"  {len(df)} sampled")
    return df


def build_oid_to_image(db_path):
    """Return dict: objectid (int) -> image_path (str)."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT objectid, image_path FROM status WHERE state='done' AND image_path IS NOT NULL"
    ).fetchall()
    conn.close()
    mapping = {}
    for oid, path in rows:
        path = str(path).replace("\\", "/")
        if int(oid) not in mapping:
            mapping[int(oid)] = path
    print(f"DB: {len(mapping):,} done images indexed")
    return mapping


def load_seg_model(device):
    print("Loading SegFormer...")
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    model = model.to(device).half().eval()
    return model, processor


def best_crop(image, model, processor, device):
    """
    Segment image, return the largest building crop (PIL Image).
    Falls back to the full image if no building is found.
    """
    w, h = image.size

    inputs = processor(images=image, return_tensors="pt")
    inputs = {
        k: v.to(device=device, dtype=torch.float16 if v.is_floating_point() else v.dtype)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        logits = model(**inputs).logits

    seg = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    probs = torch.softmax(seg[0], dim=0)

    building_prob = sum(probs[c] for c in BUILDING_IDS if c < probs.shape[0])
    mask = (building_prob > SEG_CONF).cpu().numpy()

    if mask.sum() / mask.size < MIN_AREA_FRAC:
        return image, False  # fallback

    from scipy import ndimage as ndi
    labeled, num = ndi.label(mask)
    if num == 0:
        return image, False

    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    best = int(np.argmax(sizes)) + 1
    component = labeled == best

    if component.sum() / mask.size < MIN_AREA_FRAC:
        return image, False

    rows_idx = np.where(component.any(axis=1))[0]
    cols_idx = np.where(component.any(axis=0))[0]
    y1, y2 = int(rows_idx[0]), int(rows_idx[-1])
    x1, x2 = int(cols_idx[0]), int(cols_idx[-1])

    pad_x = max(1, int((x2 - x1) * BBOX_PAD_FRAC))
    pad_y = max(1, int((y2 - y1) * BBOX_PAD_FRAC))
    x1, x2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(h, y2 + pad_y)

    return image.crop((x1, y1, x2, y2)), True


def main():
    args = parse_args()
    out_dir   = Path(args.out)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print("Sampling buildings...")
    df      = sample_buildings(args.csv, args.n, args.seed, args.min_conf)
    oid_map = build_oid_to_image(args.db)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, processor = load_seg_model(device)

    rows     = []
    skipped  = 0
    fallback = 0

    for _, bld in tqdm(df.iterrows(), total=len(df), desc="Crops"):
        oid       = int(bld["objectid"])
        crop_path = crops_dir / f"{oid}.jpg"

        # Resume: skip if crop already exists
        if crop_path.exists():
            rows.append({
                "objectid":    oid,
                "address":     bld.get("address", ""),
                "crop_path":   str(crop_path),
                "model_label": bld.get("predicted_label", ""),
                "confidence":  bld.get("confidence", ""),
            })
            continue

        img_path = oid_map.get(oid)
        if not img_path or not Path(img_path).exists():
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            crop, found = best_crop(image, model, processor, device)
            if not found:
                fallback += 1
            crop.save(crop_path, quality=92)
            rows.append({
                "objectid":    oid,
                "address":     bld.get("address", ""),
                "crop_path":   str(crop_path),
                "model_label": bld.get("predicted_label", ""),
                "confidence":  bld.get("confidence", ""),
            })
        except Exception as e:
            print(f"  Error oid={oid}: {e}")
            skipped += 1

    meta_path = out_dir / "metadata.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False)

    print(f"\nDone.")
    print(f"  Crops saved:  {len(rows)}")
    print(f"  Skipped:      {skipped}  (no image or file missing)")
    print(f"  Fallback:     {fallback}  (no building found, used full image)")
    print(f"  Metadata:     {meta_path}")
    print(f"\nNext: streamlit run label_app.py")


if __name__ == "__main__":
    main()
