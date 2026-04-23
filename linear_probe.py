"""
Linear probe on frozen CLIP ViT-B/32 features.

Encodes all training images once, trains a logistic regression classifier,
evaluates on Pexels holdout and optionally DC labeled crops.

Usage:
    python linear_probe.py --name probe_condensed --style-only --condense --dc-labels dc_labels.csv
    python linear_probe.py --name probe_style     --style-only --dc-labels dc_labels.csv
    python linear_probe.py --name probe_hier       --dc-labels dc_labels.csv
"""

import argparse
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm


# =============================================================================
# CONFIG (mirrors train_architecture.py)
# =============================================================================

VALID_PREFIXES = (
    "rowhouse_", "single_family_house_", "small_multifamily_",
    "large_multifamily_", "office_", "institutional_",
)

TYPE_KEYS = sorted([
    "single_family_house", "small_multifamily", "large_multifamily",
    "rowhouse", "office", "institutional",
], key=len, reverse=True)

KEEP_STYLES = {
    "rowhouse_vernacular", "colonial_revival", "developer_modern",
    "italianate", "victorian", "contemporary_glass", "garden_style",
    "art_deco", "developer_traditional", "neoclassical", "brutalist",
    "beaux_arts", "modernist",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name",         type=str, default="probe")
    p.add_argument("--out-dir",      type=str, default="outputs")
    p.add_argument("--data-dir",     type=str, default="data/styles")
    p.add_argument("--style-only",   action="store_true", default=False)
    p.add_argument("--condense",     action="store_true", default=False)
    p.add_argument("--probe-C",      type=float, default=0.1,
                   help="Inverse regularization (smaller = more regularized)")
    p.add_argument("--batch-size",   type=int, default=512)
    p.add_argument("--num-threads",  type=int, default=8)
    p.add_argument("--dc-labels",    type=str, default=None)
    return p.parse_args()


args    = parse_args()
OUT_DIR = f"{args.out_dir}/{args.name}"
torch.set_num_threads(args.num_threads)


# =============================================================================
# MODEL
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


# =============================================================================
# HELPERS
# =============================================================================

def _style_slug(folder_name):
    for prefix in TYPE_KEYS:
        if folder_name.startswith(prefix + "_"):
            return folder_name[len(prefix) + 1:]
    return folder_name


def _slug(s):
    return s.lower().replace(" ", "_").replace("-", "_").replace("'", "")


def condense(slug):
    return slug if slug in KEEP_STYLES else "other"


# =============================================================================
# DATA
# =============================================================================

def load_dataset(root_dir):
    root = Path(root_dir)
    image_paths, labels = [], []
    class_to_idx = {}

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if not any(folder.name.startswith(p) for p in VALID_PREFIXES):
            continue
        imgs = [p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not imgs:
            continue

        cls = _style_slug(folder.name) if args.style_only else folder.name
        if args.condense and args.style_only:
            cls = condense(cls)

        if cls not in class_to_idx:
            class_to_idx[cls] = len(class_to_idx)
        idx = class_to_idx[cls]
        for img in imgs:
            image_paths.append(img)
            labels.append(idx)

    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    return image_paths, np.array(labels), class_names


def train_test_split(paths, labels, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(paths))
    split = int(len(idx) * (1 - test_ratio))
    return ([paths[i] for i in idx[:split]], labels[idx[:split]],
            [paths[i] for i in idx[split:]], labels[idx[split:]])


# =============================================================================
# ENCODING
# =============================================================================

def load_image(path):
    try:
        return preprocess(Image.open(path).convert("RGB"))
    except Exception:
        return None


def encode(image_paths, desc="Encoding"):
    all_feats = []
    batches = [image_paths[i:i+args.batch_size] for i in range(0, len(image_paths), args.batch_size)]
    use_amp = device == "cuda"

    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        prefetch = ex.submit(lambda b: list(ex.map(load_image, b)), batches[0]) if batches else None
        for i in tqdm(range(len(batches)), desc=desc, unit="batch"):
            imgs = prefetch.result()
            if i + 1 < len(batches):
                prefetch = ex.submit(lambda b: list(ex.map(load_image, b)), batches[i + 1])
            valid = [x for x in imgs if x is not None]
            if not valid:
                continue
            batch = torch.stack(valid).pin_memory().to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                feats = model.encode_image(batch)
                feats /= feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().float().numpy())

    return np.concatenate(all_feats) if all_feats else np.zeros((0, 512))


# =============================================================================
# LINEAR PROBE
# =============================================================================

def train_probe(train_feats, train_labels):
    print(f"Training logistic regression  C={args.probe_C}  classes={len(set(train_labels))}")
    probe = LogisticRegression(
        C=args.probe_C,
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
        verbose=0,
    )
    probe.fit(train_feats, train_labels)
    return probe


def probe_probs(probe, feats):
    return probe.predict_proba(feats)


# =============================================================================
# EVALUATION
# =============================================================================

def top_k_accuracy(probs, labels, k=3):
    topk = np.argsort(-probs, axis=1)[:, :k]
    return np.mean([labels[i] in topk[i] for i in range(len(labels))])


def evaluate(probs, labels, tag=""):
    preds = probs.argmax(axis=1)
    acc  = (preds == labels).mean()
    topk = top_k_accuracy(probs, labels, k=3)
    print(f"[{tag}] Top-1: {acc:.4f}   Top-3: {topk:.4f}")
    return acc


def export(probs, labels, class_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    preds = probs.argmax(axis=1)
    metrics = {
        "accuracy":    float(accuracy_score(labels, preds)),
        "f1_macro":    float(f1_score(labels, preds, average="macro",    zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }
    np.save(f"{save_dir}/probabilities.npy", probs)
    np.save(f"{save_dir}/predictions.npy",   preds)
    np.save(f"{save_dir}/labels.npy",        labels)
    np.save(f"{save_dir}/confusion_matrix.npy", confusion_matrix(labels, preds))
    with open(f"{save_dir}/class_names.json", "w") as f:
        json.dump(class_names, f)
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> {save_dir}: {metrics}")
    return metrics


# =============================================================================
# DC EVAL
# =============================================================================

def eval_dc(labels_path, class_names, probe):
    import pandas as pd
    df = pd.read_csv(labels_path)
    df = df[~df["style"].isin(["unsure", "Other", "other"])]

    # Handle both old (building_type + style) and new (style only) CSV formats
    has_type = "building_type" in df.columns and df["building_type"].notna().any()
    if has_type:
        df = df[df["building_type"] != "unknown"]

    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    paths, true_labels = [], []
    skipped = 0

    for _, row in df.iterrows():
        style_slug = _slug(row["style"])
        if args.condense:
            style_slug = condense(style_slug)
        if args.style_only:
            slug = style_slug
        else:
            btype_slug = _slug(row.get("building_type", "")) if has_type else ""
            slug = f"{btype_slug}_{style_slug}" if btype_slug else style_slug

        if slug not in cls_to_idx:
            skipped += 1
            continue
        p = Path(row["crop_path"])
        if not p.exists():
            skipped += 1
            continue
        paths.append(p)
        true_labels.append(cls_to_idx[slug])

    print(f"\nDC crops: {len(paths)} valid  |  {skipped} skipped")
    if not paths:
        return

    feats = encode(paths, desc="DC crops")
    probs = probe_probs(probe, feats)
    true_labels = np.array(true_labels)
    evaluate(probs, true_labels, tag="DC crops")
    export(probs, true_labels, class_names, save_dir=OUT_DIR + "_dc")


# =============================================================================
# MAIN
# =============================================================================

image_paths, labels, class_names = load_dataset(args.data_dir)
num_classes = len(class_names)
print(f"\nLoaded {len(image_paths)} images  |  {num_classes} classes")
print(f"style_only={args.style_only}  condense={args.condense}  C={args.probe_C}")

train_paths, train_labels, test_paths, test_labels = train_test_split(image_paths, labels)
print(f"Train: {len(train_paths)}  |  Test: {len(test_paths)}")

print("\nEncoding training images...")
train_feats = encode(train_paths, desc="Train")

print("\nEncoding test images...")
test_feats = encode(test_paths, desc="Test")

probe = train_probe(train_feats, train_labels)

print("\nEvaluating...")
test_probs = probe_probs(probe, test_feats)
evaluate(test_probs, test_labels, tag="Pexels holdout")

os.makedirs(OUT_DIR, exist_ok=True)
export(test_probs, test_labels, class_names, save_dir=OUT_DIR)

with open(f"{OUT_DIR}/probe.pkl", "wb") as f:
    pickle.dump(probe, f)
with open(f"{OUT_DIR}/class_names.json", "w") as f:
    json.dump(class_names, f)
print(f"Saved probe -> {OUT_DIR}/probe.pkl")

if args.dc_labels:
    eval_dc(args.dc_labels, class_names, probe)
