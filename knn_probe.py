"""
kNN classifier on frozen CLIP ViT-B/32 features.

Uses cosine similarity (dot product on L2-normalized features) via batched
GPU matmul — no faiss needed, no per-query overhead.

Soft voting: each neighbor contributes its similarity score weighted by 1/rank
to the class distribution, giving smoother probabilities than hard vote.

Usage:
    python knn_probe.py --name knn_condensed --style-only --condense --dc-labels dc_labels.csv
    python knn_probe.py --name knn_condensed --style-only --condense --k 20 --dc-labels dc_labels.csv
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm


# =============================================================================
# CONFIG
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
    p.add_argument("--name",        type=str,   default="knn")
    p.add_argument("--out-dir",     type=str,   default="outputs")
    p.add_argument("--data-dir",    type=str,   default="data/styles")
    p.add_argument("--style-only",  action="store_true", default=False)
    p.add_argument("--condense",    action="store_true", default=False)
    p.add_argument("--k",           type=int,   default=20,
                   help="Number of nearest neighbours (default 20)")
    p.add_argument("--batch-size",  type=int,   default=512)
    p.add_argument("--num-threads", type=int,   default=8)
    p.add_argument("--dc-labels",   type=str,   default=None)
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
        for img in imgs:
            image_paths.append(img)
            labels.append(class_to_idx[cls])

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
            all_feats.append(feats.cpu().float())

    return torch.cat(all_feats) if all_feats else torch.zeros(0, 512)


# =============================================================================
# KNN INFERENCE
# =============================================================================

def knn_probs(query_feats, train_feats, train_labels, num_classes, k):
    """
    Batched cosine-similarity kNN via GPU matmul.
    query_feats:  [Q x D] float32
    train_feats:  [N x D] float32
    Returns:      [Q x num_classes] soft-vote probability matrix
    """
    train_gpu = train_feats.to(device)   # [N x D]
    all_probs = []

    for i in tqdm(range(0, len(query_feats), args.batch_size),
                  desc="kNN inference", unit="batch", leave=False):
        q = query_feats[i:i+args.batch_size].to(device)  # [B x D]
        sims = q @ train_gpu.T                            # [B x N] cosine sims

        # Top-k per query
        topk_sims, topk_idx = sims.topk(k, dim=1)        # [B x k]
        topk_labels = train_labels[topk_idx.cpu().numpy()]# [B x k]

        # Soft vote: weight = 1/rank (rank 1 = highest sim)
        ranks = torch.arange(1, k + 1, dtype=torch.float32, device=device)
        weights = (1.0 / ranks).unsqueeze(0).expand(len(q), -1)  # [B x k]

        probs = torch.zeros(len(q), num_classes, device=device)
        for j in range(k):
            lbls = torch.tensor(topk_labels[:, j], device=device)
            probs.scatter_add_(1, lbls.unsqueeze(1), weights[:, j:j+1])

        # Normalize to sum to 1
        probs /= probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        all_probs.append(probs.cpu().float())

    return torch.cat(all_probs).numpy()


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

def eval_dc(labels_path, class_names, train_feats, train_labels_arr):
    import pandas as pd
    df = pd.read_csv(labels_path)
    df = df[~df["style"].isin(["unsure", "Other", "other"])]
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

    dc_feats = encode(paths, desc="DC crops")
    probs = knn_probs(dc_feats, train_feats, train_labels_arr, len(class_names), args.k)
    true_labels = np.array(true_labels)
    evaluate(probs, true_labels, tag="DC crops")
    export(probs, true_labels, class_names, save_dir=OUT_DIR + "_dc")


# =============================================================================
# MAIN
# =============================================================================

image_paths, labels, class_names = load_dataset(args.data_dir)
num_classes = len(class_names)
print(f"\nLoaded {len(image_paths)} images  |  {num_classes} classes  |  k={args.k}")
print(f"style_only={args.style_only}  condense={args.condense}")

train_paths, train_labels, test_paths, test_labels = train_test_split(image_paths, labels)
print(f"Train: {len(train_paths)}  |  Test: {len(test_paths)}")

print("\nEncoding training images...")
train_feats  = encode(train_paths, desc="Train")
train_labels_t = torch.tensor(train_labels, dtype=torch.long)

print("\nEncoding test images...")
test_feats = encode(test_paths, desc="Test")

print(f"\nRunning kNN (k={args.k})...")
test_probs = knn_probs(test_feats, train_feats, train_labels_t.numpy(), num_classes, args.k)
evaluate(test_probs, test_labels, tag="Pexels holdout")

os.makedirs(OUT_DIR, exist_ok=True)
export(test_probs, test_labels, class_names, save_dir=OUT_DIR)

# Save train features for potential reuse
np.save(f"{OUT_DIR}/train_feats.npy",  train_feats.numpy())
np.save(f"{OUT_DIR}/train_labels.npy", train_labels)
with open(f"{OUT_DIR}/class_names.json", "w") as f:
    json.dump(class_names, f)
print(f"Saved train features -> {OUT_DIR}/")

if args.dc_labels:
    eval_dc(args.dc_labels, class_names, train_feats, train_labels_t.numpy())
