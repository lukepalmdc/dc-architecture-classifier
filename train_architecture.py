"""
CLIP Architecture Classifier

Trains on scraped Pexels images in data/styles/<type>_<style>/
Evaluates on:
  - 20% held-out Pexels test split
  - (optionally) labeled DC street-view crops via --dc-labels

Usage:
    python train_architecture.py --name exp1 --prompts build
    python train_architecture.py --name exp1 --prompts build --dc-labels dc_labels.csv
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

TYPE_DISPLAY = {
    "rowhouse":             "rowhouse",
    "single_family_house":  "single family house",
    "small_multifamily":    "small multifamily building",
    "large_multifamily":    "large multifamily apartment building",
    "office":               "office building",
    "institutional":        "institutional building",
}
TYPE_KEYS = sorted(TYPE_DISPLAY.keys(), key=len, reverse=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--proto-weight",   type=float, default=0.5)
    p.add_argument("--tune-alpha",     action="store_true", default=True)
    p.add_argument("--no-tune-alpha",  dest="tune_alpha", action="store_false")
    p.add_argument("--ridge-alpha",    type=float, default=1e-3)
    p.add_argument("--use-prototypes", action="store_true", default=True)
    p.add_argument("--no-prototypes",  dest="use_prototypes", action="store_false")
    p.add_argument("--use-weights",    action="store_true", default=True)
    p.add_argument("--no-weights",     dest="use_weights", action="store_false")
    p.add_argument("--prompts",        type=str, default="prompts.json",
                   help="Path to prompts JSON, or 'build' to auto-generate")
    p.add_argument("--temperature",    type=float, default=0.3)
    p.add_argument("--out-dir",        type=str, default="outputs")
    p.add_argument("--name",           type=str, default="default")
    p.add_argument("--batch-size",     type=int, default=512)
    p.add_argument("--num-threads",    type=int, default=8)
    p.add_argument("--dc-labels",      type=str, default=None,
                   help="Path to dc_labels.csv for real-world evaluation")
    return p.parse_args()


args = parse_args()

DATA_DIR     = "data/styles"
BATCH_SIZE   = args.batch_size
NUM_THREADS  = args.num_threads
TEST_SPLIT   = 0.2
TOP_K        = 3
SEED         = 42

PROTO_WEIGHT   = args.proto_weight
TUNE_ALPHA     = args.tune_alpha
RIDGE_ALPHA    = args.ridge_alpha
USE_PROTOTYPES = args.use_prototypes
USE_WEIGHTS    = args.use_weights
TEMPERATURE    = args.temperature
OUT_DIR        = f"{args.out_dir}/{args.name}"

torch.set_num_threads(NUM_THREADS)


# =============================================================================
# MODEL
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


# =============================================================================
# DATA
# =============================================================================

def load_dataset(root_dir):
    root = Path(root_dir)
    image_paths, labels, class_names = [], [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if not any(folder.name.startswith(p) for p in VALID_PREFIXES):
            continue
        imgs = [p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not imgs:
            continue
        idx = len(class_names)
        class_names.append(folder.name)
        for img in imgs:
            image_paths.append(img)
            labels.append(idx)
    return image_paths, np.array(labels), class_names


def train_test_split(paths, labels, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(paths))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return ([paths[i] for i in train_idx], labels[train_idx],
            [paths[i] for i in test_idx],  labels[test_idx])


image_paths, labels, class_names = load_dataset(DATA_DIR)
num_classes = len(class_names)
print(f"Loaded {len(image_paths)} images across {num_classes} classes")

train_paths, train_labels, test_paths, test_labels = train_test_split(
    image_paths, labels, TEST_SPLIT, SEED
)
print(f"Train: {len(train_paths)} | Test: {len(test_paths)}")


# =============================================================================
# PROMPTS
# =============================================================================

def parse_class(cls):
    """Split 'rowhouse_federal' -> ('rowhouse', 'federal style')"""
    for prefix in TYPE_KEYS:
        if cls.startswith(prefix + "_"):
            style = cls[len(prefix) + 1:].replace("_", " ")
            return TYPE_DISPLAY[prefix], style
    return None, cls.replace("_", " ")


def build_prompts(class_names):
    prompts = {}
    for cls in class_names:
        btype, style = parse_class(cls)
        if btype:
            prompts[cls] = [
                f"a {style} {btype}",
                f"a {btype} in {style} style",
                f"{style} style {btype}",
                f"photo of a {style} {btype}",
            ]
        else:
            prompts[cls] = [
                f"a {style} building",
                f"{style} architecture",
                f"a building in {style} style",
                f"photo of {style} architecture",
            ]
    return prompts


def load_prompts(json_path):
    with open(json_path) as f:
        return json.load(f)


STYLE_PROMPTS = build_prompts(class_names) if args.prompts == "build" else load_prompts(args.prompts)


# =============================================================================
# TEXT ENCODING
# =============================================================================

def encode_text_prompts(style_prompts):
    all_prompts, class_indices, prompt_counts = [], [], []
    for i, (cls, prompts) in enumerate(style_prompts.items()):
        prompt_counts.append(len(prompts))
        for p in prompts:
            all_prompts.append(p)
            class_indices.append(i)
    tokens = clip.tokenize(all_prompts).to(device)
    with torch.no_grad():
        tf = model.encode_text(tokens)
        tf /= tf.norm(dim=-1, keepdim=True)
    return tf, class_indices, prompt_counts


text_features, class_indices, prompt_counts = encode_text_prompts(STYLE_PROMPTS)


# =============================================================================
# IMAGE ENCODING
# =============================================================================

def load_image(path):
    try:
        return preprocess(Image.open(path).convert("RGB"))
    except Exception:
        return None


def encode_images_batch(image_paths, desc="Encoding"):
    all_features = []
    batch_list = [image_paths[i:i+BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)]
    use_amp = device == "cuda"

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        prefetch = executor.submit(lambda b: list(executor.map(load_image, b)), batch_list[0]) if batch_list else None
        for idx in tqdm(range(len(batch_list)), desc=desc, unit="batch", leave=False):
            images = prefetch.result()
            if idx + 1 < len(batch_list):
                prefetch = executor.submit(lambda b: list(executor.map(load_image, b)), batch_list[idx + 1])
            valid = [img for img in images if img is not None]
            if not valid:
                continue
            batch = torch.stack(valid).pin_memory().to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                feats = model.encode_image(batch)
                feats /= feats.norm(dim=-1, keepdim=True)
            all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0) if all_features else np.zeros((0, 512))


# =============================================================================
# PROTOTYPES
# =============================================================================

def build_prototypes(train_paths, train_labels, num_classes):
    print("Building prototype embeddings...")
    features = encode_images_batch(train_paths, desc="Prototypes")
    prototypes = np.zeros((num_classes, features.shape[1]))
    for c in range(num_classes):
        mask = train_labels[:len(features)] == c
        if mask.sum() > 0:
            proto = features[mask].mean(axis=0)
            norm = np.linalg.norm(proto)
            if norm > 0:
                proto /= norm
            prototypes[c] = proto
    print(f"  Built {num_classes} prototypes from {len(features)} images")
    return prototypes


# =============================================================================
# PROMPT WEIGHTS
# =============================================================================

def learn_prompt_weights(train_paths, train_labels, text_features, class_indices, num_classes):
    print("Learning prompt weights...")
    from sklearn.linear_model import Ridge
    train_features = encode_images_batch(train_paths, desc="Prompt weights")
    tf_gpu = torch.tensor(train_features, dtype=text_features.dtype, device=device)
    sims = (tf_gpu @ text_features.T).cpu().float().numpy()
    weights = np.ones(len(class_indices))
    class_prompt_indices = {}
    for j, c in enumerate(class_indices):
        class_prompt_indices.setdefault(c, []).append(j)
    for c, prompt_idxs in class_prompt_indices.items():
        if len(prompt_idxs) < 2:
            continue
        X = sims[:len(train_features), prompt_idxs]
        y = (train_labels[:len(train_features)] == c).astype(float) * 2 - 1
        reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        reg.fit(X, y)
        w = np.clip(reg.coef_, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(prompt_idxs)) / len(prompt_idxs)
        for i, idx in enumerate(prompt_idxs):
            weights[idx] = w[i]
    print(f"  Fitted weights across {num_classes} classes")
    return weights


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(image_paths, prototypes=None, prompt_weights=None, proto_weight=PROTO_WEIGHT):
    all_feats = []
    batch_list = [image_paths[i:i+BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)]
    use_amp = device == "cuda"

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        prefetch = executor.submit(lambda b: list(executor.map(load_image, b)), batch_list[0]) if batch_list else None
        for idx in tqdm(range(len(batch_list)), desc="Inference", unit="batch", leave=False):
            images = prefetch.result()
            if idx + 1 < len(batch_list):
                prefetch = executor.submit(lambda b: list(executor.map(load_image, b)), batch_list[idx + 1])
            valid = [img for img in images if img is not None]
            if not valid:
                continue
            batch = torch.stack(valid).pin_memory().to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                feats = model.encode_image(batch)
                feats /= feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())

    image_features = torch.cat(all_feats).to(device=device, dtype=text_features.dtype)
    idx_t = torch.tensor(class_indices, device=device)

    if prompt_weights is not None:
        w = torch.tensor(prompt_weights, dtype=image_features.dtype, device=device)
    else:
        w = torch.ones(len(class_indices), dtype=image_features.dtype, device=device)

    raw = image_features @ text_features.T / TEMPERATURE
    sims = torch.sigmoid(raw)
    weighted = sims * w.unsqueeze(0)
    text_scores = torch.zeros(len(image_features), num_classes, dtype=image_features.dtype, device=device)
    text_scores.scatter_add_(1, idx_t.unsqueeze(0).expand(len(image_features), -1), weighted)
    weight_sums = torch.zeros(num_classes, dtype=image_features.dtype, device=device)
    weight_sums.scatter_add_(0, idx_t, w)
    text_scores /= weight_sums.clamp(min=1e-8).unsqueeze(0)

    if prototypes is None:
        return text_scores.cpu().numpy()

    proto_t = torch.tensor(prototypes, dtype=image_features.dtype, device=device)
    proto_scores = torch.sigmoid(image_features @ proto_t.T / TEMPERATURE)
    return (proto_weight * proto_scores + (1 - proto_weight) * text_scores).cpu().numpy()


# =============================================================================
# EVALUATION
# =============================================================================

def top_k_accuracy(probs, labels, k=3):
    topk = np.argsort(-probs, axis=1)[:, :k]
    return np.mean([labels[i] in topk[i] for i in range(len(labels))])


def evaluate(probs, labels, tag=""):
    preds = probs.argmax(axis=1)
    acc  = (preds == labels[:len(preds)]).mean()
    topk = top_k_accuracy(probs, labels[:len(probs)], k=TOP_K)
    prefix = f"[{tag}] " if tag else ""
    print(f"\n{prefix}Top-1 Accuracy: {acc:.4f}   Top-{TOP_K}: {topk:.4f}")
    return acc


def export_results(probs, labels, class_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    preds = probs.argmax(axis=1)
    metrics = {
        "accuracy":    float(accuracy_score(labels, preds)),
        "f1_macro":    float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }
    np.save(f"{save_dir}/probabilities.npy", probs)
    np.save(f"{save_dir}/predictions.npy", preds)
    np.save(f"{save_dir}/labels.npy", labels)
    np.save(f"{save_dir}/confusion_matrix.npy", confusion_matrix(labels, preds))
    with open(f"{save_dir}/class_names.json", "w") as f:
        json.dump(class_names, f)
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved to {save_dir}: {metrics}")
    return metrics


# =============================================================================
# DC LABELED CROPS EVALUATION
# =============================================================================

def _slug(s):
    return s.lower().replace(" ", "_").replace("-", "_").replace("'", "")


def eval_on_dc_labels(labels_path, class_names, prototypes, prompt_weights):
    import pandas as pd
    df = pd.read_csv(labels_path)
    df = df[df["style"] != "unsure"]
    df = df[df["building_type"] != "unknown"]

    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    crop_paths, true_labels = [], []
    skipped = 0
    for _, row in df.iterrows():
        slug = f"{_slug(row['building_type'])}_{_slug(row['style'])}"
        if slug not in cls_to_idx:
            skipped += 1
            continue
        p = Path(row["crop_path"])
        if not p.exists():
            skipped += 1
            continue
        crop_paths.append(p)
        true_labels.append(cls_to_idx[slug])

    print(f"\nDC labeled crops: {len(crop_paths)} valid  |  {skipped} skipped (no class or missing file)")
    if not crop_paths:
        return

    probs = run_inference(crop_paths, prototypes=prototypes, prompt_weights=prompt_weights,
                          proto_weight=PROTO_WEIGHT)
    true_labels = np.array(true_labels)
    evaluate(probs, true_labels, tag="DC crops")
    export_results(probs, true_labels, class_names, save_dir=OUT_DIR + "_dc")


# =============================================================================
# MAIN
# =============================================================================

print(f"\nExperiment: {args.name}")
print(f"  prototypes={USE_PROTOTYPES}  weights={USE_WEIGHTS}  "
      f"tune_alpha={TUNE_ALPHA}  ridge={RIDGE_ALPHA}  temp={TEMPERATURE}")

prototypes     = build_prototypes(train_paths, train_labels, num_classes) if USE_PROTOTYPES else None
prompt_weights = learn_prompt_weights(train_paths, train_labels,
                                      text_features, class_indices, num_classes) if USE_WEIGHTS else None

if TUNE_ALPHA and prototypes is not None:
    print("\nTuning prototype blend alpha on train set...")
    train_probs_text = run_inference(train_paths, prototypes=None, prompt_weights=prompt_weights)
    train_feats = torch.tensor(
        encode_images_batch(train_paths, desc="Alpha tune"), dtype=torch.float32
    ).to(device)
    proto_t = torch.tensor(prototypes, dtype=train_feats.dtype, device=device)
    train_proto_scores = torch.sigmoid(train_feats @ proto_t.T / TEMPERATURE)
    train_text_gpu = torch.tensor(train_probs_text, dtype=train_feats.dtype, device=device)
    train_labels_gpu = torch.tensor(train_labels[:len(train_feats)], device=device)

    best_alpha, best_acc = PROTO_WEIGHT, 0.0
    for alpha in tqdm(np.arange(0.0, 1.05, 0.05), desc="Tuning alpha", unit="a"):
        blended = alpha * train_proto_scores + (1 - alpha) * train_text_gpu
        acc = (blended.argmax(dim=1) == train_labels_gpu).float().mean().item()
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha
    PROTO_WEIGHT = best_alpha
    print(f"  Best alpha: {best_alpha:.2f}  (train acc: {best_acc:.4f})")

print("\nBaseline (no weights, no prototypes):")
baseline_probs = run_inference(test_paths, prototypes=None, prompt_weights=None)
evaluate(baseline_probs, test_labels, tag="Baseline")

print("\nFull (weighted prompts + prototypes):")
test_probs = run_inference(test_paths, prototypes=prototypes,
                           prompt_weights=prompt_weights, proto_weight=PROTO_WEIGHT)
evaluate(test_probs, test_labels, tag="Full")

# Save artifacts
os.makedirs(OUT_DIR, exist_ok=True)
if prototypes is not None:
    np.save(f"{OUT_DIR}/prototypes.npy", prototypes)
if prompt_weights is not None:
    np.save(f"{OUT_DIR}/prompt_weights.npy", prompt_weights)
with open(f"{OUT_DIR}/prompts.json", "w") as f:
    json.dump(STYLE_PROMPTS, f, indent=2)

export_results(test_probs, test_labels, class_names, save_dir=OUT_DIR)

# Real-world evaluation on labeled DC crops
if args.dc_labels:
    eval_on_dc_labels(args.dc_labels, class_names, prototypes, prompt_weights)
