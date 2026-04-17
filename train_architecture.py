
"""
CLIP Architecture Classification (CPU Only)

* Loads images from: data/styles/[class folders]
* Builds prompt-based classifier
* Splits into train/test
* Evaluates performance

Author: you
"""

# =========================

# IMPORTS

# =========================

import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import torch
import clip
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# =========================

# CONFIG

# =========================

import argparse

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
                   help="Path to prompts JSON, or 'build' to use build_prompts()")
    p.add_argument("--temperature",    type=float, default=0.07)
    p.add_argument("--out-dir",        type=str, default="outputs")
    p.add_argument("--name",           type=str, default="default",
                   help="Experiment name (used as output subdirectory)")
    p.add_argument("--batch-size",     type=int, default=512,
                   help="Images per GPU batch (default 512)")
    p.add_argument("--num-threads",    type=int, default=8,
                   help="CPU threads for image loading (default 8)")
    return p.parse_args()

args = parse_args()

DATA_DIR    = "data/styles"
BATCH_SIZE  = args.batch_size
NUM_THREADS = args.num_threads
TEST_SPLIT  = 0.2
TOP_K       = 3
SEED        = 42

PROTO_WEIGHT    = args.proto_weight
TUNE_ALPHA      = args.tune_alpha
RIDGE_ALPHA     = args.ridge_alpha
USE_PROTOTYPES  = args.use_prototypes
USE_WEIGHTS     = args.use_weights
TEMPERATURE     = args.temperature
OUT_DIR         = f"{args.out_dir}/{args.name}"

torch.set_num_threads(NUM_THREADS)

# =========================

# LOAD CLIP MODEL (CPU)

# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# =========================

# LOAD DATA FROM FOLDERS

# =========================

def load_dataset(root_dir):
    """
    Reads directory structure:
    data/styles/class_name/*.jpg
    Returns:
        image_paths: list of paths
        labels: np array of class indices
        class_names: list of class names
    """

    root = Path(root_dir)

    image_paths = []
    labels = []
    class_names = []

    for i, folder in enumerate(sorted(root.iterdir())):
        if folder.is_dir():
            class_names.append(folder.name)

            for img_path in folder.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    image_paths.append(img_path)
                    labels.append(i)

    return image_paths, np.array(labels), class_names


image_paths, labels, class_names = load_dataset(DATA_DIR)
num_classes = len(class_names)

print(f"Loaded {len(image_paths)} images across {num_classes} classes")

# =========================

# TRAIN / TEST SPLIT

# =========================

def train_test_split(paths, labels, test_ratio=0.2, seed=42):
    """
    Randomly splits dataset into train/test
    """
    np.random.seed(seed)


    indices = np.arange(len(paths))
    np.random.shuffle(indices)

    split = int(len(indices) * (1 - test_ratio))

    train_idx = indices[:split]
    test_idx = indices[split:]

    return (
        [paths[i] for i in train_idx],
        labels[train_idx],
        [paths[i] for i in test_idx],
        labels[test_idx],
    )


train_paths, train_labels, test_paths, test_labels = train_test_split(
image_paths, labels, TEST_SPLIT, SEED
)

print(f"Train: {len(train_paths)} | Test: {len(test_paths)}")

# =========================

# BUILD PROMPTS

# =========================

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


def build_prompts(class_names):
    """
        Create multiple prompt variations per class
    """

    STYLE_PROMPTS = {}

    for cls in class_names:
        label = LABEL_DISPLAY.get(cls, cls.replace("_", " ").title())
        STYLE_PROMPTS[cls] = [
            f"a {label} building",
            f"{label} architecture",
            f"a building in {label} style",
            f"photo of {label} architecture",
        ]

    return STYLE_PROMPTS





def load_prompts(json_path="prompts.json"):
    """
    Loads architecture prompts from external JSON file
    """
    with open(json_path, "r") as f:
        return json.load(f)
    

if args.prompts == "build":
    STYLE_PROMPTS = build_prompts(class_names)
else:
    STYLE_PROMPTS = load_prompts(args.prompts)

# =========================

# ENCODE TEXT (ONCE)

# =========================

def encode_text_prompts(style_prompts):
    """
    Converts prompts into normalized CLIP embeddings.
    Returns:
        text_features:  [num_prompts x D] normalized
        class_indices:  list mapping prompt index -> class index
        prompt_counts:  list of how many prompts each class has
    """
    all_prompts   = []
    class_indices = []
    prompt_counts = []

    for i, (cls, prompts) in enumerate(style_prompts.items()):
        prompt_counts.append(len(prompts))
        for p in prompts:
            all_prompts.append(p)
            class_indices.append(i)

    text_tokens = clip.tokenize(all_prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features, class_indices, prompt_counts


text_features, class_indices, prompt_counts = encode_text_prompts(STYLE_PROMPTS)


# =========================

# IMAGE ENCODER (shared by prototypes + inference)

# =========================

def encode_images_batch(image_paths, desc="Encoding"):
    """
    Encode a list of image paths through CLIP vision encoder.
    Returns normalized features [N x D] as numpy array.
    Prefetches next batch on CPU while GPU processes current batch.
    """
    all_features = []
    batch_list = [image_paths[i:i+BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)]
    use_amp = device == "cuda"

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Pre-load first batch
        prefetch = executor.submit(lambda b: list(executor.map(load_image, b)), batch_list[0]) if batch_list else None

        for idx in tqdm(range(len(batch_list)), desc=desc, unit="batch", leave=False):
            images = prefetch.result()

            # Kick off next batch load immediately while GPU works
            if idx + 1 < len(batch_list):
                next_paths = batch_list[idx + 1]
                prefetch = executor.submit(lambda b: list(executor.map(load_image, b)), next_paths)

            valid_images = [img for img in images if img is not None]
            if not valid_images:
                continue

            batch = torch.stack(valid_images).pin_memory().to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                feats = model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0) if all_features else np.zeros((0, 512))


# =========================

# PROTOTYPE EMBEDDINGS

# =========================

def build_prototypes(train_paths, train_labels, num_classes):
    """
    Compute per-class prototype: mean of normalized training image embeddings,
    then re-normalize. This gives a class centroid in CLIP visual space.
    """
    print("Building prototype embeddings from training images...")
    features = encode_images_batch(train_paths, desc="Prototypes")  # [N x D]

    prototypes = np.zeros((num_classes, features.shape[1]))
    for c in range(num_classes):
        mask = train_labels[:len(features)] == c
        if mask.sum() > 0:
            proto = features[mask].mean(axis=0)
            norm  = np.linalg.norm(proto)
            if norm > 0:
                proto /= norm
        prototypes[c] = proto

    print(f"  Built {num_classes} prototypes from {len(features)} training images")
    return prototypes  # [num_classes x D]


# =========================

# LEARNED PROMPT WEIGHTS

# =========================

def learn_prompt_weights(train_paths, train_labels, text_features, class_indices, num_classes):
    """
    Fit per-class prompt weights using training image similarities.

    For each class c with K_c prompts, solve a ridge regression:
        argmin_w ||X_c w - y_c||^2 + alpha*||w||^2
    where X_c[i, k] = similarity of image i to prompt k of class c,
    and y_c[i] = 1 if image i belongs to class c else -1.

    Returns: weights array [num_prompts] (same indexing as class_indices)
    """
    print("Learning prompt weights from training images...")
    from sklearn.linear_model import Ridge

    train_features = encode_images_batch(train_paths, desc="Prompt weights")  # [N x D] numpy
    tf_gpu = torch.tensor(train_features, dtype=text_features.dtype, device=device)
    sims = (tf_gpu @ text_features.T).cpu().float().numpy()        # [N x num_prompts]

    weights = np.ones(len(class_indices))                   # fallback: uniform

    # Group prompt indices by class
    class_prompt_indices = {}
    for j, c in enumerate(class_indices):
        class_prompt_indices.setdefault(c, []).append(j)

    for c, prompt_idxs in class_prompt_indices.items():
        if len(prompt_idxs) < 2:
            continue  # nothing to weight if only one prompt

        X = sims[:len(train_features), prompt_idxs]        # [N x K_c]
        y = (train_labels[:len(train_features)] == c).astype(float) * 2 - 1  # ±1

        reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        reg.fit(X, y)

        w = reg.coef_
        # Keep weights positive and normalized so they sum to 1
        w = np.clip(w, 0, None)
        if w.sum() > 0:
            w /= w.sum()
        else:
            w = np.ones(len(prompt_idxs)) / len(prompt_idxs)

        for i, idx in enumerate(prompt_idxs):
            weights[idx] = w[i]

    print(f"  Prompt weights fitted across {num_classes} classes")
    return weights  # [num_prompts]

# =========================

# IMAGE LOADER

# =========================

def load_image(path):
    """
    Loads and preprocesses a single image
    """
    try:
        img = Image.open(path).convert("RGB")
        return preprocess(img)
    except Exception:
        return None

# =========================

# BATCH INFERENCE FUNCTION

# =========================

def run_inference(image_paths, prototypes=None, prompt_weights=None, proto_weight=PROTO_WEIGHT):
    """
    Runs CLIP on a list of image paths.

    Args:
        prototypes:     [num_classes x D] prototype embeddings (optional)
        prompt_weights: [num_prompts] learned per-prompt weights (optional)
        proto_weight:   blend factor — final = proto_weight*proto + (1-proto_weight)*text

    Returns:
        class_probs: [N x num_classes]
    """
    all_image_features = []
    batch_list = [image_paths[i:i+BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)]
    use_amp = device == "cuda"

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        prefetch = executor.submit(lambda b: list(executor.map(load_image, b)), batch_list[0]) if batch_list else None

        for idx in tqdm(range(len(batch_list)), desc="Inference", unit="batch", leave=False):
            images = prefetch.result()

            if idx + 1 < len(batch_list):
                next_paths = batch_list[idx + 1]
                prefetch = executor.submit(lambda b: list(executor.map(load_image, b)), next_paths)

            valid_images = [img for img in images if img is not None]
            if not valid_images:
                continue

            batch = torch.stack(valid_images).pin_memory().to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                feats = model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_image_features.append(feats.cpu())

    image_features = torch.cat(all_image_features).to(device)  # [N x D] on GPU

    # --- Weighted text prompt scores (all on GPU) ---
    raw_sims    = image_features @ text_features.T / TEMPERATURE   # [N x P]
    prompt_sims = torch.sigmoid(raw_sims)                          # keep sigmoid per user

    # Build weight and class-index tensors on GPU
    if prompt_weights is not None:
        w = torch.tensor(prompt_weights, dtype=image_features.dtype, device=device)  # [P]
    else:
        w = torch.ones(len(class_indices), dtype=image_features.dtype, device=device) / max(prompt_counts)

    idx = torch.tensor(class_indices, device=device)               # [P]

    # Weighted scatter-add: text_class_scores[n, c] = sum of w[j]*prompt_sims[n,j] for j->c
    weighted = prompt_sims * w.unsqueeze(0)                        # [N x P]
    text_class_scores = torch.zeros(len(image_features), num_classes, dtype=image_features.dtype, device=device)
    text_class_scores.scatter_add_(1, idx.unsqueeze(0).expand(len(image_features), -1), weighted)

    # Normalize by total weight per class
    weight_sums = torch.zeros(num_classes, dtype=image_features.dtype, device=device)
    weight_sums.scatter_add_(0, idx, w)
    weight_sums = weight_sums.clamp(min=1e-8)
    text_class_scores = text_class_scores / weight_sums.unsqueeze(0)

    if prototypes is None:
        return text_class_scores.cpu().numpy()

    # --- Prototype scores ---
    proto_tensor = torch.tensor(prototypes, dtype=torch.float32).to(device=device, dtype=image_features.dtype)
    proto_raw    = image_features @ proto_tensor.T / TEMPERATURE
    proto_scores = torch.sigmoid(proto_raw)                        # [N x C]

    # --- Blend ---
    blended = proto_weight * proto_scores + (1 - proto_weight) * text_class_scores
    return blended.cpu().numpy()


# =========================

# EVALUATION METRICS

# =========================

def top_k_accuracy(probs, labels, k=3):
    """
    Computes top-k accuracy
    """
    topk = np.argsort(-probs, axis=1)[:, :k]
    return np.mean([labels[i] in topk[i] for i in range(len(labels))])

def evaluate(probs, labels):

    """
    Prints evaluation metrics
    """
    preds = probs.argmax(axis=1)

    
    acc = (preds == labels[:len(preds)]).mean()
    topk = top_k_accuracy(probs, labels[:len(probs)], k=TOP_K)

    print("\nEvaluation Results")
    print("------------------")
    print(f"Top-1 Accuracy: {acc:.4f}")
    print(f"Top-{TOP_K} Accuracy: {topk:.4f}")


# =========================

# RUN PIPELINE

# =========================

print(f"\nExperiment: {args.name}")
print(f"  prototypes={USE_PROTOTYPES}  weights={USE_WEIGHTS}  "
      f"tune_alpha={TUNE_ALPHA}  ridge={RIDGE_ALPHA}  temp={TEMPERATURE}  "
      f"prompts={args.prompts}")

prototypes     = build_prototypes(train_paths, train_labels, num_classes) if USE_PROTOTYPES else None
prompt_weights = learn_prompt_weights(train_paths, train_labels,
                                      text_features, class_indices, num_classes) if USE_WEIGHTS else None

if TUNE_ALPHA:
    print("\nTuning prototype blend alpha on train set...")
    train_probs_text  = run_inference(train_paths, prototypes=None,
                                      prompt_weights=prompt_weights)
    train_feats        = torch.tensor(
        encode_images_batch(train_paths, desc="Alpha tune encode"), dtype=torch.float32
    ).to(device)
    train_proto_tensor = torch.tensor(prototypes, dtype=torch.float32).to(device=device, dtype=train_feats.dtype)
    train_proto_scores = torch.sigmoid(train_feats @ train_proto_tensor.T / TEMPERATURE)  # [N x C] on GPU

    # Move text scores to GPU for blending
    train_probs_text_gpu = torch.tensor(train_probs_text, dtype=train_feats.dtype, device=device)
    train_labels_gpu     = torch.tensor(train_labels[:len(train_feats)], device=device)

    best_alpha, best_acc = PROTO_WEIGHT, 0.0
    for alpha in tqdm(np.arange(0.0, 1.05, 0.05), desc="Tuning alpha", unit="α"):
        blended = alpha * train_proto_scores + (1 - alpha) * train_probs_text_gpu
        acc = (blended.argmax(dim=1) == train_labels_gpu).float().mean().item()
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha

    PROTO_WEIGHT = best_alpha
    print(f"  Best alpha: {best_alpha:.2f}  (train acc: {best_acc:.4f})")

print("\nRunning inference on test set...")

# Baseline: generic prompts, no weights, no prototypes
print("\n[Baseline] Generic prompts, max aggregation:")
baseline_probs = run_inference(test_paths, prototypes=None, prompt_weights=None)
evaluate(baseline_probs, test_labels)

# Full: weighted prompts + prototype blend
print("\n[Full] Weighted prompts + prototype embeddings:")
test_probs = run_inference(test_paths, prototypes=prototypes,
                           prompt_weights=prompt_weights, proto_weight=PROTO_WEIGHT)
evaluate(test_probs, test_labels)

# =========================

# SAVE RESULTS

# =========================

np.save("test_probs.npy", test_probs)
np.save("test_labels.npy", test_labels)
np.save("prototypes.npy", prototypes)
np.save("prompt_weights.npy", prompt_weights)

print("\nSaved: test_probs.npy, test_labels.npy, prototypes.npy, prompt_weights.npy")


# =========================

# DASHBOARD

# =========================


def export_results(probs, labels, class_names, save_dir="outputs"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    preds = probs.argmax(axis=1)

    # metrics
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted"))
    }

    # save arrays
    np.save(f"{save_dir}/probabilities.npy", probs)
    np.save(f"{save_dir}/predictions.npy", preds)
    np.save(f"{save_dir}/labels.npy", labels)

    # confusion matrix
    cm = confusion_matrix(labels, preds)
    np.save(f"{save_dir}/confusion_matrix.npy", cm)

    # class names
    with open(f"{save_dir}/class_names.json", "w") as f:
        json.dump(class_names, f)

    # metrics
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Export complete:", metrics)



export_results(test_probs, test_labels, class_names, save_dir=OUT_DIR)
 


