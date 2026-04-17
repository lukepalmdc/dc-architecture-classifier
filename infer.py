"""
Standalone inference script for architecture style classification.

Requires artifacts from train_architecture.py:
    prototypes.npy, prompt_weights.npy, outputs/class_names.json, prompts.json

Usage:
    python infer.py image.jpg
    python infer.py data/images/1.jpg --top 3
    python infer.py data/images/ --top 1 --out results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARTIFACTS = {
    "prototypes":     "prototypes.npy",
    "prompt_weights": "prompt_weights.npy",
    "class_names":    "outputs/class_names.json",
    "prompts":        "prompts.json",
}
MODEL_NAME   = "ViT-B/32"
TEMPERATURE  = 0.07
PROTO_WEIGHT = 0.5   # same blend used at training time

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

# ---------------------------------------------------------------------------
# Load artifacts (once at import time)
# ---------------------------------------------------------------------------

device = "cpu"
model, preprocess = clip.load(MODEL_NAME, device=device)
model.eval()


def _load_artifacts():
    for name, path in ARTIFACTS.items():
        if not Path(path).exists():
            print(f"ERROR: missing artifact '{path}'. Run train_architecture.py first.")
            sys.exit(1)

    prototypes     = np.load(ARTIFACTS["prototypes"])
    prompt_weights = np.load(ARTIFACTS["prompt_weights"])

    with open(ARTIFACTS["class_names"]) as f:
        class_names = json.load(f)

    with open(ARTIFACTS["prompts"]) as f:
        style_prompts = json.load(f)

    # Encode text prompts
    all_prompts, class_indices = [], []
    for i, cls in enumerate(class_names):
        for p in style_prompts.get(cls, []):
            all_prompts.append(p)
            class_indices.append(i)

    tokens = clip.tokenize(all_prompts)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return prototypes, prompt_weights, class_names, text_features.numpy(), class_indices


PROTOTYPES, PROMPT_WEIGHTS, CLASS_NAMES, TEXT_FEATURES, CLASS_INDICES = _load_artifacts()
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(image_path, top_k=3):
    """
    Predict architecture style for a single image.

    Returns list of (label, display_name, confidence) sorted by confidence desc.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
    except Exception as e:
        return None, str(e)

    with torch.no_grad():
        img_feat = model.encode_image(tensor)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)

    img_np = img_feat.numpy()   # [1 x D]

    # Weighted text scores
    raw_sims    = img_np @ TEXT_FEATURES.T / TEMPERATURE  # [1 x num_prompts]
    prompt_sims = 1 / (1 + np.exp(-raw_sims))             # sigmoid, [1 x num_prompts]

    text_scores  = np.zeros(NUM_CLASSES)
    weight_sums  = np.zeros(NUM_CLASSES)
    for j, c in enumerate(CLASS_INDICES):
        text_scores[c]  += PROMPT_WEIGHTS[j] * prompt_sims[0, j]
        weight_sums[c]  += PROMPT_WEIGHTS[j]
    for c in range(NUM_CLASSES):
        if weight_sums[c] > 0:
            text_scores[c] /= weight_sums[c]

    # Prototype scores
    proto_raw    = img_np @ PROTOTYPES.T / TEMPERATURE    # [1 x C]
    proto_scores = (1 / (1 + np.exp(-proto_raw)))[0]     # sigmoid, [C]

    # Blend
    final_scores = PROTO_WEIGHT * proto_scores + (1 - PROTO_WEIGHT) * text_scores

    top_indices = np.argsort(-final_scores)[:top_k]
    results = []
    for idx in top_indices:
        label   = CLASS_NAMES[idx]
        display = LABEL_DISPLAY.get(label, label.replace("_", " ").title())
        results.append({
            "label":      label,
            "name":       display,
            "confidence": float(final_scores[idx]),
        })

    return results, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Architecture style classifier")
    parser.add_argument("input",  help="Image file or directory")
    parser.add_argument("--top",  type=int, default=3, help="Top-K predictions")
    parser.add_argument("--out",  type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        paths = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
    elif input_path.is_file():
        paths = [input_path]
    else:
        print(f"ERROR: '{args.input}' not found")
        sys.exit(1)

    all_results = []
    for path in paths:
        preds, err = predict(path, top_k=args.top)
        if err:
            print(f"{path.name}: ERROR — {err}")
            continue

        top1 = preds[0]
        print(f"{path.name:<40} {top1['name']:<22} ({top1['confidence']:.3f})", end="")
        if args.top > 1:
            others = ", ".join(f"{p['name']} ({p['confidence']:.3f})" for p in preds[1:])
            print(f"  |  {others}", end="")
        print()

        all_results.append({"file": str(path), "predictions": preds})

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {len(all_results)} results to {args.out}")


if __name__ == "__main__":
    main()
