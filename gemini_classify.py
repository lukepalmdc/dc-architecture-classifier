"""
gemini_classify.py

Uses Gemini vision + structured output to classify building architectural style.

Evaluates on:
  - DC labeled crops  (from dc_labels.csv)
  - Pexels holdout    (sampled from data/styles/ test split)

Output: outputs/gemini_<name>/

Usage:
    export GEMINI_API_KEY=your_key
    python gemini_classify.py --name gemini_condensed --dc-labels dc_labels.csv
    python gemini_classify.py --name gemini_condensed --dc-labels dc_labels.csv --pexels-sample 200
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

from google import genai

# =============================================================================
# STYLES
# =============================================================================

CONDENSED_STYLES = [
    "Rowhouse Vernacular", "Colonial Revival", "Developer Modern",
    "Italianate", "Victorian", "Contemporary Glass", "Garden Style",
    "Art Deco", "Developer Traditional", "Neoclassical", "Brutalist",
    "Beaux-Arts", "Modernist", "Other",
]

VALID_PREFIXES = (
    "rowhouse_", "single_family_house_", "small_multifamily_",
    "large_multifamily_", "office_", "institutional_",
)

TYPE_KEYS = sorted([
    "single_family_house", "small_multifamily", "large_multifamily",
    "rowhouse", "office", "institutional",
], key=len, reverse=True)

KEEP_STYLES = {s.lower().replace(" ", "_").replace("-", "_").replace("'", "")
               for s in CONDENSED_STYLES if s != "Other"}


# =============================================================================
# PYDANTIC MODEL
# =============================================================================

StyleLiteral = Literal[
    "Rowhouse Vernacular", "Colonial Revival", "Developer Modern",
    "Italianate", "Victorian", "Contemporary Glass", "Garden Style",
    "Art Deco", "Developer Traditional", "Neoclassical", "Brutalist",
    "Beaux-Arts", "Modernist", "Other",
]


class BuildingClassification(BaseModel):
    style: StyleLiteral = Field(
        description="Architectural style of the building from the allowed list."
    )
    confidence: Literal["High", "Medium", "Low"] = Field(
        description="Confidence in the classification."
    )
    reasoning: str = Field(
        description="One sentence describing the key visual features that determined the style."
    )


PROMPT = f"""Classify the architectural style of the building shown in this image.

Choose exactly one style from this list:
{chr(10).join(f'- {s}' for s in CONDENSED_STYLES)}

Use "Other" if the building does not clearly match any listed style.
Focus on visible architectural features: facade material, window shape, roofline, ornamental details, era of construction.
"""


# =============================================================================
# ARGS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name",           type=str, default="gemini")
    p.add_argument("--out-dir",        type=str, default="outputs")
    p.add_argument("--data-dir",       type=str, default="data/styles")
    p.add_argument("--dc-labels",      type=str, default=None)
    p.add_argument("--pexels-sample",  type=int, default=None,
                   help="Number of Pexels holdout images to evaluate (default: skip)")
    p.add_argument("--model",          type=str, default="gemini-2.5-flash-preview-04-17")
    p.add_argument("--workers",        type=int, default=4,
                   help="Parallel Gemini requests (default 4, respect rate limits)")
    p.add_argument("--delay",          type=float, default=0.2,
                   help="Seconds between requests per worker (default 0.2)")
    p.add_argument("--max-retries",    type=int, default=3)
    p.add_argument("--api-key",        type=str, default=None,
                   help="Gemini API key (falls back to GEMINI_API_KEY env var)")
    return p.parse_args()


args    = parse_args()
OUT_DIR = f"{args.out_dir}/{args.name}"
os.makedirs(OUT_DIR, exist_ok=True)

_api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=_api_key)


# =============================================================================
# HELPERS
# =============================================================================

def _slug(s):
    return s.lower().replace(" ", "_").replace("-", "_").replace("'", "")


def _style_slug(folder_name):
    for prefix in TYPE_KEYS:
        if folder_name.startswith(prefix + "_"):
            return folder_name[len(prefix) + 1:]
    return folder_name


def condense_slug(slug):
    return slug if slug in KEEP_STYLES else "other"


def slug_to_display(slug):
    mapping = {_slug(s): s for s in CONDENSED_STYLES}
    return mapping.get(slug, "Other")


# =============================================================================
# GEMINI CALL
# =============================================================================

def classify_image(image_path, retries=None):
    if retries is None:
        retries = args.max_retries
    img = Image.open(image_path).convert("RGB")
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=args.model,
                contents=[img, PROMPT],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": BuildingClassification.model_json_schema(),
                },
            )
            result = BuildingClassification.model_validate_json(response.text)
            if args.delay > 0:
                time.sleep(args.delay)
            return result
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None


# =============================================================================
# BATCH CLASSIFY
# =============================================================================

def classify_batch(paths, desc="Classifying"):
    results = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(classify_image, p): i for i, p in enumerate(paths)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            idx = futures[fut]
            results[idx] = fut.result()
    return results


# =============================================================================
# EVALUATION
# =============================================================================

def top_k_accuracy(probs, labels, k=3):
    topk = np.argsort(-probs, axis=1)[:, :k]
    return np.mean([labels[i] in topk[i] for i in range(len(labels))])


def results_to_probs(classifications, class_names):
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    n = len(classifications)
    probs = np.zeros((n, len(class_names)))
    for i, clf in enumerate(classifications):
        if clf is None:
            continue
        style = clf.style
        idx = cls_to_idx.get(style)
        if idx is not None:
            conf_map = {"High": 0.9, "Medium": 0.7, "Low": 0.5}
            conf = conf_map.get(clf.confidence, 0.7)
            probs[i, idx] = conf
            # distribute remaining probability uniformly
            remaining = (1 - conf) / max(1, len(class_names) - 1)
            for j in range(len(class_names)):
                if j != idx:
                    probs[i, j] = remaining
        else:
            probs[i] = 1.0 / len(class_names)
    return probs


def evaluate_and_export(classifications, true_labels, class_names, save_dir, tag):
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    preds = []
    valid_true, valid_pred = [], []
    rows = []

    for i, (clf, true_idx) in enumerate(zip(classifications, true_labels)):
        if clf is None:
            pred_style = "API error"
            pred_idx   = -1
        else:
            pred_style = clf.style
            pred_idx   = cls_to_idx.get(pred_style, -1)

        true_style = class_names[true_idx] if 0 <= true_idx < len(class_names) else "?"
        preds.append(pred_idx)
        rows.append({
            "true_style":  true_style,
            "pred_style":  pred_style,
            "confidence":  clf.confidence if clf else None,
            "reasoning":   clf.reasoning  if clf else None,
            "correct":     pred_idx == true_idx,
        })
        if pred_idx >= 0:
            valid_true.append(true_idx)
            valid_pred.append(pred_idx)

    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/predictions.csv", index=False)

    if not valid_pred:
        print(f"[{tag}] No valid predictions")
        return

    acc = accuracy_score(valid_true, valid_pred)
    f1m = f1_score(valid_true, valid_pred, average="macro",    zero_division=0)
    f1w = f1_score(valid_true, valid_pred, average="weighted", zero_division=0)
    top1 = sum(t == p for t, p in zip(valid_true, valid_pred)) / len(valid_pred)

    metrics = {
        "accuracy":    float(acc),
        "f1_macro":    float(f1m),
        "f1_weighted": float(f1w),
        "n_valid":     len(valid_pred),
        "n_total":     len(classifications),
        "api_errors":  sum(1 for c in classifications if c is None),
    }
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{save_dir}/class_names.json", "w") as f:
        json.dump(class_names, f)

    cm = confusion_matrix(valid_true, valid_pred, labels=list(range(len(class_names))))
    np.save(f"{save_dir}/confusion_matrix.npy", cm)

    print(f"\n[{tag}]  acc={acc:.4f}  f1_macro={f1m:.4f}  f1_weighted={f1w:.4f}"
          f"  ({len(valid_pred)}/{len(classifications)} valid)")
    print(f"  -> {save_dir}/")
    return metrics


# =============================================================================
# DC EVAL
# =============================================================================

def eval_dc(labels_path):
    df = pd.read_csv(labels_path)
    df = df[~df["style"].isin(["unsure", "Other", "other"])]
    has_type = "building_type" in df.columns

    cls_to_idx = {s: i for i, s in enumerate(CONDENSED_STYLES)}
    paths, true_labels = [], []

    for _, row in df.iterrows():
        style_slug = _slug(row["style"])
        condensed  = condense_slug(style_slug)
        display    = slug_to_display(condensed)
        if display not in cls_to_idx:
            continue
        p = Path(row["crop_path"])
        if not p.exists():
            continue
        paths.append(p)
        true_labels.append(cls_to_idx[display])

    print(f"\nDC crops: {len(paths)} valid images")
    if not paths:
        return

    classifications = classify_batch(paths, desc="DC crops")
    evaluate_and_export(classifications, true_labels, CONDENSED_STYLES,
                        save_dir=f"{OUT_DIR}_dc", tag="DC crops")


# =============================================================================
# PEXELS EVAL
# =============================================================================

def eval_pexels(n_sample):
    root = Path(args.data_dir)
    cls_to_idx = {s: i for i, s in enumerate(CONDENSED_STYLES)}

    all_paths, all_labels = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if not any(folder.name.startswith(p) for p in VALID_PREFIXES):
            continue
        style_slug = _style_slug(folder.name)
        condensed  = condense_slug(style_slug)
        display    = slug_to_display(condensed)
        if display not in cls_to_idx:
            continue
        imgs = [p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        for img in imgs:
            all_paths.append(img)
            all_labels.append(cls_to_idx[display])

    # Reproduce same test split as train_architecture.py (seed=42, 20%)
    np.random.seed(42)
    idx = np.random.permutation(len(all_paths))
    split = int(len(idx) * 0.8)
    test_idx = idx[split:]

    test_paths  = [all_paths[i]  for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]

    if n_sample and n_sample < len(test_paths):
        chosen = np.random.choice(len(test_paths), n_sample, replace=False)
        test_paths  = [test_paths[i]  for i in chosen]
        test_labels = [test_labels[i] for i in chosen]

    print(f"\nPexels holdout: {len(test_paths)} images")
    classifications = classify_batch(test_paths, desc="Pexels holdout")
    evaluate_and_export(classifications, test_labels, CONDENSED_STYLES,
                        save_dir=f"{OUT_DIR}_pexels", tag="Pexels holdout")


# =============================================================================
# MAIN
# =============================================================================

print(f"Model:   {args.model}")
print(f"Workers: {args.workers}  delay: {args.delay}s")
print(f"Output:  {OUT_DIR}")

if not _api_key:
    raise SystemExit("ERROR: pass --api-key or set GEMINI_API_KEY env var")

if args.dc_labels:
    eval_dc(args.dc_labels)

if args.pexels_sample:
    eval_pexels(args.pexels_sample)

if not args.dc_labels and not args.pexels_sample:
    print("Nothing to do — pass --dc-labels and/or --pexels-sample")
