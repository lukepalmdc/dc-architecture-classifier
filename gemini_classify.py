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
import asyncio
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

from google import genai
from google.genai import types

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
    p.add_argument("--model",          type=str, default="gemini-3.1-flash-lite-preview")
    p.add_argument("--concurrency",    type=int, default=10,
                   help="Max parallel async requests (default 10)")
    p.add_argument("--delay",          type=float, default=0.0,
                   help="Seconds to sleep after each request (default 0)")
    p.add_argument("--max-retries",    type=int, default=3)
    p.add_argument("--api-key",        type=str, default=None,
                   help="Gemini API key (falls back to GEMINI_API_KEY env var)")
    p.add_argument("--test",           action="store_true",
                   help="Run a single test call and exit")
    return p.parse_args()


args    = parse_args()
OUT_DIR = f"{args.out_dir}/{args.name}"
os.makedirs(OUT_DIR, exist_ok=True)

_api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=_api_key)


# =============================================================================
# TOKEN USAGE TRACKING
# =============================================================================

_token_lock = asyncio.Lock()  # initialised before any tasks run
_usage = {"prompt": 0, "output": 0, "total": 0}


async def _add_usage(meta):
    if meta is None:
        return
    async with _token_lock:
        _usage["prompt"] += getattr(meta, "prompt_token_count",    0) or 0
        _usage["output"] += getattr(meta, "candidates_token_count", 0) or 0
        _usage["total"]  += getattr(meta, "total_token_count",      0) or 0


def print_usage():
    print(f"\nToken usage:  prompt={_usage['prompt']:,}  "
          f"output={_usage['output']:,}  total={_usage['total']:,}")


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


def _load_image_bytes(image_path):
    """Read image bytes + detect mime type (JPEG or PNG)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        return f.read(), mime


# =============================================================================
# ASYNC GEMINI CALL
# =============================================================================

async def classify_image_async(image_path, sem):
    loop = asyncio.get_running_loop()
    # Load image bytes off the event loop thread
    img_bytes, mime = await loop.run_in_executor(None, _load_image_bytes, image_path)
    image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=BuildingClassification.model_json_schema(),
    )

    async with sem:
        for attempt in range(args.max_retries):
            try:
                response = await client.aio.models.generate_content(
                    model=args.model,
                    contents=[image_part, PROMPT],
                    config=config,
                )
                await _add_usage(getattr(response, "usage_metadata", None))
                result = BuildingClassification.model_validate_json(response.text)
                if args.delay > 0:
                    await asyncio.sleep(args.delay)
                return result
            except Exception as exc:
                wait = 2 ** attempt
                if attempt < args.max_retries - 1:
                    await asyncio.sleep(wait)
                else:
                    tqdm.write(f"  [error] {Path(image_path).name}: {exc}")
                    return None


# =============================================================================
# BATCH CLASSIFY
# =============================================================================

async def classify_batch(paths, desc="Classifying"):
    sem = asyncio.Semaphore(args.concurrency)

    async def _run(i, p):
        result = await classify_image_async(p, sem)
        return i, result

    tasks = [asyncio.create_task(_run(i, p)) for i, p in enumerate(paths)]
    results = [None] * len(paths)
    pbar = tqdm(total=len(tasks), desc=desc)
    for coro in asyncio.as_completed(tasks):
        i, result = await coro
        results[i] = result
        pbar.update(1)
    pbar.close()
    return results


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_and_export(classifications, true_labels, class_names, save_dir, tag):
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    valid_true, valid_pred = [], []
    rows = []

    for clf, true_idx in zip(classifications, true_labels):
        if clf is None:
            pred_style = "API error"
            pred_idx   = -1
        else:
            pred_style = clf.style
            pred_idx   = cls_to_idx.get(pred_style, -1)

        true_style = class_names[true_idx] if 0 <= true_idx < len(class_names) else "?"
        rows.append({
            "true_style": true_style,
            "pred_style": pred_style,
            "confidence": clf.confidence if clf else None,
            "reasoning":  clf.reasoning  if clf else None,
            "correct":    pred_idx == true_idx,
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

    metrics = {
        "accuracy":    float(acc),
        "f1_macro":    float(f1m),
        "f1_weighted": float(f1w),
        "n_valid":     len(valid_pred),
        "n_total":     len(classifications),
        "api_errors":  sum(1 for c in classifications if c is None),
        "tokens_prompt": _usage["prompt"],
        "tokens_output": _usage["output"],
        "tokens_total":  _usage["total"],
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

async def eval_dc(labels_path):
    df = pd.read_csv(labels_path)
    df = df[~df["style"].isin(["unsure", "Other", "other"])]

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

    classifications = await classify_batch(paths, desc="DC crops")
    evaluate_and_export(classifications, true_labels, CONDENSED_STYLES,
                        save_dir=f"{OUT_DIR}_dc", tag="DC crops")
    print_usage()


# =============================================================================
# PEXELS EVAL
# =============================================================================

async def eval_pexels(n_sample):
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
    classifications = await classify_batch(test_paths, desc="Pexels holdout")
    evaluate_and_export(classifications, test_labels, CONDENSED_STYLES,
                        save_dir=f"{OUT_DIR}_pexels", tag="Pexels holdout")
    print_usage()


# =============================================================================
# MAIN
# =============================================================================

async def run_test():
    """Send one image through the full pipeline and print the result."""
    # Find the first valid image from dc_labels or data/styles
    test_path = None
    if args.dc_labels and Path(args.dc_labels).exists():
        df = pd.read_csv(args.dc_labels)
        df = df[~df["style"].isin(["unsure", "Other", "other"])]
        for _, row in df.iterrows():
            p = Path(row["crop_path"])
            if p.exists():
                test_path = p
                break
    if test_path is None:
        root = Path(args.data_dir)
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
            if imgs:
                test_path = imgs[0]
                break

    if test_path is None:
        raise SystemExit("No test image found. Pass --dc-labels or check --data-dir.")

    print(f"Test image: {test_path}")
    sem = asyncio.Semaphore(1)
    result = await classify_image_async(test_path, sem)
    if result is None:
        print("FAILED - check error above")
    else:
        print(f"  style:      {result.style}")
        print(f"  confidence: {result.confidence}")
        print(f"  reasoning:  {result.reasoning}")
        print_usage()


async def main():
    print(f"Model:       {args.model}")
    print(f"Concurrency: {args.concurrency}  delay: {args.delay}s")
    print(f"Output:      {OUT_DIR}")

    if not _api_key:
        raise SystemExit("ERROR: pass --api-key or set GEMINI_API_KEY env var")

    if args.test:
        await run_test()
        return

    if args.dc_labels:
        await eval_dc(args.dc_labels)

    if args.pexels_sample:
        await eval_pexels(args.pexels_sample)

    if not args.dc_labels and not args.pexels_sample:
        print("Nothing to do -- pass --dc-labels and/or --pexels-sample")


if __name__ == "__main__":
    asyncio.run(main())
