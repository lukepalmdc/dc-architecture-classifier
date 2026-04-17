"""
segment_and_classify.py

Segment buildings from street-level images using SegFormer (ADE20K),
then classify each building crop with the trained CLIP architecture classifier.

Usage:
    python segment_and_classify.py image.jpg
    python segment_and_classify.py data/images/ --exp build_prompts_full_temp_03
    python segment_and_classify.py image.jpg --top 3 --save-viz --out results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import clip
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from scipy import ndimage

# ── ADE20K building class IDs (0-indexed) ───────────────────────────────────
# 1=building/edifice  25=house  48=skyscraper  84=tower
BUILDING_IDS = {1, 25, 48, 84}

BBOX_PAD_FRAC = 0.05   # expand each crop bbox by 5%

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


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input",        help="Image file or directory")
    p.add_argument("--exp",        default=None,
                   help="Experiment name under outputs/ (default: best by accuracy)")
    p.add_argument("--top",        type=int,   default=3)
    p.add_argument("--min-conf",   type=float, default=0.3,
                   help="Reject crops below this confidence (default 0.3)")
    p.add_argument("--min-area",   type=float, default=0.02,
                   help="Ignore building regions smaller than this fraction of image (default 0.02)")
    p.add_argument("--temperature",type=float, default=0.3)
    p.add_argument("--out",        default=None, help="Save results JSON to this path")
    p.add_argument("--save-viz",   action="store_true",
                   help="Save annotated image next to each input")
    return p.parse_args()


# =============================================================================
# MODELS
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Loading SegFormer (ADE20K b2)...")
seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
).to(device).eval()

print("Loading CLIP ViT-B/32...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()


# =============================================================================
# EXPERIMENT ARTIFACTS
# =============================================================================

def find_best_exp():
    outputs = Path("outputs")
    best_exp, best_acc = None, -1.0
    for d in outputs.iterdir():
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        if m.get("accuracy", 0) > best_acc:
            best_acc = m["accuracy"]
            best_exp = d.name
    return best_exp


def load_experiment(exp_name):
    exp_dir = Path("outputs") / exp_name
    with open(exp_dir / "class_names.json") as f:
        class_names = json.load(f)
    prototypes     = np.load("prototypes.npy")     if Path("prototypes.npy").exists()     else None
    prompt_weights = np.load("prompt_weights.npy") if Path("prompt_weights.npy").exists() else None
    return class_names, prototypes, prompt_weights


def encode_text(class_names, temperature):
    """Build prompts and encode with CLIP. Returns (text_features, class_indices)."""
    all_prompts, class_indices = [], []
    for i, cls in enumerate(class_names):
        label = LABEL_DISPLAY.get(cls, cls.replace("_", " ").title())
        for prompt in [
            f"a {label} building",
            f"{label} architecture",
            f"a building in {label} style",
            f"photo of {label} architecture",
        ]:
            all_prompts.append(prompt)
            class_indices.append(i)

    tokens = clip.tokenize(all_prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features, class_indices


# =============================================================================
# SEGMENTATION
# =============================================================================

def segment_buildings(pil_image, min_area_frac):
    """
    Run SegFormer on image, return list of building crop dicts:
        {"bbox": [x1,y1,x2,y2], "area_fraction": float, "crop": PIL.Image}
    """
    W, H = pil_image.size
    total_px = W * H

    inputs = seg_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = seg_model(**inputs).logits          # [1 x C x h x w]

    upsampled = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    seg_map   = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()  # [H x W]

    building_mask = np.isin(seg_map, list(BUILDING_IDS))
    if not building_mask.any():
        return []

    labeled, n = ndimage.label(building_mask)
    crops = []

    for comp_id in range(1, n + 1):
        comp = labeled == comp_id
        area_frac = comp.sum() / total_px
        if area_frac < min_area_frac:
            continue

        rows = np.where(comp.any(axis=1))[0]
        cols = np.where(comp.any(axis=0))[0]
        y1, y2 = int(rows[0]),  int(rows[-1])
        x1, x2 = int(cols[0]),  int(cols[-1])

        # Pad bbox
        pad_y = max(1, int((y2 - y1) * BBOX_PAD_FRAC))
        pad_x = max(1, int((x2 - x1) * BBOX_PAD_FRAC))
        y1 = max(0, y1 - pad_y);  y2 = min(H, y2 + pad_y)
        x1 = max(0, x1 - pad_x);  x2 = min(W, x2 + pad_x)

        crops.append({
            "bbox":          [x1, y1, x2, y2],
            "area_fraction": float(area_frac),
            "crop":          pil_image.crop((x1, y1, x2, y2)),
        })

    # Largest buildings first
    crops.sort(key=lambda c: -c["area_fraction"])
    return crops


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_crop(pil_crop, class_names, prototypes, text_features,
                  class_indices, prompt_weights, temperature, top_k):
    img_t = clip_preprocess(pil_crop).unsqueeze(0).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda"):
        feats = clip_model.encode_image(img_t)
        feats = feats / feats.norm(dim=-1, keepdim=True)   # [1 x D]

    num_classes = len(class_names)

    # Weighted text scores (scatter-add on GPU)
    raw = feats @ text_features.T / temperature
    prompt_sims = torch.sigmoid(raw)                       # [1 x P]

    if prompt_weights is not None:
        w = torch.tensor(prompt_weights, dtype=feats.dtype, device=device)
    else:
        w = torch.ones(len(class_indices), dtype=feats.dtype, device=device)

    idx = torch.tensor(class_indices, device=device)
    weighted = prompt_sims * w.unsqueeze(0)
    text_scores = torch.zeros(1, num_classes, dtype=feats.dtype, device=device)
    text_scores.scatter_add_(1, idx.unsqueeze(0).expand(1, -1), weighted)
    weight_sums = torch.zeros(num_classes, dtype=feats.dtype, device=device)
    weight_sums.scatter_add_(0, idx, w)
    text_scores /= weight_sums.clamp(min=1e-8).unsqueeze(0)

    if prototypes is not None:
        proto_t = torch.tensor(prototypes, dtype=feats.dtype, device=device)
        proto_scores = torch.sigmoid(feats @ proto_t.T / temperature)
        scores = 0.5 * proto_scores + 0.5 * text_scores
    else:
        scores = text_scores

    scores_np = scores.squeeze(0).cpu().float().numpy()
    top_idx = np.argsort(-scores_np)[:top_k]
    return [{"label": class_names[i], "confidence": float(scores_np[i])} for i in top_idx]


# =============================================================================
# VISUALIZATION
# =============================================================================

COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

def save_viz(pil_image, buildings, out_path):
    viz = pil_image.copy()
    draw = ImageDraw.Draw(viz)
    for i, b in enumerate(buildings):
        if "predictions" not in b:
            continue
        x1, y1, x2, y2 = b["bbox"]
        color = COLORS[i % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        top = b["predictions"][0]
        draw.text((x1 + 4, y1 + 4), f"{top['label']} {top['confidence']:.2f}", fill=color)
    viz.save(out_path)


# =============================================================================
# MAIN
# =============================================================================

def process_image(image_path, class_names, prototypes, text_features,
                  class_indices, prompt_weights, args):
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"image": str(image_path), "error": str(e)}

    crops = segment_buildings(pil_image, args.min_area)

    if not crops:
        return {"image": str(image_path), "filtered": "no_buildings", "buildings": []}

    buildings = []
    for crop_info in crops:
        preds = classify_crop(
            crop_info["crop"], class_names, prototypes,
            text_features, class_indices, prompt_weights,
            args.temperature, args.top,
        )
        result = {"bbox": crop_info["bbox"], "area_fraction": crop_info["area_fraction"]}

        if preds and preds[0]["confidence"] >= args.min_conf:
            result["predictions"] = preds
        else:
            result["filtered"] = "low_confidence"
            result["top_confidence"] = preds[0]["confidence"] if preds else 0.0

        buildings.append(result)

    out = {"image": str(image_path), "buildings": buildings}

    if args.save_viz:
        viz_path = Path(image_path).with_suffix(".viz.jpg")
        save_viz(pil_image, buildings, viz_path)
        out["viz"] = str(viz_path)

    return out


def main():
    args = parse_args()

    exp_name = args.exp or find_best_exp()
    if not exp_name:
        print("No experiment outputs found. Run train_architecture.py first.")
        return
    print(f"Using experiment: {exp_name}")

    class_names, prototypes, prompt_weights = load_experiment(exp_name)
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")

    text_features, class_indices = encode_text(class_names, args.temperature)

    input_path = Path(args.input)
    image_paths = (
        sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
        if input_path.is_dir() else [input_path]
    )
    print(f"Processing {len(image_paths)} image(s)...\n")

    all_results = []
    for img_path in image_paths:
        r = process_image(img_path, class_names, prototypes, text_features,
                          class_indices, prompt_weights, args)
        all_results.append(r)

        if "error" in r:
            print(f"  ERROR  {img_path.name}: {r['error']}")
        elif r.get("filtered") == "no_buildings":
            print(f"  SKIP   {img_path.name}: no buildings detected")
        else:
            detected = [b for b in r["buildings"] if "predictions" in b]
            skipped  = [b for b in r["buildings"] if "filtered" in b]
            print(f"  OK     {img_path.name}: {len(detected)} building(s), {len(skipped)} low-conf")
            for b in detected:
                top = b["predictions"][0]
                print(f"           {top['label']:30s} {top['confidence']:.3f}  bbox={b['bbox']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
