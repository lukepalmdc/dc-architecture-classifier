"""
label_app.py — Hierarchical building architecture labeler

Loads crops produced by segment.py (manifest.jsonl).
Shows the original street-view image with the bounding box highlighted
alongside the isolated crop, then asks for building type → style.

Usage:
    streamlit run label_app.py
    streamlit run label_app.py -- --manifest dc_crops/manifest.jsonl \
                                   --labels   dc_labels.csv
"""

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

TAXONOMY = {
    "Single Family House": [
        "Developer Modern", "Tudor", "Victorian", "Neoclassical", "Modernist",
        "Craftsman", "Contemporary", "Midcentury Modern", "Colonial Revival",
        "Cape Cod", "American Foursquare",
    ],
    "Rowhouse": [
        "Developer Modern", "Rowhouse Vernacular", "Italianate", "Victorian",
        "Modernist", "Colonial Revival", "Federal", "Georgian Revival",
    ],
    "Small Multifamily Building": [
        "Modernist", "Colonial Revival", "Developer Modern", "Garden Style",
        "Italianate", "Victorian",
    ],
    "Large Multifamily Building": [
        "Postmodern", "Contemporary Glass", "Developer Modern", "Gothic",
        "Art Deco", "Brutalist", "Colonial Revival", "Neoclassical",
        "Contemporary Vernacular", "International Style",
    ],
    "Office Building": [
        "Postmodern", "Neoclassical", "International Style", "Contemporary Glass",
        "Art Deco", "Gothic Revival", "Beaux-Arts", "Brutalist", "Colonial Revival",
    ],
}

BBOX_COLOR  = "#FF4B4B"
BBOX_WIDTH  = 4


# =============================================================================
# Data
# =============================================================================

def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="dc_crops/manifest.jsonl")
    p.add_argument("--labels",   default="dc_labels.csv")
    return p.parse_args(argv)


@st.cache_data
def load_manifest(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_labels(path):
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        df = pd.read_csv(p)
        return {str(r["image_id"]): r for r in df.to_dict("records")}
    return {}


def save_label(path, record, building_type, style, labels):
    image_id = record["image_id"]
    labels[str(image_id)] = {
        "image_id":      image_id,
        "objectid":      record.get("objectid"),
        "building_type": building_type,
        "style":         style,
        "crop_path":     record["primary_crop"]["crop_path"],
        "image_path":    record["image"],
    }
    pd.DataFrame(list(labels.values())).to_csv(path, index=False)


# =============================================================================
# Image helpers
# =============================================================================

def draw_bbox_on_image(image_path, bbox, max_width=900):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    for offset in range(BBOX_WIDTH):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=BBOX_COLOR,
        )
    if img.width > max_width:
        scale = max_width / img.width
        img = img.resize((max_width, int(img.height * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    return buf


# =============================================================================
# UI
# =============================================================================

def main():
    st.set_page_config(page_title="DC Architecture Labeler", layout="wide",
                       initial_sidebar_state="collapsed")
    args     = parse_args()
    manifest = load_manifest(args.manifest)
    labels   = load_labels(args.labels)

    labeled_ids = set(labels.keys())
    unlabeled   = [r for r in manifest if str(r["image_id"]) not in labeled_ids]
    done        = len(labeled_ids)
    total       = len(manifest)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f"<h3 style='margin-bottom:4px'>DC Architecture Labeler &nbsp;"
        f"<span style='font-size:14px;font-weight:400;color:#888'>"
        f"{done} / {total} labeled</span></h3>",
        unsafe_allow_html=True,
    )
    st.progress(done / total if total else 0)

    if not unlabeled:
        st.success("All buildings labeled!")
        return

    # ── Session state ─────────────────────────────────────────────────────────
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "building_type" not in st.session_state:
        st.session_state.building_type = None

    st.session_state.idx = min(st.session_state.idx, len(unlabeled) - 1)
    record = unlabeled[st.session_state.idx]
    crop   = record["primary_crop"]

    # ── Layout ────────────────────────────────────────────────────────────────
    img_col, ctrl_col = st.columns([3, 1], gap="large")

    with img_col:
        # Original image with bbox
        buf = draw_bbox_on_image(record["image"], crop["bbox"])
        if buf:
            st.image(buf, use_container_width=True,
                     caption="Street view — red box = building being labeled")
        else:
            st.warning(f"Original image not found: {record['image']}")

        st.caption(
            f"objectid **{record.get('objectid', '—')}** · "
            f"image_id `{record['image_id']}` · "
            f"area {crop['area_fraction']:.1%} · "
            f"score {crop['score']:.2f} · "
            f"{st.session_state.idx + 1} of {len(unlabeled)} remaining"
        )

    with ctrl_col:
        # ── Step 1: building type ──────────────────────────────────────────
        st.markdown("#### 1 · Building type")
        for btype in TAXONOMY:
            active = st.session_state.building_type == btype
            if st.button(
                ("✓ " if active else "") + btype,
                key=f"type_{btype}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.building_type = btype
                st.rerun()

        st.divider()

        # ── Step 2: style (only shown after type selected) ─────────────────
        if st.session_state.building_type:
            styles = TAXONOMY[st.session_state.building_type]
            st.markdown(f"#### 2 · Style  <span style='color:#888;font-size:13px'>{st.session_state.building_type}</span>",
                        unsafe_allow_html=True)
            cols = st.columns(2)
            for i, style in enumerate(styles):
                if cols[i % 2].button(style, key=f"style_{st.session_state.building_type}_{style}",
                                      use_container_width=True):
                    save_label(args.labels, record,
                               st.session_state.building_type, style, labels)
                    st.session_state.building_type = None
                    st.session_state.idx = min(
                        st.session_state.idx, len(unlabeled) - 2
                    )
                    st.rerun()

            st.divider()

        # ── Navigation / utilities ─────────────────────────────────────────
        u1, u2 = st.columns(2)
        if u1.button("Unsure", use_container_width=True):
            save_label(args.labels, record,
                       st.session_state.building_type or "unknown", "unsure", labels)
            st.session_state.building_type = None
            st.session_state.idx = min(st.session_state.idx, len(unlabeled) - 2)
            st.rerun()
        if u2.button("Skip", use_container_width=True):
            st.session_state.building_type = None
            st.session_state.idx = min(st.session_state.idx + 1, len(unlabeled) - 1)
            st.rerun()

        n1, n2 = st.columns(2)
        if n1.button("← Prev", use_container_width=True) and st.session_state.idx > 0:
            st.session_state.building_type = None
            st.session_state.idx -= 1
            st.rerun()
        if n2.button("Next →", use_container_width=True):
            st.session_state.building_type = None
            st.session_state.idx = min(st.session_state.idx + 1, len(unlabeled) - 1)
            st.rerun()


if __name__ == "__main__":
    main()
