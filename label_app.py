"""
label_app.py  —  Manual building architecture labeler

Usage:
    streamlit run label_app.py
    streamlit run label_app.py -- --meta label_set/metadata.csv \
                                   --labels label_set/labels.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

STYLES = [
    "art_deco", "art_nouveau", "beaux_arts", "brutalist", "colonial_revival",
    "craftsman", "federal", "gothic_revival", "greek_revival", "midcentury_modern",
    "modernist", "neoclassical", "postmodern", "romanesque_revival",
    "tudor_revival", "victorian",
]

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

# Keyboard shortcut legend (display only — Streamlit doesn't support native hotkeys)
SHORTCUTS = {}
keys = list("1234567890qwertyui")
for i, style in enumerate(STYLES):
    if i < len(keys):
        SHORTCUTS[keys[i]] = style


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--meta",   default="label_set/metadata.csv")
    p.add_argument("--labels", default="label_set/labels.csv")
    return p.parse_args(argv)


def load_labels(path):
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p).set_index("objectid")["label"].to_dict()
    return {}


def save_label(path, objectid, label, labels):
    labels[int(objectid)] = label
    pd.DataFrame(
        [{"objectid": k, "label": v} for k, v in labels.items()]
    ).to_csv(path, index=False)


def main():
    st.set_page_config(page_title="Architecture Labeler", layout="wide",
                       initial_sidebar_state="collapsed")
    args   = parse_args()
    meta   = pd.read_csv(args.meta)
    labels = load_labels(args.labels)

    labeled_ids = set(int(k) for k in labels.keys())
    unlabeled   = meta[~meta["objectid"].isin(labeled_ids)].reset_index(drop=True)
    done        = len(labeled_ids)
    total       = len(meta)

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        f"<h3 style='margin-bottom:4px'>Washington DC Architecture Labeler"
        f"<span style='font-size:14px;font-weight:400;color:#888;margin-left:16px'>"
        f"{done} / {total} labeled</span></h3>",
        unsafe_allow_html=True,
    )
    st.progress(done / total if total else 0)

    if unlabeled.empty:
        st.success("All buildings labeled!")
        return

    # ── Session state ────────────────────────────────────────────────────────
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    st.session_state.idx = min(st.session_state.idx, len(unlabeled) - 1)

    row = unlabeled.iloc[st.session_state.idx]
    oid = int(row["objectid"])

    # ── Layout ───────────────────────────────────────────────────────────────
    img_col, ctrl_col = st.columns([3, 1], gap="large")

    with img_col:
        crop_path = row["crop_path"]
        if Path(crop_path).exists():
            st.image(crop_path, use_container_width=True)
        else:
            st.warning(f"Image not found: {crop_path}")

        model_label = LABEL_DISPLAY.get(str(row.get("model_label", "")),
                                        str(row.get("model_label", "")))
        conf = row.get("confidence", "")
        conf_str = f"{float(conf):.0%}" if conf != "" else "—"
        st.caption(
            f"**{row.get('address', '')}**  ·  objectid {oid}  ·  "
            f"model prediction: **{model_label}** ({conf_str})  ·  "
            f"building {st.session_state.idx + 1} of {len(unlabeled)} remaining"
        )

    with ctrl_col:
        st.markdown("#### Label this building")

        # Style buttons — 2 columns
        b1, b2 = st.columns(2)
        for i, style in enumerate(STYLES):
            col = b1 if i % 2 == 0 else b2
            label_text = LABEL_DISPLAY[style]
            key_hint   = list(SHORTCUTS.keys())[list(SHORTCUTS.values()).index(style)] \
                         if style in SHORTCUTS.values() else ""
            if col.button(f"{label_text}", key=f"btn_{style}", use_container_width=True,
                          help=f"Shortcut: {key_hint}"):
                save_label(args.labels, oid, style, labels)
                st.session_state.idx = max(0, min(
                    st.session_state.idx, len(unlabeled) - 2
                ))
                st.rerun()

        st.divider()

        sc1, sc2 = st.columns(2)
        if sc1.button("Unsure", use_container_width=True, help="Save as 'unsure'"):
            save_label(args.labels, oid, "unsure", labels)
            st.session_state.idx = max(0, min(
                st.session_state.idx, len(unlabeled) - 2
            ))
            st.rerun()
        if sc2.button("Skip", use_container_width=True, help="Skip without labeling"):
            st.session_state.idx = min(st.session_state.idx + 1, len(unlabeled) - 1)
            st.rerun()

        nc1, nc2 = st.columns(2)
        if nc1.button("← Prev", use_container_width=True) and st.session_state.idx > 0:
            st.session_state.idx -= 1
            st.rerun()
        if nc2.button("Next →", use_container_width=True):
            st.session_state.idx = min(st.session_state.idx + 1, len(unlabeled) - 1)
            st.rerun()

        # ── Keyboard shortcut legend ─────────────────────────────────────────
        with st.expander("Keyboard reference", expanded=False):
            for key, style in SHORTCUTS.items():
                st.markdown(
                    f"`{key}` &nbsp; {LABEL_DISPLAY[style]}",
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
