import streamlit as st
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(layout="wide")
st.title("Architecture Classification Dashboard")

# =====================
# EXPERIMENT SELECTOR
# =====================
outputs_dir = Path("outputs")
experiments = sorted([d.name for d in outputs_dir.iterdir() if d.is_dir() and (d / "metrics.json").exists()])

if not experiments:
    st.error("No experiment outputs found. Run train_architecture.py first.")
    st.stop()

selected = st.sidebar.selectbox("Experiment", experiments, index=len(experiments) - 1)
save_dir = outputs_dir / selected

# =====================
# LOAD DATA
# =====================
probs  = np.load(save_dir / "probabilities.npy")
labels = np.load(save_dir / "labels.npy")
preds  = np.load(save_dir / "predictions.npy")
cm     = np.load(save_dir / "confusion_matrix.npy")

with open(save_dir / "class_names.json") as f:
    class_names = json.load(f)

with open(save_dir / "metrics.json") as f:
    metrics = json.load(f)

# =====================
# SIDEBAR METRICS
# =====================
st.sidebar.title("Metrics")
st.sidebar.metric("Accuracy",    f"{metrics['accuracy']:.4f}")
st.sidebar.metric("F1 Macro",    f"{metrics['f1_macro']:.4f}")
st.sidebar.metric("F1 Weighted", f"{metrics['f1_weighted']:.4f}")
if "top3_accuracy" in metrics:
    st.sidebar.metric("Top-3 Accuracy", f"{metrics['top3_accuracy']:.4f}")

# =====================
# ALL EXPERIMENTS TABLE
# =====================
st.subheader("All Experiments")
rows = []
for exp in experiments:
    p = outputs_dir / exp / "metrics.json"
    if p.exists():
        with open(p) as f:
            m = json.load(f)
        rows.append({"experiment": exp, "accuracy": m["accuracy"],
                     "f1_macro": m["f1_macro"], "f1_weighted": m["f1_weighted"]})

if rows:
    rows.sort(key=lambda x: -x["accuracy"])
    st.dataframe(rows, use_container_width=True)

# =====================
# CONFUSION MATRIX
# =====================
st.subheader("Confusion Matrix")

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names,
            cmap="Blues", annot=True, fmt="d", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
st.pyplot(fig)

# =====================
# PER-CLASS ACCURACY
# =====================
st.subheader("Per-Class Accuracy")

class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
sorted_idx = np.argsort(class_acc)

fig2, ax2 = plt.subplots(figsize=(12, 5))
bars = ax2.barh([class_names[i] for i in sorted_idx], class_acc[sorted_idx])
ax2.set_xlabel("Accuracy")
ax2.set_xlim(0, 1)
for bar, val in zip(bars, class_acc[sorted_idx]):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:.2f}", va="center", fontsize=9)
plt.tight_layout()
st.pyplot(fig2)

# =====================
# MOST CONFUSED PAIRS
# =====================
st.subheader("Most Confused Pairs")

off_diag = cm.copy()
np.fill_diagonal(off_diag, 0)
flat_idx = np.argsort(off_diag.ravel())[::-1][:10]
confused = []
for idx in flat_idx:
    true_cls = class_names[idx // len(class_names)]
    pred_cls = class_names[idx  % len(class_names)]
    confused.append({"true": true_cls, "predicted": pred_cls, "count": int(off_diag.ravel()[idx])})
st.dataframe(confused, use_container_width=True)
