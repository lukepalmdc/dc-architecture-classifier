"""
Run training experiments and compare results.

Usage:
    python run_experiments.py
    python run_experiments.py --compare-only
    python run_experiments.py --workers 2
"""

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PYTHON = ["conda", "run", "-n", "env", "python"] if shutil.which("conda") and sys.platform == "win32" else ["python"]

DC_LABELS = "dc_labels.csv"

EXPERIMENTS = [
    # Condensed 14-class (best expected real-world accuracy)
    {
        "name": "condensed_temp02",
        "args": ["--style-only", "--condense", "--use-prototypes", "--use-weights",
                 "--tune-alpha", "--temperature", "0.2", "--prompts", "build",
                 "--dc-labels", DC_LABELS],
    },
    {
        "name": "condensed_temp03",
        "args": ["--style-only", "--condense", "--use-prototypes", "--use-weights",
                 "--tune-alpha", "--temperature", "0.3", "--prompts", "build",
                 "--dc-labels", DC_LABELS],
    },
    {
        "name": "condensed_temp04",
        "args": ["--style-only", "--condense", "--use-prototypes", "--use-weights",
                 "--tune-alpha", "--temperature", "0.4", "--prompts", "build",
                 "--dc-labels", DC_LABELS],
    },
    # Full style-only (27 classes) for comparison
    {
        "name": "style_temp03",
        "args": ["--style-only", "--use-prototypes", "--use-weights", "--tune-alpha",
                 "--temperature", "0.3", "--prompts", "build", "--dc-labels", DC_LABELS],
    },
    # Hierarchical (56 classes) for comparison
    {
        "name": "hier_temp03",
        "args": ["--use-prototypes", "--use-weights", "--tune-alpha",
                 "--temperature", "0.3", "--prompts", "build", "--dc-labels", DC_LABELS],
    },
]


def run_experiment(exp):
    cmd = PYTHON + ["train_architecture.py", "--name", exp["name"]] + exp["args"]
    print(f"  Starting: {exp['name']}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return exp["name"], None, result.stderr[-1000:]
    return exp["name"], exp, None


def load_metrics(name, suffix=""):
    path = Path(f"outputs/{name}{suffix}/metrics.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def print_comparison():
    print("\n" + "=" * 100)
    print(f"{'Experiment':<35} {'Pexels Acc':>10} {'Pexels F1':>10} {'DC Acc':>10} {'DC F1':>10}")
    print("-" * 100)

    rows = []
    for exp in EXPERIMENTS:
        m_pexels = load_metrics(exp["name"])
        m_dc     = load_metrics(exp["name"], suffix="_dc")
        if m_pexels:
            rows.append((
                exp["name"],
                m_pexels.get("accuracy", 0),
                m_pexels.get("f1_macro", 0),
                m_dc.get("accuracy", 0) if m_dc else None,
                m_dc.get("f1_macro", 0) if m_dc else None,
            ))

    for name, pa, pf, da, df in sorted(rows, key=lambda x: -x[1]):
        dc_acc = f"{da:.4f}" if da is not None else "—"
        dc_f1  = f"{df:.4f}" if df is not None else "—"
        print(f"{name:<35} {pa:>10.4f} {pf:>10.4f} {dc_acc:>10} {dc_f1:>10}")

    if rows:
        best = max(rows, key=lambda x: x[1])
        print(f"\nBest (Pexels acc): {best[0]}  acc={best[1]:.4f}")
        dc_rows = [r for r in rows if r[3] is not None]
        if dc_rows:
            best_dc = max(dc_rows, key=lambda x: x[3])
            print(f"Best (DC acc):     {best_dc[0]}  acc={best_dc[3]:.4f}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel experiments (default 2)")
    args = parser.parse_args()

    if args.compare_only:
        print_comparison()
        return

    print(f"Running {len(EXPERIMENTS)} experiments ({args.workers} parallel)...\n")

    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_experiment, exp): exp for exp in EXPERIMENTS}
        for future in as_completed(futures):
            name, exp, err = future.result()
            if err:
                print(f"  FAILED: {name}\n    {err}")
                failed.append(name)
            else:
                m = load_metrics(name)
                acc = f"{m['accuracy']:.4f}" if m else "?"
                print(f"  Done: {name}  pexels_acc={acc}")

    print_comparison()

    if failed:
        print(f"\nFailed: {failed}")


if __name__ == "__main__":
    main()
