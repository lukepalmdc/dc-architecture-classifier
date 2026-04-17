"""
Run multiple training experiments in parallel and compare results.

Usage:
    python run_experiments.py
    python run_experiments.py --compare-only   # just print table from existing results
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import shutil, sys
# Use conda env on Windows, plain python elsewhere
if shutil.which("conda") and sys.platform == "win32":
    PYTHON = ["conda", "run", "-n", "env", "python"]
else:
    PYTHON = ["python"]

EXPERIMENTS = [
    {
        "name": "baseline_no_proto_no_weights",
        "args": ["--no-prototypes", "--no-weights", "--no-tune-alpha", "--prompts", "prompts.json"],
    },
    {
        "name": "proto_only",
        "args": ["--use-prototypes", "--no-weights", "--no-tune-alpha", "--prompts", "prompts.json"],
    },
    {
        "name": "weights_only",
        "args": ["--no-prototypes", "--use-weights", "--no-tune-alpha", "--prompts", "prompts.json"],
    },
    {
        "name": "proto_and_weights",
        "args": ["--use-prototypes", "--use-weights", "--no-tune-alpha", "--prompts", "prompts.json"],
    },
    {
        "name": "proto_weights_tuned_alpha",
        "args": ["--use-prototypes", "--use-weights", "--tune-alpha", "--prompts", "prompts.json"],
    },
    {
        "name": "build_prompts_no_proto",
        "args": ["--no-prototypes", "--no-weights", "--no-tune-alpha", "--prompts", "build"],
    },
    {
        "name": "build_prompts_full",
        "args": ["--use-prototypes", "--use-weights", "--tune-alpha", "--prompts", "build"],
    },
    {
        "name": "high_temp",
        "args": ["--use-prototypes", "--use-weights", "--tune-alpha",
                 "--temperature", "0.1", "--prompts", "prompts.json"],
    },
    {
        "name": "low_temp",
        "args": ["--use-prototypes", "--use-weights", "--tune-alpha",
                 "--temperature", "0.05", "--prompts", "prompts.json"],
    },
    {
        "name": "high_ridge",
        "args": ["--use-prototypes", "--use-weights", "--tune-alpha",
                 "--ridge-alpha", "0.1", "--prompts", "prompts.json"],
    },
]


def run_experiment(exp):
    cmd = PYTHON + ["train_architecture.py", "--name", exp["name"]] + exp["args"]

    print(f"  Starting: {exp['name']}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return exp["name"], None, result.stderr[-500:]
    return exp["name"], exp, None


def load_metrics(name):
    path = Path(f"outputs/{name}/metrics.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def print_comparison():
    print("\n" + "="*80)
    print(f"{'Experiment':<40} {'Acc':>7} {'F1 macro':>10} {'F1 weighted':>12}")
    print("-"*80)

    results = []
    for exp in EXPERIMENTS:
        m = load_metrics(exp["name"])
        if m:
            results.append((exp["name"], m["accuracy"], m["f1_macro"], m["f1_weighted"]))

    # Sort by accuracy descending
    for name, acc, f1m, f1w in sorted(results, key=lambda x: -x[1]):
        print(f"{name:<40} {acc:>7.4f} {f1m:>10.4f} {f1w:>12.4f}")

    if results:
        best = max(results, key=lambda x: x[1])
        print(f"\nBest: {best[0]}  acc={best[1]:.4f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3,
                        help="Max parallel experiments (default 3, limited by CPU)")
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
                acc = m["accuracy"] if m else "?"
                print(f"  Done:   {name}  acc={acc:.4f}" if m else f"  Done: {name}")

    print_comparison()

    if failed:
        print(f"\nFailed experiments: {failed}")


if __name__ == "__main__":
    main()
