"""
Full pipeline: fetch buildings, then fetch images.
Run this to do everything in one shot.

Usage:
    python pipeline.py --token YOUR_MAPILLARY_TOKEN

Or set env var first:
    export MAPILLARY_TOKEN=your_token
    python pipeline.py
"""

import argparse
import os
import subprocess
import sys


def run(cmd):
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Step failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="DC buildings image pipeline")
    parser.add_argument("--token", default=os.environ.get("MAPILLARY_TOKEN", ""),
                        help="Mapillary Client Access Token")
    parser.add_argument("--skip-buildings", action="store_true",
                        help="Skip building fetch (use existing data/buildings.json)")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: --token required (or set MAPILLARY_TOKEN env var)")
        print("  Get a free token at: https://www.mapillary.com/developer")
        sys.exit(1)

    if not args.skip_buildings:
        run([sys.executable, "fetch_buildings.py"])
    else:
        print("Skipping building fetch (--skip-buildings set)")

    os.environ["MAPILLARY_TOKEN"] = args.token
    run([sys.executable, "fetch_images.py", "--token", args.token])

    print("\nPipeline complete.")
    print("Images are in: data/images/")
    print("Metadata DB:   data/image_status.db")
    print("Buildings:     data/buildings.json")


if __name__ == "__main__":
    main()
