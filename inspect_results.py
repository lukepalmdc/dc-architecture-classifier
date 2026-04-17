"""
Quick stats and sanity check on downloaded data.
Run after pipeline.py to see coverage summary.
"""

import json
import sqlite3
from pathlib import Path


def main():
    buildings_path = Path("data/buildings.json")
    db_path = Path("data/image_status.db")
    images_dir = Path("data/images")

    if buildings_path.exists():
        with open(buildings_path) as f:
            buildings = json.load(f)
        print(f"Buildings fetched:    {len(buildings):,}")

        types = {}
        for b in buildings:
            t = b.get("building", "yes")
            types[t] = types.get(t, 0) + 1
        print("\nTop building types (OSM 'building' tag):")
        for t, count in sorted(types.items(), key=lambda x: -x[1])[:15]:
            print(f"  {t:<30} {count:>6,}")
    else:
        print("No buildings.json found. Run fetch_buildings.py first.")
        return

    if db_path.exists():
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT state, COUNT(*) FROM status GROUP BY state").fetchall()
        conn.close()
        print("\nImage download status:")
        for state, count in rows:
            print(f"  {state:<15} {count:>6,}")
    else:
        print("\nNo image_status.db found. Run fetch_images.py first.")

    if images_dir.exists():
        images = list(images_dir.glob("*.jpg"))
        total_mb = sum(p.stat().st_size for p in images) / (1024 * 1024)
        print(f"\nImages on disk:       {len(images):,}")
        print(f"Total size:           {total_mb:,.0f} MB  ({total_mb/1024:.1f} GB)")
    else:
        print("\nNo images directory found.")


if __name__ == "__main__":
    main()
