"""
Scrape architecture style building images from Pexels using hierarchical
building type + style categories (e.g. rowhouse_federal, office_art_deco).

Usage:
    python scrape_test_data.py --key YOUR_PEXELS_KEY
    export PEXELS_KEY=your_key && python scrape_test_data.py

    python scrape_test_data.py --per-class 200
    python scrape_test_data.py --type rowhouse
    python scrape_test_data.py --type rowhouse --style federal

Output: data/styles/<type>_<style>/<id>.jpg
Resume: safe to re-run — skips already-downloaded files
"""

import argparse
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

PEXELS_SEARCH = "https://api.pexels.com/v1/search"
OUTPUT_DIR    = Path("data/styles")
PER_PAGE      = 80

TYPE_DISPLAY = {
    "Single Family House":        "single family house",
    "Rowhouse":                   "rowhouse",
    "Small Multifamily Building": "small multifamily building",
    "Large Multifamily Building": "large multifamily building",
    "Office Building":            "office building",
    "Institutional":              "institutional building",
}

TAXONOMY = {
    "Single Family House": [
        "Developer Modern", "Developer Traditional", "Tudor", "Victorian",
        "Neoclassical", "Modernist", "Craftsman", "Contemporary",
        "Midcentury Modern", "Colonial Revival", "Cape Cod",
        "American Foursquare", "Second Empire",
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
    "Institutional": [
        "Postmodern", "Neoclassical", "International Style", "Contemporary Glass",
        "Art Deco", "Gothic Revival", "Beaux-Arts", "Brutalist", "Colonial Revival",
        "Italianate",
    ],
}


def make_label(btype, style):
    slug = lambda s: s.lower().replace(" ", "_").replace("-", "_").replace("'", "")
    return f"{slug(btype)}_{slug(style)}"


def make_queries(btype, style):
    return [f"{TYPE_DISPLAY[btype]} {style.lower()} style"]


def build_categories(type_filter=None, style_filter=None):
    cats = {}
    for btype, styles in TAXONOMY.items():
        if type_filter and btype.lower() != type_filter.lower():
            continue
        for style in styles:
            if style_filter and style.lower() != style_filter.lower():
                continue
            label = make_label(btype, style)
            cats[label] = make_queries(btype, style)
    return cats


# ---------------------------------------------------------------------------
# Pexels API
# ---------------------------------------------------------------------------

def search_pexels(query, key, max_images=500):
    photos  = []
    page    = 1
    headers = {"Authorization": key}
    while len(photos) < max_images:
        params = {
            "query":    query,
            "per_page": min(PER_PAGE, max_images - len(photos)),
            "page":     page,
        }
        try:
            resp = requests.get(PEXELS_SEARCH, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  API error: {e}")
            break
        batch = data.get("photos", [])
        if not batch:
            break
        photos.extend(batch)
        page += 1
        if len(photos) >= data.get("total_results", 0):
            break
        time.sleep(0.3)
    return photos[:max_images]


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def download_images(photos, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in save_dir.iterdir()}
    to_fetch = [p for p in photos if str(p["id"]) not in existing]
    if skipped := len(photos) - len(to_fetch):
        print(f"  Skipping {skipped} already downloaded")
    for photo in tqdm(to_fetch, unit="img", leave=False):
        url  = photo["src"]["large"]
        dest = save_dir / f"{photo['id']}.jpg"
        for attempt in range(4):
            try:
                resp = requests.get(url, timeout=20, stream=True)
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    tqdm.write(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                time.sleep(0.05)
                break
            except requests.RequestException as e:
                if attempt == 3:
                    tqdm.write(f"  Failed {photo['id']}: {e}")
                    if dest.exists():
                        dest.unlink()
                else:
                    time.sleep(3 * (attempt + 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scrape_all(categories, key, per_class):
    for label, queries in categories.items():
        print(f"\n=== {label} ===")
        all_photos = []
        seen_ids   = set()
        for query in queries:
            print(f"  Searching: '{query}'")
            photos = search_pexels(query, key, max_images=per_class)
            print(f"    {len(photos)} results")
            for p in photos:
                if p["id"] not in seen_ids:
                    seen_ids.add(p["id"])
                    all_photos.append(p)
        all_photos = all_photos[:per_class]
        print(f"  Downloading {len(all_photos)} -> data/styles/{label}/")
        download_images(all_photos, OUTPUT_DIR / label)

    print("\n--- Summary ---")
    for label in categories:
        d     = OUTPUT_DIR / label
        count = len(list(d.glob("*.jpg"))) if d.exists() else 0
        print(f"  {label:<40} {count:>4} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key",       default=os.environ.get("PEXELS_KEY", ""))
    parser.add_argument("--per-class", type=int, default=320)
    parser.add_argument("--type",      type=str, default=None,
                        help="Filter by building type (e.g. rowhouse, office)")
    parser.add_argument("--style",     type=str, default=None,
                        help="Filter by style (e.g. federal, art_deco)")
    args = parser.parse_args()

    if not args.key:
        print("ERROR: Pexels API key required.")
        print("  Get a free key at: https://www.pexels.com/api/")
        print("  Then: export PEXELS_KEY=your_key")
        exit(1)

    cats = build_categories(
        type_filter=args.type.replace("_", " ") if args.type else None,
        style_filter=args.style.replace("_", " ") if args.style else None,
    )
    if not cats:
        print(f"No categories matched type='{args.type}' style='{args.style}'")
        exit(1)

    scrape_all(cats, args.key, args.per_class)
