"""
Scrape architecture style building images from Pexels.

Usage:
    python scrape_test_data.py --key YOUR_PEXELS_KEY
    # or
    export PEXELS_KEY=your_key
    python scrape_test_data.py

    python scrape_test_data.py --per-class 200          # fewer images
    python scrape_test_data.py --category brutalist     # single category

Output: data/styles/<label>/<id>.jpg
Resume: safe to re-run — skips already-downloaded files

Pexels limits: 200 requests/hour, 20,000/month (free tier)
At 2 requests/page × 17 categories = ~34 requests total for listing.
Well within limits.
"""

import argparse
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

PEXELS_SEARCH = "https://api.pexels.com/v1/search"
OUTPUT_DIR    = Path("data/styles")
PER_PAGE      = 80   # Pexels max per page

# Search queries per style — tuned for building/facade photos
CATEGORIES = {
    "neoclassical":       ["neoclassical building", "neoclassical architecture facade"],
    "federal":            ["federal style house", "federal period rowhouse", "federal style architecture home"],
    "victorian":          ["victorian building", "victorian house architecture", "queen anne house"],
    "gothic_revival":     ["gothic revival building", "gothic architecture building"],
    "romanesque_revival": ["romanesque revival building", "romanesque architecture"],
    "beaux_arts":         ["beaux arts building", "beaux arts architecture"],
    "brutalist":          ["brutalist building", "brutalist architecture concrete"],
    "modernist":          ["modernist building", "international style architecture building"],
    "postmodern":         ["postmodern building", "postmodern architecture"],
    "art_deco":           ["art deco building", "art deco architecture facade"],
    "art_nouveau":        ["art nouveau building", "art nouveau architecture"],
    "craftsman":          ["craftsman bungalow", "craftsman house"],
    "colonial_revival":   ["colonial revival building", "colonial architecture house"],
    "greek_revival":      ["greek revival building", "greek revival architecture"],
    "tudor_revival":      ["tudor revival building", "tudor style house"],
    "midcentury_modern":  ["mid century modern house", "midcentury modern building"],
}


# ---------------------------------------------------------------------------
# Pexels API
# ---------------------------------------------------------------------------

def search_pexels(query, key, max_images=500):
    """Return up to max_images photo dicts for a query."""
    photos = []
    page   = 1
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

def download_images(photos, save_dir, key):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.stem for p in save_dir.iterdir()}
    to_fetch = [p for p in photos if str(p["id"]) not in existing]

    if skipped := len(photos) - len(to_fetch):
        print(f"  Skipping {skipped} already downloaded")

    for photo in tqdm(to_fetch, unit="img", leave=False):
        url  = photo["src"]["large"]   # ~940px wide
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

def scrape_all(categories, key, per_class=500):
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
        print(f"  Downloading {len(all_photos)} images -> data/styles/{label}/")
        download_images(all_photos, OUTPUT_DIR / label, key)

    print("\n--- Summary ---")
    for label in categories:
        d     = OUTPUT_DIR / label
        count = len(list(d.glob("*.jpg"))) if d.exists() else 0
        print(f"  {label:<22} {count:>4} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default=os.environ.get("PEXELS_KEY", ""),
                        help="Pexels API key (or set PEXELS_KEY env var)")
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--category",  type=str, default=None,
                        help="Single category label (e.g. brutalist)")
    args = parser.parse_args()

    if not args.key:
        print("ERROR: Pexels API key required.")
        print("  Get a free key at: https://www.pexels.com/api/")
        print("  Then: export PEXELS_KEY=your_key")
        exit(1)

    cats = {args.category: CATEGORIES[args.category]} if args.category else CATEGORIES
    scrape_all(cats, args.key, args.per_class)
