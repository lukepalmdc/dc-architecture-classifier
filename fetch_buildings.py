"""
Fetch all building centroids in Washington DC from OpenStreetMap via Overpass API.
Saves results to data/buildings.json (list of {id, lat, lon, tags}).

DC bounding box: S=38.7916, W=-77.1198, N=38.9958, E=-76.9094
"""

import json
import time
import os
import requests
from tqdm import tqdm

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DC_BBOX = (38.7916, -77.1198, 38.9958, -76.9094)  # south, west, north, east

# Split DC into a grid of tiles to avoid Overpass timeout/memory limits
GRID_ROWS = 4
GRID_COLS = 4

OUTPUT_PATH = "data/buildings.json"


def make_query(south, west, north, east):
    bbox = f"{south},{west},{north},{east}"
    return f"""
[out:json][timeout:90];
(
  way["building"]({bbox});
  relation["building"]({bbox});
);
out center tags;
"""


def tile_bbox(south, west, north, east, rows, cols):
    lat_step = (north - south) / rows
    lon_step = (east - west) / cols
    tiles = []
    for r in range(rows):
        for c in range(cols):
            s = south + r * lat_step
            n = s + lat_step
            w = west + c * lon_step
            e = w + lon_step
            tiles.append((s, w, n, e))
    return tiles


def fetch_tile(south, west, north, east, retries=3):
    query = make_query(south, west, north, east)
    for attempt in range(retries):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  Retry {attempt+1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"  Failed tile ({south:.3f},{west:.3f}): {e}")
                return None


def parse_element(el):
    """Extract id, centroid lat/lon, and tags from an OSM element."""
    osm_id = el.get("id")
    tags = el.get("tags", {})

    if el["type"] == "way":
        center = el.get("center", {})
        lat = center.get("lat")
        lon = center.get("lon")
    elif el["type"] == "relation":
        center = el.get("center", {})
        lat = center.get("lat")
        lon = center.get("lon")
    else:
        return None

    if lat is None or lon is None:
        return None

    return {
        "osm_id": osm_id,
        "osm_type": el["type"],
        "lat": lat,
        "lon": lon,
        "name": tags.get("name", ""),
        "building": tags.get("building", "yes"),
        "amenity": tags.get("amenity", ""),
        "addr_street": tags.get("addr:street", ""),
        "addr_housenumber": tags.get("addr:housenumber", ""),
        "levels": tags.get("building:levels", ""),
    }


def main():
    os.makedirs("data", exist_ok=True)

    # Load existing progress if any
    existing = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            for b in json.load(f):
                existing[b["osm_id"]] = b
        print(f"Loaded {len(existing)} existing buildings from previous run.")

    tiles = tile_bbox(*DC_BBOX, GRID_ROWS, GRID_COLS)
    all_buildings = dict(existing)

    print(f"Fetching {GRID_ROWS * GRID_COLS} tiles from Overpass API...")
    for i, (s, w, n, e) in enumerate(tqdm(tiles, desc="Tiles")):
        tqdm.write(f"Tile {i+1}/{len(tiles)}: ({s:.3f},{w:.3f}) -> ({n:.3f},{e:.3f})")
        data = fetch_tile(s, w, n, e)
        if data is None:
            continue

        new_count = 0
        for el in data.get("elements", []):
            parsed = parse_element(el)
            if parsed and parsed["osm_id"] not in all_buildings:
                all_buildings[parsed["osm_id"]] = parsed
                new_count += 1

        tqdm.write(f"  +{new_count} new buildings (total: {len(all_buildings)})")

        # Save after each tile so progress isn't lost
        with open(OUTPUT_PATH, "w") as f:
            json.dump(list(all_buildings.values()), f)

        # Be polite to Overpass
        time.sleep(2)

    print(f"\nDone. {len(all_buildings)} buildings saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
