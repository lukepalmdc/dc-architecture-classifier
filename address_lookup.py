"""
Look up a specific address: geocode it, find the matching OSM building,
fetch the nearest Mapillary street-level image, and download it.

Usage:
    python address_lookup.py "1600 Pennsylvania Ave NW, Washington DC" --token YOUR_TOKEN
    python address_lookup.py "addresses.txt" --token YOUR_TOKEN   # one address per line

Output saved to data/address_results/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MAPILLARY_API = "https://graph.mapillary.com"
SEARCH_RADIUS_M = 80
OUTPUT_DIR = Path("data/address_results")
IMAGE_SIZE = "thumb_1024_url"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "dc-building-research/1.0"})


def geocode(address):
    """Address string → (lat, lon) via Nominatim."""
    resp = SESSION.get(NOMINATIM_URL, params={
        "q": address,
        "format": "json",
        "limit": 1,
        "countrycodes": "us",
    }, timeout=15)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None
    r = results[0]
    return float(r["lat"]), float(r["lon"])


def find_osm_building(lat, lon, radius_m=80):
    """Find the nearest OSM building within radius_m meters of (lat, lon)."""
    # Convert meters to rough degrees (~111km per degree)
    deg = radius_m / 111_000
    s, n = lat - deg, lat + deg
    w, e = lon - deg, lon + deg
    query = f"""
[out:json][timeout:30];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
);
out center tags 1;
"""
    resp = SESSION.post(OVERPASS_URL, data={"data": query}, timeout=45)
    resp.raise_for_status()
    elements = resp.json().get("elements", [])
    if not elements:
        return None
    el = elements[0]
    center = el.get("center", {})
    return {
        "osm_id": el["id"],
        "osm_type": el["type"],
        "lat": center.get("lat", lat),
        "lon": center.get("lon", lon),
        "tags": el.get("tags", {}),
    }


def find_mapillary_image(token, lat, lon):
    """Find the nearest Mapillary image within SEARCH_RADIUS_M of (lat, lon)."""
    params = {
        "fields":       f"id,{IMAGE_SIZE},compass_angle,captured_at",
        "lat":          lat,
        "lng":          lon,
        "radius":       SEARCH_RADIUS_M,
        "limit":        1,
        "access_token": token,
    }
    resp = SESSION.get(f"{MAPILLARY_API}/images", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return data[0] if data else None


def download_image(url, dest_path):
    resp = SESSION.get(url, timeout=30, stream=True)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def lookup_address(address, token):
    """Full pipeline for a single address. Returns result dict."""
    result = {"address": address, "status": "unknown"}

    print(f"\n[1/4] Geocoding: {address}")
    coords = geocode(address)
    if coords is None:
        result["status"] = "geocode_failed"
        print("  ERROR: address not found")
        return result
    lat, lon = coords
    result["geocoded_lat"] = lat
    result["geocoded_lon"] = lon
    print(f"  -> ({lat:.6f}, {lon:.6f})")

    time.sleep(1)  # Nominatim rate limit: 1 req/sec

    print("[2/4] Finding OSM building...")
    building = find_osm_building(lat, lon)
    if building is None:
        result["status"] = "no_osm_building"
        print("  WARNING: no OSM building found nearby")
    else:
        result["osm_building"] = building
        name = building["tags"].get("name", "")
        btype = building["tags"].get("building", "yes")
        print(f"  -> osm_id={building['osm_id']} type={btype}" + (f" name={name}" if name else ""))

    print("[3/4] Finding Mapillary image...")
    if token:
        image = find_mapillary_image(token, lat, lon)
        if image is None:
            result["status"] = "no_mapillary_image"
            print("  WARNING: no Mapillary image found nearby")
        else:
            result["mapillary"] = {
                "id": image["id"],
                "compass_angle": image.get("compass_angle"),
                "captured_at": image.get("captured_at"),
            }
            thumb_url = image.get(IMAGE_SIZE)
            print(f"  -> image_id={image['id']}")

            print("[4/4] Downloading image...")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = address.replace(" ", "_").replace(",", "").replace("/", "-")[:80]
            dest = OUTPUT_DIR / f"{safe_name}.jpg"
            download_image(thumb_url, dest)
            result["image_path"] = str(dest)
            result["status"] = "ok"
            print(f"  -> saved to {dest}")
    else:
        print("  (skipped — no token provided)")
        result["status"] = "ok_no_image"

    return result


def main():
    parser = argparse.ArgumentParser(description="Look up DC buildings by address")
    parser.add_argument("address", help="Address string or path to a .txt file (one address per line)")
    parser.add_argument("--token", default=os.environ.get("MAPILLARY_TOKEN", ""),
                        help="Mapillary Client Access Token (optional for geocode-only mode)")
    parser.add_argument("--no-image", action="store_true", help="Skip image download")
    args = parser.parse_args()

    token = "" if args.no_image else args.token

    # File mode
    if args.address.endswith(".txt") and Path(args.address).exists():
        with open(args.address) as f:
            addresses = [line.strip() for line in f if line.strip()]
    else:
        addresses = [args.address]

    results = []
    for addr in addresses:
        r = lookup_address(addr, token)
        results.append(r)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_DIR / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"Summary: {ok}/{len(results)} successful")


if __name__ == "__main__":
    main()
