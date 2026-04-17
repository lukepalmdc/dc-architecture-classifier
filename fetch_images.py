"""
For each building in data/buildings_enriched.geojson, find the nearest Mapillary
street-level image and download it. Progress is tracked in data/image_status.db
so the run is fully resumable.

Usage:
    python fetch_images.py --token YOUR_TOKEN
    # or
    export MAPILLARY_TOKEN=your_token
    python fetch_images.py

Tuning:
    --concurrency   parallel requests (default 10; raise to 20-30 if no rate limiting)
    --size          thumb_256_url | thumb_1024_url | thumb_2048_url  (default 1024)
    --radius        search radius in metres around centroid (default 50)
"""

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

import aiofiles
import aiohttp
from tqdm.asyncio import tqdm

MAPILLARY_API  = "https://graph.mapillary.com"
IMAGES_DIR     = Path("data/images")
DB_PATH        = "data/image_status.db"
BUILDINGS_PATH = "data/buildings_enriched.geojson"

DEFAULT_CONCURRENCY = 10
DEFAULT_RADIUS      = 50
DEFAULT_SIZE        = "thumb_1024_url"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS status (
            objectid        INTEGER PRIMARY KEY,
            state           TEXT NOT NULL,   -- done | no_image | error
            mapillary_id    TEXT,
            image_path      TEXT,
            residential_type TEXT,
            address         TEXT,
            error_msg       TEXT
        )
    """)
    conn.commit()


def load_done(conn):
    rows = conn.execute("SELECT objectid FROM status").fetchall()
    return {r[0] for r in rows}


def save_result(conn, objectid, state, mapillary_id=None, image_path=None,
                residential_type=None, address=None, error_msg=None):
    conn.execute("""
        INSERT OR REPLACE INTO status
            (objectid, state, mapillary_id, image_path, residential_type, address, error_msg)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (objectid, state, mapillary_id, image_path, residential_type, address,
          error_msg))
    conn.commit()


# ---------------------------------------------------------------------------
# Mapillary API
# ---------------------------------------------------------------------------

def _is_rate_limited(data):
    """Detect Mapillary's rate-limit error payload (arrives as HTTP 200 with error JSON)."""
    err = data.get("error", {})
    return err.get("code") == 4 or "rate" in err.get("message", "").lower()


async def find_nearest_image(session, token, lat, lon, radius, image_size, retries=5):
    params = {
        "fields":       f"id,{image_size},compass_angle",
        "lat":          lat,
        "lng":          lon,
        "radius":       radius,
        "limit":        1,
        "access_token": token,
    }
    backoff = 5
    for attempt in range(retries):
        try:
            async with session.get(f"{MAPILLARY_API}/images", params=params) as resp:
                if resp.status == 429:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                resp.raise_for_status()
                data = await resp.json()
                if _is_rate_limited(data):
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                return data.get("data", [None])[0] if data.get("data") else None
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt < retries - 1:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            else:
                raise
    return None


async def download_image(session, url, dest_path, retries=3):
    backoff = 2
    for attempt in range(retries):
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                async with aiofiles.open(dest_path, "wb") as f:
                    await f.write(await resp.read())
            return
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt < retries - 1:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
            else:
                raise


# ---------------------------------------------------------------------------
# Per-building worker
# ---------------------------------------------------------------------------

async def process_building(sem, session, token, building, conn, pbar, image_size, radius):
    props     = building["properties"]
    objectid  = props["OBJECTID"]
    lat       = props.get("centroid_lat")
    lon       = props.get("centroid_lon")
    res_type  = props.get("residential_type", "")
    address   = props.get("address", "")

    if lat is None or lon is None:
        save_result(conn, objectid, "error", error_msg="missing centroid",
                    residential_type=res_type, address=address)
        pbar.update(1)
        return

    async with sem:
        try:
            image = await find_nearest_image(session, token, lat, lon, radius, image_size)

            if image is None:
                save_result(conn, objectid, "no_image",
                            residential_type=res_type, address=address)
                pbar.update(1)
                return

            mapillary_id = image["id"]
            thumb_url    = image.get(image_size)

            if not thumb_url:
                save_result(conn, objectid, "no_image", mapillary_id=mapillary_id,
                            residential_type=res_type, address=address)
                pbar.update(1)
                return

            dest = IMAGES_DIR / f"{objectid}.jpg"
            await download_image(session, thumb_url, dest)
            save_result(conn, objectid, "done",
                        mapillary_id=mapillary_id, image_path=str(dest),
                        residential_type=res_type, address=address)

        except Exception as e:
            save_result(conn, objectid, "error", error_msg=str(e)[:500],
                        residential_type=res_type, address=address)

        pbar.update(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(token, concurrency, image_size, radius, limit=None):
    if not Path(BUILDINGS_PATH).exists():
        print(f"ERROR: {BUILDINGS_PATH} not found. Run join_buildings.py first.")
        sys.exit(1)

    print(f"Loading {BUILDINGS_PATH} ...")
    with open(BUILDINGS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    buildings = data["features"]
    print(f"  {len(buildings):,} buildings loaded")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    done_ids = load_done(conn)

    remaining = [
        b for b in buildings
        if b["properties"].get("OBJECTID") not in done_ids
    ]
    print(f"  {len(done_ids):,} already done, {len(remaining):,} remaining")

    if limit:
        remaining = remaining[:limit]
        print(f"  (limit={limit}: processing first {len(remaining)} only)")

    if not remaining:
        print("All buildings processed!")
        _print_summary(conn)
        conn.close()
        return

    sem       = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    timeout   = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        with tqdm(total=len(remaining), desc="Fetching images", unit="bldg") as pbar:
            tasks = [
                process_building(sem, session, token, b, conn, pbar, image_size, radius)
                for b in remaining
            ]
            await asyncio.gather(*tasks)

    _print_summary(conn)
    conn.close()


def _print_summary(conn):
    rows = conn.execute("SELECT state, COUNT(*) FROM status GROUP BY state").fetchall()
    print("\n--- Summary ---")
    for state, count in rows:
        print(f"  {state:<12} {count:>8,}")

    # Coverage by residential type
    rows2 = conn.execute("""
        SELECT residential_type, state, COUNT(*)
        FROM status GROUP BY residential_type, state
        ORDER BY residential_type, state
    """).fetchall()
    print("\n  Coverage by type:")
    last_type = None
    for res_type, state, count in rows2:
        if res_type != last_type:
            print(f"    {res_type or 'UNKNOWN'}:")
            last_type = res_type
        print(f"      {state:<12} {count:>7,}")


def main():
    parser = argparse.ArgumentParser(description="Download Mapillary images for DC buildings")
    parser.add_argument("--token", default=os.environ.get("MAPILLARY_TOKEN", ""),
                        help="Mapillary Client Access Token")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Parallel requests (default {DEFAULT_CONCURRENCY})")
    parser.add_argument("--size", default=DEFAULT_SIZE,
                        choices=["thumb_256_url", "thumb_1024_url", "thumb_2048_url"],
                        help="Image size (default thumb_1024_url)")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS,
                        help=f"Search radius in metres (default {DEFAULT_RADIUS})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N buildings (for testing)")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: Mapillary token required.")
        print("  Get one free at: https://www.mapillary.com/developer")
        print("  Then: export MAPILLARY_TOKEN=your_token")
        sys.exit(1)

    asyncio.run(main_async(args.token, args.concurrency, args.size, args.radius, args.limit))


if __name__ == "__main__":
    main()
