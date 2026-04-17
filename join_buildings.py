"""
Spatial join: Address_Points.csv × Building_Footprints_2023.geojson

For each building polygon, find all address points whose lat/lon falls inside it.
Attach address, RESIDENTIAL_TYPE label, unit counts, ward, zip, SSL, etc.

Output: data/buildings_enriched.geojson
  - One feature per building polygon
  - Added properties:
      address           primary street address (or empty if no match)
      residential_type  RESIDENTIAL | NON RESIDENTIAL | MIXED USE | UNKNOWN
      housing_units     total housing unit count across all matched addresses
      ward              DC ward
      zipcode
      ssl               DC parcel ID (Square/Suffix/Lot)
      address_count     number of address points matched inside this building
      centroid_lat      polygon centroid latitude
      centroid_lon      polygon centroid longitude

Usage:
    pip install shapely tqdm
    python join_buildings.py
"""

import csv
import json
import os
from collections import defaultdict

from shapely.geometry import Point, shape
from shapely.strtree import STRtree
from tqdm import tqdm

BUILDINGS_PATH = "Building_Footprints_2023.geojson"
ADDRESSES_PATH = "Address_Points.csv"
OUTPUT_PATH    = "data/buildings_enriched.geojson"

# Only join address points that are placed at/inside a building
VALID_PLACEMENTS = {"CENTER OF BUILDING", "MAIN ENTRANCE"}
VALID_STATUSES   = {"ACTIVE", "ASSIGNED"}


# ---------------------------------------------------------------------------
# 1. Load building footprints
# ---------------------------------------------------------------------------

def load_buildings(path):
    print(f"Loading {path} ...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    features = data["features"]
    print(f"  {len(features):,} features loaded")
    return features


# ---------------------------------------------------------------------------
# 2. Load address points
# ---------------------------------------------------------------------------

def load_address_points(path):
    print(f"Loading {path} ...")
    points = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["LATITUDE"])
                lon = float(row["LONGITUDE"])
            except (ValueError, KeyError):
                continue
            if row.get("STATUS", "") not in VALID_STATUSES:
                continue
            if row.get("BUILDING", "Y") == "N":
                continue
            points.append({
                "lat": lat,
                "lon": lon,
                "address":        row.get("ADDRESS", "").strip(),
                "residential_type": row.get("RESIDENTIAL_TYPE", "").strip(),
                "housing_units":  _int(row.get("HOUSING_UNIT_COUNT", "")),
                "ward":           row.get("WARD", "").strip(),
                "zipcode":        row.get("ZIPCODE", "").strip(),
                "ssl":            row.get("SSL", "").strip(),
                "has_condo":      row.get("HAS_CONDO", "N") == "Y",
                "placement":      row.get("PLACEMENT", "").strip(),
            })
    print(f"  {len(points):,} usable address points loaded")
    return points


def _int(v):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# 3. Build spatial index and join
# ---------------------------------------------------------------------------

def build_index(features):
    """Build STRtree on building polygons. Returns (tree, geometries)."""
    print("Building spatial index ...")
    geometries = []
    for feat in features:
        try:
            geom = shape(feat["geometry"])
        except Exception:
            geom = None
        geometries.append(geom)
    valid = [g for g in geometries if g is not None and g.is_valid]
    tree = STRtree([g for g in geometries if g is not None])
    print(f"  Index built on {len([g for g in geometries if g is not None]):,} geometries")
    return tree, geometries


def spatial_join(features, geometries, tree, address_points):
    """
    For each address point, find the building polygon that contains it.
    Returns: dict[feature_index -> list[address_point]]
    """
    print(f"Joining {len(address_points):,} address points to {len(features):,} buildings ...")
    matched = defaultdict(list)
    unmatched = 0

    for ap in tqdm(address_points, desc="Joining", unit="addr"):
        pt = Point(ap["lon"], ap["lat"])
        # Query candidates by bounding box, then test containment
        candidates = tree.query(pt)
        found = False
        for idx in candidates:
            geom = geometries[idx]
            if geom is not None and geom.contains(pt):
                matched[idx].append(ap)
                found = True
                break
        if not found:
            unmatched += 1

    print(f"  Matched: {sum(len(v) for v in matched.values()):,} address points")
    print(f"  Unmatched: {unmatched:,} address points (outside all building polygons)")
    print(f"  Buildings with 1+ address: {len(matched):,}")
    return matched


# ---------------------------------------------------------------------------
# 4. Aggregate multiple address points per building
# ---------------------------------------------------------------------------

TYPE_PRIORITY = {"MIXED USE": 2, "NON RESIDENTIAL": 1, "RESIDENTIAL": 0}

def aggregate(address_list):
    """Collapse multiple address points for one building into a single record."""
    if not address_list:
        return {
            "address": "",
            "residential_type": "UNKNOWN",
            "housing_units": 0,
            "ward": "",
            "zipcode": "",
            "ssl": "",
            "address_count": 0,
        }

    # Pick the highest-priority type across all addresses
    best_type = max(
        (a["residential_type"] for a in address_list if a["residential_type"]),
        key=lambda t: TYPE_PRIORITY.get(t, -1),
        default="UNKNOWN",
    )

    # Primary address: prefer MAIN ENTRANCE, else CENTER OF BUILDING, else first
    def placement_rank(a):
        return {"MAIN ENTRANCE": 0, "CENTER OF BUILDING": 1}.get(a["placement"], 2)

    primary = sorted(address_list, key=placement_rank)[0]

    return {
        "address":          primary["address"],
        "residential_type": best_type or "UNKNOWN",
        "housing_units":    sum(a["housing_units"] for a in address_list),
        "ward":             primary["ward"],
        "zipcode":          primary["zipcode"],
        "ssl":              primary["ssl"],
        "address_count":    len(address_list),
    }


# ---------------------------------------------------------------------------
# 5. Build enriched GeoJSON
# ---------------------------------------------------------------------------

def build_enriched(features, geometries, matched):
    print("Building enriched GeoJSON ...")
    out_features = []

    for idx, feat in enumerate(tqdm(features, desc="Enriching", unit="bldg")):
        geom = geometries[idx]
        addr_data = aggregate(matched.get(idx, []))

        # Compute centroid
        if geom is not None:
            c = geom.centroid
            centroid_lat = round(c.y, 7)
            centroid_lon = round(c.x, 7)
        else:
            centroid_lat = centroid_lon = None

        props = {
            # Original fields
            "OBJECTID":    feat["properties"].get("OBJECTID"),
            "DESCRIPTION": feat["properties"].get("DESCRIPTION"),
            "FEATURECODE": feat["properties"].get("FEATURECODE"),
            "GLOBALID":    feat["properties"].get("GLOBALID"),
            # Enriched fields
            "address":          addr_data["address"],
            "residential_type": addr_data["residential_type"],
            "housing_units":    addr_data["housing_units"],
            "ward":             addr_data["ward"],
            "zipcode":          addr_data["zipcode"],
            "ssl":              addr_data["ssl"],
            "address_count":    addr_data["address_count"],
            "centroid_lat":     centroid_lat,
            "centroid_lon":     centroid_lon,
        }

        out_features.append({
            "type": "Feature",
            "geometry": feat["geometry"],
            "properties": props,
        })

    return {"type": "FeatureCollection", "features": out_features}


# ---------------------------------------------------------------------------
# 6. Summary stats
# ---------------------------------------------------------------------------

def print_summary(out_features):
    from collections import Counter
    types = Counter(f["properties"]["residential_type"] for f in out_features)
    matched = sum(1 for f in out_features if f["properties"]["address_count"] > 0)
    print("\n--- Summary ---")
    print(f"  Total buildings:          {len(out_features):>8,}")
    print(f"  With address match:       {matched:>8,}")
    print(f"  Without address match:    {len(out_features)-matched:>8,}")
    print()
    print("  residential_type breakdown:")
    for t, n in types.most_common():
        print(f"    {t:<20} {n:>8,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("data", exist_ok=True)

    features      = load_buildings(BUILDINGS_PATH)
    address_points = load_address_points(ADDRESSES_PATH)
    tree, geometries = build_index(features)
    matched       = spatial_join(features, geometries, tree, address_points)
    enriched      = build_enriched(features, geometries, matched)

    # Drop buildings with no address match before writing
    before = len(enriched["features"])
    enriched["features"] = [
        f for f in enriched["features"]
        if f["properties"]["residential_type"] != "UNKNOWN"
    ]
    dropped = before - len(enriched["features"])
    print(f"Dropped {dropped:,} unmatched buildings. Keeping {len(enriched['features']):,}.")

    print(f"\nWriting {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f)
    print(f"Done. {OUTPUT_PATH} written.")

    print_summary(enriched["features"])


if __name__ == "__main__":
    main()
