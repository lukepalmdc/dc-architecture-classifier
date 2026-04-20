"""
join_results.py

Join dc_results/results.jsonl with data/buildings_enriched.geojson
to produce a flat CSV with one row per building.

Usage:
    python join_results.py
    python join_results.py --results dc_results/results.jsonl \
                           --buildings data/buildings_enriched.geojson \
                           --out dc_results/buildings_classified.csv
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import jsonlines
from shapely.geometry import shape

# Architecture style class names (order must match training)
STYLE_COLS = [
    "art_deco", "art_nouveau", "beaux_arts", "brutalist", "colonial_revival",
    "craftsman", "federal", "gothic_revival", "greek_revival", "midcentury_modern",
    "modernist", "neoclassical", "postmodern", "romanesque_revival",
    "tudor_revival", "victorian",
]

STATUS_CLASSIFIED    = "classified"
STATUS_LOW_CONF      = "low_confidence"
STATUS_NO_BUILDINGS  = "no_buildings"
STATUS_NO_IMAGE      = "no_image"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results",   default="dc_results/results.jsonl")
    p.add_argument("--buildings", default="data/buildings_enriched.geojson")
    p.add_argument("--out",       default="dc_results/buildings_classified.csv")
    return p.parse_args()


# =============================================================================
# STEP 1 — Aggregate results.jsonl per building
# =============================================================================

def aggregate_jsonl(results_path):
    """
    Re-aggregate per-image JSONL into per-building records.

    Returns dict: objectid (int) -> {
        status, predicted_label, confidence, other_p, n_crops, all_scores
    }
    """
    # Accumulate score vectors per building across all images
    score_accum  = defaultdict(list)   # objectid -> list of all_scores dicts
    status_votes = defaultdict(set)    # objectid -> set of statuses seen

    print(f"Reading {results_path} ...")
    with jsonlines.open(results_path) as reader:
        for record in reader:
            oid = record.get("objectid")
            if oid is None:
                continue
            oid = int(oid)   # normalise to int

            buildings = record.get("buildings", [])

            if not buildings and record.get("filtered") == "no_buildings":
                status_votes[oid].add(STATUS_NO_BUILDINGS)
                continue

            for b in buildings:
                if "all_scores" in b:
                    score_accum[oid].append(b["all_scores"])
                    status_votes[oid].add(STATUS_CLASSIFIED)
                elif b.get("filtered") == "low_confidence":
                    status_votes[oid].add(STATUS_LOW_CONF)
                elif not buildings:
                    status_votes[oid].add(STATUS_NO_BUILDINGS)

    print(f"  Images processed for {len(status_votes)} unique buildings")

    aggregated = {}

    for oid in status_votes:
        scores_list = score_accum.get(oid, [])

        if scores_list:
            # Average all score dicts
            avg = defaultdict(float)
            for sd in scores_list:
                for lbl, val in sd.items():
                    avg[lbl] += val / len(scores_list)

            # Fill any missing classes with 0
            for cls in STYLE_COLS:
                if cls not in avg:
                    avg[cls] = 0.0

            top_label = max(avg, key=avg.get)
            aggregated[oid] = {
                "status":          STATUS_CLASSIFIED,
                "predicted_label": top_label,
                "confidence":      round(avg[top_label], 4),
                "other_p":         round(1.0 - avg[top_label], 4),
                "n_crops":         len(scores_list),
                "all_scores":      {k: round(avg[k], 4) for k in STYLE_COLS},
            }
        else:
            # Determine best non-classified status
            votes = status_votes[oid]
            if STATUS_LOW_CONF in votes:
                status = STATUS_LOW_CONF
            else:
                status = STATUS_NO_BUILDINGS

            aggregated[oid] = {
                "status":          status,
                "predicted_label": None,
                "confidence":      None,
                "other_p":         None,
                "n_crops":         0,
                "all_scores":      {k: None for k in STYLE_COLS},
            }

    print(f"  Classified:     {sum(1 for v in aggregated.values() if v['status'] == STATUS_CLASSIFIED)}")
    print(f"  Low confidence: {sum(1 for v in aggregated.values() if v['status'] == STATUS_LOW_CONF)}")
    print(f"  No buildings:   {sum(1 for v in aggregated.values() if v['status'] == STATUS_NO_BUILDINGS)}")

    return aggregated


# =============================================================================
# STEP 2 — Load GeoJSON buildings
# =============================================================================

def load_geojson(buildings_path):
    """
    Returns dict: objectid (int) -> {address, residential_type, lat, lon, ...}
    Extracts centroid from geometry.
    """
    print(f"Reading {buildings_path} ...")
    with open(buildings_path) as f:
        gj = json.load(f)

    buildings = {}
    for feat in gj["features"]:
        props = feat.get("properties", {})
        oid   = props.get("OBJECTID") or props.get("objectid")
        if oid is None:
            continue
        oid = int(oid)

        # Centroid
        try:
            geom    = shape(feat["geometry"])
            centroid = geom.centroid
            lat, lon = round(centroid.y, 6), round(centroid.x, 6)
        except Exception:
            lat, lon = None, None

        buildings[oid] = {
            "address":          props.get("ADDRESS") or props.get("address", ""),
            "residential_type": props.get("RESIDENTIAL_TYPE") or props.get("residential_type", ""),
            "lat":              lat,
            "lon":              lon,
        }

    print(f"  Loaded {len(buildings)} buildings from GeoJSON")
    return buildings


# =============================================================================
# STEP 3 — Join and write CSV
# =============================================================================

def write_csv(buildings, aggregated, out_path):
    total        = len(buildings)
    matched      = 0
    no_image_cnt = 0

    fieldnames = [
        "objectid", "address", "residential_type", "lat", "lon",
        "status", "predicted_label", "confidence", "other_p", "n_crops",
    ] + [f"score_{cls}" for cls in STYLE_COLS]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for oid, bld in buildings.items():
            result = aggregated.get(oid)

            if result:
                matched += 1
                row = {
                    "objectid":         oid,
                    "address":          bld["address"],
                    "residential_type": bld["residential_type"],
                    "lat":              bld["lat"],
                    "lon":              bld["lon"],
                    "status":           result["status"],
                    "predicted_label":  result["predicted_label"] or "",
                    "confidence":       result["confidence"] if result["confidence"] is not None else "",
                    "other_p":          result["other_p"]    if result["other_p"]    is not None else "",
                    "n_crops":          result["n_crops"],
                }
                for cls in STYLE_COLS:
                    v = result["all_scores"].get(cls)
                    row[f"score_{cls}"] = v if v is not None else ""
            else:
                no_image_cnt += 1
                row = {
                    "objectid":         oid,
                    "address":          bld["address"],
                    "residential_type": bld["residential_type"],
                    "lat":              bld["lat"],
                    "lon":              bld["lon"],
                    "status":           STATUS_NO_IMAGE,
                    "predicted_label":  "",
                    "confidence":       "",
                    "other_p":          "",
                    "n_crops":          0,
                }
                for cls in STYLE_COLS:
                    row[f"score_{cls}"] = ""

            writer.writerow(row)

    print(f"\nCSV written to {out_path}")
    print(f"  Total buildings:  {total}")
    print(f"  Matched to results: {matched}  ({matched/total*100:.1f}%)")
    print(f"  No image/not run:   {no_image_cnt}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args       = parse_args()
    aggregated = aggregate_jsonl(args.results)
    buildings  = load_geojson(args.buildings)
    write_csv(buildings, aggregated, args.out)


if __name__ == "__main__":
    main()
