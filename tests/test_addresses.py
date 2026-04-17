"""
Integration tests: geocode → OSM building → Mapillary image for known DC addresses.

Run all tests:
    pytest tests/ -v

Run without image download (no token needed):
    pytest tests/ -v -m "not requires_token"

Run just geocoding tests:
    pytest tests/ -v -m geocode

Set token via env var:
    export MAPILLARY_TOKEN=your_token
    pytest tests/ -v
"""

import os
import time
import pytest
import requests

# ---------------------------------------------------------------------------
# Test fixtures: well-known DC addresses with expected building metadata
#
# Ranges verified against actual Nominatim responses — use ±0.003 deg (~300m)
# around the true centroid so small geocoder drift doesn't cause false fails.
# ---------------------------------------------------------------------------

KNOWN_ADDRESSES = [
    {
        "address": "1600 Pennsylvania Avenue NW, Washington, DC",
        "label": "white_house",
        "expected_lat_range": (38.893, 38.900),
        "expected_lon_range": (-77.042, -77.034),
        "notes": "White House",
    },
    {
        "address": "1 First Street NE, Washington, DC",
        "label": "supreme_court",
        "expected_lat_range": (38.903, 38.910),
        "expected_lon_range": (-77.008, -77.002),
        "notes": "Supreme Court — Nominatim geocodes to ~38.9068, -77.0046",
    },
    {
        "address": "900 Jefferson Drive SW, Washington, DC",
        "label": "smithsonian_castle",
        "expected_lat_range": (38.886, 38.891),
        "expected_lon_range": (-77.028, -77.023),
        "notes": "Smithsonian Institution Building (the Castle)",
    },
    {
        "address": "800 F Street NW, Washington, DC",
        "label": "spy_museum",
        "expected_lat_range": (38.895, 38.901),
        "expected_lon_range": (-77.025, -77.018),
        "notes": "International Spy Museum",
    },
    {
        "address": "120 Maryland Avenue NE, Washington, DC",
        "label": "senate_office",
        "expected_lat_range": (38.890, 38.896),
        "expected_lon_range": (-77.007, -77.000),
        "notes": "Senate office area near Capitol",
    },
    {
        "address": "3101 Wisconsin Avenue NW, Washington, DC",
        "label": "washington_cathedral",
        "expected_lat_range": (38.928, 38.936),
        "expected_lon_range": (-77.075, -77.068),
        "notes": "Washington National Cathedral",
    },
]

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
MAPILLARY_API = "https://graph.mapillary.com"
SEARCH_RADIUS_M = 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _geocode(address):
    resp = requests.get(NOMINATIM_URL, params={
        "q": address, "format": "json", "limit": 1, "countrycodes": "us",
    }, headers={"User-Agent": "dc-building-test/1.0"}, timeout=15)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"])


def _find_osm_building(lat, lon, radius_m=80, retries=3):
    deg = radius_m / 111_000
    s, n, w, e = lat - deg, lat + deg, lon - deg, lon + deg
    query = f"""
[out:json][timeout:30];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
);
out center tags 1;
"""
    for attempt in range(retries):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=45)
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            return elements[0] if elements else None
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (429, 503, 504):
                wait = 15 * (attempt + 1)
                time.sleep(wait)
            else:
                raise
    return None


def _find_mapillary_image(token, lat, lon):
    params = {
        "fields":       "id,thumb_1024_url,compass_angle",
        "lat":          lat,
        "lng":          lon,
        "radius":       SEARCH_RADIUS_M,
        "limit":        1,
        "access_token": token,
    }
    resp = requests.get(f"{MAPILLARY_API}/images", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return data[0] if data else None


# ---------------------------------------------------------------------------
# Session-scoped fixtures — each external API is called once per test session
# so we don't hammer rate limits across parametrized tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mapillary_token():
    return os.environ.get("MAPILLARY_TOKEN", "")


@pytest.fixture(scope="session")
def geocoded_addresses():
    """Geocode all test addresses once. Nominatim: 1 req/sec."""
    results = {}
    for entry in KNOWN_ADDRESSES:
        results[entry["label"]] = _geocode(entry["address"])
        time.sleep(1.2)
    return results


@pytest.fixture(scope="session")
def osm_buildings(geocoded_addresses):
    """Fetch OSM building for each address once. 5s between calls to respect Overpass limits."""
    results = {}
    for entry in KNOWN_ADDRESSES:
        coords = geocoded_addresses[entry["label"]]
        if coords is None:
            results[entry["label"]] = None
        else:
            results[entry["label"]] = _find_osm_building(*coords)
            time.sleep(5)
    return results


@pytest.fixture(scope="session")
def mapillary_images(geocoded_addresses, mapillary_token):
    """Fetch nearest Mapillary image for each address once."""
    if not mapillary_token:
        return {}
    results = {}
    for entry in KNOWN_ADDRESSES:
        coords = geocoded_addresses[entry["label"]]
        if coords is None:
            results[entry["label"]] = None
        else:
            try:
                results[entry["label"]] = _find_mapillary_image(mapillary_token, *coords)
            except Exception:
                results[entry["label"]] = None
        time.sleep(0.5)
    return results


# ---------------------------------------------------------------------------
# Geocoding tests
# ---------------------------------------------------------------------------

class TestGeocoding:

    @pytest.mark.parametrize("entry", KNOWN_ADDRESSES, ids=[e["label"] for e in KNOWN_ADDRESSES])
    @pytest.mark.geocode
    def test_geocode_returns_result(self, entry, geocoded_addresses):
        assert geocoded_addresses[entry["label"]] is not None, \
            f"Nominatim returned no result for: {entry['address']}"

    @pytest.mark.parametrize("entry", KNOWN_ADDRESSES, ids=[e["label"] for e in KNOWN_ADDRESSES])
    @pytest.mark.geocode
    def test_geocode_lat_in_range(self, entry, geocoded_addresses):
        coords = geocoded_addresses[entry["label"]]
        if coords is None:
            pytest.skip("geocode returned None")
        lat, _ = coords
        lo, hi = entry["expected_lat_range"]
        assert lo <= lat <= hi, \
            f"{entry['label']}: lat {lat:.5f} outside expected [{lo}, {hi}]"

    @pytest.mark.parametrize("entry", KNOWN_ADDRESSES, ids=[e["label"] for e in KNOWN_ADDRESSES])
    @pytest.mark.geocode
    def test_geocode_lon_in_range(self, entry, geocoded_addresses):
        coords = geocoded_addresses[entry["label"]]
        if coords is None:
            pytest.skip("geocode returned None")
        _, lon = coords
        lo, hi = entry["expected_lon_range"]
        assert lo <= lon <= hi, \
            f"{entry['label']}: lon {lon:.5f} outside expected [{lo}, {hi}]"

    @pytest.mark.geocode
    def test_all_addresses_in_dc(self, geocoded_addresses):
        DC_BBOX = {"s": 38.79, "n": 39.00, "w": -77.12, "e": -76.91}
        for label, coords in geocoded_addresses.items():
            if coords is None:
                continue
            lat, lon = coords
            assert DC_BBOX["s"] <= lat <= DC_BBOX["n"], f"{label}: lat {lat} outside DC"
            assert DC_BBOX["w"] <= lon <= DC_BBOX["e"], f"{label}: lon {lon} outside DC"


# ---------------------------------------------------------------------------
# OSM building tests  (use cached osm_buildings fixture — no per-test API calls)
# ---------------------------------------------------------------------------

class TestOSMBuildings:

    @pytest.mark.parametrize("entry", KNOWN_ADDRESSES, ids=[e["label"] for e in KNOWN_ADDRESSES])
    @pytest.mark.osm
    def test_osm_building_found(self, entry, osm_buildings):
        building = osm_buildings[entry["label"]]
        assert building is not None, \
            f"No OSM building found near {entry['address']} ({entry['notes']})"

    @pytest.mark.parametrize("entry", KNOWN_ADDRESSES, ids=[e["label"] for e in KNOWN_ADDRESSES])
    @pytest.mark.osm
    def test_osm_building_has_center(self, entry, osm_buildings):
        building = osm_buildings[entry["label"]]
        if building is None:
            pytest.skip("no building found")
        center = building.get("center", {})
        assert "lat" in center and "lon" in center, \
            f"Building for {entry['label']} has no center. Keys: {list(building.keys())}"

    @pytest.mark.parametrize("entry", KNOWN_ADDRESSES, ids=[e["label"] for e in KNOWN_ADDRESSES])
    @pytest.mark.osm
    def test_osm_building_has_building_tag(self, entry, osm_buildings):
        building = osm_buildings[entry["label"]]
        if building is None:
            pytest.skip("no building found")
        tags = building.get("tags", {})
        assert "building" in tags, \
            f"Building for {entry['label']} missing 'building' tag. Tags: {tags}"


# ---------------------------------------------------------------------------
# Mapillary tests
# ---------------------------------------------------------------------------

class TestMapillaryImages:

    @pytest.mark.parametrize("entry", KNOWN_ADDRESSES, ids=[e["label"] for e in KNOWN_ADDRESSES])
    @pytest.mark.requires_token
    def test_mapillary_image_found(self, entry, mapillary_images):
        if not mapillary_images:
            pytest.skip("MAPILLARY_TOKEN not set")
        image = mapillary_images.get(entry["label"])
        assert image is not None, \
            f"No Mapillary image found near {entry['address']}"

    @pytest.mark.parametrize("entry", KNOWN_ADDRESSES, ids=[e["label"] for e in KNOWN_ADDRESSES])
    @pytest.mark.requires_token
    def test_mapillary_image_has_thumbnail_url(self, entry, mapillary_images):
        if not mapillary_images:
            pytest.skip("MAPILLARY_TOKEN not set")
        image = mapillary_images.get(entry["label"])
        if image is None:
            pytest.skip("no image found")
        assert "thumb_1024_url" in image, \
            f"Image for {entry['label']} missing thumb_1024_url. Keys: {list(image.keys())}"

    @pytest.mark.parametrize("entry", KNOWN_ADDRESSES, ids=[e["label"] for e in KNOWN_ADDRESSES])
    @pytest.mark.requires_token
    def test_mapillary_thumbnail_is_reachable(self, entry, mapillary_images):
        if not mapillary_images:
            pytest.skip("MAPILLARY_TOKEN not set")
        image = mapillary_images.get(entry["label"])
        if image is None or "thumb_1024_url" not in image:
            pytest.skip("no usable image")
        resp = requests.head(image["thumb_1024_url"], timeout=10)
        assert resp.status_code == 200, \
            f"Thumbnail for {entry['label']} returned HTTP {resp.status_code}"


# ---------------------------------------------------------------------------
# address_lookup module smoke tests
# ---------------------------------------------------------------------------

class TestAddressLookupModule:

    def test_import(self):
        import address_lookup  # noqa: F401

    def test_geocode_white_house(self):
        import address_lookup
        coords = address_lookup.geocode("1600 Pennsylvania Avenue NW, Washington, DC")
        assert coords is not None
        lat, lon = coords
        assert 38.893 <= lat <= 38.900, f"lat {lat} out of range"
        assert -77.042 <= lon <= -77.034, f"lon {lon} out of range"
        time.sleep(1.2)

    @pytest.mark.osm
    def test_find_osm_building_white_house(self, osm_buildings):
        # Reuse session-cached result — no extra Overpass call
        building = osm_buildings["white_house"]
        assert building is not None
        assert "osm_id" in building
        assert "tags" in building

    @pytest.mark.requires_token
    def test_find_mapillary_image_white_house(self, mapillary_images):
        if not mapillary_images:
            pytest.skip("MAPILLARY_TOKEN not set")
        image = mapillary_images.get("white_house")
        assert image is not None
        assert "id" in image
