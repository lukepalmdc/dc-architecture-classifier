"""
pytest configuration: register custom marks so -m filtering works without warnings.
"""
import sys
import os

# Allow tests to import address_lookup from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pytest_configure(config):
    config.addinivalue_line("markers", "geocode: tests that call Nominatim geocoding API")
    config.addinivalue_line("markers", "osm: tests that call the Overpass/OSM API")
    config.addinivalue_line("markers", "requires_token: tests that need MAPILLARY_TOKEN set")
