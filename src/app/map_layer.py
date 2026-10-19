"""
Map context layer — SaayaShift

Loads the OSM Gold snapshot and renders a Folium map for the
selected work zone. Overlays roads, parks, and water features.

IMPORTANT: This map shows geographic context only.
OSM features are NOT measurements of shade, temperature,
worker presence, or safety. Local coverage may be incomplete.

Attribution: © OpenStreetMap contributors (ODbL)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
RAW_OSM_DIR = ROOT / "data" / "raw" / "osm"
CONFIG_PATH = ROOT / "config" / "karachi.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DISCLAIMER = (
    "Geographic context only — OSM features are not measurements of shade, "
    "temperature, worker presence, or safety. Local coverage may be incomplete. "
    "© OpenStreetMap contributors (ODbL)."
)

# Feature styling
STYLE_MAP = {
    "highway": {"color": "#888888", "weight": 1, "opacity": 0.6},
    "park":    {"color": "#228B22", "weight": 2, "opacity": 0.7, "fillColor": "#90EE90", "fillOpacity": 0.3},
    "water":   {"color": "#1E90FF", "weight": 2, "opacity": 0.8, "fillColor": "#ADD8E6", "fillOpacity": 0.4},
    "hospital":{"color": "#FF0000", "weight": 2},
    "industrial": {"color": "#FF8C00", "weight": 1, "fillColor": "#FFD700", "fillOpacity": 0.2},
    "default": {"color": "#666666", "weight": 1, "opacity": 0.5},
}


def load_latest_osm_snapshot() -> Optional[dict]:
    """Load the most recently fetched OSM snapshot JSON."""
    files = sorted(RAW_OSM_DIR.glob("karachi_snapshot_*.json"))
    if not files:
        log.warning("No OSM snapshot found in %s. Run `make data` first.", RAW_OSM_DIR)
        return None
    with open(files[-1]) as f:
        return json.load(f)


def build_map(
    center_lat: float = 24.8607,
    center_lon: float = 67.0011,
    zoom: int = 12,
    osm_data: Optional[dict] = None,
) -> str:
    """
    Build and return a Folium map as an HTML string.
    Adds OSM features as overlays with the geographic-context disclaimer.
    """
    try:
        import folium
    except ImportError:
        return "<p>folium not installed. Run: pip install folium</p>"

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        attr="© <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap contributors</a>",
    )

    # Work zone marker
    folium.Marker(
        location=[center_lat, center_lon],
        popup="Selected work zone (approximate grid cell)",
        icon=folium.Icon(color="red", icon="hard-hat", prefix="fa"),
        tooltip="Work zone",
    ).add_to(m)

    # OSM overlay
    if osm_data:
        elements = osm_data.get("elements", [])
        nodes = {e["id"]: e for e in elements if e["type"] == "node"}
        added = 0

        for element in elements:
            if element["type"] != "way":
                continue
            tags = element.get("tags", {})
            coords = []
            for nid in element.get("nodes", []):
                node = nodes.get(nid)
                if node:
                    coords.append([node["lat"], node["lon"]])

            if not coords:
                continue

            # Determine style
            if "highway" in tags:
                style = STYLE_MAP["highway"]
                tooltip = f"Road: {tags.get('name', tags.get('highway', 'road'))}"
            elif tags.get("leisure") == "park":
                style = STYLE_MAP["park"]
                tooltip = f"Park: {tags.get('name', 'park')}"
            elif tags.get("natural") == "water":
                style = STYLE_MAP["water"]
                tooltip = f"Water: {tags.get('name', 'water body')}"
            elif tags.get("landuse") == "industrial":
                style = STYLE_MAP["industrial"]
                tooltip = f"Industrial area: {tags.get('name', '')}"
            else:
                continue  # skip unlabelled features

            folium.PolyLine(
                locations=coords,
                tooltip=tooltip,
                **{k: v for k, v in style.items() if k in ("color", "weight", "opacity")},
            ).add_to(m)
            added += 1

        # Node markers (hospitals, fire stations)
        for element in elements:
            if element["type"] != "node":
                continue
            tags = element.get("tags", {})
            lat = element.get("lat")
            lon = element.get("lon")
            if not (lat and lon):
                continue

            if tags.get("amenity") == "hospital":
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.7,
                    tooltip=f"Hospital: {tags.get('name', 'hospital')}",
                ).add_to(m)
            elif tags.get("amenity") == "fire_station":
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    color="orange",
                    fill=True,
                    fill_color="orange",
                    fill_opacity=0.7,
                    tooltip=f"Fire station: {tags.get('name', 'fire station')}",
                ).add_to(m)

        log.info("Added %d OSM features to map", added)

    # Disclaimer box
    disclaimer_html = f"""
    <div style="
        position: fixed; bottom: 40px; left: 10px; z-index: 1000;
        background: rgba(255,255,255,0.9); padding: 8px 12px;
        border-radius: 4px; border: 1px solid #aaa;
        font-size: 11px; max-width: 380px; line-height: 1.4;">
        ⚠️ {DISCLAIMER}
    </div>
    """
    m.get_root().html.add_child(folium.Element(disclaimer_html))

    return m._repr_html_()


def get_map_html(lat: float = 24.8607, lon: float = 67.0011) -> str:
    """Convenience wrapper: load OSM data and return map HTML."""
    osm_data = load_latest_osm_snapshot()
    return build_map(center_lat=lat, center_lon=lon, osm_data=osm_data)


if __name__ == "__main__":
    html = get_map_html()
    out = Path("map_preview.html")
    out.write_text(html)
    print(f"Map saved to {out}")
