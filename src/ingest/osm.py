"""
OpenStreetMap ingest via Overpass API — Bronze layer
Fetches a bounded Karachi-area snapshot of roads, parks, water features,
and key facilities. Saves raw response unchanged and records provenance
(osm_base timestamp, fetch time, SHA-256) in docs/manifest.csv.

Attribution: © OpenStreetMap contributors (ODbL)
Docs: https://wiki.openstreetmap.org/wiki/Overpass_API
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "karachi.yaml"
RAW_DIR = ROOT / "data" / "raw" / "osm"
MANIFEST_PATH = ROOT / "docs" / "manifest.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _append_manifest(row: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not MANIFEST_PATH.exists() or MANIFEST_PATH.stat().st_size == 0
    with open(MANIFEST_PATH, "a") as f:
        if write_header:
            f.write("source,url,fetch_time_utc,sha256,licence,osm_base,notes\n")
        notes = row.get("notes", "").replace(",", ";")
        f.write(
            f"{row['source']},{row['url']},{row['fetch_time_utc']},"
            f"{row['sha256']},{row['licence']},{row.get('osm_base', '')},{notes}\n"
        )


def build_overpass_query(bbox: dict) -> str:
    """
    Build an Overpass QL query for key Karachi features.
    bbox keys: south, west, north, east
    """
    s, w, n, e = bbox["south"], bbox["west"], bbox["north"], bbox["east"]
    bbox_str = f"{s},{w},{n},{e}"

    query = f"""
[out:json][timeout:120];
(
  // Roads
  way["highway"]({bbox_str});
  // Parks and green spaces
  way["leisure"="park"]({bbox_str});
  relation["leisure"="park"]({bbox_str});
  // Water
  way["natural"="water"]({bbox_str});
  relation["natural"="water"]({bbox_str});
  // Hospitals
  node["amenity"="hospital"]({bbox_str});
  way["amenity"="hospital"]({bbox_str});
  // Fire stations
  node["amenity"="fire_station"]({bbox_str});
  // Industrial areas (outdoor work context)
  way["landuse"="industrial"]({bbox_str});
  // Construction sites
  way["landuse"="construction"]({bbox_str});
);
out body;
>;
out skel qt;
""".strip()
    return query


def fetch_osm_snapshot(cfg: dict) -> Path:
    """
    Fetch a bounded OSM snapshot for Karachi.
    Tries all known Overpass mirrors. If all fail, writes a clearly-labeled
    stub file so the pipeline can continue end-to-end.
    """
    bbox = cfg["osm_bbox"]
    query = build_overpass_query(bbox)

    log.info("Fetching OSM snapshot for Karachi bbox: %s", bbox)
    fetch_time = datetime.now(timezone.utc).isoformat()

    last_error = None
    for mirror in OVERPASS_MIRRORS:
        try:
            log.info("Trying Overpass mirror: %s", mirror)
            resp = requests.post(
                mirror,
                data={"data": query},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=180,
            )
            resp.raise_for_status()
            raw_bytes = resp.content
            data = json.loads(raw_bytes)
            osm_base = data.get("osm3s", {}).get("timestamp_osm_base", "unknown")
            element_count = len(data.get("elements", []))

            sha = _sha256(raw_bytes)
            timestamp = fetch_time.replace(":", "-").replace("+", "Z")[:19]
            out_path = RAW_DIR / f"karachi_snapshot_{timestamp}.json"
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(raw_bytes)
            query_path = RAW_DIR / f"karachi_query_{timestamp}.txt"
            query_path.write_text(query)

            log.info("Saved OSM snapshot → %s (%d elements, osm_base=%s)", out_path.name, element_count, osm_base)
            if element_count < 1000:
                log.warning("OSM element count %d below expected ≥1,000", element_count)

            _append_manifest({
                "source": "openstreetmap_overpass",
                "url": mirror,
                "fetch_time_utc": fetch_time,
                "sha256": sha,
                "licence": "ODbL",
                "osm_base": osm_base,
                "notes": f"karachi_bbox={bbox} elements={element_count} query_file={query_path.name}",
            })
            return out_path

        except Exception as e:
            log.warning("Mirror %s failed: %s", mirror, e)
            last_error = e

    # All mirrors failed — write a clearly-labeled stub for pipeline continuity
    log.error("All Overpass mirrors failed (%s). Writing labeled stub for pipeline continuity.", last_error)
    stub = {
        "_STUB": True,
        "_STUB_REASON": f"All Overpass mirrors unreachable at fetch time: {last_error}",
        "_FETCH_TIME": fetch_time,
        "elements": _karachi_stub_elements(),
        "osm3s": {"timestamp_osm_base": "STUB_NOT_REAL_OSM_DATA"},
    }
    raw_bytes = json.dumps(stub, indent=2).encode()
    sha = _sha256(raw_bytes)
    timestamp = fetch_time.replace(":", "-").replace("+", "Z")[:19]
    out_path = RAW_DIR / f"karachi_snapshot_{timestamp}_STUB.json"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw_bytes)

    _append_manifest({
        "source": "openstreetmap_overpass_STUB",
        "url": "STUB_ALL_MIRRORS_FAILED",
        "fetch_time_utc": fetch_time,
        "sha256": sha,
        "licence": "ODbL_STUB",
        "osm_base": "STUB",
        "notes": "SYNTHETIC STUB — all Overpass mirrors unreachable. Replace with real OSM fetch when network allows.",
    })
    log.warning("OSM STUB written. Status: BLOCKED — replace when Overpass is accessible.")
    return out_path


def _karachi_stub_elements() -> list:
    """
    Minimal labeled stub elements for known Karachi locations.
    SYNTHETIC DATA — clearly labeled. Not used for training or evaluation.
    For geographic context display only, until real OSM fetch succeeds.
    """
    return [
        {"type": "node", "id": -1, "lat": 24.8607, "lon": 67.0011,
         "tags": {"name": "Karachi City Centre (STUB)", "note": "SYNTHETIC_STUB"}},
        {"type": "node", "id": -2, "lat": 24.9056, "lon": 67.0822,
         "tags": {"amenity": "hospital", "name": "Jinnah Postgraduate Medical Centre (STUB)", "note": "SYNTHETIC_STUB"}},
        {"type": "node", "id": -3, "lat": 24.8615, "lon": 67.0099,
         "tags": {"amenity": "hospital", "name": "Civil Hospital Karachi (STUB)", "note": "SYNTHETIC_STUB"}},
        {"type": "node", "id": -4, "lat": 24.8553, "lon": 67.0180,
         "tags": {"leisure": "park", "name": "Frere Hall Gardens (STUB)", "note": "SYNTHETIC_STUB"}},
    ]


def count_elements(json_path: Path) -> int:
    """Return number of OSM elements in a raw snapshot file."""
    with open(json_path) as f:
        data = json.load(f)
    return len(data.get("elements", []))


def main() -> None:
    cfg = load_config()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    snap_path = fetch_osm_snapshot(cfg)
    count = count_elements(snap_path)
    log.info("OSM elements fetched: %d (expected ≥1,000)", count)
    log.info("OSM ingest complete.")


if __name__ == "__main__":
    main()
