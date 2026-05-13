#!/usr/bin/env python3
"""
Phase 1: Road-to-Geohash Extraction
=====================================
Extracts target roads (E-5, TEM, etc.) from OpenStreetMap via OSMnx,
converts node coordinates to 6-character geohashes, validates them
against IBB traffic density CSV data, and exports road_mapping.json.

Outputs:
    - road_mapping.json
"""

import logging
import time
import json
import glob
import os
from datetime import datetime, timezone

import duckdb
import osmnx as ox
import pygeohash as pgh

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
random_state = 42
START_TIME = time.time()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
CSV_PATTERN = os.path.join(DATA_DIR, "traffic_density_*.csv")
OUTPUT_FILE = "road_mapping.json"

# Istanbul bounding box for coordinate validation
LAT_MIN, LAT_MAX = 40.8, 41.6
LON_MIN, LON_MAX = 27.5, 30.0

# Road definitions with OSMnx custom filters
# Each road has a list of filters tried sequentially until one succeeds with edges.
ROADS = {
    "TEM": {
        "display_name": "TEM / O-2 Otoyolu",
        "filters": [
            '["ref"~"O.2|O 2|E.80|E 80"]',
            '["name"~"TEM Otoyolu|TEM"]',
        ],
    },
    "Buyukdere_Cad": {
        "display_name": "Büyükdere Caddesi",
        "filters": [
            '["name"~"Büyükdere Caddesi|Büyükdere Cd"]',
        ],
    },
    "Sahil_Yolu_Avrupa": {
        "display_name": "Sahil Yolu (Avrupa Yakası)",
        "filters": [
            '["name"~"Kennedy Cad|Kennedy Cd|Rauf Orbay"]',
        ],
    },
}

MAX_RETRIES = 2
RETRY_DELAY = 2  # seconds


# ---------------------------------------------------------------------------
# Step 0: Setup & IBB Validation
# ---------------------------------------------------------------------------
def load_ibb_geohashes() -> set:
    """Load unique geohashes from IBB CSV files using DuckDB. Returns empty set if no CSVs found."""
    csv_files = sorted(glob.glob(CSV_PATTERN))
    if not csv_files:
        logger.warning("No CSV files found at %s — IBB validation will be skipped.", CSV_PATTERN)
        return set()

    logger.info("Found %d IBB CSV files.", len(csv_files))

    con = duckdb.connect(database=":memory:")
    try:
        # Get unique geohashes efficiently
        query = f"SELECT DISTINCT GEOHASH FROM read_csv_auto('{CSV_PATTERN}', header=true)"
        result = con.execute(query).fetchall()
        ibb_geohashes = {row[0] for row in result if row[0] is not None}
        logger.info("Loaded %d unique geohashes from IBB data.", len(ibb_geohashes))

        # Coordinate validation — sample first file to check lat/lon swap
        sample_query = f"""
            SELECT LATITUDE, LONGITUDE
            FROM read_csv_auto('{csv_files[0]}', header=true)
            LIMIT 50000
        """
        sample = con.execute(sample_query).fetchall()

        total_rows = len(sample)
        invalid_count = 0
        for lat, lon in sample:
            if lat is not None and lon is not None:
                lat_ok = LAT_MIN <= lat <= LAT_MAX
                lon_ok = LON_MIN <= lon <= LON_MAX
                if not lat_ok or not lon_ok:
                    invalid_count += 1

        invalid_ratio = invalid_count / total_rows if total_rows > 0 else 0
        logger.info(
            "Coordinate validation (sample=%d): %d invalid (%.1f%%).",
            total_rows, invalid_count, invalid_ratio * 100,
        )
        if invalid_ratio > 0.30:
            logger.warning(
                "More than 30%% of sampled rows have out-of-range coordinates. "
                "LATITUDE and LONGITUDE columns may be swapped! (invalid_ratio=%.1f%%)",
                invalid_ratio * 100,
            )
    finally:
        con.close()

    return ibb_geohashes


# ---------------------------------------------------------------------------
# Step 1: OSMnx Road Extraction
# ---------------------------------------------------------------------------
def _geohashes_from_geometry(geom) -> set:
    """Extract geohashes by sampling coordinates along a geometry (Point or LineString)."""
    geohashes = set()

    if geom.geom_type == "Point":
        gh = pgh.encode(latitude=geom.y, longitude=geom.x, precision=6)
        geohashes.add(gh)
    elif geom.geom_type == "LineString":
        coords = list(geom.coords)
        for lon, lat in coords:
            gh = pgh.encode(latitude=lat, longitude=lon, precision=6)
            geohashes.add(gh)
        # Interpolate additional points along the line (~every 0.005 degrees ≈ 500m)
        length = geom.length
        if length > 0:
            step = 0.005  # ~500m at Istanbul latitude
            num_points = max(int(length / step), 1)
            for i in range(num_points + 1):
                fraction = i / num_points
                point = geom.interpolate(fraction, normalized=True)
                gh = pgh.encode(latitude=point.y, longitude=point.x, precision=6)
                geohashes.add(gh)
    elif geom.geom_type == "MultiLineString":
        for line in geom.geoms:
            geohashes.update(_geohashes_from_geometry(line))

    return geohashes


def extract_road_geohashes(road_name: str, filters: list) -> set:
    """Extract geohashes for a single road using OSMnx with retry logic.

    Tries each filter in the list sequentially. Uses both nodes and edge
    geometries for comprehensive geohash coverage. Interpolates points
    along edge LineStrings to fill spatial gaps. Accumulates geohashes
    from all successful filters.
    """
    logger.info("Extracting road: %s (%d filters to try)", road_name, len(filters))
    all_geohashes = set()

    for filt in filters:
        logger.info("  Trying filter: %s", filt)
        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                G = ox.graph_from_place(
                    "Istanbul, Turkey",
                    custom_filter=filt,
                    network_type="drive",
                )

                geohashes = set()

                # Extract from nodes
                nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
                for _, row in nodes_gdf.iterrows():
                    gh = pgh.encode(latitude=row.geometry.y, longitude=row.geometry.x, precision=6)
                    geohashes.add(gh)
                node_gh_count = len(geohashes)

                # Extract from edges (full geometry with interpolation)
                try:
                    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
                    edge_count = len(edges_gdf)
                    for _, row in edges_gdf.iterrows():
                        if row.geometry is not None:
                            geohashes.update(_geohashes_from_geometry(row.geometry))
                    logger.info(
                        "    Filter OK: %d nodes (%d gh) + %d edges → %d unique geohashes.",
                        len(nodes_gdf), node_gh_count, edge_count, len(geohashes),
                    )
                except Exception as edge_err:
                    logger.warning("    No edges for filter (nodes-only): %s", edge_err)
                    logger.info(
                        "    Filter partial: %d nodes → %d geohashes (no edges).",
                        len(nodes_gdf), len(geohashes),
                    )

                all_geohashes.update(geohashes)
                success = True
                break  # This filter succeeded, move to next

            except Exception as e:
                logger.error(
                    "    Attempt %d/%d failed for filter '%s': %s",
                    attempt, MAX_RETRIES, filt, e,
                )
                if attempt < MAX_RETRIES:
                    logger.info("    Retrying in %d seconds...", RETRY_DELAY)
                    time.sleep(RETRY_DELAY)

        if not success:
            logger.warning("  All attempts failed for filter '%s'.", filt)

    logger.info("  Road '%s' total: %d unique geohashes.", road_name, len(all_geohashes))
    return all_geohashes



# ---------------------------------------------------------------------------
# Step 2: IBB Validation
# ---------------------------------------------------------------------------
def validate_against_ibb(osm_geohashes: set, ibb_geohashes: set, road_name: str) -> tuple:
    """Validate OSM geohashes against IBB data. Returns (validated_set, metadata_dict)."""
    if not ibb_geohashes:
        logger.info("  IBB data not available — using all %d OSM geohashes for '%s'.", len(osm_geohashes), road_name)
        return osm_geohashes, {
            "raw_osm_geohash_count": len(osm_geohashes),
            "ibb_validated_count": len(osm_geohashes),
            "dropped_count": 0,
            "coverage_ratio": 1.0,
            "geohash_precision": 6,
        }

    validated = osm_geohashes.intersection(ibb_geohashes)
    dropped = osm_geohashes - validated
    coverage_ratio = len(validated) / len(osm_geohashes) if osm_geohashes else 0.0

    logger.info(
        "  Road '%s': raw_osm=%d, validated=%d, dropped=%d, coverage=%.2f",
        road_name, len(osm_geohashes), len(validated), len(dropped), coverage_ratio,
    )

    return validated, {
        "raw_osm_geohash_count": len(osm_geohashes),
        "ibb_validated_count": len(validated),
        "dropped_count": len(dropped),
        "coverage_ratio": round(coverage_ratio, 4),
        "geohash_precision": 6,
    }


# ---------------------------------------------------------------------------
# Step 3: Export road_mapping.json
# ---------------------------------------------------------------------------
def export_road_mapping(roads_data: dict, ibb_available: bool) -> None:
    """Write road_mapping.json to disk."""
    output = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "methodology": "osmnx_extraction + ibb_geohash_validation",
        "ibb_data_available": ibb_available,
        "roads": roads_data,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(OUTPUT_FILE)
    logger.info("Exported %s (%.1f KB).", OUTPUT_FILE, file_size / 1024)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Phase 1: Road-to-Geohash Extraction — START")
    logger.info("random_state = %d", random_state)
    logger.info("=" * 60)

    # Step 0: Load IBB geohashes
    logger.info("--- Step 0: Loading IBB geohashes ---")
    ibb_geohashes = load_ibb_geohashes()
    ibb_available = len(ibb_geohashes) > 0

    # Step 1 & 2: Extract and validate each road
    logger.info("--- Step 1 & 2: Road extraction and validation ---")
    roads_data = {}
    total_geohashes = 0

    for road_name, road_info in ROADS.items():
        osm_geohashes = extract_road_geohashes(road_name, road_info["filters"])

        if not osm_geohashes:
            logger.warning("No geohashes for '%s' — skipping.", road_name)
            continue

        validated, metadata = validate_against_ibb(osm_geohashes, ibb_geohashes, road_name)

        if not validated:
            logger.warning("No validated geohashes for '%s' — skipping.", road_name)
            continue

        roads_data[road_name] = {
            "display_name": road_info["display_name"],
            "osm_filters": road_info["filters"],
            "metadata": metadata,
            "geohashes": sorted(validated),
        }
        total_geohashes += len(validated)

    if not roads_data:
        logger.error("No roads extracted successfully! road_mapping.json will be empty.")

    # Step 3: Export
    logger.info("--- Step 3: Exporting road_mapping.json ---")
    export_road_mapping(roads_data, ibb_available)

    # Summary
    elapsed = time.time() - START_TIME
    logger.info("=" * 60)
    logger.info("Phase 1 COMPLETE in %.1f seconds.", elapsed)
    logger.info("Roads extracted: %d / %d", len(roads_data), len(ROADS))
    logger.info("Total unique geohashes across all roads: %d", total_geohashes)
    for rname, rdata in roads_data.items():
        logger.info(
            "  %s: %d geohashes (coverage: %.1f%%)",
            rname,
            len(rdata["geohashes"]),
            rdata["metadata"]["coverage_ratio"] * 100,
        )
    logger.info("Output file: %s", OUTPUT_FILE)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
