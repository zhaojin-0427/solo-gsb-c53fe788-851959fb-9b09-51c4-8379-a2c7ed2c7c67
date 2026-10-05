"""Point identify (hit-test) on top of the tile compilation pipeline.

The identify endpoint answers "which source features are visible at this
pixel of this tile?" without touching the tile pipeline or its bytes: every
source feature is compiled *individually* through
:func:`app.tiler.compile_feature_geometry` -- the exact projection /
8 px-buffer clip / 0.5 px simplify / 4096 quantisation chain that produces
the MVT bytes -- and the click is adjudicated against the resulting
*visible* geometry:

* compiling per source feature (rather than decoding the encoded MVT, whose
  deterministic re-sorting and uint64-only feature IDs would lose the
  mapping) preserves the 0-based source index, so ID-less, string-ID and
  duplicate-ID features are reported individually and never merged;
* the click point is kept at full floating-point precision on the 4096 grid
  (1 px = 16 units) and is never snapped to the integer grid, so sub-pixel
  click positions are honoured;
* a feature whose geometry does not survive quantisation (``None`` from the
  pipeline) is not visible in the tile and can never be a hit;
* hits are adjudicated by the distance from the visible geometry to the
  click point: a polygon containing the click reports distance 0, while
  lines and points (and any geometry edge, including the 8 px buffer rim)
  must lie within the circular ``radius_px``.
"""
from __future__ import annotations

from typing import Any

from shapely.geometry import Point

from .config import settings
from .tiler import compile_feature_geometry

# Tile-grid units per screen pixel (4096 / 256 = 16).
UNITS_PER_PX = settings.tile_extent / settings.tile_size_px

# Slack for floating-point noise when a distance lands exactly on the radius
# boundary (a few ulps of a 4096-unit coordinate; far below one sub-pixel).
_EPS_UNITS = 1e-9

# Reported distances are rounded to 1e-6 px: clean JSON, deterministic sort.
_DISTANCE_DECIMALS = 6


def identify_features(
    features: list[dict[str, Any]],
    z: int,
    x: int,
    y: int,
    px: float,
    py: float,
    radius_px: float,
) -> list[dict[str, Any]]:
    """Return the hits for a click at (``px``, ``py``) tile pixels.

    ``features`` are the parsed source features of one revision *in source
    document order*; the 0-based position in that list is reported as
    ``source_index``.  Each hit carries the source ID, the source properties
    and the pixel distance between the visible geometry and the click point.
    Hits are sorted by ``distance_px`` and then by ``source_index``.
    """
    click = Point(px * UNITS_PER_PX, py * UNITS_PER_PX)
    radius_units = radius_px * UNITS_PER_PX

    hits: list[dict[str, Any]] = []
    for source_index, feature in enumerate(features):
        geom = compile_feature_geometry(feature["geometry"], z, x, y)
        if geom is None or geom.is_empty:
            # Quantised away or outside tile+buffer: not visible, never a hit.
            continue
        distance_units = geom.distance(click)
        if distance_units > radius_units + _EPS_UNITS:
            continue
        hits.append(
            {
                "source_index": source_index,
                "id": feature.get("id"),
                "properties": feature.get("properties") or {},
                "distance_px": round(
                    distance_units / UNITS_PER_PX, _DISTANCE_DECIMALS
                ),
            }
        )
    hits.sort(key=lambda hit: (hit["distance_px"], hit["source_index"]))
    return hits
