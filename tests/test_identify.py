"""Point identify endpoint: hit adjudication, ordering and validation.

Test collection layout at z=0 (one world tile, 1 px = 16 grid units):

* idx 0: Point (0, 0)            -> pixel (128, 128), integer id 1
* idx 1: Point (11.25, 0)        -> pixel (136, 128), id 2
* idx 2: LineString on equator   -> horizontal line at py 128, string id
* idx 3: Point (0, 0)            -> no id
* idx 4: Point (0, 0)            -> duplicate id 1
* idx 5: Polygon lng +-45/lat+-20 -> covers the click area, id 9
"""
import json

import mapbox_vector_tile
import pytest

IDENTIFY_COLLECTION = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "id": 1,
            "properties": {"name": "origin"},
            "geometry": {"type": "Point", "coordinates": [0, 0]},
        },
        {
            "type": "Feature",
            "id": 2,
            "properties": {"name": "east"},
            "geometry": {"type": "Point", "coordinates": [11.25, 0]},
        },
        {
            "type": "Feature",
            "id": "line",
            "properties": {"kind": "equator"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[-45, 0], [45, 0]],
            },
        },
        {
            "type": "Feature",
            "properties": {"name": "no-id"},
            "geometry": {"type": "Point", "coordinates": [0, 0]},
        },
        {
            "type": "Feature",
            "id": 1,
            "properties": {"name": "duplicate"},
            "geometry": {"type": "Point", "coordinates": [0, 0]},
        },
        {
            "type": "Feature",
            "id": 9,
            "properties": {"name": "box"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[-45, -20], [45, -20], [45, 20], [-45, 20], [-45, -20]]
                ],
            },
        },
    ],
}


@pytest.fixture
def identify_ds(client):
    """Publish the identify collection; return its dataset id."""
    resp = client.post(
        "/datasets?layer=demo",
        content=json.dumps(IDENTIFY_COLLECTION).encode("utf-8"),
        headers={"content-type": "application/geo+json"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _identify(client, ds, query, z=0, x=0, y=0):
    return client.get(f"/datasets/{ds}/tiles/{z}/{x}/{y}/identify?{query}")


# ---------------------------------------------------------------------------
# Hit adjudication
# ---------------------------------------------------------------------------


def test_identify_exact_click_radius_zero(identify_ds, client):
    resp = _identify(client, identify_ds, "px=128&py=128&radius_px=0")
    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == 1
    assert body["z"] == 0 and body["x"] == 0 and body["y"] == 0
    # Points at (0,0), the equator line, the no-id point, the duplicate-id
    # point and the containing polygon all sit exactly under the click.
    assert [h["source_index"] for h in body["hits"]] == [0, 2, 3, 4, 5]
    assert all(h["distance_px"] == 0.0 for h in body["hits"])
    # The point 8 px away does not hit with radius 0.
    assert 1 not in [h["source_index"] for h in body["hits"]]


def test_identify_reports_source_id_and_properties(identify_ds, client):
    body = _identify(client, identify_ds, "px=128&py=128&radius_px=0").json()
    by_index = {h["source_index"]: h for h in body["hits"]}
    assert by_index[0]["id"] == 1
    assert by_index[0]["properties"] == {"name": "origin"}
    # String IDs are reported (not merged, not dropped).
    assert by_index[2]["id"] == "line"
    assert by_index[2]["properties"] == {"kind": "equator"}
    # ID-less feature: id is null, still its own hit.
    assert by_index[3]["id"] is None
    assert by_index[3]["properties"] == {"name": "no-id"}
    # Duplicate IDs are not merged: two hits share id 1.
    assert by_index[4]["id"] == 1
    assert by_index[4]["properties"] == {"name": "duplicate"}


def test_identify_sorts_by_distance_then_source_index(identify_ds, client):
    # Click 4 px above the equator: polygon contains it (0.0); the origin
    # points and the line are exactly 4.0 px away; the east point is farther
    # than 8 px and must not hit.
    body = _identify(client, identify_ds, "px=128&py=124&radius_px=8").json()
    hits = body["hits"]
    assert [h["source_index"] for h in hits] == [5, 0, 2, 3, 4]
    assert hits[0]["distance_px"] == 0.0
    assert all(h["distance_px"] == 4.0 for h in hits[1:])


def test_identify_radius_boundary_is_inclusive(identify_ds, client):
    # The east point sits exactly 8.0 px from the click.
    far = _identify(client, identify_ds, "px=128&py=128&radius_px=7.9").json()
    assert 1 not in [h["source_index"] for h in far["hits"]]
    edge = _identify(client, identify_ds, "px=128&py=128&radius_px=8").json()
    hits = {h["source_index"]: h for h in edge["hits"]}
    assert hits[1]["distance_px"] == 8.0
    # ... and it sorts last.
    assert [h["source_index"] for h in edge["hits"]][-1] == 1


def test_identify_line_distance_within_radius(identify_ds, client):
    # 4 px off the equator: the line hits only once the radius reaches 4 px.
    small = _identify(client, identify_ds, "px=128&py=124&radius_px=3.9").json()
    assert [h["source_index"] for h in small["hits"]] == [5]
    exact = _identify(client, identify_ds, "px=128&py=124&radius_px=4").json()
    assert 2 in [h["source_index"] for h in exact["hits"]]


def test_identify_subpixel_click_position(identify_ds, client):
    # Half a pixel next to the origin point: only a radius >= 0.5 hits it,
    # and the reported distance keeps the sub-pixel precision.
    resp = _identify(client, identify_ds, "px=128.5&py=128&radius_px=0.5")
    assert resp.status_code == 200
    hits = {h["source_index"]: h for h in resp.json()["hits"]}
    assert hits[0]["distance_px"] == 0.5
    too_tight = _identify(
        client, identify_ds, "px=128.5&py=128&radius_px=0.4"
    ).json()
    assert 0 not in [h["source_index"] for h in too_tight["hits"]]


def test_identify_empty_when_nothing_within_radius(identify_ds, client):
    body = _identify(client, identify_ds, "px=10&py=10&radius_px=0").json()
    assert body["hits"] == []


def test_identify_polygon_inside_is_zero(identify_ds, client):
    # Deep inside the polygon but far from every other geometry.
    body = _identify(client, identify_ds, "px=160&py=120&radius_px=0").json()
    assert [h["source_index"] for h in body["hits"]] == [5]
    assert body["hits"][0]["distance_px"] == 0.0


# ---------------------------------------------------------------------------
# Quantised-away geometries never hit
# ---------------------------------------------------------------------------


def test_identify_quantised_away_geometry_never_hits(client):
    eps = 1e-7  # ~0.01 mm triangle: collapses to one grid point at any zoom
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": 1,
                "properties": {"name": "tiny"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[-eps, -eps], [eps, -eps], [0, eps], [-eps, -eps]]
                    ],
                },
            },
            {
                "type": "Feature",
                "id": 2,
                "properties": {"name": "origin"},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            },
        ],
    }
    ds = client.post(
        "/datasets?layer=demo",
        content=json.dumps(collection).encode("utf-8"),
        headers={"content-type": "application/geo+json"},
    ).json()["id"]
    body = _identify(client, ds, "px=128&py=128&radius_px=8").json()
    # The collapsed polygon is not visible in the tile and cannot hit; the
    # co-located point can.
    assert [h["source_index"] for h in body["hits"]] == [1]
    assert body["hits"][0]["id"] == 2


# ---------------------------------------------------------------------------
# Buffer rim adjudication
# ---------------------------------------------------------------------------


def test_identify_buffer_rim_uses_same_radius(client):
    # A point just east of tile (1,0,0)'s edge: inside the 8 px buffer, so
    # it is compiled into the tile, ~1.72 px outside the 256 px area.
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": 7,
                "properties": {"name": "rim"},
                "geometry": {"type": "Point", "coordinates": [0.5, 0.5]},
            }
        ],
    }
    ds = client.post(
        "/datasets?layer=demo",
        content=json.dumps(collection).encode("utf-8"),
        headers={"content-type": "application/geo+json"},
    ).json()["id"]

    # The feature is visible in the tile bytes (via the buffer) ...
    tile = client.get(f"/datasets/{ds}/tiles/1/0/0.mvt")
    decoded = mapbox_vector_tile.decode(
        tile.content, default_options={"y_coord_down": True}
    )
    assert len(decoded["demo"]["features"]) == 1

    # ... and the click at the tile corner hits it only within the radius.
    miss = _identify(client, ds, "px=255&py=255&radius_px=1", z=1, x=0, y=0)
    assert miss.json()["hits"] == []
    hit = _identify(client, ds, "px=255&py=255&radius_px=2", z=1, x=0, y=0)
    hits = hit.json()["hits"]
    assert [h["source_index"] for h in hits] == [0]
    assert hits[0]["distance_px"] == pytest.approx(1.716191, abs=1e-6)


# ---------------------------------------------------------------------------
# Revision handling
# ---------------------------------------------------------------------------


def test_identify_defaults_to_head_and_honours_revision_pin(client):
    v1 = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": 1,
                "properties": {"name": "old"},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            }
        ],
    }
    v2 = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": 2,
                "properties": {"name": "new"},
                "geometry": {"type": "Point", "coordinates": [11.25, 0]},
            }
        ],
    }
    ds = client.post(
        "/datasets?layer=demo", content=json.dumps(v1).encode("utf-8")
    ).json()["id"]
    resp = client.post(
        f"/datasets/{ds}/revisions",
        files={"file": ("d.geojson", json.dumps(v2).encode("utf-8"))},
        data={"expected_revision": "1"},
    )
    assert resp.status_code == 201

    # No pin -> head revision (2): only the moved point hits.
    head = _identify(client, ds, "px=136&py=128&radius_px=0").json()
    assert head["revision"] == 2
    assert [h["id"] for h in head["hits"]] == [2]
    old_spot = _identify(client, ds, "px=128&py=128&radius_px=0").json()
    assert old_spot["hits"] == []

    # Pinned to revision 1: the old geometry answers.
    pinned = _identify(
        client, ds, "px=128&py=128&radius_px=0&revision=1"
    ).json()
    assert pinned["revision"] == 1
    assert [h["id"] for h in pinned["hits"]] == [1]


def test_identify_unknown_revision_404(identify_ds, client):
    resp = _identify(client, identify_ds, "px=1&py=1&radius_px=1&revision=99")
    assert resp.status_code == 404
    assert resp.json()["latest_revision"] == 1


def test_identify_unknown_dataset_404(client):
    resp = _identify(client, "nope", "px=1&py=1&radius_px=1")
    assert resp.status_code == 404
    assert resp.json()["error"] == "dataset not found"


# ---------------------------------------------------------------------------
# Parameter validation (400)
# ---------------------------------------------------------------------------


def test_identify_invalid_tile_numbers_400(identify_ds, client):
    for path in ["15/0/0", "-1/0/0", "2/4/0", "2/0/4", "abc/0/0", "0/0/-1"]:
        resp = _identify(
            client, identify_ds, "px=1&py=1&radius_px=1",
            z=path.split("/")[0], x=path.split("/")[1], y=path.split("/")[2],
        )
        assert resp.status_code == 400, path


def test_identify_invalid_pixel_params_400(identify_ds, client):
    for query in [
        "py=1&radius_px=1",          # missing px
        "px=1&radius_px=1",          # missing py
        "px=-1&py=1&radius_px=1",    # below range
        "px=256&py=1&radius_px=1",   # 256 is out of range (0 <= px < 256)
        "px=1&py=255.5&radius_px=9", # radius above 8
        "px=1&py=1&radius_px=-0.5",  # negative radius
        "px=abc&py=1&radius_px=1",   # non-numeric px
        "px=1&py=1&radius_px=soon",  # non-numeric radius
        "px=1&py=1",                 # missing radius
    ]:
        resp = _identify(client, identify_ds, query)
        assert resp.status_code == 400, query


def test_identify_invalid_revision_400(identify_ds, client):
    for revision in ["0", "-2", "abc", "1.5"]:
        resp = _identify(
            client, identify_ds, f"px=1&py=1&radius_px=1&revision={revision}"
        )
        assert resp.status_code == 400, revision


def test_identify_boundary_pixel_values_accepted(identify_ds, client):
    assert _identify(
        client, identify_ds, "px=0&py=0&radius_px=0"
    ).status_code == 200
    assert _identify(
        client, identify_ds, "px=255.999&py=255&radius_px=8"
    ).status_code == 200


# ---------------------------------------------------------------------------
# Identify is read-only: tile bytes and ETags are untouched
# ---------------------------------------------------------------------------


def test_identify_does_not_change_tile_bytes_or_etag(identify_ds, client):
    before = client.get(f"/datasets/{identify_ds}/tiles/0/0/0.mvt")
    resp = _identify(client, identify_ds, "px=128&py=128&radius_px=8")
    assert resp.status_code == 200 and resp.json()["hits"]
    after = client.get(f"/datasets/{identify_ds}/tiles/0/0/0.mvt")
    assert after.content == before.content
    assert after.headers["etag"] == before.headers["etag"]
