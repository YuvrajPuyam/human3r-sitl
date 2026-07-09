"""Unit tests for the spatial-analytics math (backend/workers/analytics.py).

These are the numbers the dashboard reports, so they get concrete,
geometry-checked assertions rather than smoke tests.
"""
import os
import json
import struct

import numpy as np
import pytest

from backend.workers import analytics as A


# ── Proxemics zones (Hall 1966) ───────────────────────────────────────────────

@pytest.mark.parametrize("dist,zone", [
    (0.0,   "intimate"),
    (0.44,  "intimate"),
    (0.45,  "personal"),   # boundary is exclusive-below
    (1.19,  "personal"),
    (1.20,  "social"),
    (3.69,  "social"),
    (3.70,  "public"),
    (10.0,  "public"),
])
def test_proxemics_zone_boundaries(dist, zone):
    assert A._proxemics_zone(dist) == zone


# ── Gaze / facing direction ───────────────────────────────────────────────────

def test_gaze_from_joints_is_unit_and_perpendicular():
    # pelvis at origin, head above (Y-DOWN → negative Y), shoulders along +X.
    joints = [None] * 18
    joints[0]  = [0.0,  0.0, 0.0]     # pelvis
    joints[15] = [0.0, -1.0, 0.0]     # head
    joints[16] = [-0.2, -0.8, 0.0]    # l_shoulder
    joints[17] = [0.2, -0.8, 0.0]     # r_shoulder
    v = A._compute_gaze_direction([0, -1, 0], [0, 0, 0], joints)
    assert np.isclose(np.linalg.norm(v), 1.0)
    # cross(body_up=[0,-1,0], shoulder=[1,0,0]) = [0,0,1]
    np.testing.assert_allclose(v, [0.0, 0.0, 1.0], atol=1e-6)


def test_gaze_fallback_flips_y_to_ydown():
    # No joints → use head-pelvis vertex delta (Y-UP) and flip Y to Y-DOWN.
    v = A._compute_gaze_direction([0, 1, 0], [0, 0, 0], None)
    assert np.isclose(np.linalg.norm(v), 1.0)
    np.testing.assert_allclose(v, [0.0, -1.0, 0.0], atol=1e-6)


def test_gaze_degenerate_head_equals_pelvis():
    v = A._compute_gaze_direction([1, 1, 1], [1, 1, 1], None)
    np.testing.assert_allclose(v, [0.0, -1.0, 0.0], atol=1e-6)


# ── Gaze-ray convergence ───────────────────────────────────────────────────────

def test_pair_rays_converge_when_facing():
    # Two rays that cross at [1,0,0] (kept non-collinear: exactly antiparallel
    # rays are degenerate — denom==0 — and correctly report no convergence).
    p1, g1 = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])     # +X axis
    p2, g2 = np.array([1.0, -1, 0]), np.array([0.0, 1.0, 0])  # vertical line x=1
    assert A._pair_rays_converge(p1, g1, p2, g2, threshold=1.0) is True


def test_pair_rays_diverge_when_facing_away():
    p1, g1 = np.array([0.0, 0, 0]), np.array([-1.0, 0, 0])
    p2, g2 = np.array([2.0, 0, 0]), np.array([1.0, 0, 0])
    assert A._pair_rays_converge(p1, g1, p2, g2, threshold=1.0) is False


def test_pair_rays_parallel_do_not_converge():
    p1, g1 = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    p2, g2 = np.array([0.0, 5, 0]), np.array([1.0, 0, 0])
    assert A._pair_rays_converge(p1, g1, p2, g2, threshold=1.0) is False


def test_rays_converge_array_form():
    positions = np.array([[0.0, 0, 0], [1.0, -1, 0]])
    gazes     = np.array([[1.0, 0, 0], [0.0, 1.0, 0]])
    assert A._rays_converge(positions, gazes, threshold=1.0) is True
    # far-apart parallel rays: no convergence
    positions2 = np.array([[0.0, 0, 0], [0.0, 9, 0]])
    gazes2     = np.array([[1.0, 0, 0], [1.0, 0, 0]])
    assert A._rays_converge(positions2, gazes2, threshold=1.0) is False


# ── Connected components (F-formation grouping) ───────────────────────────────

def test_connected_components_two_groups():
    labels = A._connected_components(4, [(0, 1), (2, 3)])
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_connected_components_no_edges_all_singletons():
    labels = A._connected_components(3, [])
    assert len(set(labels)) == 3


def test_connected_components_chain_is_one_group():
    labels = A._connected_components(4, [(0, 1), (1, 2), (2, 3)])
    assert len(set(labels)) == 1


# ── Track stitching ────────────────────────────────────────────────────────────

def _frame(*people):
    return {"humans": [{"id": pid, "world_pos": pos} for pid, pos in people]}


def test_stitch_merges_fragmented_track():
    # id 0 walks +X for 3 frames, disappears, id 1 continues the motion.
    frames = [
        _frame((0, [0.0, 0, 0])),
        _frame((0, [1.0, 0, 0])),
        _frame((0, [2.0, 0, 0])),
        _frame(),                       # gap
        _frame((1, [4.0, 0, 0])),
        _frame((1, [5.0, 0, 0])),
        _frame((1, [6.0, 0, 0])),
    ]
    merges = A._stitch_tracks(frames, max_gap=10, thresh=0.6)
    assert merges == 1
    seen = {h["id"] for f in frames for h in f["humans"]}
    assert seen == {0}


def test_stitch_does_not_merge_co_occurring_ids():
    # Two people present at the same time are different physical people.
    frames = [
        _frame((0, [0.0, 0, 0]), (1, [3.0, 0, 0])),
        _frame((0, [0.1, 0, 0]), (1, [3.1, 0, 0])),
    ]
    merges = A._stitch_tracks(frames, max_gap=10, thresh=0.6)
    assert merges == 0
    seen = {h["id"] for f in frames for h in f["humans"]}
    assert seen == {0, 1}


# ── PLY loader (ASCII + binary) ───────────────────────────────────────────────

_PTS = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]


def _write_ascii_ply(path):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex 3\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in _PTS:
            f.write(f"{x} {y} {z}\n")


def _write_binary_ply(path):
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
              "property float x\nproperty float y\nproperty float z\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for x, y, z in _PTS:
            f.write(struct.pack("<fff", x, y, z))


def test_load_ply_ascii(tmp_path):
    p = tmp_path / "a.ply"
    _write_ascii_ply(str(p))
    pts = A._load_ply_xyz(str(p))
    np.testing.assert_allclose(pts, _PTS, atol=1e-6)


def test_load_ply_binary_matches_ascii(tmp_path):
    p = tmp_path / "b.ply"
    _write_binary_ply(str(p))
    pts = A._load_ply_xyz(str(p))
    np.testing.assert_allclose(pts, _PTS, atol=1e-6)


def test_load_ply_empty(tmp_path):
    p = tmp_path / "empty.ply"
    with open(p, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex 0\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n")
    pts = A._load_ply_xyz(str(p))
    assert pts.shape == (0, 3)


# ── End-to-end analytics on a synthetic job ───────────────────────────────────

def _synthetic_dashboard(n_frames=6, sep=1.0):
    """Two static people `sep` metres apart on the X axis, standing."""
    from conftest import standing_smplx_joints
    j0 = standing_smplx_joints((0.0, 0.0, 0.0)).tolist()
    j1 = standing_smplx_joints((sep, 0.0, 0.0)).tolist()
    frames = []
    for fi in range(n_frames):
        frames.append({
            "frame_id": fi,
            "humans": [
                {"id": 0, "world_pos": [0.0, 0.0, 0.0], "head_world": [0.0, 0.7, 0.0],
                 "joints": j0, "pose": [[0.0, 0.0, 0.0]] * 53},
                {"id": 1, "world_pos": [sep, 0.0, 0.0], "head_world": [sep, 0.7, 0.0],
                 "joints": j1, "pose": [[0.0, 0.0, 0.0]] * 53},
            ],
        })
    return {
        "metadata": {"total_frames": n_frames, "effective_fps": 30.0, "up_axis": "Y"},
        "camera_trajectory": [{"R": np.eye(3).tolist(), "t": [0, 0, 0]}
                              for _ in range(n_frames)],
        "frames": frames,
    }


def test_compute_analytics_end_to_end(tmp_path, monkeypatch, no_motionbert):
    monkeypatch.chdir(tmp_path)
    job_id = "unittest01"
    out_dir = tmp_path / "outputs" / job_id
    out_dir.mkdir(parents=True)
    with open(out_dir / "dashboard_data.json", "w") as f:
        json.dump(_synthetic_dashboard(n_frames=6, sep=1.0), f)

    A._compute_spatial_analytics_sync(job_id, {})

    enriched_path = out_dir / "enriched_data.json"
    assert enriched_path.exists()
    with open(enriched_path) as f:
        out = json.load(f)

    s = out["summary"]
    # Two people held exactly 1.0 m apart every frame.
    assert s["avg_inter_human_distance"] == pytest.approx(1.0, abs=1e-6)
    # 1.0 m is inside Hall's personal zone → engaged every frame.
    assert s["social_engagement_pct"] == pytest.approx(100.0)
    assert s["peak_occupancy"] == 2
    assert s["effective_fps"] == pytest.approx(30.0)
    # Static subjects → no motion.
    assert s["avg_speed_mps"] == pytest.approx(0.0)

    # Per-frame enrichment is well-formed.
    for fr in out["frames"]:
        assert len(fr["interactions"]) == 1
        inter = fr["interactions"][0]
        assert inter["zone"] == "personal"
        assert inter["distance"] == pytest.approx(1.0, abs=1e-6)
        for h in fr["humans"]:
            assert np.isclose(np.linalg.norm(h["gaze_vec"]), 1.0)
            assert h["action"] in A.__dict__.get("ACTIONS",
                ['stationary', 'walking', 'running', 'sitting', 'reaching', 'bending'])
