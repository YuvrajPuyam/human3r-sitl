"""Shared pytest fixtures + import path setup for the SITL test suite.

The backend uses namespace packages (no __init__.py) and is imported as
`backend.workers.analytics` when uvicorn runs from `sitl/`. We replicate that
here by putting the `sitl/` directory (parent of this tests/ dir) on sys.path,
so `import backend...` resolves the same way in tests.
"""
import os
import sys

import numpy as np
import pytest

SITL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SITL_DIR not in sys.path:
    sys.path.insert(0, SITL_DIR)


# ── Synthetic SMPL-X skeletons (Y-DOWN world coords: larger Y = physically lower)

def standing_smplx_joints(pelvis=(0.0, 0.0, 0.0)):
    """A plausible upright SMPL-X body as a (45, 3) array in Y-DOWN coords.

    Only the joints referenced by SMPLX_TO_H36M are placed meaningfully; the
    rest stay at the pelvis. Head is *above* the pelvis, so its Y is smaller
    (more negative) than the pelvis Y; knees are *below*, so larger Y.
    """
    px, py, pz = pelvis
    j = np.tile(np.array([px, py, pz], dtype=np.float64), (45, 1))
    j[0]  = [px,        py,        pz]        # pelvis
    j[1]  = [px - 0.10, py,        pz]        # l_hip
    j[2]  = [px + 0.10, py,        pz]        # r_hip
    j[4]  = [px - 0.10, py + 0.45, pz]        # l_knee  (below pelvis)
    j[5]  = [px + 0.10, py + 0.45, pz]        # r_knee
    j[7]  = [px - 0.10, py + 0.90, pz]        # l_ankle
    j[8]  = [px + 0.10, py + 0.90, pz]        # r_ankle
    j[6]  = [px,        py - 0.20, pz]        # spine
    j[9]  = [px,        py - 0.40, pz]        # thorax
    j[12] = [px,        py - 0.55, pz]        # neck
    j[15] = [px,        py - 0.70, pz]        # head    (above pelvis)
    j[16] = [px - 0.15, py - 0.50, pz]        # l_shoulder
    j[17] = [px + 0.15, py - 0.50, pz]        # r_shoulder
    j[18] = [px - 0.20, py - 0.30, pz]        # l_elbow
    j[19] = [px + 0.20, py - 0.30, pz]        # r_elbow
    j[20] = [px - 0.20, py - 0.10, pz]        # l_wrist (below shoulder)
    j[21] = [px + 0.20, py - 0.10, pz]        # r_wrist
    return j


@pytest.fixture
def standing_joints():
    return standing_smplx_joints()


@pytest.fixture
def no_motionbert(monkeypatch):
    """Force the heuristic path so tests never load torch / the checkpoint."""
    import backend.workers.action_recognition as ar
    monkeypatch.setattr(ar, "_try_load_motionbert", lambda: None)
    monkeypatch.setattr(ar, "_mb_model", None, raising=False)
    return None
