"""Unit tests for action recognition (backend/workers/action_recognition.py).

Runs entirely on the biomechanical-heuristic path (the `no_motionbert` fixture
forces the checkpoint off) so no torch / GPU is required.
"""
import numpy as np
import pytest

from backend.workers import action_recognition as AR
from conftest import standing_smplx_joints


# ── SMPL-X → H36M remap ────────────────────────────────────────────────────────

def test_smplx_to_h36m_shape_and_mapping():
    # Tag each SMPL-X joint with its index so we can verify the gather.
    T = 2
    j = np.zeros((T, 45, 3))
    for k in range(45):
        j[:, k, :] = k
    h = AR._smplx_to_h36m(j)
    assert h.shape == (T, 17, 3)
    # H36M pelvis(0)=SMPLX0, r_ankle(3)=SMPLX8, head(10)=SMPLX15
    assert h[0, 0, 0] == 0
    assert h[0, 3, 0] == 8
    assert h[0, 10, 0] == 15


def test_normalize_h36m_centers_pelvis():
    j17 = np.random.RandomState(0).randn(3, 17, 3)
    norm = AR._normalize_h36m(j17)
    np.testing.assert_allclose(norm[:, AR.H36M["pelvis"], :], 0.0, atol=1e-9)


# ── Body-geometry helpers on a canonical standing pose ────────────────────────

@pytest.fixture
def standing_h36m():
    return AR._smplx_to_h36m(standing_smplx_joints()[None, ...])[0]


def test_body_height_positive_when_standing(standing_h36m):
    # pelvis_y(0) - head_y(-0.70) = 0.70
    assert AR._body_height(standing_h36m) == pytest.approx(0.70, abs=1e-6)


def test_knee_lift_ratio_negative_when_standing(standing_h36m):
    # knees below pelvis → knee_lift = pelvis_y - min(knee_y) < 0
    r = AR._knee_lift_ratio(standing_h36m, AR._body_height(standing_h36m))
    assert r == pytest.approx(-0.45 / 0.70, abs=1e-6)


def test_knee_drop_ratio_positive_when_standing(standing_h36m):
    r = AR._knee_drop_ratio(standing_h36m, AR._body_height(standing_h36m))
    assert r == pytest.approx(0.45 / 0.70, abs=1e-6)


def test_wrist_below_shoulder_when_standing(standing_h36m):
    # wrists hang below shoulders → not reaching
    assert AR._wrist_above_shoulder(standing_h36m, AR._body_height(standing_h36m)) < 0


# ── Knee-flexion (rotation-vector sitting signal) ─────────────────────────────

def test_knee_flexion_takes_max_of_both_knees():
    pose = np.zeros((6, 3))
    pose[4] = [1.2, 0, 0]   # left knee bent
    pose[5] = [0.3, 0, 0]   # right knee straighter
    assert AR._knee_flexion(pose) == pytest.approx(1.2)


def test_knee_flexion_none_when_missing():
    assert AR._knee_flexion(None) is None
    assert AR._knee_flexion(np.zeros((3, 3))) is None   # too few rows


# ── Pose labels ────────────────────────────────────────────────────────────────

def test_pose_label_sitting_from_knee_flexion(standing_h36m):
    pose = np.zeros((6, 3))
    pose[4] = [1.2, 0, 0]   # strong knee bend → sitting regardless of positions
    assert AR._pose_label(standing_h36m, pose) == "sitting"


def test_pose_label_sitting_from_knee_drop():
    # Knees pulled up level with the pelvis → knee-drop ratio small → sitting.
    j = standing_smplx_joints()
    j[4] = [-0.10, 0.03, 0.0]   # l_knee near pelvis height
    j[5] = [0.10, 0.03, 0.0]    # r_knee near pelvis height
    h17 = AR._smplx_to_h36m(j[None, ...])[0]
    assert AR._pose_label(h17, None) == "sitting"


def test_pose_label_reaching_when_wrist_raised():
    j = standing_smplx_joints()
    j[20] = [-0.20, -0.80, 0.0]   # l_wrist above shoulders (Y-DOWN → smaller Y)
    h17 = AR._smplx_to_h36m(j[None, ...])[0]
    assert AR._pose_label(h17, None) == "reaching"


def test_pose_label_none_for_plain_standing(standing_h36m):
    assert AR._pose_label(standing_h36m, None) is None


# ── Locomotion label ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("speed,knee_lift,expected", [
    (0.50, -0.50, "walking"),     # moving, moderate knee lift
    (0.50, -0.10, "running"),     # knees high
    (0.05, -0.50, "stationary"),  # too slow to be walking
    (0.50, -0.80, "stationary"),  # knees too low (deep stance) + gate
])
def test_locomotion_label(speed, knee_lift, expected):
    assert AR._locomotion_label(None, speed, knee_lift) == expected


# ── Temporal smoothing ────────────────────────────────────────────────────────

def test_smooth_removes_single_frame_spike():
    seq = ["walking", "walking", "running", "walking", "walking"]
    assert AR._smooth(seq, window=3) == ["walking"] * 5


def test_smooth_empty():
    assert AR._smooth([], window=3) == []


# ── Public API: classify_person_sequence ──────────────────────────────────────

def test_classify_empty_sequence(no_motionbert):
    assert AR.classify_person_sequence([], []) == []


def test_classify_standing_person_is_stationary(no_motionbert):
    joints = [standing_smplx_joints().tolist() for _ in range(20)]
    speeds = [0.0] * 20
    labels = AR.classify_person_sequence(joints, speeds)
    assert len(labels) == 20
    assert set(labels) == {"stationary"}


def test_classify_sitting_from_pose(no_motionbert):
    joints = [standing_smplx_joints().tolist() for _ in range(20)]
    speeds = [0.0] * 20
    bent = np.zeros((53, 3)); bent[4] = [1.2, 0, 0]
    poses = [bent.tolist() for _ in range(20)]
    labels = AR.classify_person_sequence(joints, speeds, poses)
    assert set(labels) == {"sitting"}


def test_classify_walking_person(no_motionbert):
    joints = [standing_smplx_joints().tolist() for _ in range(20)]
    speeds = [0.5] * 20   # clearly moving, above the stationary gate
    labels = AR.classify_person_sequence(joints, speeds)
    assert labels.count("walking") >= 15   # dominant label after smoothing


def test_classify_falls_back_to_legacy_for_few_joints(no_motionbert):
    # Fewer than 22 joints → legacy speed-only classifier.
    joints = [np.zeros((10, 3)).tolist() for _ in range(10)]
    speeds = [0.0] * 10
    labels = AR.classify_person_sequence(joints, speeds)
    assert set(labels) == {"stationary"}
