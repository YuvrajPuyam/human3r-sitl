"""
Action recognition for SITL analytics.

Two modes (automatic fallback):
  1. MotionBERT backbone — when checkpoint is present, extracts rich per-joint
     temporal features from 3D SMPL-X joints, then classifies via improved
     biomechanical rules applied in feature space.
  2. Biomechanical heuristics — always available, uses knee-lift ratio
     (fps-agnostic relative geometry) for walk/run/stationary, plus joint
     angles for sitting/reaching/bending.

Checkpoint download (optional):
  Download MB_pretrain.bin (~162 MB) or MB_lite.bin (~61 MB) from:
    https://github.com/Walter0807/MotionBERT
  Place at: sitl/third_party/MotionBERT/checkpoint/MB_lite.bin

Joint conventions (SMPL-X, Y-DOWN world coords):
  - Larger Y = physically lower
  - pelvis (joint 0):  lower body centre
  - head   (joint 15): head centre
  - joints 0-21: body joints (rest: hands + face)
"""

import os
import sys
import logging
import numpy as np
from collections import Counter

log = logging.getLogger(__name__)

# ── SMPL-X body joints (0-21) → H36M 17-joint order ─────────────────────────
# H36M: [pelvis, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle,
#         spine,  thorax, neck,  head,
#         l_shldr, l_elbow, l_wrist, r_shldr, r_elbow, r_wrist]
SMPLX_TO_H36M = [0, 2, 5, 8, 1, 4, 7, 6, 9, 12, 15, 16, 18, 20, 17, 19, 21]

# H36M joint roles (for heuristic rules)
H36M = dict(
    pelvis=0, r_hip=1, r_knee=2, r_ankle=3,
    l_hip=4,  l_knee=5, l_ankle=6,
    spine=7,  thorax=8, neck=9, head=10,
    l_shldr=11, l_elbow=12, l_wrist=13,
    r_shldr=14, r_elbow=15, r_wrist=16,
)

ACTIONS        = ['stationary', 'walking', 'running', 'sitting', 'reaching', 'bending']
WINDOW         = 15      # smoothing window (majority vote)
MB_CHECKPOINT  = os.path.join(
    os.path.dirname(__file__), '../../third_party/MotionBERT/checkpoint/MB_lite.bin')
MB_CHECKPOINT  = os.path.normpath(MB_CHECKPOINT)


# ── MotionBERT loader (lazy, only when weights present) ───────────────────────

_mb_model = None   # cached after first load

def _try_load_motionbert():
    global _mb_model
    if _mb_model is not None:
        return _mb_model
    if not os.path.exists(MB_CHECKPOINT):
        return None
    try:
        import torch
        mb_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '../../third_party/MotionBERT'))
        if mb_dir not in sys.path:
            sys.path.insert(0, mb_dir)
        from lib.model.DSTformer import DSTformer

        backbone = DSTformer(
            dim_in=3, dim_out=3,
            dim_feat=256, dim_rep=512,
            depth=5, num_heads=8, mlp_ratio=2,
            num_joints=17, maxlen=243,
            att_fuse=True,
        )
        ckpt = torch.load(MB_CHECKPOINT, map_location='cpu')
        # checkpoint may be wrapped under 'model_pos' or directly
        state = ckpt.get('model_pos', ckpt.get('model', ckpt))
        # strip 'module.' prefix from DataParallel saves
        state = {k.replace('module.', ''): v for k, v in state.items()}
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing:
            log.warning('MotionBERT: %d missing keys (may be OK for backbone-only load)', len(missing))
        backbone.eval()
        _mb_model = backbone
        log.info('MotionBERT backbone loaded from %s', MB_CHECKPOINT)
        return _mb_model
    except Exception as e:
        log.warning('MotionBERT load failed (%s) — using heuristics', e)
        return None


# ── Joint geometry helpers ────────────────────────────────────────────────────

def _smplx_to_h36m(joints_T):
    """
    joints_T: (T, 127, 3) SMPL-X joints → (T, 17, 3) H36M joints.
    All coordinates remain in the original Y-DOWN world frame.
    """
    return joints_T[:, SMPLX_TO_H36M, :]


def _normalize_h36m(joints_T17):
    """
    Pelvis-center and scale by mean hip width.
    Returns (T, 17, 3) in a normalised body-relative frame.
    """
    pelvis = joints_T17[:, H36M['pelvis']:H36M['pelvis']+1, :]   # (T,1,3)
    centered = joints_T17 - pelvis
    hip_widths = np.linalg.norm(
        joints_T17[:, H36M['r_hip'], :] - joints_T17[:, H36M['l_hip'], :],
        axis=-1)
    scale = hip_widths.mean()
    if scale > 1e-6:
        centered /= scale
    return centered


def _body_height(j17_t):
    """Approximate body height in Y-DOWN: pelvis_y - head_y (positive = standing)."""
    return float(j17_t[H36M['pelvis'], 1] - j17_t[H36M['head'], 1])


def _knee_lift_ratio(j17_t, body_h):
    """
    Max knee lift relative to pelvis, normalised by body height.
    In Y-DOWN:  smaller Y = physically higher.
    knee_lift = pelvis_y - min(l_knee_y, r_knee_y)   → positive when knee is raised.
    Normalised by body_h so it is fps-agnostic.
    """
    if body_h < 0.1:
        return 0.0
    pelvis_y  = j17_t[H36M['pelvis'], 1]
    l_knee_y  = j17_t[H36M['l_knee'], 1]
    r_knee_y  = j17_t[H36M['r_knee'], 1]
    knee_lift = pelvis_y - min(l_knee_y, r_knee_y)   # positive when knee raised
    return float(knee_lift) / body_h


def _knee_drop_ratio(j17_t, body_h):
    """
    How much lower are the knees compared to the pelvis (standing posture).
    In Y-DOWN:  knee_y > pelvis_y  → knee is physically lower.
    knee_drop = avg(knee_y) - pelvis_y  → positive when standing.
    For sitting this approaches 0 (knees level with pelvis).
    """
    if body_h < 0.1:
        return 0.5
    pelvis_y = j17_t[H36M['pelvis'], 1]
    avg_knee = (j17_t[H36M['l_knee'], 1] + j17_t[H36M['r_knee'], 1]) / 2
    return float(avg_knee - pelvis_y) / body_h


def _wrist_above_shoulder(j17_t, body_h):
    """
    Normalised height of highest wrist above the shoulder line.
    In Y-DOWN: smaller Y = higher.  Positive = wrist is above shoulder.
    """
    if body_h < 0.1:
        return 0.0
    avg_shldr_y = (j17_t[H36M['l_shldr'], 1] + j17_t[H36M['r_shldr'], 1]) / 2
    min_wrist_y = min(j17_t[H36M['l_wrist'], 1], j17_t[H36M['r_wrist'], 1])
    return float(avg_shldr_y - min_wrist_y) / body_h   # positive = wrist higher


def _torso_lean(j17_t, body_h):
    """
    Head-to-pelvis vertical clearance relative to body height.
    Upright: head is ~55% of body_h above pelvis.
    Bending forward: clearance shrinks.
    In Y-DOWN: pelvis_y - head_y ≈ 0.55 * body_h when upright.
    """
    if body_h < 0.1:
        return 0.5
    head_clearance = j17_t[H36M['pelvis'], 1] - j17_t[H36M['head'], 1]
    return float(head_clearance) / body_h


# ── Per-frame pose classification (sitting / reaching / bending) ──────────────

def _knee_flexion(pose_53):
    """
    Max knee flexion angle from SMPL-X rotation vectors (radians).
    pose_53: (53, 3) array — pose[4]=left_knee, pose[5]=right_knee.
    Returns float, or None if pose unavailable.
    Sitting: ~1.0–1.6 rad. Walking: ~0.1–0.4 rad. Threshold: 0.65 rad.
    """
    if pose_53 is None:
        return None
    l = float(np.linalg.norm(pose_53[4]))
    r = float(np.linalg.norm(pose_53[5]))
    return max(l, r)


def _pose_label(j17_t, pose_53=None):
    """
    Returns 'sitting', 'reaching', or 'bending' if the pose is clearly one of
    those, else None (locomotion label determined separately from sequence).

    Sitting detection uses two signals in priority order:
      1. Knee flexion angle from rotation vectors (pose_53) — most stable,
         unaffected by joint-position reconstruction instability.
         Threshold: max(l_knee, r_knee) > 0.65 rad (≈37°).
         Walking: 0.1–0.4 rad. Sitting: 1.0–1.6 rad. Gap is large.
      2. Knee-drop ratio from joint positions — fallback when pose unavailable.
         Threshold raised to 0.40 (was 0.18) to tolerate reconstruction noise
         where knees aren't fully folded in the SMPL-X estimate.
    """
    body_h = _body_height(j17_t)

    # Sitting — primary: rotation-vector knee flexion
    kf = _knee_flexion(pose_53)
    if kf is not None and kf > 0.65:
        return 'sitting'

    # Sitting — fallback: joint-position knee-drop ratio
    kd = _knee_drop_ratio(j17_t, body_h)
    if kd < 0.40:
        return 'sitting'

    # Reaching: wrist clearly above shoulder
    wa = _wrist_above_shoulder(j17_t, body_h)
    if wa > 0.20:
        return 'reaching'

    # Bending: head vertical clearance from pelvis < 35% body height
    tl = _torso_lean(j17_t, body_h)
    if tl < 0.35:
        return 'bending'

    return None


# ── Sequence-level locomotion classification ──────────────────────────────────

def _locomotion_label(j17_t, speed, knee_lift):
    """
    Classify stationary / walking / running using two signals:
      - speed:      pelvis displacement per processed frame (m/frame)
      - knee_lift:  normalised max knee lift (body-height-relative, fps-agnostic)

    Knee lift values in Y-DOWN world coordinates:
      -0.8 to -0.6 → knees far below pelvis = standing still
      -0.5 to -0.3 → knees partially raised = walking gait swing phase
      > -0.2       → knees near or above pelvis = running / jumping

    Speed alone is NOT sufficient for running — a tracking blip can produce
    a large pelvis displacement on a clearly stationary pose. Running requires
    the knees to actually rise (knee_lift > -0.20).
    """
    # Running: knees raised to within 20% of body height from pelvis
    if knee_lift > -0.20:
        return 'running'

    # Walking: moderate knee lift + meaningful speed
    if knee_lift > -0.55 and speed > 0.008:
        return 'walking'

    return 'stationary'


# ── MotionBERT feature-space classification ───────────────────────────────────

def _classify_with_motionbert(j_seq_T17_norm, speed_seq, backbone, pose_seq=None):
    """
    Extract backbone features (B=1, F, 17, 512) then classify per-frame
    using improved heuristics applied in the normalised H36M joint space.
    The backbone is used to validate/refine the locomotion call via
    feature-space temporal variance (running has higher feature jerk).
    """
    import torch

    T = len(j_seq_T17_norm)
    CLIP = 64   # max frames per forward pass

    # Slide a window across the sequence
    feature_vels = []   # temporal gradient magnitude of pooled features
    for start in range(0, T, CLIP):
        clip = j_seq_T17_norm[start:start+CLIP]     # (C, 17, 3)
        x    = torch.from_numpy(clip).float().unsqueeze(0)  # (1, C, 17, 3)
        with torch.no_grad():
            feats = backbone(x, return_rep=True)    # (1, C, 17, 512)
        # Mean-pool over joints → (1, C, 512)
        f_pooled = feats.mean(dim=2).squeeze(0).numpy()     # (C, 512)
        # Frame-to-frame feature velocity (proxy for motion intensity)
        if len(f_pooled) > 1:
            vel = np.linalg.norm(np.diff(f_pooled, axis=0), axis=-1)  # (C-1,)
        else:
            vel = np.array([0.0])
        feature_vels.append(vel)

    feat_vel = np.concatenate(feature_vels)   # (T-1,) or shorter
    # Pad to T by repeating last value
    feat_vel = np.append(feat_vel, feat_vel[-1] if len(feat_vel) else 0.0)

    # Normalise feature velocity to [0,1] for the sequence
    fv_max = feat_vel.max()
    fv_norm = feat_vel / (fv_max + 1e-8)

    # Classify per frame
    labels = []
    for t in range(T):
        j17 = j_seq_T17_norm[t]   # already normalised

        # Pose-based first
        p53 = np.array(pose_seq[t]) if pose_seq and t < len(pose_seq) else None
        pose = _pose_label(j17, p53)
        if pose is not None:
            labels.append(pose)
            continue

        # Locomotion: use joint geometry + feature velocity as running signal
        body_h    = max(j17[H36M['pelvis'], 1] - j17[H36M['head'], 1], 0.1)
        knee_lift = _knee_lift_ratio(j17, body_h)
        speed     = speed_seq[t]

        # Feature velocity boosts the running signal
        if fv_norm[t] > 0.60 and (knee_lift > 0.10 or speed > 0.05):
            labels.append('running')
        else:
            labels.append(_locomotion_label(j17, speed, knee_lift))

    return labels


# ── Heuristic fallback (no MotionBERT) ───────────────────────────────────────

def _classify_heuristic(j_seq_T17, speed_seq, pose_seq=None):
    """
    Improved biomechanical heuristics using the H36M 17-joint sequence.
    Knee-lift ratio is the primary locomotion discriminator (fps-agnostic).
    Pose rotation vectors (pose_seq) used for robust sitting detection.

    Sequence-level stationary gate: a genuine walking gait produces a cyclic
    knee-lift pattern with std > 0.04. SMPL-X reconstruction jitter on a
    standing person yields std < 0.02. When the whole sequence shows no gait
    cycle, non-sitting frames are forced to 'stationary' regardless of speed.
    """
    # ── Sequence-level knee-lift statistics ───────────────────────────────────
    kl_vals = []
    for j17 in j_seq_T17:
        body_h = max(_body_height(j17), 0.1)
        kl_vals.append(_knee_lift_ratio(j17, body_h))

    kl_arr  = np.array(kl_vals)
    kl_std  = float(np.std(kl_arr))
    kl_mean = float(np.mean(kl_arr))

    # Low variance + deeply dropped knees = no walking gait present
    is_stationary_seq = kl_std < 0.04 and kl_mean < -0.50

    labels = []
    for t, j17 in enumerate(j_seq_T17):
        p53 = np.array(pose_seq[t]) if pose_seq and t < len(pose_seq) else None
        pose = _pose_label(j17, p53)
        if pose is not None:
            labels.append(pose)
            continue

        if is_stationary_seq:
            labels.append('stationary')
            continue

        body_h    = _body_height(j17)
        knee_lift = _knee_lift_ratio(j17, max(body_h, 0.1))
        labels.append(_locomotion_label(j17, speed_seq[t], knee_lift))

    return labels


# ── Temporal smoothing ────────────────────────────────────────────────────────

def _smooth(seq, window=15):
    """Majority-vote smoothing over a sliding window."""
    n    = len(seq)
    half = window // 2
    return [
        Counter(seq[max(0, i-half): min(n, i+half+1)]).most_common(1)[0][0]
        for i in range(n)
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def classify_person_sequence(joints_seq, speed_seq, pose_seq=None):
    """
    Classify actions for a single person across all frames.

    Args:
        joints_seq: list of (127, 3) or (45, 3) SMPL-X joint arrays (Y-DOWN).
                    Frames where the person is absent should be omitted.
        speed_seq:  list of per-frame pelvis speeds (m/processed_frame),
                    same length as joints_seq.
        pose_seq:   list of (53, 3) SMPL-X rotation-vector arrays (optional).
                    When provided, used for robust sitting detection via
                    knee flexion angle (more stable than joint positions).

    Returns:
        list of action strings (one per frame), smoothed.
        Actions: 'stationary' | 'walking' | 'running' | 'sitting' |
                 'reaching'   | 'bending'
    """
    if not joints_seq:
        return []

    # Build (T, 17, 3) in H36M format from SMPL-X body joints
    arr = np.array(joints_seq, dtype=np.float64)    # (T, J_in, 3)
    if arr.shape[1] < max(SMPLX_TO_H36M) + 1:
        log.debug('Not enough joints (%d) for H36M remap; using legacy heuristic', arr.shape[1])
        return _smooth(_legacy_classify(joints_seq, speed_seq))

    j_h36m   = _smplx_to_h36m(arr)          # (T, 17, 3)
    j_norm   = _normalize_h36m(j_h36m)      # pelvis-centred, unit scale

    backbone = _try_load_motionbert()
    if backbone is not None:
        raw = _classify_with_motionbert(j_norm, speed_seq, backbone, pose_seq)
    else:
        raw = _classify_heuristic(j_h36m, speed_seq, pose_seq)

    return _smooth(raw, window=WINDOW)


def _legacy_classify(joints_seq, speed_seq):
    """Minimal fallback when joint count is too small for H36M remapping."""
    labels = []
    for t, (j, speed) in enumerate(zip(joints_seq, speed_seq)):
        j = np.array(j)
        if speed < 0.008:
            labels.append('stationary')
        elif speed < 0.10:
            labels.append('walking')
        else:
            labels.append('running')
    return labels
