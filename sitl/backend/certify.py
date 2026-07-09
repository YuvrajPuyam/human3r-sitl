"""Data-integrity certification for a processed SITL job.

Validates the contract that the frontend viewer relies on: enriched_data.json
schema, coordinate/geometry sanity (finite values, unit gaze vectors), proxemics
consistency, interaction referential integrity, and (optionally) the point cloud.

Usage:
    cd sitl
    python -m backend.certify <job_id>            # validates outputs/<job_id>/
    python -m backend.certify path/to/job_dir     # or an explicit directory

Exit code 0 = all checks passed, 1 = at least one failure. Importable too:
    from backend.certify import certify_job
    checks = certify_job("outputs/06c9ccd8")
    ok = all(c.ok for c in checks)
"""
import os
import sys
import json
import math
from dataclasses import dataclass

import numpy as np

from .workers.analytics import _proxemics_zone, _load_ply_xyz
from .workers.action_recognition import ACTIONS

VALID_ZONES = {"intimate", "personal", "social", "public"}
VALID_TYPES = {"contact", "proximity"}
REQUIRED_SUMMARY_KEYS = {
    "social_engagement_pct", "avg_inter_human_distance", "scene_utilization_pct",
    "gaze_convergence_events", "peak_occupancy", "effective_fps",
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _finite_vec(v, n):
    """True if v is a length-n sequence of finite numbers."""
    try:
        if len(v) != n:
            return False
        return all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
    except TypeError:
        return False


def certify_job(job_dir: str) -> list[Check]:
    """Run every integrity check against a job directory. Never raises —
    failures are reported as Check(ok=False) so the whole report always returns."""
    checks: list[Check] = []
    enriched = os.path.join(job_dir, "enriched_data.json")

    # 1. File present + parseable
    if not os.path.exists(enriched):
        checks.append(Check("enriched_data.json exists", False, enriched))
        return checks
    try:
        with open(enriched) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        checks.append(Check("enriched_data.json parses", False, str(e)))
        return checks
    checks.append(Check("enriched_data.json parses", True))

    # 2. Top-level schema
    missing = [k for k in ("metadata", "camera_trajectory", "frames", "summary")
               if k not in data]
    checks.append(Check("top-level keys present", not missing,
                        f"missing: {missing}" if missing else ""))
    if missing:
        return checks

    meta   = data["metadata"]
    frames = data["frames"]
    cams   = data["camera_trajectory"]
    summ   = data["summary"]

    # 3. Frame count matches metadata
    total = meta.get("total_frames")
    checks.append(Check("total_frames matches frames[]", total == len(frames),
                        f"metadata={total} frames={len(frames)}"))

    # 4. Camera trajectory well-formed and aligned
    cam_ok = len(cams) == len(frames) and all(
        _finite_vec(c.get("t", []), 3)
        and np.asarray(c.get("R", [])).shape == (3, 3)
        and np.isfinite(np.asarray(c.get("R", []), dtype=float)).all()
        for c in cams)
    checks.append(Check("camera_trajectory aligned & finite", cam_ok,
                        f"cams={len(cams)} frames={len(frames)}"))

    # 5-7. Per-frame human + interaction integrity
    bad_human, bad_gaze, bad_action, bad_inter, bad_zone = [], [], [], [], []
    for fr in frames:
        fid = fr.get("frame_id", "?")
        ids_here = set()
        for h in fr.get("humans", []):
            ids_here.add(h.get("id"))
            if not (_finite_vec(h.get("world_pos", []), 3)
                    and _finite_vec(h.get("head_world", []), 3)):
                bad_human.append(fid)
            g = h.get("gaze_vec")
            if g is not None:
                if not _finite_vec(g, 3) or abs(np.linalg.norm(g) - 1.0) > 1e-2:
                    bad_gaze.append(fid)
            if h.get("action") not in ACTIONS:
                bad_action.append(fid)
        for it in fr.get("interactions", []):
            if it.get("source") not in ids_here or it.get("target") not in ids_here:
                bad_inter.append(fid)
            dist = it.get("distance")
            zone = it.get("zone")
            if (not isinstance(dist, (int, float)) or not math.isfinite(dist)
                    or dist < 0 or zone not in VALID_ZONES
                    or it.get("type") not in VALID_TYPES):
                bad_zone.append(fid)
            elif _proxemics_zone(dist) != zone:      # proxemics consistency
                bad_zone.append(fid)

    checks.append(Check("human positions finite", not bad_human,
                        f"bad frames: {bad_human[:5]}" if bad_human else ""))
    checks.append(Check("gaze vectors unit-length", not bad_gaze,
                        f"bad frames: {bad_gaze[:5]}" if bad_gaze else ""))
    checks.append(Check("actions in vocabulary", not bad_action,
                        f"bad frames: {bad_action[:5]}" if bad_action else ""))
    checks.append(Check("interactions reference present ids", not bad_inter,
                        f"bad frames: {bad_inter[:5]}" if bad_inter else ""))
    checks.append(Check("proxemics zone matches distance", not bad_zone,
                        f"bad frames: {bad_zone[:5]}" if bad_zone else ""))

    # 8. Summary scalar keys
    smissing = REQUIRED_SUMMARY_KEYS - set(summ)
    checks.append(Check("summary keys present", not smissing,
                        f"missing: {sorted(smissing)}" if smissing else ""))

    # 9. Optional point cloud
    ply = os.path.join(job_dir, "scene.ply")
    if os.path.exists(ply):
        try:
            pts = _load_ply_xyz(ply)
            ply_ok = len(pts) > 0 and bool(np.isfinite(pts).all())
            checks.append(Check("scene.ply loads & finite", ply_ok,
                                f"{len(pts)} points"))
        except Exception as e:
            checks.append(Check("scene.ply loads & finite", False, str(e)))

    return checks


def _resolve_dir(arg: str) -> str:
    """Accept either a directory path or a bare job_id (→ outputs/<job_id>)."""
    if os.path.isdir(arg):
        return arg
    return os.path.join("outputs", arg)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m backend.certify <job_id | job_dir>")
        return 2
    job_dir = _resolve_dir(argv[0])
    checks = certify_job(job_dir)
    print(f"\nCertifying {job_dir}\n" + "─" * 48)
    for c in checks:
        mark = "✓" if c.ok else "✗"
        line = f"  {mark}  {c.name}"
        if c.detail:
            line += f"   [{c.detail}]"
        print(line)
    passed = sum(c.ok for c in checks)
    ok = all(c.ok for c in checks)
    print("─" * 48)
    print(f"{passed}/{len(checks)} checks passed — {'PASS' if ok else 'FAIL'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
