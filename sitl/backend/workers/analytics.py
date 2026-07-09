import os
import json
import asyncio
import logging
import numpy as np
from collections import Counter
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from .action_recognition import classify_person_sequence

log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False


# ── PLY loader ────────────────────────────────────────────────────────────────

def _load_ply_xyz(path: str) -> np.ndarray:
    """Load XYZ from ASCII or binary_little_endian PLY."""
    n_verts = 0
    fmt = "ascii"
    properties = []

    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))
            elif line == "end_header":
                break

        if n_verts == 0:
            return np.empty((0, 3), dtype=np.float32)

        if fmt == "ascii":
            pts = []
            for _ in range(n_verts):
                row = f.readline().decode("ascii", errors="replace").split()
                pts.append([float(row[0]), float(row[1]), float(row[2])])
            return np.array(pts, dtype=np.float32)

        endian = "<" if "little" in fmt else ">"
        _np_map = {
            "float": "f4", "float32": "f4",
            "double": "f8", "float64": "f8",
            "int": "i4", "int32": "i4",
            "uint": "u4", "uint32": "u4",
            "short": "i2", "int16": "i2",
            "ushort": "u2", "uint16": "u2",
            "char": "i1", "int8": "i1",
            "uchar": "u1", "uint8": "u1",
        }
        dt = np.dtype([(n, endian + _np_map.get(t, "f4")) for t, n in properties])
        raw = f.read(dt.itemsize * n_verts)
        records = np.frombuffer(raw, dtype=dt, count=n_verts)
        return np.stack([records["x"].astype(np.float32),
                         records["y"].astype(np.float32),
                         records["z"].astype(np.float32)], axis=1)


# ── Gaze / facing direction ───────────────────────────────────────────────────

def _compute_gaze_direction(
    head_pos: list, pelvis_pos: list, joints: list | None = None
) -> np.ndarray:
    """
    Compute the body-forward (facing) direction in OpenCV Y-DOWN world coordinates.

    Uses SMPL-X joints when available:
      body_up  = joint[15] - joint[0]   (head − pelvis in Y-DOWN, so body_up.y < 0)
      shldr    = joint[17] - joint[16]  (right − left shoulder)
      facing   = cross(shldr, body_up)  → forward direction perpendicular to both

    Falls back to body-up derived from Y-DOWN joints, then to vertex-based body-up.
    The result must be in Y-DOWN convention so that Viewer.jsx's fy() flip converts
    it to the correct upward/forward direction in Three.js Y-UP space.
    """
    # ── Preferred path: use joints (reliably in OpenCV Y-DOWN) ────────────────
    if joints and len(joints) >= 18:
        try:
            j0  = np.array(joints[0],  dtype=np.float64)   # pelvis joint
            j15 = np.array(joints[15], dtype=np.float64)   # head joint
            j16 = np.array(joints[16], dtype=np.float64)   # left shoulder
            j17 = np.array(joints[17], dtype=np.float64)   # right shoulder

            body_up = j15 - j0          # Y-DOWN: head_y < pelvis_y → body_up.y < 0 = "up"
            bu_norm = np.linalg.norm(body_up)
            if bu_norm > 1e-6:
                body_up /= bu_norm
                sv      = j17 - j16     # right − left shoulder
                sv_norm = np.linalg.norm(sv)
                if sv_norm > 1e-6:
                    sv  /= sv_norm
                    # cross(body_up, shoulder_vec) → forward-facing direction
                    # Note: cross(sv, body_up) is opposite (away from face); use body_up × sv
                    fwd  = np.cross(body_up, sv)
                    fn   = np.linalg.norm(fwd)
                    if fn > 1e-6:
                        return fwd / fn
                # no shoulder info — fall back to body-up from joints
                return body_up
        except (IndexError, ValueError):
            pass

    # ── Fallback: body-up from vertex positions ────────────────────────────────
    # SMPL-X vertices (world_pos, head_world) are in Y-UP convention; joints are Y-DOWN.
    # To get a Y-DOWN result (so fy() in Viewer.jsx converts it back correctly):
    # d = head_vertex − pelvis_vertex  → +Y in Y-UP (pointing up)
    # Negate Y component to get the equivalent Y-DOWN direction.
    d = np.array(head_pos, dtype=np.float64) - np.array(pelvis_pos, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        return np.array([0.0, -1.0, 0.0])   # Y-DOWN "up" direction
    d /= norm
    d[1] = -d[1]                             # flip Y: Y-UP → Y-DOWN
    n2 = np.linalg.norm(d)
    return d / (n2 if n2 > 1e-6 else 1.0)


# ── Gaze convergence ──────────────────────────────────────────────────────────

def _rays_converge(positions: np.ndarray, gazes: np.ndarray,
                   threshold: float = 1.0) -> bool:
    """True if any pair of gaze rays passes within `threshold` metres of each other."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            d1, d2 = gazes[i], gazes[j]
            w      = positions[i] - positions[j]
            a, b   = d1 @ d1, d1 @ d2
            c, d_  = d2 @ d2, d1 @ w
            e      = d2 @ w
            denom  = a * c - b * b
            if abs(denom) < 1e-8:
                continue
            s = (b * e - c * d_) / denom
            t = (a * e - b * d_) / denom
            if s < 0 or t < 0:
                continue
            if np.linalg.norm(w + s * d1 - t * d2) < threshold:
                return True
    return False


def _pair_rays_converge(p1, g1, p2, g2, threshold: float = 1.0) -> bool:
    """Check gaze convergence for a single pair."""
    w = p1 - p2
    a, b = g1 @ g1, g1 @ g2
    c, d_ = g2 @ g2, g1 @ w
    e = g2 @ w
    denom = a * c - b * b
    if abs(denom) < 1e-8:
        return False
    s = (b * e - c * d_) / denom
    t = (a * e - b * d_) / denom
    if s < 0 or t < 0:
        return False
    return bool(np.linalg.norm(w + s * g1 - t * g2) < threshold)


# ── Action classification ─────────────────────────────────────────────────────

# SMPL-X joint indices: 0=pelvis, 4=L_knee, 5=R_knee, 15=head,
# 16=L_shoulder, 17=R_shoulder, 20=L_wrist, 21=R_wrist



# ── Proxemics helpers ─────────────────────────────────────────────────────────

def _proxemics_zone(dist: float) -> str:
    """Hall (1966) proxemics zone for a given inter-person distance."""
    if dist < 0.45:
        return "intimate"
    if dist < 1.20:
        return "personal"
    if dist < 3.70:
        return "social"
    return "public"


# ── Bird's-eye floor heatmap ──────────────────────────────────────────────────

# Mirror of Viewer.jsx PALETTE_HEX so bird's-eye trajectory colours match the
# per-subject colours in the 3D viewer (vivid, non-Claude cool palette).
_PERSON_COLORS = [
    '#3b82f6', '#ec4899', '#22c55e', '#8b5cf6', '#06b6d4',
    '#f43f5e', '#eab308', '#d946ef', '#14b8a6', '#818cf8',
]


def _compute_floor_heatmap(enriched_frames: list, output_path: str,
                            cell_size: float = 0.12) -> dict | None:
    """
    Generate a bird's-eye density heatmap from human pelvis XZ positions.
    Saves PNG to output_path; returns XZ bounding box for Three.js plane positioning.
    """
    if not _MATPLOTLIB:
        return None

    all_xz: list[tuple[float, float]] = []
    traj: dict[int, list[tuple[float, float]]] = {}

    for fd in enriched_frames:
        for h in fd.get("humans", []):
            pos = h["world_pos"]
            x, z = float(pos[0]), float(pos[2])
            all_xz.append((x, z))
            traj.setdefault(h["id"], []).append((x, z))

    if not all_xz:
        return None

    xs = [p[0] for p in all_xz]
    zs = [p[1] for p in all_xz]
    margin = 0.7
    x_min, x_max = min(xs) - margin, max(xs) + margin
    z_min, z_max = min(zs) - margin, max(zs) + margin

    x_range = max(0.5, x_max - x_min)
    z_range = max(0.5, z_max - z_min)

    nx = max(40, int(x_range / cell_size))
    nz = max(40, int(z_range / cell_size))

    density = np.zeros((nz, nx), dtype=np.float32)
    for x, z in all_xz:
        xi = int((x - x_min) / x_range * (nx - 1))
        zi = int((z - z_min) / z_range * (nz - 1))
        density[np.clip(zi, 0, nz-1), np.clip(xi, 0, nx-1)] += 1

    sigma = max(1.5, nx * 0.025)
    density = gaussian_filter(density, sigma=sigma)

    dpi   = 130
    fig_w = 7.0
    fig_h = float(np.clip(fig_w * z_range / x_range, 2.5, 11.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor('#070710')
    ax.set_facecolor('#070710')

    if density.max() > 0:
        ax.imshow(
            density, origin='lower', cmap='hot',
            extent=[x_min, x_max, z_min, z_max],
            interpolation='bilinear', alpha=0.82,
            vmin=0, vmax=density.max(),
        )

    for pid, pts in sorted(traj.items()):
        if len(pts) < 2:
            continue
        col = _PERSON_COLORS[pid % len(_PERSON_COLORS)]
        tx  = [p[0] for p in pts]
        tz  = [p[1] for p in pts]
        ax.plot(tx, tz, color=col, linewidth=1.4, alpha=0.75, zorder=3,
                label=f"Person {pid}")
        ax.scatter(tx[0],  tz[0],  c=col, s=50, marker='o',  zorder=5, alpha=0.9)
        ax.scatter(tx[-1], tz[-1], c=col, s=50, marker='x',
                   linewidths=2.0, zorder=5, alpha=0.9)

    ax.legend(fontsize=7.5, facecolor='#111', edgecolor='#333',
              labelcolor='white', loc='upper right', framealpha=0.85)
    ax.set_xlabel('X  (m)', color='#475569', fontsize=8)
    ax.set_ylabel('Z  (m)', color='#475569', fontsize=8)
    ax.tick_params(colors='#334155', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
    ax.set_title("Floor Occupancy — Bird's-Eye View",
                 color='#64748b', fontsize=9, pad=6)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='#070710', edgecolor='none')
    plt.close(fig)

    return {
        "x_min": round(float(x_min), 3),
        "x_max": round(float(x_max), 3),
        "z_min": round(float(z_min), 3),
        "z_max": round(float(z_max), 3),
    }


# ── Track stitching ───────────────────────────────────────────────────────────

def _stitch_tracks(frames: list, max_gap: int = 10, thresh: float = 0.6) -> int:
    """Greedily merge fragmented person tracks in place.

    Monocular tracking drops IDs across occlusions/blinks, so one physical person
    may carry several IDs over a clip — which silently corrupts every per-person
    metric (speed, action, trajectory). For each ID we take its first/last frame
    and position; a later segment B is merged into an earlier A when B starts
    within `max_gap` frames of A ending and B's start position is within `thresh`
    metres of A's velocity-extrapolated end position (and they never co-occur).

    Returns the number of merges applied. Mutates humans["id"] in `frames`.
    """
    # Build per-id segments: frame indices + positions in appearance order.
    seg: dict[int, list[tuple[int, np.ndarray]]] = {}
    for fi, frame in enumerate(frames):
        for h in frame.get("humans", []):
            seg.setdefault(h["id"], []).append((fi, np.asarray(h["world_pos"], float)))

    ids = sorted(seg)
    frame_sets = {i: {fi for fi, _ in seg[i]} for i in ids}
    remap: dict[int, int] = {}

    def resolve(i):
        while i in remap:
            i = remap[i]
        return i

    merges = 0
    for b in ids:                       # candidate "later" track
        bf = seg[b]
        b_start_fi, b_start_pos = bf[0]
        best, best_d = None, thresh
        for a in ids:
            if a == b:
                continue
            ar = resolve(a)
            if ar == b:
                continue
            af = seg[a]
            a_end_fi, a_end_pos = af[-1]
            gap = b_start_fi - a_end_fi
            if gap < 1 or gap > max_gap:
                continue
            if frame_sets[ar] & frame_sets[b]:      # they co-exist → different people
                continue
            vel = (af[-1][1] - af[max(0, len(af) - 4)][1]) / max(1, min(3, len(af) - 1))
            pred = a_end_pos + vel * gap
            d = float(np.linalg.norm(pred - b_start_pos))
            if d < best_d:
                best, best_d = ar, d
        if best is not None:
            remap[b] = best
            frame_sets[best] |= frame_sets[b]
            merges += 1

    if merges:
        for frame in frames:
            for h in frame.get("humans", []):
                h["id"] = resolve(h["id"])
    return merges


# ── Group / F-formation detection (union-find) ────────────────────────────────

def _connected_components(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """Return a component label per node index given undirected edges."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return [find(i) for i in range(n)]


# ── Main analytics function ───────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "intimate_zone":     0.45,   # Hall (1966)
    "personal_zone":     1.20,
    "social_zone":       3.70,
    "contact_thresh":    0.70,   # interaction "contact" vs "proximity"
    "gaze_thresh":       1.00,   # ray closest-approach for mutual gaze (m)
    "stitch_max_gap":    10,     # processed frames
    "stitch_thresh":     0.60,   # m
    "group_edge_dist":   2.00,   # m — max spacing for an F-formation edge
    "group_edge_orient": 0.35,   # min orientation score (1 = facing each other)
    "move_speed_mps":    0.20,   # above this a subject counts as "moving"
    "max_jump_mps":      12.00,  # sprint cap; faster = tracking blip
    "fps_override":      None,   # force effective fps (else read from metadata)
}


async def compute_spatial_analytics(job_id: str, params: dict | None = None):
    """Async entry point — runs the heavy compute in a worker thread so the
    FastAPI event loop (and the SSE status stream) stays responsive."""
    return await asyncio.to_thread(_compute_spatial_analytics_sync, job_id, params or {})


def _compute_spatial_analytics_sync(job_id: str, params: dict):
    P = {**DEFAULT_PARAMS, **(params or {})}

    output_dir = f"outputs/{job_id}"
    data_path  = os.path.join(output_dir, "dashboard_data.json")
    ply_path   = os.path.join(output_dir, "scene.ply")

    with open(data_path) as f:
        data = json.load(f)

    frames    = data["frames"]
    meta_in   = data.get("metadata", {})
    # Effective frame rate (processed frames/sec) → converts displacement to m/s.
    eff_fps   = P["fps_override"] or float(meta_in.get("effective_fps") or 30.0)
    eff_fps   = eff_fps if eff_fps > 1e-6 else 30.0

    # ── Track stitching: repair fragmented person IDs before any metric ────────
    n_merges = _stitch_tracks(frames, P["stitch_max_gap"], P["stitch_thresh"])
    if n_merges:
        log.info("Track stitching merged %d fragmented track(s)", n_merges)

    scene_pts  = None
    scene_tree = None
    if os.path.exists(ply_path):
        scene_pts = _load_ply_xyz(ply_path)
        if len(scene_pts) > 0:
            scene_tree = KDTree(scene_pts)

    # SMPL-X ankle joints (Y-DOWN, same frame as scene.ply) for ground contact.
    L_ANKLE, R_ANKLE = 7, 8

    VOXEL = 0.5
    all_pair_dists          = []
    social_frames           = 0       # pairs within Hall's personal zone
    personal_space_frames   = 0       # pairs within Hall's intimate zone
    gaze_convergence_events = 0
    approach_events         = 0
    occupied_voxels         = set()
    peak_occupancy          = 0

    prev_positions    = {}
    prev_frame_seen   = {}            # h_id → last frame index where person was seen
    speed_samples     = []            # m/s
    pair_dist_win     = {}            # pk → recent distances (smoothing window)
    pair_states       = {}            # pk → "approaching"|"stable"|"retreating"
    pair_close        = {}            # pk → bool, currently within personal zone

    # Dyad accumulators (per unordered pair)
    dyads: dict[tuple, dict] = {}
    # Per-person trajectory accumulators
    subj: dict[int, dict] = {}
    events: list[dict] = []
    prev_num_groups = 0
    global_min = {"dist": float("inf"), "frame": -1, "pair": None}

    # Maps person_id → list of (ef_idx, human_idx, joints, speed_mps, pose_53)
    person_seq_data: dict[int, list[tuple[int, int, list, float, list]]] = {}
    enriched_frames = []

    for frame in frames:
        humans = frame.get("humans", [])
        n      = len(humans)
        peak_occupancy = max(peak_occupancy, n)

        ef = dict(frame)
        ef["humans"]       = [{k: v for k, v in h.items()
                               if k not in ("verts", "joints")} for h in humans]
        ef["interactions"] = []
        ef["groups"]       = []
        ef_idx = len(enriched_frames)

        if n == 0:
            enriched_frames.append(ef)
            prev_num_groups = 0
            continue

        positions = np.array([h["world_pos"] for h in humans], dtype=np.float64)

        # ── a) Gaze / facing directions ───────────────────────────────────────
        gaze_vecs = np.zeros((n, 3))
        for i in range(n):
            gaze_vecs[i] = _compute_gaze_direction(
                humans[i]["head_world"], humans[i]["world_pos"], humans[i].get("joints"))

        # ── b) Per-person: contact score, speed, scene voxel ──────────────────
        for i in range(n):
            h_id   = humans[i]["id"]
            joints = humans[i].get("joints")

            # Ground contact from real ankle joints (Y-DOWN, matches scene.ply).
            contact_score = 0.0
            if scene_tree is not None:
                query_pts = None
                if joints and len(joints) > R_ANKLE:
                    query_pts = np.array([joints[L_ANKLE], joints[R_ANKLE]], float)
                else:
                    # Fallback: pelvis vertex is Y-UP — flip Y to match Y-DOWN cloud.
                    p = positions[i].copy(); p[1] = -p[1] + 0.9
                    query_pts = p[None, :]
                dists, _ = scene_tree.query(query_pts)
                contact_score = float(np.max(np.exp(-(np.atleast_1d(dists) ** 2))))

            # Speed in m/s (displacement × effective fps), only across consecutive frames.
            speed_mps = 0.0
            if (h_id in prev_positions
                    and prev_frame_seen.get(h_id, -1) == ef_idx - 1):
                step = float(np.linalg.norm(positions[i] - prev_positions[h_id]))
                cand = step * eff_fps
                if cand <= P["max_jump_mps"]:
                    speed_mps = cand
                    speed_samples.append(speed_mps)
            prev_positions[h_id]  = positions[i].copy()
            prev_frame_seen[h_id] = ef_idx

            ef["humans"][i]["gaze_vec"]      = gaze_vecs[i].tolist()
            ef["humans"][i]["contact_score"] = round(contact_score, 4)
            ef["humans"][i]["speed"]         = round(speed_mps, 4)   # m/s
            ef["humans"][i]["action"]        = 'stationary'          # filled post-loop

            occupied_voxels.add(tuple((positions[i] / VOXEL).astype(int)))

            # Trajectory accumulation
            s = subj.setdefault(h_id, {"frames": 0, "path": 0.0, "moving": 0,
                                       "peak": 0.0, "xz": []})
            s["frames"] += 1
            s["xz"].append((float(positions[i][0]), float(positions[i][2])))
            if speed_mps > 0:
                s["path"]  += speed_mps / eff_fps           # back to metres for this step
                s["peak"]   = max(s["peak"], speed_mps)
                if speed_mps > P["move_speed_mps"]:
                    s["moving"] += 1

            person_seq_data.setdefault(h_id, []).append(
                (ef_idx, i, joints or [], speed_mps, humans[i].get("pose", [])))

        # ── c) Pairwise interactions (proxemics + gaze + dynamics) ────────────
        any_personal = any_intimate = any_mutual_gaze = False
        group_edges  = []

        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                all_pair_dists.append(dist)

                zone = _proxemics_zone(dist)
                if zone in ("intimate", "personal"):
                    any_personal = True
                if zone == "intimate":
                    any_intimate = True

                ida, idb = humans[i]["id"], humans[j]["id"]
                pk = (min(ida, idb), max(ida, idb))

                # Approach/retreat from a smoothed distance window (jitter-robust).
                win = pair_dist_win.setdefault(pk, [])
                win.append(dist)
                if len(win) > 5:
                    win.pop(0)
                old_state = pair_states.get(pk, "stable")
                new_state = "stable"
                if len(win) >= 3:
                    half = len(win) // 2
                    delta = float(np.mean(win[half:]) - np.mean(win[:half]))
                    if delta < -0.04:
                        new_state = "approaching"
                    elif delta > 0.04:
                        new_state = "retreating"
                if old_state != "approaching" and new_state == "approaching":
                    approach_events += 1
                pair_states[pk] = new_state

                mutual_gaze = bool(_pair_rays_converge(
                    positions[i], gaze_vecs[i], positions[j], gaze_vecs[j],
                    threshold=P["gaze_thresh"]))
                if mutual_gaze:
                    any_mutual_gaze = True

                cos_a        = float(np.clip(gaze_vecs[i] @ gaze_vecs[j], -1.0, 1.0))
                orient_score = round((1.0 - cos_a) / 2.0, 3)
                facing_angle = round(float(np.degrees(np.arccos(abs(cos_a)))), 1)
                dist_score   = max(0.0, 1.0 - dist / P["social_zone"])
                social_score = round(dist_score * 0.55 + orient_score * 0.45, 3)

                ef["interactions"].append({
                    "source": ida, "target": idb,
                    "distance": round(dist, 3), "zone": zone,
                    "type": "contact" if dist < P["contact_thresh"] else "proximity",
                    "mutual_gaze": mutual_gaze, "approach_state": new_state,
                    "facing_angle": facing_angle, "social_score": social_score,
                })

                # F-formation edge: close enough and oriented toward each other.
                if dist < P["group_edge_dist"] and orient_score >= P["group_edge_orient"]:
                    group_edges.append((i, j))

                # Dyad accumulation
                dy = dyads.setdefault(pk, {
                    "frames": 0, "min_dist": float("inf"), "sum_dist": 0.0,
                    "mutual_gaze": 0, "approaches": 0,
                    "zones": {"intimate": 0, "personal": 0, "social": 0, "public": 0},
                })
                dy["frames"]     += 1
                dy["min_dist"]    = min(dy["min_dist"], dist)
                dy["sum_dist"]   += dist
                dy["zones"][zone] += 1
                if mutual_gaze:
                    dy["mutual_gaze"] += 1
                if old_state != "approaching" and new_state == "approaching":
                    dy["approaches"] += 1

                # Meeting event: pair crosses into the personal zone.
                was_close = pair_close.get(pk, False)
                now_close = dist < P["personal_zone"]
                if now_close and not was_close:
                    events.append({"frame": ef_idx, "type": "meeting",
                                   "subjects": [ida, idb], "distance": round(dist, 2)})
                pair_close[pk] = now_close

                if dist < global_min["dist"]:
                    global_min.update({"dist": dist, "frame": ef_idx, "pair": [ida, idb]})

        if any_personal:
            social_frames += 1
        if any_intimate:
            personal_space_frames += 1
        if any_mutual_gaze:                       # dedup: reuse per-pair result
            gaze_convergence_events += 1

        # ── d) Groups (connected components on F-formation edges) ─────────────
        labels = _connected_components(n, group_edges)
        comp_members: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            comp_members.setdefault(lab, []).append(idx)
        gid = 0
        for members in comp_members.values():
            if len(members) >= 2:
                ids_in = [humans[m]["id"] for m in members]
                ef["groups"].append(ids_in)
                for m in members:
                    ef["humans"][m]["group"] = gid
                gid += 1
        for idx in range(n):
            ef["humans"][idx].setdefault("group", -1)

        num_groups = len(ef["groups"])
        if num_groups > prev_num_groups:
            events.append({"frame": ef_idx, "type": "group_form",
                           "count": num_groups})
        prev_num_groups = num_groups

        enriched_frames.append(ef)

    # ── Post-loop: batch action classification per person ─────────────────────
    action_distributions: dict[int, dict[str, float]] = {}
    for h_id, entries in person_seq_data.items():
        joints_seq = [e[2] for e in entries]
        speed_seq  = [e[3] for e in entries]    # m/s
        pose_seq   = [e[4] for e in entries]
        try:
            smoothed = classify_person_sequence(joints_seq, speed_seq, pose_seq)
        except Exception as exc:
            log.warning('Action classification failed for person %s: %s', h_id, exc)
            smoothed = ['stationary'] * len(entries)

        prev_act = None
        for idx, (ef_idx, hi, _, _, _) in enumerate(entries):
            action = smoothed[idx] if idx < len(smoothed) else 'stationary'
            enriched_frames[ef_idx]["humans"][hi]["action"] = action
            # Sit event on transition into sitting.
            if action == 'sitting' and prev_act != 'sitting':
                events.append({"frame": ef_idx, "type": "sit", "subjects": [h_id]})
            prev_act = action

        counts  = Counter(smoothed)
        total_p = len(smoothed) or 1
        action_distributions[h_id] = {
            k: round(100.0 * v / total_p, 1) for k, v in counts.items()}

    # Closest-contact highlight event
    if global_min["frame"] >= 0:
        events.append({"frame": global_min["frame"], "type": "closest_contact",
                       "subjects": global_min["pair"],
                       "distance": round(global_min["dist"], 2)})
    events.sort(key=lambda e: e["frame"])

    # ── Per-person trajectory metrics (D) ─────────────────────────────────────
    subjects = {}
    for h_id, s in subj.items():
        area = 0.0
        pts = np.array(s["xz"]) if s["xz"] else np.empty((0, 2))
        if len(pts) >= 3:
            try:
                from scipy.spatial import ConvexHull
                area = round(float(ConvexHull(pts).volume), 3)   # 2D hull area
            except Exception:
                rng = pts.max(0) - pts.min(0)
                area = round(float(rng[0] * rng[1]), 3)
        subjects[h_id] = {
            "frames_tracked": s["frames"],
            "path_length_m":  round(s["path"], 3),
            "pct_moving":     round(100.0 * s["moving"] / max(1, s["frames"]), 1),
            "peak_speed_mps": round(s["peak"], 3),
            "area_m2":        area,
        }

    # ── Dyad report (B) ───────────────────────────────────────────────────────
    dyad_list = []
    for (a, b), dy in dyads.items():
        fr = dy["frames"] or 1
        dyad_list.append({
            "pair":          [a, b],
            "frames":        dy["frames"],
            "closest_m":     round(dy["min_dist"], 3),
            "avg_dist_m":    round(dy["sum_dist"] / fr, 3),
            "pct_mutual_gaze": round(100.0 * dy["mutual_gaze"] / fr, 1),
            "approaches":    dy["approaches"],
            "zone_frames":   dy["zones"],
        })
    dyad_list.sort(key=lambda d: d["frames"], reverse=True)

    # ── Summary metrics ────────────────────────────────────────────────────────
    nf         = len(frames) or 1
    social_pct = round(100.0 * social_frames / nf, 1)
    avg_dist   = round(float(np.mean(all_pair_dists)), 3) if all_pair_dists else 0.0
    ps_pct     = round(100.0 * personal_space_frames / nf, 1)
    avg_speed  = round(float(np.mean(speed_samples)), 3) if speed_samples else 0.0

    if scene_pts is not None and len(scene_pts) > 0:
        bbox         = scene_pts.max(0) - scene_pts.min(0)
        total_voxels = max(1, int(np.prod(np.ceil(bbox / VOXEL))))
        scene_util   = round(min(100.0, 100.0 * len(occupied_voxels) / total_voxels), 1)
    else:
        scene_util = 0.0

    summary = {
        "social_engagement_pct":    social_pct,
        "avg_inter_human_distance": avg_dist,
        "scene_utilization_pct":    scene_util,
        "gaze_convergence_events":  gaze_convergence_events,
        "personal_space_pct":       ps_pct,
        "avg_speed_mps":            avg_speed,
        "approach_events":          approach_events,
        "peak_occupancy":           peak_occupancy,
        "action_distributions":     action_distributions,
        # New analytics
        "tracks_stitched":          n_merges,
        "event_count":              len(events),
        "group_count":              sum(1 for e in events if e["type"] == "group_form"),
        "effective_fps":            round(eff_fps, 3),
        "events":                   events,
        "dyads":                    dyad_list,
        "subjects":                 subjects,
    }

    # ── Bird's-eye heatmap ─────────────────────────────────────────────────────
    heatmap_path = os.path.join(output_dir, "heatmap.png")
    heatmap_meta = _compute_floor_heatmap(enriched_frames, heatmap_path)

    # Keep heavy nested analytics out of metadata (it mirrors summary); only the
    # scalar metrics + heatmap bounds belong there.
    scalar_summary = {k: v for k, v in summary.items()
                      if k not in ("events", "dyads", "subjects", "action_distributions")}
    metadata = {**data["metadata"], **scalar_summary}
    if heatmap_meta:
        metadata["heatmap"] = heatmap_meta

    output = {
        "metadata":          metadata,
        "camera_trajectory": data["camera_trajectory"],
        "frames":            enriched_frames,
        "summary":           summary,
    }

    with open(os.path.join(output_dir, "enriched_data.json"), "w") as f:
        json.dump(output, f)
