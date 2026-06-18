# SITL — Semantic Interaction Topology Lab

A spatial-interaction analytics platform built on [Human3R](https://arxiv.org/abs/2510.06219) (ICLR 2026). Drop in a monocular video — one phone camera, one CCTV feed — and get a fully navigable 3D scene with SMPL-X body meshes, per-person action labels, real-time interaction analytics, and a browser dashboard. No depth sensor, no multi-camera rig, no body-worn sensors.

---

## Overview

```
Video upload
    ↓ POST /upload
FastAPI  →  engine.py (subprocess)
               ↓ Human3R forward pass
               ↓ SMPL-X body meshes + 45 joints per person per frame
               ↓ Fused colored point cloud of the scene
               ↓ Camera trajectory (R, t per frame)
               → dashboard_data.json + scene.ply
    ↓ analytics.py
       ↓ Track stitching (repair fragmented person IDs)
       ↓ Hall (1966) proxemics zones per pair per frame
       ↓ Gaze direction from shoulder × body-up cross product
       ↓ Contact score (KDTree query from ankle joints to scene)
       ↓ Action classification (biomechanical heuristics + MotionBERT refinement)
       ↓ Groups (F-formations) · dyad report · event detection · trajectory metrics
       ↓ Bird's-eye floor heatmap (matplotlib → PNG)
       → enriched_data.json + heatmap.png
    ↓ Three.js dashboard
       → point cloud + SMPL-X meshes + overlays + timeline + metrics panel
```

Human3R performs one-shot monocular reconstruction: SMPL-X body meshes for every person, a dense colored point cloud of the environment, and the full camera trajectory — all from a single RGB video in one forward pass, in shared world coordinates.

---

## Features

### 3D Viewer

**Rendering layers** (each independently togglable):

| Layer | Description |
|-------|-------------|
| Scene cloud | Colored point cloud from fused depth maps; Y-flipped for Three.js |
| SMPL-X mesh | Full 10,475-vertex surface mesh per person, MeshPhongMaterial |
| Skeleton | 22 joint spheres + 21 cylinder bones from SMPL-X body joints |
| Gaze arrows | ArrowHelper from pelvis, direction = shoulder × body-up cross product |
| Trajectory trails | Last 30 world positions per person, fading opacity |
| Proximity edges | Red < 0.45 m · Orange < 1.2 m · Yellow < 3.7 m; white sphere when mutual gaze detected |
| Floor heatmap | Bird's-eye density texture on a PlaneGeometry at floor level |

**Playback controls:**
- Play/pause: `Space`
- Frame step: `←` / `→`
- Speed presets: 0.25× · 0.5× · 1× · 2× · 4× (keys `1`–`5`)
- Loop toggle, screenshot (PNG), live FPS counter

**Camera presets:** Perspective · Front · Follow (all with −Z offset so subjects face camera on first load)

**Interaction timeline:** Clickable color-coded strip above the scrubber — grey = no interaction, yellow = social zone, red = contact range. Click any segment to seek.

**Action timeline:** Per-person horizontal strips color-coded by current action; expandable below the interaction timeline. Click any segment to seek.

**Action labels overlay:** 2D canvas overlay drawn each animation frame; projects each person's head position to screen space and renders a colored pill badge with the current action label.

**Grid floor:** Positioned at −box.max.y after PLY load so subjects stand on it.

**Memory-safe fetching:** AbortController cancels stale JSON fetches on unmount/job change; geometry and material disposed on every frame transition.

---

### Analytics (analytics.py)

Summary metrics, all grounded in published social science:

| Metric | Formula | Research basis |
|--------|---------|----------------|
| Social Engagement % | % frames where any pair distance < 1.2 m | Hall (1966) personal zone |
| Avg Inter-Person Distance | Mean 3D pairwise distance across all frames, all pairs | — |
| Scene Utilization % | % of 0.5 m voxels within reach of any person | — |
| Gaze Convergence Events | Frames where ≥ 2 gaze rays pass within 1 m of each other | Argyle & Cook (1976) |
| Intimate Zone % | % frames where any pair distance < 0.45 m | Hall (1966) intimate zone |
| Avg Movement Speed | Mean pelvis speed in m/s, normalised by effective fps (fps ÷ subsample) | — |
| Approach Events | Transitions from stable/retreating → approaching per pair (window-smoothed) | Goffman (1971) |
| Peak Occupancy | Max simultaneous tracked persons in any frame | — |
| Group Formations | Times a new F-formation (close + mutually oriented cluster) forms | Kendon (1990) |
| Detected Events | Total auto-detected highlights (meetings, group forms, sits, closest contact) | — |

**Higher-order analytics** (in `enriched_data.json` → `summary`):
- **Groups (F-formations):** per-frame connected components on the proximity + orientation graph; each human carries a `group` id, each frame a `groups` list.
- **Dyad report (`summary.dyads`):** per pair — frames together, closest/avg distance, % mutual gaze, approach count, frames in each proxemics zone.
- **Events (`summary.events`):** `meeting` (pair enters personal zone), `group_form`, `sit`, and the single `closest_contact` moment, each with a frame index.
- **Per-person trajectory (`summary.subjects`):** path length (m), % time moving, peak speed (m/s), area covered (m²).
- **Track stitching:** fragmented person IDs are greedily merged by position/velocity continuity before any metric is computed (`summary.tracks_stitched`).

**Per-frame interaction record:**
```
distance · zone (intimate/personal/social/public)
mutual_gaze (bool) · approach_state (approaching/stable/retreating)
facing_angle (degrees) · social_score (distance × 0.55 + orientation × 0.45)
```

**Contact score:** KDTree built once from scene.ply; per person, query from the real SMPL-X ankle joints (Y-down, same frame as the cloud). `score = exp(−dist²)`.

**Gaze direction:** Preferred path uses joints (Y-down): `cross(body_up, shoulder_vec)`. Falls back to `head_world − world_pos` with Y-component flipped when joints unavailable.

**Speed normalisation:** displacement is converted to m/s using the clip's `effective_fps` (recorded by engine.py), so speed and the locomotion thresholds are correct at any subsample.

**Floor heatmap:** Gaussian-smoothed density grid from all pelvis XZ positions across the video; saved as `heatmap.png` and referenced in `enriched_data.json` metadata for Three.js plane positioning.

---

### Action Recognition (action_recognition.py)

Two-mode classifier with automatic fallback:

**Mode 1 — MotionBERT-assisted** (when `checkpoint/MB_lite.bin` is present):
- SMPL-X joints → H36M 17-joint remapping → DSTformer (depth=5, dim_feat=256, dim_rep=512)
- Feature velocity (temporal gradient of pooled backbone features) sharpens the run-vs-walk call
- Note: this is **not** a trained action head — labels still come from the biomechanical rules; the backbone only refines locomotion intensity (a trained head is future work)

**Mode 2 — Biomechanical heuristics** (always available, no checkpoint needed):
- Sitting: knee flexion from SMPL-X rotation vectors (primary) or knee-drop ratio (fallback)
- Reaching: wrist above shoulder line
- Bending: head-to-pelvis vertical clearance < 35% body height
- Walking/running: knee-lift ratio (pelvis_y − knee_y) / body_h, fps-agnostic
- Sequence-level gate: suppresses false walking labels on stationary reconstructions (low knee-lift std AND low mean speed)

**Six action classes:** `stationary · walking · running · sitting · reaching · bending`

**Temporal smoothing:** majority-vote over a 15-frame sliding window.

**Coordinate convention:** joints are Y-down (OpenCV world). Knee-lift ratio is negative for knees-below-pelvis (standing), less negative for walking swing phase, positive for running.

---

### Upload & Job Management (App.jsx)

- Drag-and-drop or click-to-browse video upload
- Subsample presets: 1× · 2× · 3× · 4× · 8× (higher = fewer frames = faster + lower VRAM)
- Animated progress ring + stage pills during inference
- URL hash (`#job_id`) set on complete; auto-loaded on page refresh
- Job history in `localStorage` (last 10 jobs, one-click reload)
- **Processed-cases panel** (always visible): lists completed jobs on disk via `/dev/jobs`, click to load
- Delete job button — removes outputs from disk and history
- **Dev mode:** enter any previously computed job ID and press Enter or click Load; viewer opens in ~1 second with no GPU usage

---

## Stack

| Layer | Technology |
|-------|-----------|
| 3D reconstruction | Human3R (DUSt3R backbone + Multi-HMR human encoder, ViT-L @ 896 px) |
| Body model | SMPL-X (10,475 vertices · 127 joints · 53 rotation parameters) |
| Action recognition | MotionBERT DSTformer + biomechanical heuristics |
| Backend | FastAPI · uvicorn · SSE (sse-starlette) |
| Analytics | NumPy · SciPy (KDTree · gaussian_filter) · Matplotlib |
| Frontend | React 18 UMD · Three.js r134 · Babel standalone · no build step |

---

## Hardware Requirements

- Linux with CUDA 12.4+
- VRAM: ~12 GB for `896L` (ViT-L); `672B` works at ~8 GB; `672S` at ~5 GB
- RAM: minimum 16 GB; 32 GB recommended for long videos

**Video length guidance (subsample rate vs. VRAM pressure):**

| Video length | Recommended subsample | Approx processed frames |
|-------------|----------------------|------------------------|
| < 1 min | 1–2 | < 120 |
| 1–3 min | 3–4 | 90–180 |
| 3–10 min | 6–8 | 100–200 |
| > 10 min | 10–16 | 100–200 |

Keeping processed frames under ~200 avoids OOM during post-processing. See [Memory Management](#memory-management) below.

---

## Installation

```bash
git clone https://github.com/YuvrajPuyam/human3r-sitl.git
cd human3r-sitl

conda create -n human3r128 python=3.11 cmake
conda activate human3r128

conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
conda install 'llvm-openmp<16'
pip install fastapi uvicorn sse-starlette scipy
pip install git+https://github.com/nerfstudio-project/gsplat.git
pip install evo open3d

# Compile CUDA RoPE kernels (required)
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../

# Download SMPL-X body model weights
bash scripts/fetch_smplx.sh

# Download Human3R checkpoint (default: 896L, ~1.6 GB)
huggingface-cli download faneggg/human3r human3r_896L.pth --local-dir ./src
```

### Checkpoints

| Model | Resolution | Backbone | Approx speed | VRAM |
|-------|-----------|----------|-------------|------|
| `human3r_672S.pth` | 672 px | ViT-S | ~15 FPS | ~5 GB |
| `human3r_672B.pth` | 672 px | ViT-B | ~11 FPS | ~8 GB |
| `human3r_672L.pth` | 672 px | ViT-L | ~7 FPS | ~12 GB |
| `human3r_896L.pth` | 896 px | ViT-L | ~5 FPS | ~12 GB — **default** |

### MotionBERT (optional — improves action recognition)

```bash
# Download MB_lite.bin (~61 MB) from the MotionBERT release page
# Place at:
sitl/third_party/MotionBERT/checkpoint/MB_lite.bin
```

Without the checkpoint, the system falls back to biomechanical heuristics automatically.

---

## Running

### SITL Dashboard

```bash
conda activate human3r128
cd sitl
uvicorn backend.main:app --reload --port 8000
```

Open **http://localhost:8000/app**, drop a video, choose a subsample rate, and run. The SSE stream pushes stage and progress updates to the UI in real time.

**Dev mode — skip inference entirely:**
In the sidebar, enter a previously computed job ID and press Enter (or click Load). The viewer opens in ~1 second. Useful for iterating on frontend without re-running the 4–15 minute pipeline.

Current verified job on disk: **`06c9ccd8`**

### Headless inference only

```bash
# Run from project root
CUDA_VISIBLE_DEVICES=0 python engine.py \
    --model_path src/human3r_896L.pth \
    --seq_path examples/your_video.mp4 \
    --output_dir sitl/outputs/my_job \
    --subsample 3 --size 512
```

Outputs written to `--output_dir`: `dashboard_data.json` + `scene.ply` (+ `verts.bin` and `verts_index.json` when `--export_verts` is set).

#### engine.py options

| Flag | Default | Description |
|------|---------|-------------|
| `--subsample N` | 1 | Use every Nth frame |
| `--size N` | 512 | Input resolution |
| `--ply_stride N` | 3 | Fuse every Nth frame into PLY (~40 MB at 3) |
| `--conf_thresh F` | 1.0 | Confidence threshold for PLY inclusion |
| `--voxel_size F` | 0.02 | Voxel dedup size in metres |
| `--max_depth F` | 12.0 | Max point depth in metres |
| `--export_verts` | off | Export all 10,475 SMPL-X vertices per person per frame to a Float32 binary side-car (`verts.bin` + `verts_index.json`), ~5× smaller than JSON and loadable via `arrayBuffer()`. Off by default. (A full-resolution clip as inline JSON exceeds Chrome's ~512 MB max string length and won't parse — hence the binary.) |
| `--use_ttt3r` | off | Enable test-time training for higher quality |

---

## API Reference

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/upload` | Save video, returns `{job_id}` |
| `POST` | `/run/{job_id}?subsample=N` | Start inference + analytics pipeline |
| `GET` | `/status/{job_id}` | SSE stream: `{status, stage, progress, logs}` |
| `GET` | `/results/{job_id}` | URLs to `enriched_data.json`, `scene.ply`, `heatmap.png` |
| `POST` | `/rerun-analytics/{job_id}` | Re-run analytics only; optional query overrides (`personal_zone`, `fps_override`, …) |
| `GET` | `/dev/load/{job_id}` | Register existing `outputs/{job_id}/` as completed (dev only) |
| `GET` | `/dev/jobs` | List completed jobs on disk, newest first (powers the cases panel) |
| `DELETE` | `/jobs/{job_id}` | Remove outputs from disk and job store |
| `GET` | `/jobs` | List all jobs and status (debug) |
| `GET` | `/app` | Serve frontend |

---

## Output File Schemas

### `dashboard_data.json` — written by engine.py

```json
{
  "metadata": {
    "total_frames": 229,
    "up_axis": "Y",
    "head_vertex": 4840,
    "exporter": "SITL_engine_v2",
    "source_fps": 30.0,
    "subsample": 3,
    "effective_fps": 10.0,
    "smpl_faces": [[i0, i1, i2], "...10k triangles, static, exported once"]
  },
  "camera_trajectory": [
    { "R": [[3x3 matrix]], "t": [x, y, z] }
  ],
  "frames": [{
    "frame_id": 0,
    "humans": [{
      "id": 0,
      "world_pos":  [x, y, z],
      "head_world": [x, y, z],
      "pose":       [[53 x 3 rotation vectors]],
      "shape":      [10 betas],
      "joints":     [[x,y,z], "...45 world-frame body joints"]
    }]
  }]
}
```

With `--export_verts`, the 10,475 vertices per person go to `verts.bin` (Float32LE) plus a small `verts_index.json` that maps each frame's humans to a block offset — **not** inline in this JSON.

### `enriched_data.json` — written by analytics.py

`verts` and `joints` are **not** copied here (the file stays under 5 MB for the browser). Viewer.jsx fetches `dashboard_data.json` in parallel and merges them client-side.

```json
{
  "metadata": {
    "total_frames": 229,
    "smpl_faces": ["...passed through from dashboard_data"],
    "social_engagement_pct": 63.2,
    "avg_inter_human_distance": 1.84,
    "scene_utilization_pct": 41.0,
    "gaze_convergence_events": 2,
    "personal_space_pct": 12.1,
    "avg_speed_mps": 1.26,
    "approach_events": 7,
    "peak_occupancy": 3,
    "group_count": 4,
    "event_count": 23,
    "tracks_stitched": 0,
    "effective_fps": 10.0,
    "heatmap": { "x_min": -1.2, "x_max": 3.4, "z_min": 0.1, "z_max": 4.8 }
  },
  "camera_trajectory": ["..."],
  "frames": [{
    "frame_id": 0,
    "humans": [{
      "id": 0,
      "world_pos":     [x, y, z],
      "head_world":    [x, y, z],
      "gaze_vec":      [x, y, z],
      "contact_score": 0.85,
      "speed":         1.26,
      "action":        "walking",
      "group":         0,
      "pose":          [[53 x 3]],
      "shape":         [10]
    }],
    "interactions": [{
      "source": 0, "target": 1,
      "distance": 1.24,
      "zone": "personal",
      "type": "proximity",
      "mutual_gaze": false,
      "approach_state": "approaching",
      "facing_angle": 34.7,
      "social_score": 0.61
    }],
    "groups": [[0, 1]]
  }],
  "summary": {
    "...scalar metrics as above": true,
    "action_distributions": { "0": { "walking": 54.2, "stationary": 45.8 } },
    "events":   [{ "frame": 21, "type": "closest_contact", "subjects": [0, 2], "distance": 0.29 }],
    "dyads":    [{ "pair": [0, 2], "frames": 278, "closest_m": 0.29, "pct_mutual_gaze": 31.3 }],
    "subjects": { "0": { "path_length_m": 8.87, "pct_moving": 90.4, "peak_speed_mps": 4.17, "area_m2": 0.16 } }
  }
}
```

(The nested `events`/`dyads`/`subjects`/`action_distributions` live in `summary` only; `metadata` keeps just the scalar metrics so it stays small.)

---

## Coordinate System

Human3R outputs in OpenCV convention (Y-down, Z-forward). SITL carries this through to analytics, then Y-flips for Three.js at render time.

| Data | Convention | Notes |
|------|-----------|-------|
| SMPL-X vertices (`world_pos`, `head_world`) | Y-up | Vertex positions from the SMPL-X model |
| SMPL-X joints | Y-down | Joint positions from the forward pass |
| PLY point cloud | Y-down | Fused from camera-frame depth maps |
| Analytics (distances, gaze, speed) | Y-down | Computed pre-flip |
| Three.js viewer | Y-up | `fy([x,y,z])` flips joints; `scale.y = -1` flips PLY |

```javascript
// Applied to all joint/position data in Viewer.jsx:
const fy = ([x, y, z]) => new THREE.Vector3(x, -y, z);

// Applied to the point cloud geometry:
points.scale.y = -1;
```

SMPL-X constants (verified by diagnostic):
- Pelvis: vertex/joint **0**
- Head: vertex **4840** (confirmed Y-delta ~0.6 m above pelvis)

---

## Memory Management

engine.py frees memory in three explicit stages to prevent OOM kills on long videos:

```
inference → del views, model → cuda.empty_cache()   # free ~7 GB VRAM + view images
         → process_outputs()
         → del outputs → cuda.empty_cache()          # free raw inference tensors
         → save_fused_scene_ply()
         → del pts3ds_other, conf, msks, colors       # free per-frame arrays
         → export_dashboard_json()
```

If a run crashes with only `inference_logs.txt` in the output directory (no PLY, no JSON), the process was OOM-killed. Check the last line of the log:
- Ends at model load → OOM during `.to(device)`, try a smaller checkpoint
- Ends at `Starting inference` / mid-inference → too many frames, increase `--subsample`
- Ends at `Processing inference outputs...` → post-processing OOM, increase `--subsample` or use `--ply_stride 5`

---

## Repository Layout

```
human3r-sitl/
├── engine.py                    ← Headless SITL runner (runs from project root)
├── demo.py                      ← Original Human3R interactive viewer (unmodified)
├── add_ckpt_path.py             ← Adds model checkpoint path to sys.path
├── requirements.txt
├── scripts/                     ← SMPL-X + data download helpers
├── src/
│   ├── dust3r/                  ← Core model: ARCroco3DStereo, inference, losses
│   │   ├── model.py
│   │   ├── smpl_model.py        ← SMPL-X wrapper (SMPL_Layer)
│   │   └── inference.py        ← inference_recurrent_lighter()
│   ├── croco/                   ← CroCoNet backbone + CUDA RoPE kernels
│   └── mhmr/                    ← Multi-HMR human encoder blocks
├── eval/                        ← Evaluation scripts (3DPW, RICH, EMDB, TUM, Bonn)
└── sitl/
    ├── backend/
    │   ├── main.py              ← FastAPI app + all routes
    │   ├── pipeline.py          ← Stage orchestrator (inference → analytics → done)
    │   └── workers/
    │       ├── inference.py     ← Async subprocess wrapper; parses stdout progress
    │       ├── analytics.py     ← Proxemics · gaze · contact · heatmap
    │       ├── action_recognition.py  ← MotionBERT + biomechanical heuristics
    │       └── export.py        ← (legacy, superseded by engine.py)
    ├── frontend/
    │   ├── index.html           ← CDN loader: Three.js r134 + React 18 + Babel standalone@7
    │   ├── Viewer.jsx           ← Three.js scene, playback, overlays, analytics panel
    │   └── App.jsx              ← Upload UI, SSE state machine, job history, cases panel
    ├── tools/
    │   └── pack_existing_verts.py  ← Convert a job's inline-JSON verts → verts.bin side-car
    ├── third_party/
    │   └── MotionBERT/          ← DSTformer backbone for action recognition
    └── outputs/                 ← Runtime job outputs, one subdirectory per job_id
        └── <job_id>/
            ├── inference_logs.txt     ← Streamed stdout from engine.py
            ├── dashboard_data.json    ← Per-frame SMPL-X + camera data (verts excluded)
            ├── verts.bin             ← Float32 SMPL-X vertices (with --export_verts)
            ├── verts_index.json      ← Frame→block map for verts.bin
            ├── scene.ply              ← Fused colored scene point cloud (binary PLY)
            ├── enriched_data.json     ← dashboard_data + analytics (< 5 MB)
            └── heatmap.png            ← Bird's-eye floor occupancy map
```

---

## Frontend Development Notes

- **No build step.** React 18 UMD + Babel standalone — edit `.jsx`, hard-refresh, done.
- **Cache busting:** bump `?v=N` on the script tag in `index.html` after every JSX edit, then `Ctrl+Shift+R`. Babel output is cached aggressively.
- **Babel is pinned to `@babel/standalone@7`** — Babel 8's preset-react defaults to the automatic JSX runtime, which injects `import` statements that crash a plain (non-module) script and blank the page.
- **Verts** load from `verts.bin` via `arrayBuffer()` (no JSON string-length limit); the Viewer flattens them to a per-human `vbuf` and builds the mesh from it.
- Three.js `Object3D.position` is non-writable — never use `Object.assign`. Use `.position.copy(vec)`.
- OrbitControls and PLYLoader loaded as `examples/js` UMD globals extending `window.THREE`.

---

## Evaluation

```bash
bash eval/global_human/run.sh   # MPJPE / PA-MPJPE / PVE on 3DPW, RICH, EMDB
bash eval/relpose/run.sh        # Camera pose on TUM-dynamics, Bonn
bash eval/video_depth/run.sh    # Depth estimation benchmarks
```

---

## Acknowledgements

Built on top of:

- **Human3R** — *Generalizable 3D Human Reconstruction in the Wild*, Fan et al. (ICLR 2026, [arXiv:2510.06219](https://arxiv.org/abs/2510.06219))
- **DUSt3R** — Dense Unconstrained Stereo 3D Reconstruction, Wang et al.
- **Multi-HMR** — Multi-person Human Mesh Recovery from a Single Image, Baradel et al.
- **SMPL-X** — Expressive Body: Unified Modeling of the Face, Body and Hands, Pavlakos et al. (2019)
- **MotionBERT** — Unified Pretraining for Human Motion Analysis, Zhu et al. (2023)

Interaction metrics grounded in:
- E.T. Hall (1966) — *The Hidden Dimension* (proxemics zones)
- Argyle & Cook (1976) — *Gaze and Mutual Gaze*
- Goffman (1971) — *Relations in Public* (approach–avoidance dynamics)
- Kendon (1967) — *Some Functions of Gaze-Direction in Social Interaction*
- Kendon (1990) — *Conducting Interaction* (F-formations / o-space, group detection)

---

## License

SITL dashboard code (`sitl/`, `engine.py`) — MIT.  
Human3R model code in `src/` is subject to the original [Human3R license](https://github.com/fanegg/Human3R).
