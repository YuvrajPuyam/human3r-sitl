# CLAUDE.md
# Human3R × SITL — Claude Code Session Guide

This file is the single source of truth for Claude Code working in this
repository. Read it fully before touching any file.

---

## What This Repository Contains

Two things that share the same codebase:

**1. Human3R** — a published research model (ICLR 2026, arXiv 2510.06219).
Given a monocular video it reconstructs in one forward pass:
- Multiple people as SMPL-X body meshes in world coordinates
- Dense 3D scene point cloud in the same world coordinate system
- Camera trajectory (R, t per frame)
Do not modify the core model code.

**2. SITL (Semantic Interaction Topology Lab)** — a local web app we built
on top of Human3R. It takes a video as input, runs Human3R headlessly via
`engine.py`, computes interaction analytics, and visualizes everything in a
React + Three.js dashboard. Lives entirely in `sitl/`.

---

## Environment

```bash
conda activate human3r128      # always use this env
```

Full setup (only needed on fresh machine):
```bash
conda create -n human3r128 python=3.11 cmake
conda activate human3r128
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
conda install 'llvm-openmp<16'
pip install git+https://github.com/nerfstudio-project/gsplat.git
pip install evo open3d
pip install fastapi uvicorn sse-starlette scipy

# Compile CUDA RoPE kernels (required)
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

Download models:
```bash
bash scripts/fetch_smplx.sh
huggingface-cli download faneggg/human3r human3r_896L.pth --local-dir ./src
```

Available checkpoints:

| Model | Resolution | Speed |
|---|---|---|
| `human3r_672S.pth` | 672 / ViT-S | ~15 FPS |
| `human3r_672B.pth` | 672 / ViT-B | ~11 FPS |
| `human3r_672L.pth` | 672 / ViT-L | ~7 FPS |
| `human3r_896L.pth` | 896 / ViT-L | ~5 FPS — best quality, default |

---

## Repository Layout

```
Human3R/                              ← project root, run everything from here
├── CLAUDE.md                         ← this file
├── demo.py                           ← original Human3R interactive viewer (DO NOT MODIFY)
├── engine.py                         ← our headless runner for SITL (alongside demo.py
│                                        so it can import add_ckpt_path and demo.py)
├── certify_week1.py                  ← data pipeline certification (9-check validator)
├── src/
│   ├── dust3r/                       ← core model, losses, dataset loaders
│   │   ├── model.py                  ← ARCroco3DStereo — the main model class
│   │   ├── smpl_model.py             ← SMPL-X wrapper
│   │   └── inference.py             ← inference_recurrent_lighter()
│   ├── croco/                        ← CroCoNet backbone + CUDA RoPE kernels
│   └── mhmr/                         ← Multi-HMR human encoder blocks
├── eval/                             ← evaluation scripts
├── config/                           ← Hydra training configs
├── scripts/                          ← data download helpers
└── sitl/                             ← SITL dashboard (our addition)
    ├── backend/
    │   ├── main.py                   ← FastAPI app
    │   ├── inference.py              ← async subprocess wrapper for engine.py
    │   ├── pipeline.py               ← orchestrator: inference → analytics → done
    │   └── workers/
    │       └── analytics.py          ← proximity + gaze + contact math
    ├── frontend/
    │   ├── index.html                ← loads Three.js r134 + React 18 UMD + Babel
    │   ├── App.jsx                   ← upload UI + pipeline stages + SSE state machine
    │   └── Viewer.jsx                ← Three.js scene + timeline + analytics panel
    └── outputs/                      ← created at runtime, one subdir per job_id
        └── <job_id>/
            ├── dashboard_data.json   ← per-frame humans, camera, tracking IDs
            ├── scene.ply             ← fused colored room point cloud
            └── enriched_data.json   ← dashboard_data + proximity + gaze + contact
```

---

## How to Run

### Original Human3R viewer (interactive, opens Viser on port 8080)
```bash
# From project root
CUDA_VISIBLE_DEVICES=0 python demo.py \
    --model_path src/human3r_896L.pth --size 512 \
    --seq_path examples/GoodMornin1.mp4 \
    --subsample 1 --use_ttt3r --vis_threshold 2 \
    --downsample_factor 1 --reset_interval 100
```

### Headless inference only (used by SITL)
```bash
# From project root — engine.py must run from here
CUDA_VISIBLE_DEVICES=0 python engine.py \
    --model_path src/human3r_896L.pth \
    --seq_path examples/GoodMornin1.mp4 \
    --output_dir sitl/outputs/test01 \
    --subsample 3 --size 512
```

### SITL dashboard (the web app)
```bash
# From sitl/ directory
cd sitl
uvicorn backend.main:app --reload --port 8000
# Open http://localhost:8000/app
```

### Dev mode — skip inference entirely (CRITICAL for frontend development)
Inference takes ~4 minutes. Never re-run it just to test UI changes.
Use the **Dev — Load Existing Job** panel in the sidebar (visible when idle): enter a job ID and click Load. The viewer opens in ~1 second with no pipeline run.

Current working job_id with verified outputs on disk: **06c9ccd8**

Under the hood it hits `GET /dev/load/{job_id}`, which registers the existing
`outputs/{job_id}/` directory as a completed job in memory.

**Cache busting:** whenever you edit a JSX file, bump `?v=N` on its `<script>` tag
in `frontend/index.html`, then hard-refresh (`Ctrl+Shift+R`). The browser caches
Babel-transpiled JSX aggressively.

### Evaluation
```bash
bash eval/global_human/run.sh   # MPJPE/PA-MPJPE/PVE on 3DPW, RICH, EMDB
bash eval/relpose/run.sh        # Camera pose on TUM-dynamics, Bonn
bash eval/video_depth/run.sh    # Depth estimation benchmarks
```

### Training
```bash
cd src/
accelerate launch --multi_gpu train.py --config-name trian_human3r
```

---

## SITL Architecture

### Data flow
```
Video upload
    ↓ POST /upload
FastAPI saves to uploads/<job_id>.ext
    ↓ POST /run/{job_id}
pipeline.py orchestrates:
    Stage 1: inference.py spawns engine.py as subprocess
             engine.py streams "Label: N/M" progress lines to stdout
             inference.py parses stdout → updates jobs[job_id]
    Stage 2: analytics.py reads dashboard_data.json + scene.ply
             writes enriched_data.json
    Stage 3: jobs[job_id]["status"] = "completed"
    ↓ GET /results/{job_id}
Browser fetches enriched_data.json + scene.ply
    ↓
Three.js viewer renders point cloud + overlays
```

### API routes (main.py)
```
POST   /upload                     → saves video, returns {job_id}
POST   /run/{job_id}?subsample=N   → starts pipeline background task
GET    /status/{job_id}            → SSE stream of job state (1 event/sec)
GET    /results/{job_id}           → URLs to enriched_data.json + scene.ply
GET    /dev/load/{job_id}          → DEV ONLY: register existing outputs as completed
GET    /jobs                       → list all jobs (debug)
DELETE /jobs/{job_id}              → cleanup outputs + uploaded video
GET    /app                        → serves sitl/frontend/index.html
/frontend/*                        → static: Viewer.jsx, App.jsx
/outputs/*                         → static: PLY + JSON files
```

### Job state schema (in-memory dict, fine for local use)
```python
jobs[job_id] = {
    "status":     "uploaded|queued|processing|completed|failed",
    "video_path": "uploads/<job_id>.mp4",
    "filename":   "original_name.mp4",
    "stage":      0,    # 1=inference  2=analytics  3=complete
    "progress":   0,    # 0-100, parsed from engine.py stdout
    "logs":       [],   # last 50 sanitized log lines
}
```

### pipeline.py call signature (used by main.py)
```python
await run_pipeline(job_id, jobs, subsample, video_path)
# Must update jobs[job_id]["stage"] at each step
# Must set jobs[job_id]["status"] = "completed" or "failed"
# Must wrap in try/except — one failure must not crash the server
```

---

## Output File Schemas

### dashboard_data.json (written by engine.py — exporter v2)
```json
{
  "metadata": {
    "total_frames": 229,
    "up_axis": "Y",
    "head_vertex": 4840,
    "exporter": "SITL_engine_v2",
    "smpl_faces": [[i0,i1,i2], ...]   // ~10k triangles, static topology, exported ONCE
  },
  "camera_trajectory": [
    {"R": [[3x3 matrix]], "t": [x, y, z]}
  ],
  "frames": [{
    "frame_id": 0,
    "humans": [{
      "id": 0,
      "world_pos":  [x, y, z],        // pelvis (vertex 0)
      "head_world": [x, y, z],        // head (vertex 4840)
      "pose":  [[53 x 3 rotvec]],
      "shape": [10 betas],
      "verts": [[x,y,z], ...],        // all 10475 SMPL-X vertices (world frame)
      "joints": [[x,y,z], ...]        // 45 body joints (world frame), if smpl_j3d available
    }]
  }]
}
```

### enriched_data.json (analytics.py adds gaze/contact/interactions to dashboard_data)
```json
{
  "metadata": {
    "total_frames": 229,
    "up_axis": "Y",
    "smpl_faces": [...],              // passed through from dashboard_data
    "social_engagement_pct": 63.2,
    "avg_inter_human_distance": 1.84,
    "scene_utilization_pct": 41.0,
    "gaze_convergence_events": 2
  },
  "camera_trajectory": [...],
  "frames": [{
    "frame_id": 0,
    "humans": [{
      "id": 0,
      "world_pos":  [x, y, z],
      "head_world": [x, y, z],
      "gaze_vec":   [x, y, z],        // normalize(head - pelvis)
      "contact_score": 0.85,
      // verts and joints are NOT in enriched_data.json (too large — 440 MB+)
      // Viewer.jsx fetches dashboard_data.json in parallel and merges them
      "pose":  [[53 x 3]],
      "shape": [10]
    }],
    "interactions": [{
      "source": 0, "target": 1,
      "distance": 1.24,
      "type": "contact"               // "contact" < 0.7m, "proximity" < 1.5m
    }]
  }],
  "summary": {
    "social_engagement_pct": 63.2,
    "avg_inter_human_distance": 1.84,
    "scene_utilization_pct": 41.0,
    "gaze_convergence_events": 2
  }
}
```

---

## Verified Data Facts — Do Not Change These

These were confirmed by running the pipeline and diagnostic scripts.
Do not "fix" them without re-running the diagnostics.

| Fact | Value | How confirmed |
|---|---|---|
| SMPL-X pelvis vertex | 0 | Always vertex 0 |
| SMPL-X head vertex | **4840** | Vertex search diagnostic, Y-delta 0.6m |
| World coordinate Y axis | **Y-down** (OpenCV) | head[1] - pelvis[1] is negative in raw data |
| Three.js Y-flip required | **Yes** | `points.scale.y = -1` on PLY mesh; `fy([x,y,z])` on all human positions |
| Head-pelvis Y delta | ~0.6m | Confirmed plausible (top of skull) |
| Max inter-frame jump | 0.309m | Velocity check passed — no teleport |
| Longest tracked run | 104 frames | Tracking stable |
| Camera travel (test video) | 8.604m | Camera motion confirmed |
| Color tensor shape | [1, H, W, 3] | Always squeeze dim 0 before imwrite |
| JSON camera key | `camera_trajectory` | Not `camera` — certification checks this |
| engine.py final line | `print("1/1")` | Signals 100% to inference.py parser |
| PLY format | **binary_little_endian** | Switched from ASCII; PLYLoader r134 handles both |

---

## engine.py — Key Behaviours

- Must be run from `Human3R/` root (imports `add_ckpt_path` and `demo.py`)
- Streams all output to stdout with `flush=True` — never use `print()` without flush
- Progress lines must match `r'(\d+)/(\d+)'` regex for inference.py to parse them
  - Use: `print(f"Label: {current}/{total}", flush=True)`
- Final line must be `print("1/1", flush=True)` to signal completion
- Writes two files: `scene.ply` (binary PLY) and `dashboard_data.json` to `--output_dir`

### PLY fusion parameters (with baselines)
| Parameter | Current | Baseline | Effect |
|---|---|---|---|
| `--ply_stride` | 3 | 5 | Every 3rd frame fused → ~40 MB PLY |
| `--conf_thresh` | 1.0 | 1.2 | Includes more points |
| `--voxel_size` | 0.02m | 0.02m | Merged at 2 cm voxels |
| `--max_depth` | 12.0m | 12.0m | Unchanged |
| `--export_verts` | False | — | Off by default; 10k verts/frame = 500+ MB. Use `--export_verts` only when SMPL-X mesh rendering is needed |
| `msk_thresh` | 0.2 | 0.5 | Lower = more human pixels masked; passed via `args.msk_threshold` |
| dilation iters | 5 | 2 | More px around human silhouette excluded |

- Human mask convention: **high msk value (≥ 0.2) = human pixel** (model outputs sigmoid near 1 for human). Threshold was confirmed by demo.py which keeps scene via `msk < threshold`. Do not invert.
- Human mask exclusion uses **scipy `binary_dilation(iterations=5)`** to fill silhouette edges and thin limbs (was 2 — insufficient for arms/legs)
- Mask spatial resolution may differ from pts3d resolution — engine.py bilinearly resizes mask to `(H, W)` before thresholding if shapes differ
- PLY written as **binary_little_endian** via numpy structured array (baseline: ASCII for-loop — 3-5× larger, 10× slower)
- Voxel key formula uses offset 10,000 to handle negative coordinates safely

### SMPL-X outputs extracted per forward pass
- `smpl_v3d` → all 10475 world-frame surface vertices per human (`"verts"` in JSON)
- `smpl_j3d` → 45 world-frame body joints per human (`"joints"` in JSON, if key exists)
- `smpl_layer.bm_x.faces` → triangle face indices, exported ONCE in metadata (`"smpl_faces"`)

---

## analytics.py — What It Must Compute

Input: `outputs/<job_id>/dashboard_data.json` + `outputs/<job_id>/scene.ply`
Output: `outputs/<job_id>/enriched_data.json`

Four computations:

**1. Pairwise distances** (per frame)
```python
dist = np.linalg.norm(person_a["world_pos"] - person_b["world_pos"])
type = "close" if dist < 1.5 else "near" if dist < 3.0 else "far"
```

**2. Gaze vectors** (per person per frame)
```python
gaze = np.array(h["head_world"]) - np.array(h["world_pos"])
gaze_vec = (gaze / np.linalg.norm(gaze)).tolist()
```

**3. Contact scores** (per person per frame)
```python
# Build KDTree ONCE from scene.ply points
tree = KDTree(scene_points)
# Query from estimated foot position and pelvis
foot_pos = np.array(h["world_pos"]) - np.array([0, 0.95, 0])
dist, _ = tree.query(foot_pos)
contact_score = float(np.exp(-dist**2))
```

**4. Summary metrics** (in metadata)
- `social_engagement_pct`: % frames where any pair distance < 1.5m
- `avg_inter_human_distance`: mean of all pairwise distances across all frames
- `scene_utilization_pct`: % of PLY voxels within 0.5m of any human
- `gaze_convergence_events`: frames where 2+ gaze rays have closest approach < 1m

---

## frontend — Three.js Viewer Notes

- Three.js version: **r134 UMD** (loaded from CDN in index.html)
- OrbitControls and PLYLoader loaded as `examples/js` globals extending `window.THREE`
- React 18 UMD + Babel standalone — no build step, no webpack
- Static files served at `/frontend/*` by FastAPI
- Script tags in index.html must use `/frontend/Viewer.jsx` and `/frontend/App.jsx`
  (not relative `./` paths — those 404 because the browser resolves from `/app`)
- Current cache-bust version: **`?v=12`** (Viewer.jsx) / **`?v=6`** (App.jsx) — bump whenever JSX files change, then `Ctrl+Shift+R`
- PLYLoader handles both ASCII and binary_little_endian PLY with vertex colors
- After pipeline completes: App.jsx SSE handler sets state → Viewer mounts →
  fetches enriched_data.json + scene.ply from `/outputs/<job_id>/`

### Viewer.jsx rendering layers (in order)
1. **Scene point cloud** — PLY loaded via PLYLoader, `scale.y = -1` to flip OpenCV→Three.js
2. **SMPL-X surface mesh** — MeshPhongMaterial, built from `h.verts` + `metadata.smpl_faces`; requires new inference run with engine v2
3. **Skeleton** — 22 joint spheres + 21 cylinder bones from `h.joints[0:22]`; requires new inference run
4. **Gaze arrows** — ArrowHelper from pelvis, direction = normalize(head - pelvis)
5. **Trajectory trails** — last 30 world positions per person ID, fading opacity
6. **Proximity/contact edges** — yellow (< 1.5m) or red (< 0.7m) lines between humans

### New features in Viewer.jsx (v9)
- **Playback**: play/pause (Space), step (←/→), speed presets 0.25×–4× (keys 1–5), loop toggle, 30 fps base rate
- **Layer panel** (floating top-right): toggle each of the 7 layers + grid; live point-size slider
- **Camera presets**: Persp / Front / Follow (Top removed; all presets use –Z offset so subjects face camera)
- **Grid floor fix**: grid.position.y set to –box.max.y after PLY load so subjects stand ON the grid
- **Default camera**: negative-Z offset so first view shows subject fronts, not backs
- **Interaction timeline**: clickable color-coded strip above scrubber (grey=none, yellow=proximity, red=contact)
- **Screenshot**: 📷 button saves current frame as PNG
- **FPS counter**: live DOM write (no React re-renders) next to frame count
- **AbortController**: cancels stale JSON fetches on unmount/reload
- **Proper disposal**: geometry + material dispose on every frame transition

### New features in Viewer.jsx (v10)
- **Action labels canvas**: 2D transparent canvas overlay drawn in the 60 fps animation loop; projects each person's head_world to screen space and renders a colored action label pill (stationary/walking/running/sitting/reaching/bending)
- **Action timeline**: per-person colored horizontal strips above the interaction timeline; color = ACTION_COLORS[action]; legend row below; clickable to seek
- **Floor heatmap layer**: when `enriched_data.json` metadata includes `heatmap` bounds, loads `heatmap.png` as a TextureLoader texture and renders a flat PlaneGeometry at floor level (floorY + 0.01). Toggled via "Floor Heatmap" in layer panel.
- **`heatmap` added to DEFAULT_LAYERS and LAYER_DEFS**
- **`fy()` moved to module level** so both the per-frame overlay useEffect and the animation loop can use it
- **`frameRef`** keeps current frame index available to animation loop without React closure capture
- **`frameData` state** (mirror of `frameDataRef`) passed to ActionTimeline so it redraws on new job load
- **`personIds` state** — sorted unique person IDs extracted from all frames on JSON load
- **`heatmapMeta` state** — `{x_min, x_max, z_min, z_max}` from `enriched.metadata.heatmap`
- **`threeRef.current.floorY`** stored after PLY bounding box computed; used by heatmap plane positioning
- **Bottom dashboard zIndex: 3** so labels canvas (no explicit z-index) doesn't overlap controls

### New features in App.jsx (v6)
- **Sidebar moved to right** — viewer/canvas takes full left area
- **Main canvas upload screen** (idle): large centered drop zone + subsample presets + dev-load + recent jobs
- **Main canvas processing screen** (inference running): animated SVG progress ring + stage pills + last log line
- **URL hash**: `window.location.hash = '#' + job_id` set on load/complete; auto-loaded on mount
- **Job history**: last 10 jobs in `localStorage.sitl_history`; clickable panels on both main canvas and sidebar
- **Subsample presets**: quick-select buttons [1,2,3,4,8]
- **Delete job button**: 🗑 calls `DELETE /jobs/{job_id}` and removes from history
- **Enter key** on dev-load input triggers load

### Y-flip: all human positions must go through `fy()`
```javascript
const fy = ([x, y, z]) => new THREE.Vector3(x, -y, z);
// Applied to: world_pos, head_world, gaze_vec, joints, verts, interactions
// PLY cloud: points.scale.y = -1 (flip entire geometry, not individual points)
```

### THREE.Object3D.position is non-writable — never use Object.assign
```javascript
// WRONG — crashes with "Cannot assign to read only property 'position'"
Object.assign(new THREE.Mesh(...), { position: pos })
// CORRECT
const m = new THREE.Mesh(...); m.position.copy(pos);
```

---

## Current Status (Last Checkpoint)

### Working end-to-end
- Full inference pipeline: video → engine.py → dashboard_data.json + scene.ply → enriched_data.json
- FastAPI backend: all routes functional including `/dev/load/{job_id}`
- SSE progress streaming to frontend
- Three.js viewer renders: scene point cloud (Y-flipped), human overlays, gaze arrows, proximity edges, trajectory trails
- Dev mode sidebar panel in App.jsx — enter job ID, click Load, viewer opens in ~1 second
- analytics.py `_load_ply_xyz` handles both ASCII and binary_little_endian PLY (numpy structured-array read, no per-row Python loop)
- Viewer has playback engine, layer toggles, camera presets, timeline strip, screenshot, FPS counter
- App has URL-hash auto-load, localStorage job history, subsample presets, delete button

### Bugs fixed this session (require new inference to take effect)
- **Human mask corrected** in `save_fused_scene_ply`: correct form is `msk >= 0.2` — high msk = human, exclude these. Lower threshold (0.2 vs old 0.5) catches more borderline/boundary human pixels.
- **Dilation increased** from 2 → 5 iterations to properly cover limb edges.
- **Mask resize guard** added: if mask spatial dims differ from pts3d dims, bilinear-interpolate before threshold.
- **enriched_data.json bloat fixed**: `verts` and `joints` are no longer written into enriched_data.json (was 440 MB+, unparseable by browser). Analytics file is now ~3-5 MB. Viewer.jsx fetches dashboard_data.json in parallel and merges verts/joints client-side.

### What requires a new inference run
The existing `06c9ccd8` outputs were produced with engine v1 (ASCII PLY, no `verts`/`joints`/`smpl_faces`, broken human masking).
Run a new inference to get:
- Binary PLY with correct human holes (scene only, no human geometry)
- Full SMPL-X mesh rendering in viewer
- Skeleton overlay (45 joints)

### What to do first in a new session
1. `conda activate human3r128`
2. `cd sitl && uvicorn backend.main:app --reload --port 8000`
3. Open http://localhost:8000/app, enter job ID `06c9ccd8` in Dev panel, click Load
4. Verify viewer renders before making any changes

---

## Rules — What Not To Do

- **Do not modify `demo.py`** — it is the upstream Human3R code
- **Do not modify `src/dust3r/`, `src/croco/`, `src/mhmr/`** — core model, no touch
- **Do not change SMPL-X vertex indices** (pelvis=0, head=4840) — verified correct
- **Do not add a build step** to the frontend — no webpack, no vite, no npm
- **Do not re-run inference** for frontend debugging — use `/dev/load/{job_id}`
- **Do not change the JSON key** `camera_trajectory` to `camera` — certification depends on it
- **Do not use Object.assign to set position** on Three.js objects — it is non-writable; use `.position.copy()`
- **Do not forget to bump `?v=N`** in index.html after editing JSX — browser caches Babel output aggressively