# SITL — Semantic Interaction Topology Lab

A spatial-interaction analytics dashboard built on [Human3R](https://arxiv.org/abs/2510.06219). Upload a monocular video and get a fully navigable 3D scene with SMPL-X body meshes, real-time interaction analytics, and a browser-based dashboard — no camera rig, depth sensor, or manual annotation required.

---

## Overview

```
Video  →  Human3R inference  →  Spatial analytics  →  Three.js dashboard
           (3D scene + SMPL-X      (Hall proxemics,       (point cloud +
            meshes + camera         gaze, contact,          overlays +
            trajectory)             speed, heatmap)         playback)
```

Human3R performs one-shot monocular reconstruction: SMPL-X body meshes for every person, a dense colored point cloud of the environment, and the full camera trajectory — in a single forward pass. SITL wraps this with a headless pipeline, enriches the output with interaction metrics grounded in proxemics theory, and serves everything through a local web UI.

---

## Features

### 3D Viewer
- Colored point cloud of the reconstructed scene (binary PLY)
- SMPL-X surface meshes per person (10,475 vertices, anatomically-sized skeleton)
- Gaze arrows from head joint, direction from shoulder cross product
- Fading trajectory trails (last 30 frames) per tracked person
- Proximity edges colored by Hall (1966) proxemics zone:
  - **Red** — intimate (< 0.45 m)
  - **Orange** — personal (0.45–1.2 m)
  - **Yellow** — social (1.2–3.7 m)
- White midpoint sphere on edges where mutual gaze is detected
- Floor heatmap showing spatial activity density
- Layer toggles: cloud · skeleton · mesh · gaze · trails · interactions · heatmap · grid
- Camera presets: Persp · Front · Follow
- Live point-size slider

### Playback Engine
- Play / pause (`Space`), frame step (`←` / `→`), loop toggle
- Speed presets: 0.25× · 0.5× · 1× · 2× · 4× (keys `1`–`5`)
- Live FPS counter, screenshot export (PNG)

### Analytics Dashboard
Eight metrics with hover tooltips and click-to-expand modal showing formula and research citation:

| Metric | Definition |
|--------|------------|
| Social Engagement | % frames where any pair is within Hall's personal zone (≤ 1.2 m) |
| Avg Inter-Person Distance | Mean 3D pairwise distance across all frames |
| Scene Utilization | % of 0.5 m voxels occupied by at least one person |
| Gaze Convergence Events | Frames where ≥ 2 gaze rays converge within 1 m |
| Intimate Zone Violations | % frames where any pair is within intimate zone (< 0.45 m) |
| Average Movement Speed | Mean pelvis speed in m/s at 30 fps |
| Approach Events | Times any pair transitions from diverging → converging |
| Peak Occupancy | Max simultaneous persons in any single frame |

### Timelines
- **Interaction timeline**: color-coded strip above scrubber, zones per frame — click to seek
- **Action display**: compact per-person action badge chips (stationary / walking / running / sitting / reaching / bending); expandable to full per-person timeline strips

### Upload & Job Management
- Drag-and-drop video upload with subsample presets (`1×`–`8×`)
- Job history in `localStorage` (last 10 jobs, one-click reload)
- URL hash (`#job_id`) auto-loads on page refresh
- Delete job button (removes outputs from disk)
- **Dev mode**: load any previously computed job in ~1 second — no GPU usage

---

## Stack

| Layer | Technology |
|-------|-----------|
| 3D reconstruction | Human3R (DUSt3R + Multi-HMR, ViT-L @ 896px) |
| Body model | SMPL-X (10,475 vertices, 45 joints) |
| Backend | FastAPI + uvicorn + SSE |
| Analytics | NumPy + SciPy (KDTree, gaussian filter) |
| Frontend | React 18 UMD + Three.js r134 + Babel standalone |
| Build | None — pure CDN, no webpack or npm |

---

## Requirements

- Linux with CUDA 12.4
- Conda
- ~12 GB VRAM (ViT-L); ViT-S/B checkpoints available for lower VRAM

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

# Compile CUDA RoPE kernels
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../

# Download SMPL-X body model
bash scripts/fetch_smplx.sh

# Download Human3R checkpoint
huggingface-cli download faneggg/human3r human3r_896L.pth --local-dir ./src
```

### Checkpoints

| Model | Resolution | Backbone | Speed |
|-------|-----------|----------|-------|
| `human3r_672S.pth` | 672 | ViT-S | ~15 FPS |
| `human3r_672B.pth` | 672 | ViT-B | ~11 FPS |
| `human3r_672L.pth` | 672 | ViT-L | ~7 FPS |
| `human3r_896L.pth` | 896 | ViT-L | ~5 FPS — **default** |

---

## Running

### SITL Dashboard

```bash
conda activate human3r128
cd sitl
uvicorn backend.main:app --reload --port 8000
```

Open **http://localhost:8000/app**, drop a video, choose subsample rate, and let the pipeline run.

**Dev mode** — skip inference entirely: enter any previously computed job ID in the sidebar and press Enter. Viewer opens in ~1 second.

### Headless inference only

```bash
# Run from project root
CUDA_VISIBLE_DEVICES=0 python engine.py \
    --model_path src/human3r_896L.pth \
    --seq_path examples/your_video.mp4 \
    --output_dir sitl/outputs/my_job \
    --subsample 3 --size 512
```

Outputs: `dashboard_data.json` + `scene.ply` in `--output_dir`.

---

## API

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/upload` | Save video, returns `{job_id}` |
| `POST` | `/run/{job_id}?subsample=N` | Start pipeline |
| `GET` | `/status/{job_id}` | SSE stream of stage/progress/logs |
| `GET` | `/results/{job_id}` | URLs to `enriched_data.json` + `scene.ply` |
| `GET` | `/dev/load/{job_id}` | Register existing outputs as completed |
| `DELETE` | `/jobs/{job_id}` | Remove outputs + uploaded video |
| `GET` | `/app` | Serve frontend |

---

## Output Files

### `dashboard_data.json`
Per-frame SMPL-X meshes, joints, camera trajectory. Written by `engine.py`.

```json
{
  "metadata": { "total_frames": 229, "smpl_faces": [[i0,i1,i2], ...] },
  "camera_trajectory": [{ "R": [[3×3]], "t": [x,y,z] }],
  "frames": [{
    "frame_id": 0,
    "humans": [{
      "id": 0,
      "world_pos": [x,y,z],
      "head_world": [x,y,z],
      "pose": [[53×3]],
      "shape": [10],
      "verts": [[x,y,z] × 10475],
      "joints": [[x,y,z] × 45]
    }]
  }]
}
```

### `enriched_data.json`
`dashboard_data` + analytics, `verts`/`joints` stripped (kept in `dashboard_data.json` to stay under 5 MB). Written by `analytics.py`.

```json
{
  "metadata": { "social_engagement_pct": 63.2, ... },
  "frames": [{
    "humans": [{ "gaze_vec": [x,y,z], "contact_score": 0.85, "speed": 0.003 }],
    "interactions": [{
      "source": 0, "target": 1, "distance": 1.24,
      "zone": "personal", "mutual_gaze": false, "approach_state": "approaching"
    }]
  }],
  "summary": { ... }
}
```

---

## Coordinate System

Human3R outputs in OpenCV world coordinates (Y-down). SITL Y-flips for Three.js:

```javascript
const fy = ([x, y, z]) => new THREE.Vector3(x, -y, z);  // joints, gaze vecs
points.scale.y = -1;                                       // PLY cloud
```

- SMPL-X vertices (`world_pos`, `head_world`) — Y-up
- SMPL-X joints — Y-down
- PLY point cloud — Y-down

---

## Repository Layout

```
human3r-sitl/
├── engine.py              ← Headless SITL runner
├── demo.py                ← Original Human3R interactive demo (unmodified)
├── src/                   ← Human3R model code (dust3r, croco, mhmr)
├── scripts/               ← Data download helpers
└── sitl/
    ├── backend/
    │   ├── main.py        ← FastAPI routes
    │   ├── pipeline.py    ← Stage orchestrator
    │   ├── inference.py   ← Subprocess wrapper for engine.py
    │   └── workers/
    │       └── analytics.py  ← Proxemics · gaze · contact · heatmap
    ├── frontend/
    │   ├── index.html     ← CDN loader (Three.js r134 + React 18 + Babel)
    │   ├── Viewer.jsx     ← Three.js scene + dashboard
    │   └── App.jsx        ← Upload UI + pipeline state machine
    └── outputs/           ← Runtime job outputs (gitignored)
```

---

## Acknowledgements

Built on top of:
- **Human3R** — *Generalizable 3D Human Reconstruction in the Wild* (ICLR 2026, [arXiv:2510.06219](https://arxiv.org/abs/2510.06219))
- **DUSt3R** — Dense Unconstrained Stereo 3D Reconstruction
- **Multi-HMR** — Multi-person Human Mesh Recovery
- **SMPL-X** — Expressive body model (Pavlakos et al., 2019)

Interaction metrics grounded in:
- Hall (1966) — proxemics zones
- Kendon (1967), Argyle & Cook (1976) — mutual gaze and social attention
- Goffman (1971) — approach–avoidance dynamics

---

## License

SITL dashboard code (`sitl/`, `engine.py`) — MIT. Human3R model code in `src/` is subject to the original [Human3R license](https://github.com/fanegg/Human3R).
