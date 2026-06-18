#!/usr/bin/env python3
"""
engine.py — Headless Human3R inference engine for the SITL dashboard.

Called by inference.py as a subprocess. Runs the full pipeline:
  1. Human3R inference (demo.py logic, no viewer)
  2. Fused scene PLY export (human-masked, voxel-deduplicated)
  3. Dashboard JSON export (world-frame positions, gaze vertices)

Streams progress to stdout so inference.py can parse it.

Usage (called by inference.py, not directly):
    python engine.py \
        --model_path ../src/human3r_896L.pth \
        --seq_path /path/to/video.mp4 \
        --output_dir outputs/<job_id> \
        --subsample 1 \
        --size 512
"""

import os
import sys
import json
import time
import glob
import random
import tempfile
import shutil

import cv2
import numpy as np
import torch
import roma

from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r

import argparse

random.seed(42)


# ── 1. ARGUMENT PARSING ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SITL headless inference engine")
    parser.add_argument("--model_path",    type=str,   default="../src/human3r_896L.pth")
    parser.add_argument("--seq_path",      type=str,   required=True)
    parser.add_argument("--output_dir",    type=str,   required=True)
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--size",          type=int,   default=512)
    parser.add_argument("--subsample",     type=int,   default=1)
    parser.add_argument("--max_frames",    type=int,   default=None)
    parser.add_argument("--reset_interval",type=int,   default=10_000_000)
    parser.add_argument("--use_ttt3r",     action="store_true", default=False)
    parser.add_argument("--save",          action="store_true",
                        help="Also write per-frame .npz files (depth, conf, camera, smpl)")
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    parser.add_argument("--msk_threshold", type=float, default=0.1)
    # PLY fusion settings
    parser.add_argument("--ply_stride",    type=int,   default=3)    # 3→sparse ~40MB; 2→dense ~120MB
    parser.add_argument("--voxel_size",    type=float, default=0.02)  # 0.02→~40MB; 0.01→~120MB
    parser.add_argument("--conf_thresh",   type=float, default=1.0)   # baseline: 1.2
    parser.add_argument("--max_depth",     type=float, default=12.0)
    # Vertex export — off by default (10k verts/human/frame → 500+ MB)
    # Enable only when SMPL-X mesh rendering is needed; joints alone suffice for skeleton
    parser.add_argument("--export_verts",  action="store_true", default=False)
    return parser.parse_args()


# ── 2. PROGRESS LOGGING ───────────────────────────────────────────────────────
# inference.py parses lines containing "N/M" for progress.
# We emit lines like "Frame: 15/229" so extract_progress() picks them up.

def log(msg: str):
    """Print immediately — no buffering so inference.py readline() gets it."""
    print(msg, flush=True)

def log_progress(current: int, total: int, label: str = "Frame"):
    """Emit a progress line in the N/M format inference.py expects."""
    log(f"{label}: {current}/{total}")


# ── 3. VIDEO / DIRECTORY PARSING ──────────────────────────────────────────────

def parse_seq_path(p):
    # Returns (img_paths, tmpdirname, source_fps). source_fps is the native video
    # frame rate; for an image directory it's unknown so we assume 30 (the common
    # default) — downstream speed metrics scale by fps/subsample.
    DEFAULT_FPS = 30.0
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        return img_paths, None, DEFAULT_FPS

    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {p}")

    video_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps == 0:
        cap.release()
        raise ValueError(f"Video FPS is 0 for {p}")

    log(f"Video: {total_frames} frames at {video_fps:.1f} FPS")

    tmpdirname = tempfile.mkdtemp()
    img_paths  = []
    for i in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        path = os.path.join(tmpdirname, f"frame_{i:06d}.jpg")
        cv2.imwrite(path, frame)
        img_paths.append(path)
    cap.release()
    return img_paths, tmpdirname, float(video_fps)


# ── 4. INPUT PREPARATION ──────────────────────────────────────────────────────

def prepare_input(img_paths, size, img_res=None, reset_interval=10_000_000):
    from src.dust3r.utils.image import load_images, pad_image
    from dust3r.utils.geometry import get_camera_parameters

    log(f"Loading {len(img_paths)} images at size {size}...")
    images = load_images(img_paths, size=size)

    K_mhmr = None
    if img_res is not None:
        K_mhmr = get_camera_parameters(img_res, device="cpu")

    views = []
    for i, img in enumerate(images):
        view = {
            "img":         img["img"],
            "ray_map":     torch.full(
                (img["img"].shape[0], 6,
                 img["img"].shape[-2], img["img"].shape[-1]),
                torch.nan),
            "true_shape":  torch.from_numpy(img["true_shape"]),
            "idx":         i,
            "instance":    str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask":    torch.tensor(True).unsqueeze(0),
            "ray_mask":    torch.tensor(False).unsqueeze(0),
            "update":      torch.tensor(True).unsqueeze(0),
            "reset":       torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
        }
        if img_res is not None:
            view["img_mhmr"] = pad_image(view["img"], img_res)
            view["K_mhmr"]   = K_mhmr
        views.append(view)

        # Overlap view at reset boundary (matches demo.py logic)
        if (i + 1) % reset_interval == 0:
            overlap = deepcopy(view)
            overlap["reset"] = torch.tensor(False).unsqueeze(0)
            views.append(overlap)

    return views


# ── 5. OUTPUT PROCESSING ──────────────────────────────────────────────────────

def process_outputs(outputs, args, img_res):
    """
    Mirrors prepare_output() from demo.py.
    Returns everything needed for export — no viewer, no render.
    """
    from src.dust3r.utils.camera   import pose_encoding_to_camera
    from src.dust3r.post_process   import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    from src.dust3r.utils          import SMPL_Layer
    from src.dust3r.utils.image    import unpad_image

    log("Processing inference outputs...")

    # Keep only the last (valid) pass
    outputs["pred"]  = outputs["pred"][-len(outputs["pred"]):]
    outputs["views"] = outputs["views"][-len(outputs["views"]):]

    # Remove reset-overlap duplicates
    reset_mask = torch.cat([v["reset"] for v in outputs["views"]], 0)
    shifted    = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], 0)
    outputs["pred"]  = [p for p, m in zip(outputs["pred"],  shifted) if not m]
    outputs["views"] = [v for v, m in zip(outputs["views"], shifted) if not m]
    reset_mask = reset_mask[~shifted]

    pts3ds_self_ls = [o["pts3d_in_self_view"] for o in outputs["pred"]]
    pts3ds_other   = [o["pts3d_in_other_view"] for o in outputs["pred"]]
    conf_self      = [o["conf_self"]           for o in outputs["pred"]]
    pts3ds_self    = torch.cat(pts3ds_self_ls, 0)

    # Camera poses
    pr_poses = [pose_encoding_to_camera(p["camera_pose"].clone()).cpu()
                for p in outputs["pred"]]

    if reset_mask.any():
        pr_poses_t   = torch.cat(pr_poses, 0)
        identity     = torch.eye(4, device=pr_poses_t.device)
        reset_poses  = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1),
                                   pr_poses_t, identity)
        cum_bases    = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cum_bases[:-1]], 0)
        pr_poses_t   = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses_t)
        pr_poses     = list(pr_poses_t.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([p[:, :3, :3] for p in pr_poses], 0)
    t_c2w = torch.cat([p[:, :3, 3]  for p in pr_poses], 0)

    # Transform pts to world frame
    transformed = []
    for pose, pself in zip(pr_poses, pts3ds_self):
        transformed.append(geotrf(pose, pself.unsqueeze(0)))
    pts3ds_other = transformed
    conf_other   = conf_self        # world-frame conf = self conf after geotrf

    B, H, W, _ = pts3ds_self.shape
    pp    = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [0.5 * (o["img"].permute(0, 2, 3, 1) + 1.0) for o in outputs["views"]]

    cam_dict = {
        "focal": focal.numpy(),
        "pp":    pp.numpy(),
        "R":     R_c2w.numpy(),
        "t":     t_c2w.numpy(),
    }

    cam2world    = torch.cat(pr_poses)
    intrinsics   = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    intrinsics[:, 0, 0] = focal.detach()
    intrinsics[:, 1, 1] = focal.detach()
    intrinsics[:, 0, 2] = pp[:, 0]
    intrinsics[:, 1, 2] = pp[:, 1]

    # SMPL-X parameters
    smpl_shape = [o.get("smpl_shape",      torch.empty(1, 0, 10))[0]       for o in outputs["pred"]]
    smpl_rotvec= [roma.rotmat_to_rotvec(
                      o.get("smpl_rotmat", torch.empty(1, 0, 53, 3, 3))[0])for o in outputs["pred"]]
    smpl_transl= [o.get("smpl_transl",     torch.empty(1, 0, 3))[0]        for o in outputs["pred"]]
    smpl_expr  = [o.get("smpl_expression", [None])[0]                       for o in outputs["pred"]]
    smpl_id    = [o.get("smpl_id",         torch.empty(1, 0))[0]           for o in outputs["pred"]]

    # Masks
    has_mask = "msk" in outputs["pred"][0]
    if has_mask:
        msks = [o["msk"][..., 0] for o in outputs["pred"]]
        if img_res is not None:
            msks = [unpad_image(m, [H, W]) for m in msks]
    else:
        msks = [torch.zeros(1, H, W) for _ in range(B)]

    # SMPL-X forward pass → world-frame vertices + joints
    smpl_layer = SMPL_Layer(type='smplx', gender='neutral',
                             num_betas=smpl_shape[0].shape[-1],
                             kid=False, person_center='head')

    # Extract static mesh face topology once (shared across all frames/humans)
    smpl_faces = []
    try:
        smpl_faces = smpl_layer.bm_x.faces.tolist()
        log(f"SMPL-X faces extracted: {len(smpl_faces):,} triangles")
    except AttributeError:
        log("Warning: could not extract SMPL-X faces — mesh rendering disabled")

    all_verts  = []
    all_joints = []   # 45 world-frame joint positions per frame
    log(f"Running SMPL-X forward pass for {B} frames...")

    for f_id in range(B):
        log_progress(f_id + 1, B, "SMPL frame")
        n = smpl_shape[f_id].shape[0]

        if n > 0:
            with torch.no_grad():
                smpl_out = smpl_layer(
                    smpl_rotvec[f_id], smpl_shape[f_id], smpl_transl[f_id],
                    None, None,
                    K=intrinsics[f_id].expand(n, -1, -1),
                    expression=smpl_expr[f_id])
            # Transform vertices to world coordinates
            all_verts.append(geotrf(pr_poses[f_id],
                                    smpl_out['smpl_v3d'].unsqueeze(0))[0])
            # Transform 45 SMPL-X body joints to world coordinates
            if 'smpl_j3d' in smpl_out:
                all_joints.append(geotrf(pr_poses[f_id],
                                         smpl_out['smpl_j3d'].unsqueeze(0))[0])
            else:
                all_joints.append(torch.empty(0))
        else:
            all_verts.append(torch.empty(0))
            all_joints.append(torch.empty(0))

        # Optional per-frame .npz saves (--save flag)
        if args.save:
            _save_frame_npz(f_id, args.output_dir,
                            pts3ds_self[f_id], conf_self[f_id],
                            colors[f_id], cam2world[f_id], intrinsics[f_id],
                            smpl_shape[f_id], smpl_rotvec[f_id],
                            smpl_transl[f_id], smpl_expr[f_id], msks[f_id],
                            has_mask)

    smpl_params = {
        "shapes":  smpl_shape,
        "rotvecs": smpl_rotvec,
        "transls": smpl_transl,
    }

    return (pts3ds_other, colors, conf_other, cam_dict,
            all_verts, all_joints, smpl_faces, smpl_id, msks, smpl_params)


def _save_frame_npz(f_id, outdir, depth_pts, conf, color,
                    c2w, intrins, shape, rotvec, transl,
                    expr, msk, has_mask):
    """Write per-frame .npz files (optional, matches demo.py --save format)."""
    import imageio.v2 as iio
    for sub in ("depth", "conf", "color", "camera", "smpl"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)

    np.save(os.path.join(outdir, "depth",  f"{f_id:06d}.npy"), depth_pts[..., 2].numpy())
    np.save(os.path.join(outdir, "conf",   f"{f_id:06d}.npy"), conf.numpy())
    iio.imwrite(os.path.join(outdir, "color", f"{f_id:06d}.png"),
                (color.numpy() * 255).astype(np.uint8))
    np.savez(os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
             pose=c2w.numpy(), intrinsics=intrins.numpy())
    np.savez(os.path.join(outdir, "smpl",   f"{f_id:06d}.npz"),
             shape=shape.numpy(), rotvec=rotvec.numpy(),
             transl=transl.numpy(),
             expression=expr.numpy() if expr is not None else None,
             msk=msk.numpy() if has_mask else None)


# ── 6. SCENE PLY EXPORT ───────────────────────────────────────────────────────

def save_fused_scene_ply(pts3ds_other, conf_list, msks, colors, output_path,
                         stride=2,          # baseline: 5 — more frames = denser cloud
                         conf_thresh=1.0,   # baseline: 1.2 — include more surface points
                         voxel_size=0.01,   # baseline: 0.02 — finer geometry detail
                         max_depth=12.0,
                         msk_thresh=0.2):   # pixels with msk >= thresh treated as human
    """
    Fuse all frames into one human-masked, confidence-filtered,
    voxel-deduplicated world-frame point cloud. Written as binary PLY.
    """
    from scipy.ndimage import binary_dilation

    log(f"Fusing scene PLY (stride={stride}, voxel={voxel_size}m, "
        f"conf>{conf_thresh}, depth<{max_depth}m)...")

    all_pts  = []
    all_cols = []
    n_frames = len(pts3ds_other)

    for i in range(0, n_frames, stride):
        log_progress(i + 1, n_frames, "PLY frame")

        pts  = pts3ds_other[i]
        conf = conf_list[i]
        msk  = msks[i]

        if hasattr(pts,  'cpu'): pts  = pts.cpu()
        if hasattr(conf, 'cpu'): conf = conf.cpu()
        if hasattr(msk,  'cpu'): msk  = msk.cpu()

        pts_np  = pts.reshape(-1, 3).numpy()
        conf_np = conf.flatten().numpy()

        # Derive spatial dims for morphological mask cleanup
        shape = pts.shape  # (..., H, W, 3)
        H, W  = int(shape[-3]), int(shape[-2])

        # Resize mask to match point-cloud spatial dims if model outputs a different res
        msk_squeezed = msk.squeeze()  # remove batch/channel dims → (h, w)
        if msk_squeezed.shape != (H, W):
            import torch.nn.functional as F
            msk_squeezed = F.interpolate(
                msk.reshape(1, 1, *msk_squeezed.shape[-2:]).float(),
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze()
        msk_np = msk_squeezed.numpy().flatten()

        # Human mask: high msk values = human pixel (model sigmoid output near 1 = human).
        # Use a low threshold (0.2) so partial/boundary human pixels are also masked out.
        # Dilate 5px to fully cover silhouette edges and thin limbs.
        msk_2d       = msk_np.reshape(H, W) >= msk_thresh
        msk_dilated  = binary_dilation(msk_2d, iterations=5).reshape(-1)

        dists = np.linalg.norm(pts_np, axis=1)
        valid = ((conf_np   > conf_thresh) &
                 (~msk_dilated)            &   # exclude humans + 5px border
                 (dists     > 0.1)         &
                 (dists     < max_depth))

        clean_pts = pts_np[valid]
        if len(clean_pts) == 0:
            continue
        all_pts.append(clean_pts)

        if colors is not None:
            col = colors[i]
            if hasattr(col, 'cpu'): col = col.cpu()
            col_np = col.reshape(-1, 3).numpy()
            all_cols.append((col_np[valid] * 255).clip(0, 255).astype(np.uint8))

    if not all_pts:
        log("WARNING: No points survived PLY filters — scene.ply will be empty.")
        return

    all_pts = np.concatenate(all_pts, axis=0)
    log(f"Raw fused points: {len(all_pts):,}")

    # Voxel deduplication
    vox = np.floor(all_pts / voxel_size).astype(np.int32)
    off = 10_000
    keys = ((vox[:, 0] + off).astype(np.int64) * 1_000_000_000 +
            (vox[:, 1] + off).astype(np.int64) * 100_000 +
            (vox[:, 2] + off).astype(np.int64))
    _, uid = np.unique(keys, return_index=True)

    pts_final  = all_pts[uid]
    has_color  = len(all_cols) > 0
    cols_final = np.concatenate(all_cols, axis=0)[uid] if has_color else None

    log(f"After voxel dedup: {len(pts_final):,} points")

    # Binary PLY — baseline was ASCII written in a Python for-loop (3-5x larger, 10x slower)
    with open(output_path, 'wb') as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(pts_final)}\n"
            "property float x\nproperty float y\nproperty float z\n"
        )
        if has_color:
            header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += "end_header\n"
        f.write(header.encode('ascii'))

        if has_color:
            dtype = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                              ('red','u1'),('green','u1'),('blue','u1')])
            buf = np.empty(len(pts_final), dtype=dtype)
            buf['x'] = pts_final[:,0]; buf['y'] = pts_final[:,1]; buf['z'] = pts_final[:,2]
            buf['red']   = cols_final[:,0]
            buf['green'] = cols_final[:,1]
            buf['blue']  = cols_final[:,2]
        else:
            dtype = np.dtype([('x','<f4'),('y','<f4'),('z','<f4')])
            buf = np.empty(len(pts_final), dtype=dtype)
            buf['x'] = pts_final[:,0]; buf['y'] = pts_final[:,1]; buf['z'] = pts_final[:,2]
        f.write(buf.tobytes())

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log(f"Scene PLY saved: {output_path} ({len(pts_final):,} pts, {size_mb:.1f} MB)")


# ── 7. DASHBOARD JSON EXPORT ──────────────────────────────────────────────────

def export_dashboard_json(output_dir, cam_dict, smpl_params,
                           smpl_ids, all_smpl_verts, all_joints, smpl_faces,
                           export_verts=False,
                           source_fps=30.0, subsample=1, effective_fps=30.0):
    """
    Write dashboard_data.json with world-frame positions, 45 skeleton joints,
    full SMPL-X vertices (for mesh rendering), and face topology (once in metadata).
    """
    assert len(all_smpl_verts) == len(cam_dict['R']), \
        f"Frame/camera count mismatch: {len(all_smpl_verts)} vs {len(cam_dict['R'])}"

    HEAD_VERTEX   = 4840  # confirmed: top-of-skull in SMPL-X topology
    PELVIS_VERTEX = 0     # root joint — always vertex 0
    # VERT_STRIDE was 20 (~524 sampled pts); now 1 = all 10475 verts for proper mesh rendering

    manifest = {
        "metadata": {
            "total_frames": len(all_smpl_verts),
            "up_axis":      "Y",
            "head_vertex":  HEAD_VERTEX,
            "exporter":     "SITL_engine_v2",
            # Frame-rate provenance so analytics can convert per-frame displacement
            # into physical m/s regardless of subsample.
            "source_fps":    round(float(source_fps), 3),
            "subsample":     int(subsample),
            "effective_fps": round(float(effective_fps), 3),
            # Static face topology shared by every human every frame (~10k triangles, exported once)
            "smpl_faces":   smpl_faces,
        },
        "camera_trajectory": [],
        "frames": [],
    }

    for i in range(len(cam_dict['R'])):
        manifest["camera_trajectory"].append({
            "R": cam_dict['R'][i].tolist(),
            "t": cam_dict['t'][i].tolist(),
        })

    # Verts are exported to a Float32 binary side-car (verts.bin) rather than inline
    # JSON. A full-resolution mesh (10475 verts/human) over a long clip pushes
    # dashboard_data.json past Chrome's ~512 MB max JS-string length, so the browser
    # cannot JSON.parse it. Binary loads via arrayBuffer() with no string limit and
    # is ~5x smaller. verts_index.json maps each frame's humans to a block in the bin.
    vert_blocks = []          # list of np.float32 [V,3] arrays, in iteration order
    vert_index  = {"frames": []}
    vert_count  = None

    total = len(all_smpl_verts)
    for f_id in range(total):
        log_progress(f_id + 1, total, "JSON frame")
        frame_data = {"frame_id": f_id, "humans": []}
        frame_blocks = []     # block index per human, aligned to humans order
        verts  = all_smpl_verts[f_id]
        joints = all_joints[f_id]

        if verts.numel() == 0:
            manifest["frames"].append(frame_data)
            vert_index["frames"].append(frame_blocks)
            continue

        for h_idx in range(len(smpl_ids[f_id])):
            hv = verts[h_idx]   # [10475, 3] world-frame SMPL-X vertices

            human_entry = {
                "id":         int(smpl_ids[f_id][h_idx]),
                "world_pos":  hv[PELVIS_VERTEX].tolist(),
                "head_world": hv[HEAD_VERTEX].tolist(),
                "pose":       smpl_params["rotvecs"][f_id][h_idx].cpu().tolist(),
                "shape":      smpl_params["shapes"][f_id][h_idx].cpu().tolist(),
            }
            if export_verts:
                hv_np = hv.cpu().numpy().astype(np.float32)
                if vert_count is None:
                    vert_count = hv_np.shape[0]
                frame_blocks.append(len(vert_blocks))
                vert_blocks.append(hv_np)

            # 45 world-frame body joints for skeleton overlay
            if joints.numel() > 0 and h_idx < joints.shape[0]:
                human_entry["joints"] = joints[h_idx].tolist()

            frame_data["humans"].append(human_entry)

        manifest["frames"].append(frame_data)
        vert_index["frames"].append(frame_blocks)

    json_path = os.path.join(output_dir, "dashboard_data.json")
    with open(json_path, "w") as f:
        json.dump(manifest, f)
    log(f"Dashboard JSON saved: {json_path}")

    if export_verts and vert_blocks:
        bin_path = os.path.join(output_dir, "verts.bin")
        np.concatenate([b.reshape(-1) for b in vert_blocks]).astype(
            "<f4").tofile(bin_path)
        vert_index.update({"dtype": "float32", "vert_count": int(vert_count),
                           "stride": int(vert_count) * 3})
        with open(os.path.join(output_dir, "verts_index.json"), "w") as f:
            json.dump(vert_index, f)
        log(f"Verts binary saved: {bin_path} "
            f"({os.path.getsize(bin_path) / 1048576:.0f} MB, {len(vert_blocks)} blocks)")


# ── 8. MAIN ENTRY POINT ───────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Device setup
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log("CUDA not available — falling back to CPU.")
        device = "cpu"

    # Add checkpoint path for dust3r imports
    add_path_to_dust3r(args.model_path)

    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model     import ARCroco3DStereo

    # ── Parse input ──────────────────────────────────────────────────────────
    img_paths, tmpdirname, source_fps = parse_seq_path(args.seq_path)
    if not img_paths:
        log(f"ERROR: No images found at {args.seq_path}")
        sys.exit(1)

    if args.max_frames is not None:
        img_paths = img_paths[:args.max_frames]
    img_paths = img_paths[::args.subsample]
    # Effective frame rate of the exported sequence (processed frames per second).
    effective_fps = source_fps / max(1, args.subsample)
    log(f"Using {len(img_paths)} frames (subsample={args.subsample}, "
        f"source_fps={source_fps:.1f}, effective_fps={effective_fps:.2f})")

    # ── Load model ───────────────────────────────────────────────────────────
    log(f"Loading model: {args.model_path}")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()
    img_res = getattr(model, 'mhmr_img_res', None)
    log("Model loaded.")

    # ── Prepare views ────────────────────────────────────────────────────────
    views = prepare_input(img_paths, args.size, img_res, args.reset_interval)

    if tmpdirname:
        shutil.rmtree(tmpdirname)

    # ── Run inference ────────────────────────────────────────────────────────
    log(f"Starting inference on {len(views)} views...")
    t0 = time.time()
    outputs, _ = inference_recurrent_lighter(
        views, model, device, use_ttt3r=args.use_ttt3r)
    elapsed = time.time() - t0
    log(f"Inference complete: {elapsed:.1f}s ({elapsed/len(views):.2f}s/frame)")

    # Free the model and view tensors — both are large GPU allocations that are
    # no longer needed after inference finishes.  Releasing them before
    # process_outputs() prevents an OOM kill during post-processing on long videos.
    del views, model
    torch.cuda.empty_cache()

    # ── Process outputs ──────────────────────────────────────────────────────
    (pts3ds_other, colors, conf_other, cam_dict,
     all_smpl_verts, all_joints, smpl_faces, smpl_id, msks, smpl_params) = process_outputs(
        outputs, args, img_res)

    # Free the raw inference outputs — process_outputs() has already extracted
    # all needed data into separate variables.
    del outputs
    torch.cuda.empty_cache()
    log("Memory freed after post-processing.")

    # ── Export scene PLY ─────────────────────────────────────────────────────
    ply_path = os.path.join(args.output_dir, "scene.ply")
    save_fused_scene_ply(
        pts3ds_other, conf_other, msks, colors, ply_path,
        stride=args.ply_stride,
        conf_thresh=args.conf_thresh,
        voxel_size=args.voxel_size,
        max_depth=args.max_depth,
        msk_thresh=args.msk_threshold,
    )

    # Free the large per-frame arrays once PLY is written
    del pts3ds_other, conf_other, msks, colors
    torch.cuda.empty_cache()

    # ── Export dashboard JSON ────────────────────────────────────────────────
    export_dashboard_json(
        args.output_dir, cam_dict, smpl_params, smpl_id,
        all_smpl_verts, all_joints, smpl_faces,
        export_verts=args.export_verts,
        source_fps=source_fps, subsample=args.subsample,
        effective_fps=effective_fps)

    log(f"Engine complete. Outputs in: {args.output_dir}")
    log("1/1")   # final progress signal so inference.py sees 100%


if __name__ == "__main__":
    main()