#!/usr/bin/env python3
"""
Modified from CUT3R: https://github.com/CUT3R/CUT3R

Online Human-Scene Reconstruction Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D scene point clouds and SMPLX sequences with the SceneHumanViewer. 
Use the command-line arguments to adjust parameters 
such as the model checkpoint path, image sequence directory, image size, device, etc.

Example:
    python demo.py --model_path src/human3r_896L.pth --size 512 \
        --seq_path examples/GoodMornin1.mp4 --subsample 1 --vis_threshold 2 \
        --downsample_factor 1 --use_ttt3r --reset_interval 100
"""

import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import roma
import json 

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--msk_threshold",
        type=float,
        default=0.1,
        help="Mask threshold. Ranging from 0 to 1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output results.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Save smpl mesh projection.",
    )
    parser.add_argument(
        "--render_video",
        action="store_true",
        help="Save smpl mesh projection video.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames to use. Default is None (use all images).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample factor for input images. Default is 1 (use all images).",
    )
    parser.add_argument(
        "--reset_interval", 
        type=int, 
        default=10000000
        )
    parser.add_argument(
        "--use_ttt3r",
        action="store_true",
        help="Use TTT3R.",
        default=False
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=10,
        help="Point cloud downsample factor for the viewer",
    )
    parser.add_argument(
        "--smpl_downsample",
        type=int,
        default=1,
        help="SMPL sequence downsample factor for the viewer",
    )
    parser.add_argument(
        "--camera_downsample",
        type=int,
        default=1,
        help="Camera motion downsample factor for the viewer",
    )
    parser.add_argument(
        "--mask_morph",
        type=int,
        default=10,
        help="Mask morphology for the viewer",
    )
    parser.add_argument(
    "--no_vis",
    action="store_true",
    help="If set, skip launching the Viser 3D viewer.",
    
    )
    return parser.parse_args()
    


def prepare_input(
    img_paths, 
    img_mask, 
    size, 
    raymaps=None, 
    raymap_mask=None, 
    revisit=1, 
    update=True, 
    img_res=None, 
    reset_interval=100
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images, pad_image
    from dust3r.utils.geometry import get_camera_parameters

    images = load_images(img_paths, size=size)
    if img_res is not None:
        K_mhmr = get_camera_parameters(img_res, device="cpu") # if use pseudo K

    views = []
    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views

def prepare_output(
        outputs, outdir, revisit=1, use_pose=True, 
        save=False, render=False, render_video=False, img_res=None, subsample=1):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.
        save (bool): Whether to save output results.
        render (bool): Whether to save smpl mesh projection.
        render_video (bool): Whether to save smpl mesh projection video.
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    from src.dust3r.utils import SMPL_Layer, vis_heatmap, render_meshes
    from src.dust3r.utils.image import unpad_image
    from viser_utils import get_color

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self_ls = [output["pts3d_in_self_view"] for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"] for output in outputs["pred"]]
    conf_self = [output["conf_self"] for output in outputs["pred"]]
    conf_other = [output["conf"] for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    # reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        # keeps only reset_mask=False pr_poses
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.numpy(),
        "pp": pp.numpy(),
        "R": R_c2w.numpy(),
        "t": t_c2w.numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach()
    intrinsics_tosave[:, 1, 1] = focal.detach()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    # get SMPL parameters from outputs
    smpl_shape = [output.get(
        "smpl_shape", torch.empty(1,0,10))[0] for output in outputs["pred"]]
    smpl_rotvec = [roma.rotmat_to_rotvec(
        output.get(
            "smpl_rotmat", torch.empty(1,0,53,3,3))[0]) for output in outputs["pred"]]
    smpl_transl = [output.get(
        "smpl_transl", torch.empty(1,0,3))[0] for output in outputs["pred"]]
    smpl_expression = [output.get(
        "smpl_expression", [None])[0] for output in outputs["pred"]]
    smpl_id = [output.get(
        "smpl_id", torch.empty(1,0))[0] for output in outputs["pred"]]
    # smpl_loc = [output.get(
    #     "smpl_loc", torch.empty(1,0,2))[0] for output in outputs["pred"]]
    # K_mhmr = [output.get(
    #     "K_mhmr", torch.empty(1,0,3))[0] for output in outputs["views"]]
        
    if render or save:
        smpl_scores = [
            output.get("smpl_scores", torch.zeros(1, H, W, 1))[...,0] for output in outputs["pred"]]
        if img_res is not None:
            smpl_scores = [
                unpad_image(s, [H, W])[0] for s in smpl_scores]

    has_mask = "msk" in outputs["pred"][0]
    if has_mask:
        msks = [output["msk"][...,0] for output in outputs["pred"]]
        if img_res is not None:
            msks = [unpad_image(m, [H, W]) for m in msks]
    else:
        msks = [torch.zeros(1, H, W) for _ in range(B)]

    # SMPL layer
    smpl_layer = SMPL_Layer(type='smplx', 
                            gender='neutral', 
                            num_betas=smpl_shape[0].shape[-1], 
                            kid=False, 
                            person_center='head')
    smpl_faces = smpl_layer.bm_x.faces

    if save:
        print(f"Saving output to {outdir}...")
        os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "smpl"), exist_ok=True)

    all_verts = []
    for f_id in range(B):
        n_humans_i = smpl_shape[f_id].shape[0]
        
        if n_humans_i > 0:
            with torch.no_grad():
                smpl_out = smpl_layer(
                    smpl_rotvec[f_id], 
                    smpl_shape[f_id], 
                    smpl_transl[f_id], 
                    None, None, 
                    K=intrinsics_tosave[f_id].expand(n_humans_i, -1 , -1), 
                    expression=smpl_expression[f_id])
                
            # if f_id == 0:
            #     verts_world = geotrf(
            #         pr_poses[f_id],
            #         smpl_out['smpl_v3d'].unsqueeze(0)
            #     )[0]  # [N_humans, 10475, 3]

            #     person_verts = verts_world[0]   # first person
            #     pelvis_pos   = person_verts[0]  # vertex 0

            #     print(f"\n=== VERTEX SEARCH ===")
            #     print(f"Pelvis (v0): {pelvis_pos.numpy()}")

            #     dists = torch.norm(person_verts - pelvis_pos, dim=1)
            #     candidates = torch.where((dists > 0.25) & (dists < 0.65))[0]
            #     print(f"Vertices 0.25–0.65m from pelvis: {len(candidates)} candidates")

            #     for axis, name in enumerate(['X', 'Y', 'Z']):
            #         axis_vals = person_verts[candidates, axis] - pelvis_pos[axis]
            #         best_idx  = candidates[axis_vals.argmax()]
            #         best_val  = axis_vals.max().item()
            #         print(f"  Best head candidate along {name}: "
            #             f"vertex {best_idx.item()}, delta={best_val:.3f}m")

            #     top5 = torch.topk(dists, 5)
            #     print(f"\nTop 5 most distant vertices from pelvis:")
            #     for dist, idx in zip(top5.values, top5.indices):
            #         v = person_verts[idx].numpy()
            #         print(f"  vertex {idx.item():5d}: dist={dist:.3f}m  "
            #             f"pos={np.round(v, 3)}")
            #     print("=== END VERTEX SEARCH ===\n")
        
        depth = depths_tosave[f_id].numpy()
        conf = conf_self_tosave[f_id].numpy()
        color = colors_tosave[f_id].numpy()
        c2w = cam2world_tosave[f_id].numpy()
        intrins = intrinsics_tosave[f_id].numpy()

        if n_humans_i > 0:
            # transform smpl verts to world coordinates
            all_verts.append(geotrf(pr_poses[f_id], smpl_out['smpl_v3d'].unsqueeze(0))[0])
            pr_verts = [t.numpy() for t in smpl_out['smpl_v3d'].unbind(0)]
            pr_faces = [smpl_faces] * n_humans_i
        else:
            pr_verts = []
            pr_faces = []
            all_verts.append(torch.empty(0))

        if render:
            hm = vis_heatmap(colors_tosave[f_id], smpl_scores[f_id]).numpy()
            img_array_np = (color * 255).astype(np.uint8)
            smpl_rend = render_meshes(img_array_np.copy(), pr_verts, pr_faces,
                                        {'focal': intrins[[0,1],[0,1]], 
                                        'princpt': intrins[[0,1],[-1,-1]]},
                                        color=[get_color(i)/255 for i in smpl_id[f_id]])
            if has_mask:
                msk_array_np = vis_heatmap(colors_tosave[f_id], msks[f_id][0]).numpy()
                color_smpl = np.concatenate([
                    img_array_np, 
                    (msk_array_np * 255).astype(np.uint8), 
                    (hm * 255).astype(np.uint8), 
                    smpl_rend], 1)
            else:
                color_smpl = np.concatenate([
                    img_array_np, 
                    (hm * 255).astype(np.uint8), 
                    smpl_rend], 1)
        
        if save:
            np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
            np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
            iio.imwrite(
                os.path.join(outdir, "color", f"{f_id:06d}.png"),
                (color * 255).astype(np.uint8),
            )
            np.savez(
                os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
                pose=c2w,
                intrinsics=intrins,
            )
            np.savez(
                os.path.join(outdir, "smpl", f"{f_id:06d}.npz"),
                scores=smpl_scores[f_id].numpy(),
                msk=msks[f_id].numpy() if has_mask else None,
                shape=smpl_shape[f_id].numpy(),
                rotvec=smpl_rotvec[f_id].numpy(),
                transl=smpl_transl[f_id].numpy(),
                expression=smpl_expression[f_id].numpy() if smpl_expression[f_id] is not None else None
            )

        # Save smpl projection
        if render:
            os.makedirs(os.path.join(outdir, "color_smpl"), exist_ok=True)
            iio.imwrite(
                os.path.join(outdir, "color_smpl", f"{f_id:06d}.png"),
                color_smpl,
            )

    if render and render_video:
        print(f"Saving smpl mesh projection to {outdir}...")
        frames_dir = os.path.join(outdir, "color_smpl")
        video_path = os.path.join(outdir, "output_video.mp4")
        output_fps = 30 // subsample
        os.system(f'/usr/bin/ffmpeg -y -framerate {output_fps} -i "{frames_dir}/%06d.png" '
                f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                f'-movflags +faststart -b:v 5000k "{video_path}"')
    
    # return (
    #     pts3ds_other,
    #     colors, 
    #     conf_other, 
    #     cam_dict, 
    #     all_verts, 
    #     smpl_faces,
    #     smpl_id,
    #     msks
    # )

    smpl_params = {
        "shapes": smpl_shape, 
        "rotvecs": smpl_rotvec, 
        "transls": smpl_transl
    }
    return (
        pts3ds_other, colors, conf_other, cam_dict, 
        all_verts, smpl_faces, smpl_id, msks, smpl_params
    )



def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname

import json

def save_fused_scene_ply(pts3ds_other, msks, colors, output_path, 
                         stride=5, voxel_size=0.05, max_depth=10.0, mask_threshold=0.2):
    """
    Fuses multiple frames into a clean, human-free world-frame point cloud.
    """
    import numpy as np

    print(f"🛠️  Fusing {len(pts3ds_other)} frames into a clean background...")

    all_pts = []
    all_cols = []

    for i in range(0, len(pts3ds_other), stride):
        # 1. Coordinate Extraction
        pts = pts3ds_other[i]
        if hasattr(pts, 'cpu'): pts = pts.cpu()
        pts_np = pts.reshape(-1, 3).numpy()

        # 2. Human Masking Logic
        # msks[i] is [1, H, W] -> flatten to match the 512x512 point cloud
        msk_np = msks[i].flatten().cpu().numpy()
        
        # 3. Spatial Filtering (Distance + Human Removal)
        dists = np.linalg.norm(pts_np, axis=1)
        # We only keep points that are:
        # - Close enough (< max_depth)
        # - Not a human (mask < threshold)
        # - Valid (> 0.1 to avoid origin noise)
        valid_mask = (dists > 0.1) & (dists < max_depth) & (msk_np < mask_threshold)
        
        clean_pts = pts_np[valid_mask]
        all_pts.append(clean_pts)

        if colors is not None:
            col = colors[i]
            if hasattr(col, 'cpu'): col = col.cpu()
            col_np = col.reshape(-1, 3).numpy()[valid_mask]
            all_cols.append((col_np * 255).clip(0, 255).astype(np.uint8))

    if not all_pts:
        print("⚠️  Warning: No background points found after masking.")
        return

    all_pts = np.concatenate(all_pts, axis=0)

    # 4. Voxel Grid Deduplication (The "Stitching")
    # This collapses redundant observations into a clean surface
    voxel_indices = np.floor(all_pts / voxel_size).astype(np.int32)
    offset = 1000  
    keys = ((voxel_indices[:, 0] + offset).astype(np.int64) * 100_000_000 +
            (voxel_indices[:, 1] + offset).astype(np.int64) * 10_000 +
            (voxel_indices[:, 2] + offset).astype(np.int64))

    _, unique_indices = np.unique(keys, return_index=True)
    pts_deduped = all_pts[unique_indices]

    has_color = len(all_cols) > 0
    if has_color:
        all_cols = np.concatenate(all_cols, axis=0)
        cols_deduped = all_cols[unique_indices]

    # 5. Write PLY
    with open(output_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts_deduped)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        if has_color:
            for p, c in zip(pts_deduped, cols_deduped):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        else:
            for p in pts_deduped:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

    print(f"✅ Clean Scene PLY saved: {output_path} ({len(pts_deduped):,} points)")


def export_for_dashboard(output_dir, cam_dict, smpl_params, smpl_ids,
                          all_smpl_verts, pts3ds_other, msks, colors=None):
    """
    Main export controller for the React/Three.js dashboard.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Data Integrity Check
    assert len(all_smpl_verts) == len(cam_dict['R']), "Timeline desync detected."

    # 2. Global Scene Reconstruction (CLEANED)
    # We use a 5cm voxel size and a 0.2 mask threshold for the 'Proper Scene' look
    save_fused_scene_ply(
        pts3ds_other, 
        msks, 
        colors, 
        os.path.join(output_dir, "scene.ply"),
        stride=5,         # Higher stride = faster fusion
        voxel_size=0.05,  # 5cm grid balances fidelity and file size
        max_depth=12.0,   # Increase if the room is large
        mask_threshold=0.2 # Strict masking to remove humans entirely
    )

    # 3. Build JSON Manifest
    manifest = {
        "metadata": {
            "total_frames": len(all_smpl_verts),
            "up_axis": "Y",
            "exporter": "Human3R_Dashboard_v1"
        },
        "camera_trajectory": [],
        "frames": []
    }

    # Export Camera Path
    for i in range(len(cam_dict['R'])):
        manifest["camera_trajectory"].append({
            "R": cam_dict['R'][i].tolist(),
            "t": cam_dict['t'][i].tolist()
        })

    # Export Per-Frame Human Data
    for f_id in range(len(all_smpl_verts)):
        frame_data = {"frame_id": f_id, "humans": []}
        current_frame_verts = all_smpl_verts[f_id]

        if current_frame_verts.numel() == 0:
            manifest["frames"].append(frame_data)
            continue

        for h_idx in range(len(smpl_ids[f_id])):
            # Pelvis (0) for World Position, Head (4840) for Gaze
            world_pos  = current_frame_verts[h_idx][0].tolist()
            head_world = current_frame_verts[h_idx][4840].tolist()

            human = {
                "id": int(smpl_ids[f_id][h_idx]),
                "world_pos": world_pos,
                "head_world": head_world,
                "pose": smpl_params["rotvecs"][f_id][h_idx].cpu().tolist(),
                "shape": smpl_params["shapes"][f_id][h_idx].cpu().tolist(),
            }
            frame_data["humans"].append(human)

        manifest["frames"].append(frame_data)

    # Final Write
    json_path = os.path.join(output_dir, "dashboard_data.json")
    with open(json_path, "w") as f:
        json.dump(manifest, f)
        
    print(f"🚀 DASHBOARD EXPORT SUCCESSFUL → {output_dir}")

def save_point_cloud_as_ply(points, path, colors=None):
    if hasattr(points, 'cpu'):
        points = points.cpu()
    pts = points.reshape(-1, 3).numpy()

    # downsample: every 8th point keeps file small (~18k pts from a 512×288 frame)
    pts = pts[::8]

    # remove outliers beyond 20m (common noise in background)
    dists = np.linalg.norm(pts, axis=1)
    mask = dists < 20.0
    pts = pts[mask]

    has_color = colors is not None
    if has_color:
        if hasattr(colors, 'cpu'):
            colors = colors.cpu()
        cols = colors.reshape(-1, 3).numpy()[::8][mask]
        cols = (cols * 255).clip(0, 255).astype(np.uint8)

    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, p in enumerate(pts):
            line = f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
            if has_color:
                c = cols[i]
                line += f" {c[0]} {c[1]} {c[2]}"
            f.write(line + "\n")

    print(f"   Scene PLY: {path} ({len(pts)} points, color={'yes' if has_color else 'no'})")

def save_fused_scene_automated(pts_list, conf_list, mask_list, color_list, output_path, 
                               stride=5, conf_thresh=1.2, voxel_size=0.02):
    """
    Automated version of the Viser-style reconstructor.
    Uses tensors directly from the inference pipeline.
    """
    import numpy as np
    import torch

    print(f"🏗️  Automating scene reconstruction (Stride={stride}, Voxel={voxel_size}m)...")
    
    all_fused_pts = []
    all_fused_cols = []

    # Loop through the results with the specified stride
    for i in range(0, len(pts_list), stride):
        # 1. Prepare Tensors (Move to CPU and flatten)
        pts = pts_list[i].reshape(-1, 3).cpu().numpy()
        conf = conf_list[i].flatten().cpu().numpy()
        
        # Human mask is [1, H, W] -> flatten to [H*W]
        msk = mask_list[i].flatten().cpu().numpy()
        
        # 2. Apply Viser Filtering Logic
        # (High confidence AND not human AND valid depth)
        valid_mask = (conf > conf_thresh) & (msk < 0.5) & (pts[:, 2] < 12.0)
        
        clean_pts = pts[valid_mask]
        all_fused_pts.append(clean_pts)

        if color_list is not None:
            # Colors are usually [1, 3, H, W] or [H, W, 3]
            # Based on prepare_output, they are likely [H, W, 3]
            cols = color_list[i].reshape(-1, 3).cpu().numpy()
            all_fused_cols.append((cols[valid_mask] * 255).astype(np.uint8))

    if not all_fused_pts:
        print("⚠️  Warning: No points survived the confidence/mask filter.")
        return

    # 3. Stitch and Voxel-Deduplicate
    p_all = np.concatenate(all_fused_pts, axis=0)
    c_all = np.concatenate(all_fused_cols, axis=0) if all_fused_cols else None

    # Efficient Voxelization
    voxel_indices = np.floor(p_all / voxel_size).astype(np.int32)
    # Using a 1D hash for fast uniqueness check
    keys = (voxel_indices[:, 0].astype(np.int64) * 100_000_000 +
            voxel_indices[:, 1].astype(np.int64) * 10_000 +
            voxel_indices[:, 2].astype(np.int64))

    _, unique_idx = np.unique(keys, return_index=True)
    p_final = p_all[unique_idx]
    c_final = c_all[unique_idx] if c_all is not None else None

    # 4. Write the PLY file
    with open(output_path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(p_final)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if c_final is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(p_final)):
            p, c = p_final[i], c_final[i]
            line = f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
            if c_final is not None:
                line += f" {c[0]} {c[1]} {c[2]}"
            f.write(line + "\n")

    print(f"✅ Automated Scene Reconstructed: {output_path} ({len(p_final):,} points)")


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import SceneHumanViewer

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return
    
    if args.max_frames is not None:
        img_paths = img_paths[:args.max_frames]
    img_paths = img_paths[::args.subsample]

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    # Prepare input views.
    print("Preparing input views...")
    img_res = getattr(model, 'mhmr_img_res', None)
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=args.reset_interval
    )

    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, _ = inference_recurrent_lighter(
        views, model, device, use_ttt3r=args.use_ttt3r)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    (
        pts3ds_other, 
        colors, 
        conf, 
        cam_dict, 
        all_smpl_verts, 
        smpl_faces,
        smpl_id,
        msks,
        smpl_params,
        ) = prepare_output(
        outputs, args.output_dir, 1, True, 
        args.save, args.render, args.render_video, img_res, args.subsample
    )

    scene_ply_path = os.path.join(args.output_dir, "final_proper_scene.ply")
    save_fused_scene_automated(
        pts3ds_other, 
        conf, 
        msks, 
        colors, 
        scene_ply_path,
        stride=5,         # Keep every 5th frame for the stitch
        conf_thresh=1.2,  # Viser-quality threshold
        voxel_size=0.02   # 2cm grid for high fidelity
    )

    export_for_dashboard(
    args.output_dir,
    cam_dict,
    smpl_params,
    smpl_id,
    all_smpl_verts,
    pts3ds_other,
    msks,
    colors=colors      # add this — gives you RGB in the PLY
    )


    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    msks_to_vis = [m.cpu().numpy() for m in msks]
    conf_to_vis = [c.cpu().numpy() for c in conf]
    edge_colors = [None] * len(pts3ds_to_vis)
    verts_to_vis = [p.cpu().numpy() for p in all_smpl_verts]

    # Create and run the point cloud viewer.
    print("Launching Human3R viewer...")
    viewer = SceneHumanViewer(
        pts3ds_to_vis,
        colors_to_vis,
        conf_to_vis,
        cam_dict,
        verts_to_vis,
        smpl_faces,
        smpl_id,
        msks_to_vis,
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        msk_threshold=args.msk_threshold,
        mask_morph=args.mask_morph,
        size = args.size,
        downsample_factor=args.downsample_factor,
        smpl_downsample_factor=args.smpl_downsample,
        camera_downsample_factor=args.camera_downsample
    )

    # Wrap this at the bottom of demo.py
    if not args.no_vis:
        print("Launching Human3R viewer...")
        viewer = SceneHumanViewer(...)
        viewer.run()
    else:
        print("Skipping viewer launch as --no_vis is set.")
    viewer.run()



def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
