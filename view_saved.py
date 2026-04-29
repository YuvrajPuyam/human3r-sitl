#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import numpy as np
import torch
import imageio.v2 as iio

# Add the 'src' directory to Python's path to fix absolute import errors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.dust3r.utils import SMPL_Layer
from viser_utils import SceneHumanViewer

def parse_args():
    parser = argparse.ArgumentParser(description="Offline Human-Scene Viewer for saved outputs.")
    parser.add_argument("--output_dir", type=str, default="outputs/test10s", help="Path to the saved inference outputs.")
    parser.add_argument("--vis_threshold", type=float, default=2.0, help="Visualization threshold.")
    parser.add_argument("--msk_threshold", type=float, default=0.1, help="Mask threshold.")
    parser.add_argument("--downsample_factor", type=int, default=2, help="Point cloud downsample factor.")
    parser.add_argument("--smpl_downsample", type=int, default=1, help="SMPL downsample factor.")
    parser.add_argument("--camera_downsample", type=int, default=1, help="Camera downsample factor.")
    parser.add_argument("--mask_morph", type=int, default=10, help="Mask morphology size.")
    parser.add_argument("--size", type=int, default=224, help="Input size used during inference.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for SMPL calculation.")
    return parser.parse_args()

def unproject_depth(depth, intrinsics):
    """Reconstructs 3D points in camera space from a depth map and intrinsics."""
    H, W = depth.shape
    v, u = np.indices((H, W))
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    
    return np.stack([x, y, depth], axis=-1)

def transform_to_world(pts3d_cam, pose):
    """Transforms 3D points from camera space to world space."""
    R = pose[:3, :3]
    t = pose[:3, 3]
    return pts3d_cam @ R.T + t

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    
    color_files = sorted(glob.glob(os.path.join(args.output_dir, "color", "*.png")))
    if not color_files:
        print(f"No rendered images found in {args.output_dir}/color. Please check the path.")
        return
        
    num_frames = len(color_files)
    print(f"Found {num_frames} saved frames. Loading data...")

    pts3ds_to_vis = []
    colors_to_vis = []
    conf_to_vis = []
    msks_to_vis = []
    verts_to_vis = []
    smpl_id_list = []
    
    focal_list, pp_list, R_list, t_list = [], [], [], []
    
    smpl_layer = None
    smpl_faces = None

    for f_id in range(num_frames):
        # 1. Load Color and add batch dim
        color_img = iio.imread(color_files[f_id])
        colors_to_vis.append((color_img.astype(np.float32) / 255.0)[None, ...]) 
        
        # 2. Load Depth, Confidence, and Camera extrinsics/intrinsics
        depth = np.load(os.path.join(args.output_dir, "depth", f"{f_id:06d}.npy"))
        
        conf = np.load(os.path.join(args.output_dir, "conf", f"{f_id:06d}.npy"))
        conf_to_vis.append(conf[None, ...])
        
        cam_data = np.load(os.path.join(args.output_dir, "camera", f"{f_id:06d}.npz"))
        pose = cam_data["pose"]             
        intrinsics = cam_data["intrinsics"] 
        
        # 3. Reconstruct 3D Point Cloud for this frame and add batch dim
        pts3d_cam = unproject_depth(depth, intrinsics)
        pts3d_world = transform_to_world(pts3d_cam, pose)
        pts3ds_to_vis.append(pts3d_world[None, ...]) 
        
        # 4. Extract Camera parameters
        focal_list.append(np.array([intrinsics[0,0], intrinsics[1,1]]))
        pp_list.append(np.array([intrinsics[0,2], intrinsics[1,2]]))
        R_list.append(pose[:3, :3])
        t_list.append(pose[:3, 3])
        
        # 5. Load SMPL parameters and reconstruct meshes
        smpl_data = np.load(os.path.join(args.output_dir, "smpl", f"{f_id:06d}.npz"), allow_pickle=True)
        shape = torch.from_numpy(smpl_data["shape"]).to(device)
        rotvec = torch.from_numpy(smpl_data["rotvec"]).to(device)
        transl = torch.from_numpy(smpl_data["transl"]).to(device)
        
        exp_raw = smpl_data["expression"]
        expression = torch.from_numpy(exp_raw).to(device) if exp_raw[()] is not None else None
            
        msk = smpl_data["msk"]
        if msk is not None:
            # The original script saves msk as (1, H, W), so we enforce that shape
            msk_array = msk.reshape(1, depth.shape[0], depth.shape[1])
        else:
            # If no mask exists, create an empty one with the correct (1, H, W) shape
            msk_array = np.zeros((1, depth.shape[0], depth.shape[1]), dtype=np.float32)
            
        msks_to_vis.append(msk_array) # Notice we removed the [None, ...] here
            
        n_humans = shape.shape[0]
        if n_humans > 0:
            if smpl_layer is None:
                smpl_layer = SMPL_Layer(type='smplx', gender='neutral', 
                                        num_betas=shape.shape[-1], kid=False, 
                                        person_center='head').to(device)
                smpl_faces = smpl_layer.bm_x.faces
                
            with torch.no_grad():
                K_torch = torch.from_numpy(intrinsics).float().unsqueeze(0).expand(n_humans, -1, -1).to(device)
                smpl_out = smpl_layer(rotvec, shape, transl, None, None, K=K_torch, expression=expression)
                
            v3d_cam = smpl_out['smpl_v3d'].cpu().numpy() 
            v3d_world = transform_to_world(v3d_cam, pose)
            verts_to_vis.append(v3d_world)
            smpl_id_list.append(np.arange(n_humans))
        else:
            # Mimic `torch.empty(0).numpy()` exactly: flat 1D empty array
            verts_to_vis.append(np.empty(0, dtype=np.float32))
            smpl_id_list.append(np.empty(0, dtype=np.int64))

    # Fallback if no humans were found in the entire sequence
    if smpl_faces is None:
        smpl_faces = np.empty((0, 3))

    cam_dict = {
        "focal": np.stack(focal_list),
        "pp": np.stack(pp_list),
        "R": np.stack(R_list),
        "t": np.stack(t_list),
    }

    edge_colors = [None] * num_frames
    
    # Safety Check: Guarantee all lists are synchronized before launching viewer
    assert len(pts3ds_to_vis) == num_frames, "Mismatch in point clouds!"
    assert len(verts_to_vis) == num_frames, "Mismatch in SMPL vertices!"
    assert len(smpl_id_list) == num_frames, "Mismatch in SMPL IDs!"

    print("Launching Viser...")
    viewer = SceneHumanViewer(
        pts3ds_to_vis,
        colors_to_vis,
        conf_to_vis,
        cam_dict,
        verts_to_vis,
        smpl_faces,
        smpl_id_list,
        msks_to_vis,
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        msk_threshold=args.msk_threshold,
        mask_morph=args.mask_morph,
        size=args.size,
        downsample_factor=args.downsample_factor,
        smpl_downsample_factor=args.smpl_downsample,
        camera_downsample_factor=args.camera_downsample
    )
    viewer.run()

if __name__ == "__main__":
    main()