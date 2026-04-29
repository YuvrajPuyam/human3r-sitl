import os
import numpy as np
import glob
from tqdm import tqdm

def reconstruct_viser_scene(root_dir, output_path, conf_thresh=1.5, stride=10):
    # 1. Setup paths
    depth_files = sorted(glob.glob(os.path.join(root_dir, "depth/*.npy")))
    conf_files  = sorted(glob.glob(os.path.join(root_dir, "conf/*.npy")))
    cam_files   = sorted(glob.glob(os.path.join(root_dir, "camera/*.npz")))
    color_files = sorted(glob.glob(os.path.join(root_dir, "color/*.png")))
    smpl_files  = sorted(glob.glob(os.path.join(root_dir, "smpl/*.npz")))

    import cv2 # for color loading

    fused_pts = []
    fused_cols = []

    print(f"🏗️ Reconstructing scene from {len(depth_files)} frames...")

    for i in tqdm(range(0, len(depth_files), stride)):
        # Load Data
        depth = np.load(depth_files[i])  # Shape: (H, W)
        conf  = np.load(conf_files[i])   # Shape: (H, W)
        cam   = np.load(cam_files[i])    # Pose and Intrinsics
        color = cv2.imread(color_files[i])[..., ::-1] / 255.0
        smpl  = np.load(smpl_files[i])
        
        # 1. Get Camera Parameters
        pose = cam['pose']             # 4x4 Camera-to-World matrix
        K = cam['intrinsics']          # 3x3 Intrinsics matrix
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 2. Create Pixel Grid
        H, W = depth.shape
        v, u = np.indices((H, W))
        
        # 3. Apply Viser-style filtering
        raw_mask = (smpl['msk'][0] * 255).astype(np.uint8)
        
        # Create a dilation kernel (5x5 or 7x7 depending on how much 'halo' you see)
        # Increasing iterations will 'grow' the mask further.
        kernel = np.ones((7, 7), np.uint8) 
        dilated_mask = cv2.dilate(raw_mask, kernel, iterations=2)
        
        # Now use the expanded mask to hide the human
        h_mask_expanded = dilated_mask > 128 
        
        # Updated strategy with the expanded mask
        clean_mask = (conf > 1.2) & (~h_mask_expanded) & (depth < 12.0) & (depth > 0.1)

        z = depth[clean_mask]
        u_c = u[clean_mask]; v_c = v[clean_mask]

        x_c = (u_c - cx) * z / fx
        y_c = (v_c - cy) * z / fy
        
        pts_cam = np.stack([x_c, y_c, z, np.ones_like(z)], axis=-1)
        pts_world = (pts_cam @ pose.T)[:, :3]
        cols = color[clean_mask]

        fused_pts.append(pts_world)
        fused_cols.append(cols)

    # --- THE STITCH: FINER VOXEL GRID ---
    p_final = np.concatenate(fused_pts, axis=0)
    c_final = np.concatenate(fused_cols, axis=0)

    # 2cm grid is the 'sweet spot' for indoor scenes
    voxel_size = 0.02 
    indices = np.floor(p_final / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(indices, axis=0, return_index=True)
    
    p_final = p_final[unique_idx]
    c_final = (c_final[unique_idx] * 255).astype(np.uint8)

    # 5. Export to PLY
    write_ply(output_path, p_final, c_final)
    print(f"✅ Scene reconstructed: {output_path}")

def write_ply(path, pts, cols):
    header = f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    with open(path, 'w') as f:
        f.write(header)
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")

# Run it on your output folder
reconstruct_viser_scene("./outputs/GoodMornin1", "final_proper_scene.ply")