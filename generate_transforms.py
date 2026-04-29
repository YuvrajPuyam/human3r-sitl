import os
import glob
import json
import numpy as np
import cv2
import open3d as o3d


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR        = "outputs/GoodMornin1"
SAMPLE_EVERY_N  = 2      # every Nth frame for point cloud
CONF_PERCENTILE = 60     # keep top (100-X)% confident points per frame
DEPTH_MAX       = None   # float to clip far depth, or None
VOXEL_SIZE      = None   # float to force voxel size, or None = auto

# How many frames to sample when computing pose-depth scale
SCALE_SAMPLE_FRAMES = 20

# =============================================================================


def opencv_to_opengl(c2w_cv):
    c2w_gl = c2w_cv.copy()
    c2w_gl[:, 1] *= -1
    c2w_gl[:, 2] *= -1
    return c2w_gl


def estimate_pose_to_depth_scale(color_files, camera_dir, depth_dir, conf_dir,
                                  n_samples=20, depth_max=None):
    """
    Human3R camera translations are in a normalized unit that does not match
    the metric depth values. This estimates the scale factor S such that:
        true_translation = pose_translation * S

    For each sampled frame, take the median depth of high-confidence pixels.
    Compute the camera baseline (distance from frame 0) from raw pose translations.
    Scale = median(depth / baseline) across frames.
    """
    indices = np.linspace(1, len(color_files) - 1, n_samples, dtype=int)

    median_depths  = []
    pose_baselines = []

    cam0 = np.load(os.path.join(
        camera_dir,
        os.path.splitext(os.path.basename(color_files[0]))[0] + ".npz"
    ))
    t0 = cam0["pose"][:3, 3]

    for idx in indices:
        base       = os.path.splitext(os.path.basename(color_files[idx]))[0]
        depth_path = os.path.join(depth_dir,  f"{base}.npy")
        conf_path  = os.path.join(conf_dir,   f"{base}.npy")
        cam_path   = os.path.join(camera_dir, f"{base}.npz")

        if not all(os.path.exists(p) for p in [depth_path, conf_path, cam_path]):
            continue

        depth = np.load(depth_path)
        conf  = np.load(conf_path)
        cam   = np.load(cam_path)

        thresh = np.percentile(conf, 70)
        valid  = (conf >= thresh) & (depth > 0)
        if depth_max is not None:
            valid &= depth <= depth_max
        if valid.sum() == 0:
            continue

        median_depths.append(np.median(depth[valid]))

        t_i      = cam["pose"][:3, 3]
        baseline = np.linalg.norm(t_i - t0)
        pose_baselines.append(baseline)

    median_depths  = np.array(median_depths)
    pose_baselines = np.array(pose_baselines)

    nonzero = pose_baselines > 1e-9
    if nonzero.sum() < 3:
        fallback = float(np.median(median_depths))
        print(f"  WARNING: Camera barely moves — using depth median as scale: {fallback:.2f}")
        return fallback

    ratios = median_depths[nonzero] / pose_baselines[nonzero]
    scale  = float(np.median(ratios))

    print(f"  Depth range  (sampled) : [{median_depths.min():.3f}, {median_depths.max():.3f}]")
    print(f"  Pose baseline range    : [{pose_baselines[nonzero].min():.6f}, "
          f"{pose_baselines[nonzero].max():.6f}]")
    print(f"  Estimated scale factor : {scale:.2f}")
    return scale


def unproject_to_world_gl(depth, intrinsics, c2w_cv_scaled,
                           mask=None, conf=None, conf_threshold=None,
                           depth_max=None):
    """Back-project depth to OpenGL world-space points."""
    h, w   = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    uu, vv = np.meshgrid(np.arange(w), np.arange(h))

    valid = depth > 0
    if mask      is not None: valid &= mask > 0
    if depth_max is not None: valid &= depth <= depth_max
    if conf      is not None and conf_threshold is not None:
        valid &= conf >= conf_threshold

    if valid.sum() == 0:
        return None, valid

    u_v = uu[valid]; v_v = vv[valid]; z_c = depth[valid]
    x_c = (u_v - cx) * z_c / fx
    y_c = (v_v - cy) * z_c / fy
    pts_cam_cv = np.stack((x_c, y_c, z_c), axis=-1)

    R = c2w_cv_scaled[:3, :3]
    t = c2w_cv_scaled[:3, 3]
    pts_world_cv = (pts_cam_cv @ R.T) + t

    # Flip Y and Z to go from OpenCV to OpenGL world space
    pts_world_gl = pts_world_cv.copy()
    pts_world_gl[:, 1] *= -1
    pts_world_gl[:, 2] *= -1

    return pts_world_gl, valid


def auto_voxel_size(pts, target_points=600_000):
    if len(pts) <= target_points:
        return None
    diagonal = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    return float(diagonal / (target_points ** (1 / 3)))


def print_diagnostics(ns_data, final_pts, scale):
    print("\n--- DIAGNOSTICS ---")
    print(f"Scale applied : {scale:.2f}x")
    if final_pts is not None and len(final_pts) > 0:
        print(f"Point cloud   : {len(final_pts):,} points")
        print(f"  X : [{final_pts[:,0].min():.3f}, {final_pts[:,0].max():.3f}]")
        print(f"  Y : [{final_pts[:,1].min():.3f}, {final_pts[:,1].max():.3f}]")
        print(f"  Z : [{final_pts[:,2].min():.3f}, {final_pts[:,2].max():.3f}]")
    else:
        print("Point cloud : EMPTY")

    print("\nCamera positions (first 5 frames, after scaling):")
    for frame in ns_data["frames"][:5]:
        mat = np.array(frame["transform_matrix"])
        pos = mat[:3, 3]
        print(f"  {frame['file_path']} -> [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    if final_pts is not None and len(final_pts) > 0:
        cam_positions = np.array([
            np.array(f["transform_matrix"])[:3, 3] for f in ns_data["frames"]
        ])
        overlap = all(
            cam_positions[:, ax].max() >= final_pts[:, ax].min() and
            cam_positions[:, ax].min() <= final_pts[:, ax].max()
            for ax in range(3)
        )
        if overlap:
            print("\n  OK: Camera positions overlap point cloud — scale looks correct.")
        else:
            print("\n  WARNING: Cameras and point cloud still don't overlap in world space.")
            print("  Try running with SCALE_SAMPLE_FRAMES=40 or inspect poses manually.")
    print("-------------------\n")


def prepare_nerfstudio_data():
    color_dir   = os.path.join(BASE_DIR, "color")
    mask_dir    = os.path.join(BASE_DIR, "masks_background")
    depth_dir   = os.path.join(BASE_DIR, "depth")
    conf_dir    = os.path.join(BASE_DIR, "conf")
    camera_dir  = os.path.join(BASE_DIR, "camera")
    output_json = os.path.join(BASE_DIR, "transforms.json")
    output_ply  = os.path.join(BASE_DIR, "init.ply")

    color_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    num_frames  = len(color_files)
    print(f"Found {num_frames} frames.")

    # ------------------------------------------------------------------
    # STEP 1: Estimate scale between pose translations and depth units
    # ------------------------------------------------------------------
    print("\nEstimating pose-to-depth scale...")
    scale = estimate_pose_to_depth_scale(
        color_files, camera_dir, depth_dir, conf_dir,
        n_samples=SCALE_SAMPLE_FRAMES, depth_max=DEPTH_MAX
    )

    # ------------------------------------------------------------------
    # STEP 2: Build transforms.json and point cloud
    # ------------------------------------------------------------------
    ns_data = {
        "camera_model": "OPENCV",
        "orientation_override": "none",
        "ply_file_path": "init.ply",
        "frames": [],
    }

    all_pts    = []
    all_colors = []

    for i, img_path in enumerate(color_files):
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        cam_path = os.path.join(camera_dir, f"{base_name}.npz")
        if not os.path.exists(cam_path):
            continue

        cam_data   = np.load(cam_path)
        intrinsics = cam_data["intrinsics"]
        c2w_cv_raw = cam_data["pose"].copy()

        # Apply scale to translation only — rotation is dimensionless
        c2w_cv_scaled = c2w_cv_raw.copy()
        c2w_cv_scaled[:3, 3] *= scale

        # OpenGL version goes into transforms.json
        c2w_gl = opencv_to_opengl(c2w_cv_scaled)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]

        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

        ns_data["frames"].append({
            "file_path":       f"color/{base_name}.png",
            "mask_path":       f"masks_background/{base_name}.png",
            "depth_file_path": f"depth/{base_name}.npy",
            "fl_x": fx, "fl_y": fy,
            "cx":   cx, "cy":   cy,
            "w": w, "h": h,
            "transform_matrix": c2w_gl.tolist(),
        })

        # Point cloud every SAMPLE_EVERY_N frames
        if i % SAMPLE_EVERY_N != 0:
            continue

        depth_path = os.path.join(depth_dir, f"{base_name}.npy")
        conf_path  = os.path.join(conf_dir,  f"{base_name}.npy")
        mask_path  = os.path.join(mask_dir,  f"{base_name}.png")

        if not all(os.path.exists(p) for p in [depth_path, conf_path, mask_path]):
            continue

        depth = np.load(depth_path)
        conf  = np.load(conf_path)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        conf_threshold = np.percentile(conf, CONF_PERCENTILE)

        pts_gl, valid = unproject_to_world_gl(
            depth, intrinsics, c2w_cv_scaled,
            mask=mask, conf=conf, conf_threshold=conf_threshold,
            depth_max=DEPTH_MAX,
        )
        if pts_gl is None:
            continue

        all_pts.append(pts_gl)
        all_colors.append(img_rgb[valid] / 255.0)

        if i % 20 == 0:
            print(f"  frame {i:04d}/{num_frames} — {len(pts_gl):,} pts "
                  f"(conf_thresh={conf_threshold:.3f})")

    # ------------------------------------------------------------------
    # STEP 3: Save transforms.json
    # ------------------------------------------------------------------
    print(f"\nSaving {output_json} ({len(ns_data['frames'])} frames)...")
    with open(output_json, "w") as f:
        json.dump(ns_data, f, indent=4)

    if not all_pts:
        print("ERROR: No valid points — check paths and filter settings.")
        return

    final_pts    = np.concatenate(all_pts,    axis=0)
    final_colors = np.concatenate(all_colors, axis=0)

    print_diagnostics(ns_data, final_pts, scale)

    # ------------------------------------------------------------------
    # STEP 4: Build and save point cloud
    # ------------------------------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_pts)
    pcd.colors = o3d.utility.Vector3dVector(final_colors)

    print("Removing statistical outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  After outlier removal : {len(pcd.points):,} points")

    voxel = VOXEL_SIZE or auto_voxel_size(np.asarray(pcd.points))
    if voxel:
        print(f"Voxel downsampling (voxel_size={voxel:.5f})...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel)
        print(f"  After downsampling    : {len(pcd.points):,} points")

    print(f"Saving {output_ply}...")
    o3d.io.write_point_cloud(output_ply, pcd)

    print("\nDone! Run:")
    print(f"  ns-train splatfacto --data {BASE_DIR}")


if __name__ == "__main__":
    prepare_nerfstudio_data()