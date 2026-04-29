import os
import glob
import argparse
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def interpolate_missing_data(data_array, valid_indices, total_frames):
    """Linearly interpolates missing frames for continuous temporal filtering."""
    if len(valid_indices) == total_frames:
        return data_array
    
    # Create an interpolation function based on the valid frames
    interpolator = interp1d(valid_indices, data_array, axis=0, 
                            kind='linear', fill_value='extrapolate')
    
    # Evaluate the function across all frame indices
    return interpolator(np.arange(total_frames))

def main():
    parser = argparse.ArgumentParser(description="Prepare SMPL data for ZJU-MoCap format.")
    parser.add_argument("--input_dir", type=str, required=True, help="Base directory containing 'smpl' and 'camera' folders.")
    args = parser.parse_args()

    smpl_dir = os.path.join(args.input_dir, "smpl")
    cam_dir = os.path.join(args.input_dir, "camera")
    output_file = os.path.join(args.input_dir, "annots.npy")
    
    # Savitzky-Golay filter parameters
    # window_length must be odd. 11 frames is ~1/3 of a second at 30fps.
    window_length = 11 
    polyorder = 3
    
    smpl_files = sorted(glob.glob(os.path.join(smpl_dir, "*.npz")))
    cam_files = sorted(glob.glob(os.path.join(cam_dir, "*.npz")))
    num_frames = len(smpl_files)
    
    if num_frames == 0:
        print(f"Error: Found 0 frames in {smpl_dir}. Please check your input directory.")
        return

    print(f"Loading {num_frames} frames from {args.input_dir}...")

    # Lists to hold raw sequences
    raw_poses = []
    raw_transls = []
    valid_indices = []
    
    # ZJU-MoCap camera dictionaries
    cams = {}
    global_shape = None

    for i in range(num_frames):
        # 1. Process Camera Data
        cam_data = np.load(cam_files[i])
        c2w = cam_data["pose"]         # Shape: (4, 4)
        K = cam_data["intrinsics"]     # Shape: (3, 3)
        
        # Convert Camera-to-World (c2w) to World-to-Camera (w2c) for ZJU format
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]                # Shape: (3, 3)
        T = w2c[:3, 3].reshape(3, 1)   # Shape: (3, 1)
        
        cams[i] = {
            "K": K,
            "R": R,
            "T": T
        }

        # 2. Process SMPL Data
        smpl_data = np.load(smpl_files[i])
        shape_arr = smpl_data["shape"]     # Shape: (N_humans, 10)
        rotvec_arr = smpl_data["rotvec"]   # Shape: (N_humans, J, 3)
        transl_arr = smpl_data["transl"]   # Shape: (N_humans, 3)
        
        if shape_arr.shape[0] > 0:
            # We assume the primary subject is index 0
            if global_shape is None:
                global_shape = shape_arr[0] # Shape: (10,)
                
            # Flatten the joint rotvecs into a single 1D vector per frame
            pose_flat = rotvec_arr[0].flatten() 
            transl_flat = transl_arr[0]
            
            raw_poses.append(pose_flat)
            raw_transls.append(transl_flat)
            valid_indices.append(i)

    # Convert to NumPy arrays for temporal processing
    poses_np = np.stack(raw_poses)
    transls_np = np.stack(raw_transls)
    
    missing_frames = num_frames - len(valid_indices)
    if missing_frames > 0:
        print(f"Interpolating {missing_frames} missing frames...")
    
    poses_full = interpolate_missing_data(poses_np, valid_indices, num_frames)
    transls_full = interpolate_missing_data(transls_np, valid_indices, num_frames)

    print("Applying Savitzky-Golay filter...")
    # Prevent filter window length from exceeding the number of available frames
    current_window = min(window_length, num_frames)
    if current_window % 2 == 0:
        current_window -= 1

    if current_window > polyorder:
        smooth_poses = savgol_filter(poses_full, current_window, polyorder, axis=0)
        smooth_transls = savgol_filter(transls_full, current_window, polyorder, axis=0)
    else:
        print("Warning: Sequence too short for Savitzky-Golay filtering. Skipping smoothing.")
        smooth_poses = poses_full
        smooth_transls = transls_full

    # 3. Package into ZJU-MoCap format
    annots_data = {
        "cams": cams,
        "ims": [{"cam_idx": i, "time": i} for i in range(num_frames)],
        "params": {
            "betas": global_shape,
            "Rh": smooth_poses[:, :3],
            "Th": smooth_transls,
            "poses": smooth_poses[:, 3:]
        }
    }

    print(f"Saving packaged data to {output_file}...")
    np.save(output_file, annots_data, allow_pickle=True)
    print("Done.")

if __name__ == "__main__":
    main()