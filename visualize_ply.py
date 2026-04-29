#!/usr/bin/env python3
import open3d as o3d
import argparse
import sys
import os

def visualize_ply(file_path, show_axes=False):
    """Loads and visualizes a .ply file as either a point cloud or a mesh."""
    
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        sys.exit(1)
        
    print(f"Loading {file_path}...")
    
    geometries_to_draw = []

    # 1. Attempt to load the file as a Point Cloud
    pcd = o3d.io.read_point_cloud(file_path)
    
    if not pcd.is_empty():
        print(f"Successfully loaded a point cloud with {len(pcd.points)} points.")
        geometries_to_draw.append(pcd)
    else:
        # 2. Fallback: Attempt to load as a Triangle Mesh
        print("Point cloud is empty. Attempting to load as a 3D mesh...")
        mesh = o3d.io.read_triangle_mesh(file_path)
        
        if not mesh.is_empty():
            print("Successfully loaded a 3D mesh.")
            # Computing normals helps the lighting look correct in the viewer
            mesh.compute_vertex_normals()
            geometries_to_draw.append(mesh)
        else:
            print("Failed to read the .ply file as either a point cloud or a mesh.")
            sys.exit(1)

    # 3. Optionally add coordinate frames (X=Red, Y=Green, Z=Blue)
    # This is highly useful for debugging camera orientations (OpenCV vs OpenGL)
    if show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geometries_to_draw.append(axes)

    # Launch the interactive Open3D visualizer
    print("Launching visualizer. Use your mouse to rotate and zoom.")
    o3d.visualization.draw_geometries(geometries_to_draw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A lightweight script to view .ply files.")
    parser.add_argument("--path", type=str, required=True, help="Path to the .ply file.")
    parser.add_argument("--axes", action="store_true", help="Display the XYZ coordinate axes.")
    
    args = parser.parse_args()
    visualize_ply(args.path, show_axes=args.axes)