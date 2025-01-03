import cv2
import numpy as np
import open3d as o3d
import random

def downsample_mask(mask, factor=2):
    """Downsample the mask by a specified factor."""
    return cv2.resize(mask.astype(np.uint8), (mask.shape[1] // factor, mask.shape[0] // factor), interpolation=cv2.INTER_NEAREST)

def adjust_intrinsic_matrix(K, factor=2):
    """Adjust the intrinsic matrix for downsampling."""
    K_new = K.copy()
    K_new[0, 0] /= factor
    K_new[1, 1] /= factor
    K_new[0, 2] /= factor
    K_new[1, 2] /= factor
    return K_new

def save_point_clouds_with_colors(point_clouds, intersection_points, original_color=[0.5, 0.5, 0.5],
                                  intersection_color=[1, 0, 0], include_source=False):
    """Save multiple point clouds with different colors to a single PLY file."""
    all_points = []
    all_colors = []

    if include_source:
        for pcd in point_clouds:
            all_points.append(pcd)
            all_colors.append(np.tile([random.random(), random.random(), random.random()], (pcd.shape[0], 1)))

    if intersection_points.size > 0:
        all_points.append(intersection_points)
        all_colors.append(np.tile(intersection_color, (intersection_points.shape[0], 1)))

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    o3d.io.write_point_cloud("test.ply", pcd)


def mask_to_3d_points(mask, K, RT, depth_range=(0.1, 20), num_samples=512):
    """Convert a 2D mask into a set of 3D points using the given camera matrix and a range of depths."""
    rows, cols = np.where(mask > 0)
    depth_values = 1. / np.linspace(1./depth_range[1], 1./depth_range[0], num_samples)
    points_3d = []

    for depth in depth_values:
        # Convert pixel coordinates to normalized device coordinates
        normalized_coords = np.vstack((cols, rows, np.ones_like(cols))) * depth * -1
        cam_coords = np.linalg.inv(K) @ normalized_coords
        cam_coords_homogeneous = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
        world_coords = RT @ cam_coords_homogeneous
        points_3d.append(world_coords[:3, :].T)

    return np.vstack(points_3d)

def create_point_cloud_from_frustum(points_3d):
    """Convert a set of 3D points to a point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    return pcd

def voxel_intersection(point_clouds, voxel_size):
    """Compute the intersection of point clouds and find the bounding box."""
    voxel_maps = []
    origin = np.min(np.vstack(point_clouds), axis=0)
    print(f"Voxel grid origin: {origin}")
    for idx, pcd in enumerate(point_clouds):
        new_pcd = np.vstack([pcd, origin]) - origin
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(new_pcd)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_o3d, voxel_size)
        voxel_indices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        voxel_maps.append(set(map(tuple, voxel_indices)))
        print(f"Voxel map {idx + 1}: {len(voxel_indices)} voxels")

    intersection_voxels = voxel_maps[0]
    for voxel_set in voxel_maps[1:]:
        intersection_voxels &= voxel_set

    if not intersection_voxels:
        return np.array([])  # No intersection

    # Convert voxel centers back to points
    intersection_points = np.array(list(intersection_voxels)) * voxel_size + voxel_size / 2 + origin
    intersection_points = np.delete(intersection_points, intersection_points.argmin(0), axis=0)
    return intersection_points
    # intersection_voxels = set(map(tuple, np.asarray(voxel_maps[0].get_voxels())))
    # # Find the intersection of voxel sets
    # intersected_voxels = set.intersection(*voxel_maps)
    # if not intersected_voxels:
    #     return None
    # # Convert voxels back to points
    # intersection_points = np.array([np.fromstring(v, dtype=int) * voxel_size for v in intersected_voxels])
    # return intersection_points