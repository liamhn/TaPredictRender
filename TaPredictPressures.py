import numpy as np
import logging
logging.getLogger("streamlit.elements.lib.policies").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
import streamlit as st
import numpy as np
import plotly.graph_objects as go

import open3d as o3d
import pandas as pd
import xgboost as xgb
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def get_pressure_lattice(mesh,regularity,multiplier):
    mesh.compute_vertex_normals()
    mesh_vertices = np.asarray(mesh.vertices)
    mesh.compute_triangle_normals()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.orient_triangles()
    # --- Compute triangle areas ---
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    def triangle_area(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    areas = triangle_area(a, b, c)
    
    # --- Project triangle normals into XY plane ---
    normals = np.asarray(mesh.triangle_normals)
    xy_normals = normals[:, :2]
    xy_norms = np.linalg.norm(xy_normals, axis=1)
    valid_mask = xy_norms > 1e-6
    xy_normals = xy_normals[valid_mask]
    areas = areas[valid_mask]
    xy_normals /= xy_norms[valid_mask][:, np.newaxis]
    
    # --- Compute angles of XY normals ---
    angles = (np.degrees(np.arctan2(xy_normals[:,1], xy_normals[:,0])) + 360) % 360
    
    # --- Bin into histogram and find dominant direction ---
    hist, bin_edges = np.histogram(angles, bins=360*10, weights=areas) # multiple by 10 to get decimal angles in 0.1 intervals
    dominant_angle = bin_edges[np.argmax(hist)]
    
    # --- Snap to closest axis-aligned version (0, 90, 180, 270) ---
    closest_axis = min([0, 90, 180, 270], key=lambda x: abs((dominant_angle - x + 180) % 360 - 180))
    rotation_needed = dominant_angle - closest_axis
    
    # --- Apply rotation around Z axis ---
    theta = np.radians(rotation_needed)
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    R = np.eye(3)
    R[:2, :2] = [[cos_t, -sin_t], [sin_t, cos_t]]
    
    center = vertices.mean(axis=0)
    centered = vertices - center
    rotated = centered @ R.T
    mesh.vertices = o3d.utility.Vector3dVector(rotated)
    
    dense_mesh_pcd=mesh.sample_points_uniformly(int(multiplier*mesh.get_surface_area())) # make a dense, but roughly sampled PCD
    
    lattice_mesh_pcd = dense_mesh_pcd.voxel_down_sample(voxel_size=regularity) #use that rough PCD to make regular lattice PCD with voxel downsampling
    
    
    # Get the sampled points
    sampled_points = np.asarray(lattice_mesh_pcd.points)
    
    # Apply inverse rotation (transpose of rotation matrix)
    R_inverse = R.T
    points_unrotated = sampled_points @ R_inverse.T
    unrotated = np.asarray(mesh.vertices) @ R_inverse.T  # only forr testing and visualization
    
    # Add back the original center (since we subtracted it before rotation)
    points_original_location = points_unrotated + center
    uncentered = unrotated + center # only for testing and visualization
    
    
    # Update the point cloud with the transformed points
    lattice_mesh_pcd.points = o3d.utility.Vector3dVector(points_original_location)
    mesh.vertices = o3d.utility.Vector3dVector(uncentered)



    # Parameters:

    #regularity = 2. # We are going to use a point spacing of 2. in this case -- since we are goign to be interpolating from far away top these points anyway it is ok if we have relatively coarse points. This will also cut down massively on ram and computational requiements
    #multiplier = 1000   # We'll still use a relatively high (but considerably smaller) multiplier here, since the points are spaced much further, the noise created byas  lower multiplier affects the resaults much less, however, we still want good spacing.

    #mesh_angle_sharp_corner_threshold = 10 # Anglke threshold of 10 degrees for dewt4ecting sharp edges. for mesh generation

    return lattice_mesh_pcd
    
    



def get_positions_knn_vectors_and_normal_vector_distances(mesh_pcd,K=25):

    """ """

    # Build a KDTree for the point cloud
    kdtree = o3d.geometry.KDTreeFlann(mesh_pcd)
    
    # Parameters
    all_vectors = []  # Store vectors instead of distances
    all_normal_vector_differences = [] # These will be in degrees
    
    
    #print("computing KNN vectors")
    # Compute K-nearest neighbors for each point
    for i, point in enumerate(mesh_pcd.points):
        #print(i,len(mesh_pcd.points))
        [k, idx, distances] = kdtree.search_knn_vector_3d(point, K)
        neighbors = np.asarray(mesh_pcd.points)[idx[1:]]  # Exclude the point itself
        vectors = neighbors - point  # Compute vectors to neighbors
        all_vectors.append(vectors)  # Append vectors for the current point
        
        # Compute normal vector differences
        query_normal = np.array(mesh_pcd.normals[i])
        neighbor_normals = np.array([mesh_pcd.normals[j] for j in idx[1:]])
    
        # Compute dot product to get cosine similarity; for normal vectors, angle between them contains all the information about how they differ
        cos_theta = np.einsum('ij,j->i', neighbor_normals, query_normal)  # Efficient dot product
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Avoid floating point errors
    
        # Convert to angles (in degrees)
        normal_vector_differences = np.arccos(cos_theta) * (180.0 / np.pi)
        all_normal_vector_differences.append(normal_vector_differences)
    
    # Convert to NumPy arrays for easier analysis
    all_vectors = np.array(all_vectors)
    all_normal_vector_differences = np.array(all_normal_vector_differences)
    
    # Name the columns cause why not
    columnNames=['x','y','z']
    for i in range(K-1):
        columnNames+=[f"KNNDvecX{i}",f"KNNDvecY{i}",f"KNNDvecZ{i}"]
    for i in range(K-1):
        columnNames+=[f"KNNNormDifference{i}"]
    data_df = pd.DataFrame(np.hstack([mesh_pcd.points,all_vectors.reshape(len(all_vectors),-1,),all_normal_vector_differences]),columns=columnNames)

    return data_df


def predict_base_model(knn_data):

    model = xgb.XGBRegressor()
    model.load_model("./xbg_mostbasic_firstpass_mesh_nest100_maxdepth6_lr0d1.json")


    y_pred = model.predict(knn_data.values[:,2:])

    return y_pred

def recolour_mesh(mesh, mesh_pcd_pressures, y_pred):
    mesh_vertices = np.asarray(mesh.vertices)
    tree_lattice = cKDTree(np.asarray(mesh_pcd_pressures.points))
    _, idx2 = tree_lattice.query(mesh_vertices, k=4)
    vertex_pressures = np.mean(y_pred[idx2], axis=1)

    cmap = plt.get_cmap('jet')
    norm = (vertex_pressures - vertex_pressures.min()) / (vertex_pressures.max() - vertex_pressures.min() + 1e-9)
    colors = cmap(norm)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh, vertex_pressures  # ✅ return pressures too

def find_local_maxima(mesh, pressures, k=6, percentile=85):
    """
    Find local maxima on a mesh that are also above a percentile threshold.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Mesh with vertices.
        pressures (np.ndarray): Pressure values per vertex (N,).
        k (int): Number of nearest neighbors to consider.
        percentile (float): Percentile threshold (0-100), only keep maxima above this.
    
    Returns:
        np.ndarray: Indices of selected vertices.
    """
    vertices = np.asarray(mesh.vertices)
    tree = cKDTree(vertices)

    _, neighbors = tree.query(vertices, k=k+1)
    neighbors = neighbors[:, 1:]  # drop self

    # Threshold for top pressures
    threshold = np.percentile(pressures, percentile)

    # Local max AND above threshold
    is_local_max = np.array([
        pressures[i] >= np.max(pressures[neighbors[i]]) and pressures[i] >= threshold
        for i in range(len(vertices))
    ])

    maxima_indices = np.where(is_local_max)[0]
    return maxima_indices

from scipy.spatial import cKDTree
import numpy as np

def find_local_maxima_away_from_taps(mesh, pressures, taps=None, k=6, percentile=85, min_dist_to_tap=0.5):
    """
    Find local maxima on a mesh that are:
      1) Above a percentile threshold
      2) Local maxima compared to neighbors
      3) Not within min_dist_to_tap of existing taps
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Mesh with vertices.
        pressures (np.ndarray): Pressure values per vertex (N,).
        taps (np.ndarray): Coordinates of existing taps (M,3), optional.
        k (int): Number of nearest neighbors to consider.
        percentile (float): Percentile threshold (0-100).
        min_dist_to_tap (float): Minimum distance from existing taps.
    
    Returns:
        np.ndarray: Indices of selected vertices.
    """
    vertices = np.asarray(mesh.points)#np.asarray(mesh.vertices)

    tree = cKDTree(vertices)

    _, neighbors = tree.query(vertices, k=k+1)
    neighbors = neighbors[:, 1:]  # drop self

    # Threshold for top pressures
    threshold = np.percentile(pressures, percentile)

    # Local max AND above threshold
    is_local_max = np.array([
        pressures[i] >= np.max(pressures[neighbors[i]]) and pressures[i] >= threshold
        for i in range(len(vertices))
    ])

    # Exclude vertices too close to taps
    if taps is not None and len(taps) > 0:
        tap_tree = cKDTree(taps)
        distances, _ = tap_tree.query(vertices)
        is_local_max &= distances >= min_dist_to_tap

    maxima_indices = np.where(is_local_max)[0]
    return maxima_indices


def add_maxima_to_taps(building, maxima_points):
    """
    Add maxima points to building.PredictedTapPCD and allow undo.
    
    Args:
        building: Your building object with PredictedTapPCD.
        maxima_points (np.ndarray): Points to add (N,3).
    """
    if not hasattr(building, "PredictedTapPCD") or building.PredictedTapPCD is None:
        building.PredictedTapPCD = o3d.geometry.PointCloud()

    # Store original points for undo
    if not hasattr(building, "deleted_maxima_points"):
        building.deleted_maxima_points = []

    # Add points
    current_points = np.asarray(building.PredictedTapPCD.points)
    new_points = np.vstack([current_points, maxima_points]) if len(current_points) > 0 else maxima_points
    building.PredictedTapPCD.points = o3d.utility.Vector3dVector(new_points)

    # Save added points for undo
    building.deleted_maxima_points.append(maxima_points.copy())

def undo_last_added_maxima(building):
    """
    Undo the last set of maxima points added to PredictedTapPCD.
    """
    if hasattr(building, "deleted_maxima_points") and building.deleted_maxima_points:
        last_added = building.deleted_maxima_points.pop()
        current_points = np.asarray(building.PredictedTapPCD.points)
        
        # Remove the last added points
        mask = np.ones(len(current_points), dtype=bool)
        for p in last_added:
            idx = np.all(current_points == p, axis=1)
            mask[idx] = False
        building.PredictedTapPCD.points = o3d.utility.Vector3dVector(current_points[mask])
    else:
        print("No added maxima to undo.")