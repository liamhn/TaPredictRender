import sys
import os

from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull, cKDTree, Voronoi
from scipy.spatial import KDTree as SCKDTree
from scipy.interpolate import interp1d

from shapely.geometry import Polygon, Point

import networkx as nx

import open3d as o3d
import trimesh

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree as SklearnKDTree

from concave_hull import concave_hull_indexes

















































































class Building:

    def __init__(self,file_path):
        
        self.mesh_file_path = file_path
        self.mesh_file_path = self.mesh_file_path.replace("\\","/")
        self.mesh_folder = os.path.dirname(self.mesh_file_path)

        self.mesh_pcd_generation_multiplier = 1000 # Default Value
        self.mesh_pcd_spacing = 0.5 # Default Value
        self.mesh_pcd = None

        self.mesh_edge_pcd_angle_threshold = 20
        self.mesh_edge_pcd = None
        self.total_pcd = None
        
        self.roof_lines = None
        self.facade_z_levels = None
        self.roof_clusters = None

        self.ZLevels = None
        self.Roofs = None
        self.PredictedTaps = None

        self.PredictedTapPCD=None
        self.AllCornerPCD = None

        # Set the default heuristic paramters
        # Roof identification - clustering
        self.roof_nz_cutoff = 0.2                                           # nz value for first pass roof identification ( IE does the point point upward to some extent)
        self.roof_z_eps = 0.1                                               # Clustering value (epsilon) for clustering identified roof points in z, larger values might join roofs together, smaller values might omit roofs. Ideally this should be slightly larger than the mesh lattice bnoise in Z
        self.roof_z_min_samples = 5                                         # Number of points in a z-cluster required to be cojnsidered a z cluster
        self.roof_xy_eps = self.mesh_pcd_spacing+0.01                       # Epsilon value foe the hierarchical clustering of roof points in XY, should probably be just higher than the spacing oif the mesh lattice. Larger values may agglomerate roofs of the same z-value that are actually distinct, smaller values may omit roofs
        self.roof_xy_min_samples = 4                                        # cluster min_samples value for the hierarchical clustering of roof points in XT. Since we are using an epsilon just above the point spacing, this should probably just be 4 always ( since we expect 4 points around a point in a plane for any roof.)

        # Roof identification - filtering
        self.roof_dense_point_radius=1.5                                    # radius for determining if a point is in a "dense region". The distance around the point to look for other points. needs to be adjuested along with roof_dense_min_neighbors   
        self.roof_dense_min_neighbors=25                                    # Threshold number of points within roof_dense_min_neighbours. If this value is lower, more balconies and "thin sections" such as parapets wikll be counted as roofs, if it is higher, larger roofs will be omitted (0, all z-pointing things are roofs, infinity - nothing is a roof). If a point on a roof has at least this many points within a roof_dense_point_radius radius, then it is considered to be a dense point 
        self.roof_min_dense_points=60                                       # Number of dense points in a roof cluster required for it to be considered a roof. Lower values will make more permissible (more balconies and small roof structures will be considered roofs, etc.). Higher values will filter out more larger structures
        self.roof_overlap_thresh = 0.5                                      #If None -- will not remove roofs below it, if 0.5 (or another number) -- will remove roofs below it if the points are within this value of the points above in XY. THis should probably be kept as either 0.5 or None ( probably 0.5)

        # Concave Hull Parameters
        self.roof_concave_hull_length_threshold = self.mesh_pcd_spacing-.02 # Length threshold for concave_hull for roof major corner detection. Probably doesn't need to change for a given spacing
        self.roof_concave_hull_concavity = 2                                # concavity for concave_hull for roof major corner detection. Probably doesn't need to change for a given spacing.

        # Parameters for corner detection and filtering
        self.roof_corner_angle_threshold=30                                 # The angle cutoff for detecting corners
        self.roof_vector_width_threshold = 0.05                             # Width of vectors for counting points in a direction, probably should be relatively close to epsilon
        self.roof_vector_length_threshold =2.                               # Length of vector for counting points in a direction, higher value means longer segments tend to be considered as major corners, smaller value will include shorter segments
        self.roof_major_cutoff_multiplier=.6                                # A scale factor for the "major corner cutoff". Cutoff is computed as major_cutoff_multiplier*vector_length/avg_pointspacing. If there are at least thjat many points in the "vector zone" it is considered a major corner

        # Parameters for genrating horizontal rows of taps on roofs
        self.roof_horizontal_spacing=12                                     # horizontal spacing of generated facade taps
        self.roof_anchor_distance=.5                                        # Distance from major corners where corner roof taps are placed
        self.roof_double_tap_dist = 2.0                                     # additional distance for double corner taps
        self.roof_double_tap_list = False                                   # allow sto specify 'which' roof corner taps get doulbe tapped. Probasbly always false
        self.roof_max_distance_from_model=self.mesh_pcd_spacing-.04         # BEfore merging roof edge points, remove any points that are not near the roof (for whatever reason, usually weird point detection). THis is the threshold distance. THis probably shouldn't be a user parameter but I will need to tune it.
        self.roof_edge_maximum_filtering_distance=2.                        # This is the distance roof edge-points can be from each other before they are replaced by the average of the points. Higher values will mean many points get averaged together, lower valuers mean fewer points do. 0 means this doesn't happen at all. infinity means all roof edges are replaced by a single point ( for each roof)
        self.roof_distance_to_concave_hull_filtration=(2*self.mesh_pcd_spacing) + (.5*self.mesh_pcd_spacing)                  # THis will remove any points that are more than X away from the concave hull itself ( for roof edge points only -- prevents crazy tap placement for badly computed concave hulls)

        self.roof_center_upperdist = 15                                     # THe largest distance to attempt voronoi tap sampling for (Larger values means fewer central center roof taps). 
        self.roof_center_mindist = 3                                        # The smallest distance to attempt voronoi tap sampling for (larger values means smaller roofs are less likely to have central taps). Taps will not be placed closer than this distance to the edges of the roof (for central taps, edge taps are separate)
        self.roof_center_min_spacing_2 = 10                                 # The distance target of the ridge line iterations of the center roof sampling ( Larger values mean less total roof taps, smaller values mean more total roof taps, this value should probably be similar to roof_center_upperdist, but can be lower if you want to get a lot of taps)
        self.roof_center_max_iter = 5                                       #  Maximum number mof times the ridge line roof tap placement will iterate
        self.roof_polyline_neighbour_radius = 10                            # an in-line parameter that probably will never need to change, determines how close points have to be to be considered part of the same polyline.


        # Parameters for roof/vertical ring assignment
        self.facade_zLevel_lower = 1                                           # Determines how far below rooflines the row of points is placed
        self.facade_min_roof_points = 25                                       # Ought to be adjusted for the number of points, I started off using 50 for 0.5, but never changed it for 0.25
        self.facade_zLevel_gap_threshold = 16                                  # doesn't need to be any particular value, I think 20 looks right on most bnuildings
        self.facade_z_levels = None

        # Parameters for genrating horizontal taps
        self.facade_horizontal_spacing=12                                   # horizontal spacing of generated facade taps
        self.facade_anchor_distance= 1.                                     # Distance from major corners where corner facade taps are placed
        self.facade_double_tap_dist = 2.0                                   # additional distance for double corner taps
        self.facade_double_tap_int = 2                                      # frequency of double tap z-levels (2 is alternating, 3 is every 3, etc.)                 
        self.facade_double_tap_list = None

        # Determine algorithm for major corner detection
        self.facade_basic_major_corners= False         # Set to True if you want to use the "basic major corner detection",False for advanced. The basic may be faster and potentially even better on certain buildings. True is likely more accurate for ornate buildings.

        # Parameters for basic corner detection
        self.facade_basic_angle_threshold=180 - 20,    # the angle used to detect corners (vs 180 rather than vs 0)
        self.facade_basic_min_segment_length=15        # the minimum segment length for determining if a corner is major or not
        self.facade_basic_smoothing_window=5           # window size for smoothing analysis (default 5) - used in significance analysis
        self.facade_basic_significance_ratio=0.0       # ratio of local vs global variation to consider significant (default 0.3), keep at 0 probably

        # Parameters for advanced corner detection and filtering
        self.facade_corner_angle_threshold=45          # The angle cutoff for detecting corners
        self.facade_vector_width_threshold = 0.05      # Width of vectors for counting points in a direction, probably should be relatively close to epsilon
        self.facade_vector_length_threshold =2.        # Length of vector for counting points in a direction, higher value means longer segments tend to be considered as major corners, smaller value will include shorter segments
        self.facade_major_cutoff_multiplier=.6         # A scale factor for the "major corner cutoff". Cutoff is computed as major_cutoff_multiplier*vector_length/avg_pointspacing. If there are at least thjat many points in the "vector zone" it is considered a major corner

        # Parameters for concave hull determination in major corner detection -- To be honest these probably shouldn't be changed by the user, but they were useful for testing and they SHOULD be determined by the spacing of the mesh pointcloud
        self.facade_epsilon = (self.mesh_pcd_spacing/2.)+.01                                # Is ideally equal to or slightly higher than HALF the spacing of the points
        self.facade_concave_hull_length_threshold = self.mesh_pcd_spacing-.02               # Is ideally equal to or slightly lower than the spacing of the points
        self.facade_hull_concavity = 2                                                      # probably best not to touch this
        self.facade_nzFiltercutoff = 10                                                     # This removes points from concave hulls that points up to a particular extent, might not be required since meshes were improved, but further testing required at this point. A large number means the filter is off.

        # Z-level filtering parameters for remove
        self.filter_whole_z_levels = False                       # Activate logic for filtering whole ZLevels
        self.whole_z_level_filter_distance = 12.                 # Filter distance if you are removing whole ZLevels
        self.facade_filtration_xy_dist = 2.1                   # Determines the width of the cyllinder for z-filtration
        self.facade_filtration_z_dist = 6.                    # determines the height of the cyllinder for z-filtration.


    def load_mesh(self):
        self.mesh = o3d.io.read_triangle_mesh(self.mesh_file_path)
        return self.mesh


    def filter_whole_zlevels(self):

        if len(self.facade_z_levels) > 0:
            filtered_z_levels = [self.facade_z_levels[0]]

            if len(self.facade_z_levels)>1:
                for i in range(1,len(self.facade_z_levels)):
                    if np.abs(self.facade_z_levels[i] - filtered_z_levels[-1]) > self.whole_z_level_filter_distance:
                        filtered_z_levels.append(self.facade_z_levels[i])

            self.facade_z_levels = filtered_z_levels
        return filtered_z_levels

    def get_predicted_tap_pcd(self):

        predicted_tap_pcd = o3d.geometry.PointCloud()
        predicted_tap_pcd.points = o3d.utility.Vector3dVector(self.PredictedTaps)
        predicted_tap_pcd.paint_uniform_color([.1,.1,.8])
        self.PredictedTapPCD = predicted_tap_pcd
        return predicted_tap_pcd

    def get_all_corner_pcd(self):
        all_corner_coords = []
        all_corner_colors = []
        for i in range(len(self.ZLevels)):
            for j in range(len(self.ZLevels[i].Corners)):
                # Append coordinates
                all_corner_coords.append(self.ZLevels[i].Corners[j].coordinates)
                
                # Color by major
                if self.ZLevels[i].Corners[j].major == np.True_: myColor = [1.,.1,.1]
                else: myColor = [.4,.1,.1]
                all_corner_colors.append(myColor)

    def collect_taps(self):
        self.predicted_facade_taps = []
        self.predicted_roof_edge_taps=[]
        self.predicted_roof_central_taps=[]

        for i in range(len(self.ZLevels)):
            # Don't include empty arrays
            if len(self.ZLevels[i].tap_coords) > 0:
                self.predicted_facade_taps.append(self.ZLevels[i].tap_coords)

        for i in range(len(self.Roofs)):
            if len(self.Roofs[i].edge_tap_coords) >0:
                self.predicted_roof_edge_taps.append(self.Roofs[i].edge_tap_coords)
            if len(self.Roofs[i].central_tap_coords) > 0:
                self.predicted_roof_central_taps.append(self.Roofs[i].central_tap_coords)

        

        all_predicted_taps = np.vstack(self.predicted_facade_taps + self.predicted_roof_edge_taps + self.predicted_roof_central_taps)

        # Filter out any that are fr from the surface of the building
        tree = cKDTree(np.asarray(self.total_pcd.points))
        taps_array = np.asarray(all_predicted_taps)
        dists, _ = tree.query(taps_array)
        good_predicted_taps = taps_array[dists <= self.mesh_pcd_spacing]

        return good_predicted_taps


    def resource_path(relative_path):
        """
        Return absolute path to resource relative to the exe/script.
        Works in dev and when running the compiled exe.
        """
        if getattr(sys, "frozen", False):
            # sys.executable is the path to the exe
            base_path = os.path.dirname(sys.executable)
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, relative_path)


    def filter_z_rings(self):
        min_xy_dist = self.facade_filtration_xy_dist
        max_z_dist = self.facade_filtration_z_dist
        allFacadePoints = []

        # Extract all of the tap locations from the ZLevel objects
        for i in range(len(self.ZLevels)):
            allFacadePoints.append(self.ZLevels[i].tap_coords)

        # Now loop over the ZLevels from top to bottom and filter ]taps that already have taps above
        for i in range(len(self.ZLevels)):

            # If i = 0 then there are no facade points above.
            if i == 0:
                facade_points_above = []
            else:
                facade_points_above+=allFacadePoints[i-1].tolist()


            filtered_current_ring_points = self.z_filter_current_ring(allFacadePoints[i],facade_points_above,min_xy_dist, max_z_dist)
            self.ZLevels[i].tap_coords = filtered_current_ring_points



    def z_filter_current_ring(self, currentRing,allFacadePoints, min_xy_dist, max_z_dist):
        
        all_facade_taps_array = np.array(allFacadePoints)
        if len(all_facade_taps_array) == 0:
            return currentRing

        else:
            tree = cKDTree(all_facade_taps_array[:, :2])
            # For each point in the ring, find all previous points within XY radius
            neighbors = tree.query_ball_point(currentRing[:, :2], r=min_xy_dist)

            # Check Z distances for all neighbors
            mask = np.ones(len(currentRing), dtype=bool)
            for idx, nbr_idxs in enumerate(neighbors):
                if nbr_idxs:
                    dzs = all_facade_taps_array[nbr_idxs, 2] - currentRing[idx, 2]
                    if np.any((dzs > 0) & (dzs <= max_z_dist)):
                        mask[idx] = False  # mark for removal

            filtered_downSampledRing = currentRing[mask]
            return filtered_downSampledRing



    def initialize_Roofs(self):

        roof_objects = []
        for i in range(len(self.roof_clusters)):
            roof_objects.append(Roof(self.roof_clusters[i], self.total_pcd,
                    self.roof_concave_hull_length_threshold, self.roof_concave_hull_concavity,
                    self.roof_corner_angle_threshold,
                    self.roof_vector_length_threshold, self.roof_vector_width_threshold, self.roof_major_cutoff_multiplier,
                    self.roof_horizontal_spacing, self.roof_anchor_distance, self.roof_double_tap_dist, self.roof_double_tap_list, self.roof_max_distance_from_model, self.roof_edge_maximum_filtering_distance,
                    self.roof_distance_to_concave_hull_filtration,
                    self.roof_center_upperdist, self.roof_center_mindist, self.roof_center_min_spacing_2,self.roof_center_max_iter, self.roof_polyline_neighbour_radius
                    ))
        return roof_objects


    def initialize_ZLevels(self):

        zlevel_objects = []

        if self.facade_double_tap_list is not None:
            if len(self.facade_double_tap_list) != len(self.facade_z_levels):
                self.facade_double_tap_list = [i%self.facade_double_tap_int == 0 for i in range(len(self.facade_z_levels))]

            for i in range(len(self.facade_z_levels)):
                new_z_level = ZLevel(self.facade_z_levels[i]-self.facade_zLevel_lower, self.total_pcd, 
                                             self.facade_epsilon, self.facade_concave_hull_length_threshold,self.facade_hull_concavity, self.facade_corner_angle_threshold,
                                             self.facade_nzFiltercutoff,self.facade_corner_angle_threshold,
                                             self.facade_vector_length_threshold, self.facade_vector_width_threshold, self.facade_major_cutoff_multiplier,
                                             self.facade_horizontal_spacing,self.facade_anchor_distance, self.facade_double_tap_dist, self.facade_double_tap_list[i])
                if new_z_level.tap_coords is not None and len(new_z_level.tap_coords) > 0:
                    zlevel_objects.append(new_z_level)


            
        else:
            self.facade_double_tap_list = [i%self.facade_double_tap_int == 0 for i in range(len(self.facade_z_levels))]
            for i in range(len(self.facade_z_levels)):
                new_z_level = ZLevel(self.facade_z_levels[i]-self.facade_zLevel_lower, self.total_pcd, 
                                             self.facade_epsilon, self.facade_concave_hull_length_threshold,self.facade_hull_concavity, self.facade_corner_angle_threshold,
                                             self.facade_nzFiltercutoff,self.facade_corner_angle_threshold, 
                                             self.facade_vector_length_threshold, self.facade_vector_width_threshold, self.facade_major_cutoff_multiplier,
                                             self.facade_horizontal_spacing,self.facade_anchor_distance, self.facade_double_tap_dist, self.facade_double_tap_list[i])
                if new_z_level.tap_coords is not None and len(new_z_level.tap_coords) > 0:
                    zlevel_objects.append(new_z_level)

        return zlevel_objects

    def align_and_sample_mesh_lattice(self):

        if os.path.isfile(self.mesh_file_path + f'mesh_pointcloud_spacing'+str(self.mesh_pcd_spacing).replace('.','d')+f'_multiplier{self.mesh_pcd_generation_multiplier}.pcd'):
            print('Mesh point cloud already saved, loading..')
            lattice_mesh_pcd = o3d.io.read_point_cloud(self.mesh_file_path+str(self.mesh_pcd_spacing).replace('.','d')+f'_multiplier{self.mesh_pcd_generation_multiplier}.pcd')

        else:
            print('Computing mesh point cloud')
            def triangle_area(a, b, c):
                return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

            mesh = self.mesh
            multiplier = self.mesh_pcd_generation_multiplier
            regularity = self.mesh_pcd_spacing

            # --- Clean mesh ---
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


            # Check if the dense mesh has already bewen computed ( this is the slowest step)
            if not os.path.isfile(self.mesh_folder + f'dense_mesh_pointcloud_spacing'+str(self.mesh_pcd_spacing).replace('.','d')+f'_multiplier{self.mesh_pcd_generation_multiplier}.pcd'):
                dense_mesh_pcd=mesh.sample_points_uniformly(int(multiplier*mesh.get_surface_area())) # make a dense, but roughly sampled PCD
                self.dense_mesh_pcd = dense_mesh_pcd
                # The file is huge -- for now let's leave it out of the testing.
                #o3d.io.write_point_cloud(self.mesh_folder + f'dense_mesh_pointcloud_spacing'+str(self.mesh_pcd_spacing).replace('.','d')+f'_multiplier{self.mesh_pcd_generation_multiplier}.pcd', dense_mesh_pcd)
            # If it has, load it in. Save it as an attribute because it will be useful later for checking proximity to mesh.
            else:
                dense_mesh_pcd = o3d.io.read_point_cloud(self.mesh_folder + f'dense_mesh_pointcloud_spacing'+str(self.mesh_pcd_spacing).replace('.','d')+f'_multiplier{self.mesh_pcd_generation_multiplier}.pcd')
                self.dense_mesh_pcd = dense_mesh_pcd
            lattice_mesh_pcd = dense_mesh_pcd.voxel_down_sample(voxel_size=regularity) #use that rough PCD to make regular lattice PCD with vocel downsampling

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

            # Save the file in the test folder
            o3d.io.write_point_cloud(self.mesh_folder + f'mesh_pointcloud_spacing'+str(self.mesh_pcd_spacing).replace('.','d')+f'_multiplier{self.mesh_pcd_generation_multiplier}.pcd', lattice_mesh_pcd)

        return lattice_mesh_pcd
    


    def align_edge_points_to_lattice_excluding_endpoints(self):
        """
        Add edge points that align with existing lattice points.
        
        Args:
            sharp_edges: List of [vertex_idx1, vertex_idx2] pairs (edge indices into mesh)
            mesh_vertices: Nx3 numpy array of mesh vertex positions
            existing_points: Mx3 numpy array of existing lattice points
            desired_spacing: Desired spacing between points along edges
        
        Returns:
            edge_points: Numpy array of new edge points aligned to lattice
        """

        if os.path.isfile(self.mesh_folder + f'mesh_edge_pointcloud_spacing'+str(self.mesh_pcd_spacing).replace('.','d')+f'_multiplier{self.mesh_pcd_generation_multiplier}.pcd'):
            print('Mesh edge point cloud already saved, loading..')
            aligned_edge_pcd = o3d.io.read_point_cloud(self.mesh_folder + f'mesh_edge_pointcloud_spacing'+str(self.mesh_pcd_spacing).replace('.','d')+f'_multiplier{self.mesh_pcd_generation_multiplier}.pcd')

        else:
            print('Computing mesh edge point cloud')

            desired_spacing = self.mesh_pcd_spacing
            existing_points = np.asarray(self.mesh_pcd.points)
            mesh_vertices = self.mesh.vertices
            mesh = self.mesh
            mesh.compute_triangle_normals()
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_non_manifold_edges()


            def get_sharp_edges(mesh,angle_threshold_deg):
                triangle_normals = np.asarray(mesh.triangle_normals)
                triangles = np.asarray(mesh.triangles)
                edges = dict()

                for i, tri in enumerate(triangles):
                    for j in range(3):
                        edge = tuple(sorted((tri[j], tri[(j + 1) % 3])))
                        if edge in edges:
                            prev_idx = edges[edge]
                            n1 = triangle_normals[prev_idx]
                            n2 = triangle_normals[i]
                            angle = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)) * 180.0 / np.pi
                            if angle > angle_threshold_deg:
                                edges[edge] = -1  # keep edge, mark as sharp
                            else:
                                edges[edge] = None  # smooth, discard
                        else:
                            edges[edge] = i  # store first triangle index

                # Keep only sharp edges
                sharp_edges = [edge for edge, val in edges.items() if val == -1]
                return sharp_edges
            
            sharp_edges = get_sharp_edges(mesh,self.mesh_edge_pcd_angle_threshold)
            # Build KDTree for fast nearest neighbor queries
            tree = cKDTree(existing_points)
            
            all_edge_points = []
            
            for edge_idx1, edge_idx2 in sharp_edges:
                edge_start = mesh_vertices[edge_idx1]
                edge_end = mesh_vertices[edge_idx2]
                edge_vector = edge_end - edge_start
                edge_length = np.linalg.norm(edge_vector)
                edge_direction = edge_vector / edge_length
                
                # Always include endpoints (these never move)
                edge_points = [edge_start.copy(), edge_end.copy()]
                
                # Calculate number of intermediate points needed
                num_intermediate = max(0, int(edge_length / desired_spacing) - 1)
                
                if num_intermediate > 0:
                    # Space out points evenly along the edge
                    t_values = np.linspace(0, 1, num_intermediate + 2)[1:-1]  # Exclude endpoints
                    
                    for t in t_values:
                        # Initial position along edge
                        initial_point = edge_start + t * edge_vector
                        
                        # Find nearest existing lattice point
                        _, nearest_idx = tree.query(initial_point)
                        nearest_point = existing_points[nearest_idx]
                        
                        # Project the nearest lattice point onto the edge line to find 
                        # the point on the edge that is closest to the lattice point
                        to_nearest = nearest_point - edge_start
                        edge_param = np.dot(to_nearest, edge_direction)
                        
                        # Clamp to edge bounds and find the actual point on the edge segment
                        edge_param = np.clip(edge_param, 0, edge_length)
                        adjusted_point = edge_start + edge_param * edge_direction
                        
                        edge_points.append(adjusted_point)
                
                # Sort points along edge direction to maintain order, but keep endpoints fixed
                if len(edge_points) > 2:
                    # Separate endpoints from intermediate points
                    endpoints = [edge_points[0], edge_points[1]]  # Original start and end
                    intermediate_points = edge_points[2:]
                    
                    # Sort intermediate points along the edge
                    edge_params = []
                    for point in intermediate_points:
                        t = np.dot(point - edge_start, edge_direction) / edge_length
                        edge_params.append(t)
                    
                    sorted_indices = np.argsort(edge_params)
                    sorted_intermediate = [intermediate_points[i] for i in sorted_indices]
                    
                    # Reconstruct: start, sorted intermediates, end
                    edge_points = [endpoints[0]] + sorted_intermediate + [endpoints[1]]
                
                all_edge_points.extend(edge_points)

            aligned_edge_pcd = o3d.geometry.PointCloud()
            aligned_edge_pcd.points = o3d.utility.Vector3dVector(all_edge_points)
            o3d.io.write_point_cloud(self.mesh_folder + f'mesh_edge_pointcloud_spacing'+str(self.mesh_pcd_spacing).replace('.','d')+f'_multiplier{self.mesh_pcd_generation_multiplier}.pcd',aligned_edge_pcd)
        return aligned_edge_pcd
    
    def get_total_pcd(self):

        if os.path.isfile(self.mesh_folder + f'total_pcd_spacing'+str(self.mesh_pcd_spacing).replace('.','d')+f'_multiplier{self.mesh_pcd_generation_multiplier}.pcd'):
            total_pcd = o3d.io.read_point_cloud(self.mesh_folder + f'total_pcd_spacing'+str(self.mesh_pcd_spacing).replace('.','d')+f'_multiplier{self.mesh_pcd_generation_multiplier}.pcd')
        else:
            def remove_close_points_with_normals(E, M, M_normals, threshold):
                """
                Removes points in M (and their normals) that are within `threshold` distance of any point in E.
                
                Args:
                    E (np.ndarray): Nx3 array of reference points.
                    M (np.ndarray): Mx3 array of candidate points.
                    M_normals (np.ndarray): Mx3 array of normals corresponding to M.
                    threshold (float): Distance threshold.
                
                Returns:
                    tuple: (filtered_M, filtered_M_normals)
                """
                tree_E = cKDTree(E)
                distances, _ = tree_E.query(M, distance_upper_bound=threshold)
                keep_mask = distances > threshold
                return M[keep_mask], M_normals[keep_mask]  

            # Check for the mesh edge file and compute if it doesn't exist, or if it is forced to recompute.
            if self.mesh_pcd is None:
                self.mesh_pcd = self.align_and_sample_mesh_lattice()
            if self.mesh_edge_pcd is None:
                self.mesh_edge_pcd = self.align_edge_points_to_lattice_excluding_endpoints()

            # We need the trimesh mesh for this
            try:
                self.mesh_trimesh = trimesh.load(os.path.join(self.mesh_folder,"mesh_scaled.stl"))
                self.mesh_trimesh.fix_normals()
            except Exception as e:
                    raise RuntimeError(f"Could not load the scaled mesh as trimesh, run Building.get_scaled_mesh_and_taps: {e}") from e

            mesh_tm = self.mesh_trimesh
            mesh_pcd = self.mesh_pcd
            edge_pcd = self.mesh_edge_pcd

            points_main = np.asarray(mesh_pcd.points)
            closest_points, distances, face_indices = mesh_tm.nearest.on_surface(points_main)

            # Get face normals of those triangles
            face_normals = mesh_tm.face_normals[face_indices]

            # Assign these normals back to your Open3D point cloud
            mesh_pcd.normals = o3d.utility.Vector3dVector(face_normals)
            normals_main = np.asarray(mesh_pcd.normals)
            edge_points = np.asarray(edge_pcd.points)


            # Compute edge normals as the average of the nearest 2 face normals
            pcd_tree = o3d.geometry.KDTreeFlann(mesh_pcd)

            normals_out = []
            k = 2  # number of neighbors to average

            for pt in edge_points:
                [_, idx, _] = pcd_tree.search_knn_vector_3d(pt, k)
                neighbor_normals = normals_main[idx]
                avg_normal = np.mean(neighbor_normals, axis=0)
                norm = np.linalg.norm(avg_normal)
                if norm > 0:
                    avg_normal /= norm
                else:
                    avg_normal = np.array([0, 0, 1])  # fallback
                normals_out.append(avg_normal)

            edge_pcd.normals = o3d.utility.Vector3dVector(np.array(normals_out))
            
            filtered_mesh_points,filtered_mesh_normals = remove_close_points_with_normals(np.asarray(edge_pcd.points), np.asarray(mesh_pcd.points),np.asarray(mesh_pcd.normals), threshold=.01 + (self.mesh_pcd_spacing/2.))

            total_pcd = o3d.geometry.PointCloud()

            total_pcd.points = o3d.utility.Vector3dVector(np.vstack([filtered_mesh_points,np.asarray(edge_pcd.points)]))
            total_pcd.normals = o3d.utility.Vector3dVector(np.vstack([filtered_mesh_normals,edge_pcd.normals]))

            o3d.io.write_point_cloud(self.mesh_folder + f'total_pcd_spacing'+str(self.mesh_pcd_spacing).replace('.','d')+f'_multiplier{self.mesh_pcd_generation_multiplier}.pcd', total_pcd)

        return total_pcd


# Now add in the detection for rooflines
    def identify_roof_points_by_projection(self,total_pcd,xy_resolution,min_roof_points,z_cluster_eps):

        """
        Identifies horizontal rooflines from a mesh or point cloud by projecting to XY and clustering Z values.

        Returns:
            List of sorted roofline Z-values (highest to lowest).
        """

        points = np.asarray(total_pcd.points)
        df = pd.DataFrame(points, columns=["x", "y", "z"])

        # Step 2: Bin by (x, y) to get topmost Z per cell
        df["xy_bin"] = list(zip(
            (df.x / xy_resolution).round().astype(int),
            (df.y / xy_resolution).round().astype(int)
        ))
        top_df = df.loc[df.groupby("xy_bin")["z"].idxmax()]  # top Z per XY bin

        # Step 3: Cluster Z values to identify horizontal layers
        z_vals = top_df["z"].values.reshape(-1, 1)
        db = DBSCAN(eps=z_cluster_eps, min_samples=min_roof_points).fit(z_vals)
        top_df["z_cluster"] = db.labels_

        return top_df

    def get_concave_hull(self,z_level,total_pcd,epsilon,concavity,length_threshold):

        all_points = np.asarray(total_pcd.points)

        # Step 1: Filter points near z_level
        mask = (all_points[:, 2] >= z_level - epsilon) & (all_points[:, 2] <= z_level + epsilon)
        slice_points = all_points[mask]

        # Step 2: Compute concave hull on 2D projection
        idxes_local = concave_hull_indexes(slice_points[:, :2], length_threshold=length_threshold,concavity =concavity)
        points = slice_points[idxes_local]

        #Get the indices of those points in the original mesh_pcd
        mesh_pcd_indices = np.flatnonzero(mask)[idxes_local]

        return points, mesh_pcd_indices

    def check_any_close(self,roof_points, wire, threshold = 0.5):
        tree_wire = cKDTree(wire)

        # Query for distances from roof point to wire points
        distances, _ = tree_wire.query(roof_points, k=1)

        # Check if all distances are within threshold
        any_close = np.any(distances <= threshold)
        
        return any_close
        
    def identify_roofline_zs(self,roof_zs,top_df,total_pcd,spacing,epsilon, concavity, length_threshold):
                            
        roofline_zs=[]
        for i in range(len(roof_zs)):
            z_level = roof_zs[i]
            cluster_num = top_df.query(f"z < {roof_zs[i]+(spacing + 0.1)} and z > {roof_zs[i] - (spacing + 0.1)}").z_cluster.mode().values[0]

            roof_points = top_df.query(f'z_cluster == {cluster_num}')[['x','y','z']].values
            wire =  self.get_concave_hull(z_level,total_pcd,epsilon,concavity,length_threshold)[0]
            any_close = self.check_any_close(roof_points,wire,threshold = 0.5)
            if any_close == True:
                roofline_zs.append(roof_zs[i])

        return roofline_zs    

    def compute_and_identify_roofs_and_rooflines(self):
        
        total_pcd = self.total_pcd
        min_roof_points = self.facade_min_roof_points
        z_cluster_eps = (self.mesh_pcd_spacing/2) +.01
        xy_resolution = self.mesh_pcd_spacing
        spacing = self.mesh_pcd_spacing
        epsilon = self.facade_epsilon
        length_threshold = self.facade_concave_hull_length_threshold
        concavity = self.facade_hull_concavity


        #Identify roof points by using a top down projection
        top_df = self.identify_roof_points_by_projection(total_pcd, xy_resolution=xy_resolution, min_roof_points=min_roof_points, z_cluster_eps=z_cluster_eps)
        
        # Get the z-values of each identified roof
        roof_zs = (top_df[top_df["z_cluster"] != -1].groupby("z_cluster")["z"].mean().tolist())

        # Filter out the roofs that are not rooflines
        roofline_zs = self.identify_roofline_zs(roof_zs,top_df,total_pcd,spacing,epsilon,concavity,length_threshold)
        
        return roofline_zs
    
    def fill_z_gaps(self):
        
        z_values = self.roof_lines
        threshold = self.facade_zLevel_gap_threshold


        filled = []
        last_z = None

        for z in sorted(z_values, reverse=True):
            if last_z is not None and (last_z - z) > threshold:
                new_z = last_z - threshold
                while new_z > z:
                    filled.append(new_z)
                    new_z -= threshold
            filled.append(z)
            last_z = z

        # Extend downward using the same threshold until just above 0
        if last_z is not None:
            new_z = last_z - threshold
            while new_z >= 0:
                filled.append(new_z)
                new_z -= threshold
        return sorted(filled, reverse=True)



    def are_colinear(self, points, tol=1e-9):
        points = np.array(points)
        if len(points) < 3:
            return True  # 2 points are always colinear

        # Take direction vector from first two points
        v = points[1] - points[0]

        for i in range(2, len(points)):
            w = points[i] - points[0]
            if np.linalg.norm(np.cross(v, w)) > tol:  # non-zero cross product → not colinear
                return False
        return True
    def is_cluster_dense(self, points, radius=1.8, min_neighbors=15, min_dense_points=10):
        """Check if a cluster is dense enough."""
        if len(points) == 0:
            return False
        tree = SklearnKDTree(points)
        counts = tree.query_radius(points, r=radius, count_only=True)
        dense_points = np.sum(counts >= min_neighbors)
        return dense_points >= min_dense_points

    def filter_interior_points(self, points_2d):
        """Returns a boolean mask for points inside the convex hull."""
        if len(points_2d) < 4:
            return np.ones(len(points_2d), dtype=bool)
        try:
            hull = ConvexHull(points_2d, qhull_options='QJ')
            polygon = Polygon(points_2d[hull.vertices])
            interior_mask = np.array([polygon.contains(Point(p)) for p in points_2d])
            return interior_mask
        except Exception:
            return np.ones(len(points_2d), dtype=bool)

    def get_roof_clusters_simple(self, total_pcd,
                        nz_cutoff=0.2,
                        z_eps=0.1,
                        z_min_samples=5,
                        xy_eps=0.8,
                        xy_min_samples=5,
                        dense_point_radius=1.5,
                        dense_min_neighbors=20,
                        min_dense_points=60):
        """
        Returns: list of lists of indices into total_pcd.points for each dense roof sub-cluster.
        """

        total_points = np.asarray(total_pcd.points)

        # Step 1: Select points with upward-facing normals
        upward_normal_mask = np.asarray(total_pcd.normals)[:, 2] > nz_cutoff
        upwards_points = total_points[upward_normal_mask]

        # Step 2: Cluster by Z
        z_db = DBSCAN(eps=z_eps, min_samples=z_min_samples).fit(upwards_points[:, 2].reshape(-1, 1))
        z_clusters = z_db.labels_
        unique_z_clusters = np.unique(z_clusters)

        roof_clusters_indices = []

        # Step 3: Inside each Z cluster, cluster by XY and check density
        for z_cluster_id in unique_z_clusters:
            mask_z = (z_clusters == z_cluster_id)
            if z_cluster_id == -1 or np.sum(mask_z) == 0:
                continue

            points_in_z = upwards_points[mask_z]
            xy = points_in_z[:, :2]

            xy_db = DBSCAN(eps=xy_eps, min_samples=xy_min_samples).fit(xy)
            xy_labels = xy_db.labels_

            unique_xy = np.unique(xy_labels)

            for xy_sub_id in unique_xy:
                if xy_sub_id == -1:
                    continue

                mask_xy = (xy_labels == xy_sub_id)
                idxs_upwards = np.where(mask_z)[0][mask_xy]  # indices within upwards_points

                cluster_points = upwards_points[idxs_upwards]
                if self.is_cluster_dense(cluster_points,
                                    radius=dense_point_radius,
                                    min_neighbors=dense_min_neighbors,
                                    min_dense_points=min_dense_points):
                    # Map upwards_points indices back to total_pcd indices
                    idxs_total = np.where(upward_normal_mask)[0][idxs_upwards]
                    roof_clusters_indices.append(list(idxs_total))

        return roof_clusters_indices
    def get_underroof_filtered_roof_indices(self, total_pcd,nz_cutoff,
                            z_eps=0.3, z_min_samples=5,
                            xy_eps=0.76, xy_min_samples=5,
                            density_radius=1.5, density_min_neighbors=20, density_min_dense_points=70,
                            roof_overlap_thresh=0.5):

        #def filter_interior_points(points_2d):
        #    if len(points_2d) < 4:
        #        return np.ones(len(points_2d), dtype=bool)
        #    hull = ConvexHull(points_2d)
        #    polygon = Polygon(points_2d[hull.vertices])
        #    return np.array([polygon.contains(Point(p)) for p in points_2d])

        def is_cluster_dense(self, points, radius, min_neighbors, min_dense_points):
            if len(points) == 0:
                return False
            tree = SklearnKDTree(points)
            counts = tree.query_radius(points, r=radius, count_only=True)
            dense_points = np.sum(counts >= min_neighbors)
            return dense_points >= min_dense_points

        upward_normal_mask = np.asarray(total_pcd.normals)[:, 2] > nz_cutoff
        upwards_points = np.asarray(total_pcd.points)[upward_normal_mask]

        # Cluster by Z
        z_db = DBSCAN(eps=z_eps, min_samples=z_min_samples).fit(upwards_points[:, 2].reshape(-1, 1))
        z_clusters = z_db.labels_
        unique_z_clusters = np.unique(z_clusters)

        dense_subclusters = set()
        for z_cluster_id in unique_z_clusters:
            mask_z = (z_clusters == z_cluster_id)
            if z_cluster_id == -1 or np.sum(mask_z) == 0:
                continue

            points_in_z = upwards_points[mask_z]
            xy_db = DBSCAN(eps=xy_eps, min_samples=xy_min_samples).fit(points_in_z[:, :2])
            xy_labels = xy_db.labels_
            unique_xy = np.unique(xy_labels)

            for xy_sub_id in unique_xy:
                mask_xy = (xy_labels == xy_sub_id)
                if xy_sub_id == -1:
                    continue

                idxs = np.where(mask_z)[0][mask_xy]
                cluster_points = upwards_points[idxs]
                if self.is_cluster_dense(cluster_points, radius=density_radius,
                                    min_neighbors=density_min_neighbors,
                                    min_dense_points=density_min_dense_points):
                    dense_subclusters.add((z_cluster_id, xy_sub_id))

        # Build dict: (z_cluster, xy_subcluster) -> points
        roof_clusters = {}
        for (z_id, xy_id) in dense_subclusters:
            mask_z = (z_clusters == z_id)
            points_in_z = upwards_points[mask_z]
            xy_db = DBSCAN(eps=xy_eps, min_samples=xy_min_samples).fit(points_in_z[:, :2])
            mask_xy = (xy_db.labels_ == xy_id)
            roof_points = points_in_z[mask_xy]
            if len(roof_points) > 0:
                roof_clusters[(z_id, xy_id)] = roof_points

        # Sort roofs by mean Z height
        sorted_roofs = sorted(roof_clusters.items(), key=lambda x: np.mean(x[1][:, 2]))

        keep_mask = np.ones(len(upwards_points), dtype=bool)

        for i, ((z1, xy1), points1) in enumerate(sorted_roofs):
            mask_z1 = (z_clusters == z1)
            points_in_z1 = upwards_points[mask_z1]
            xy_db1 = DBSCAN(eps=xy_eps, min_samples=xy_min_samples).fit(points_in_z1[:, :2])
            mask_xy1 = (xy_db1.labels_ == xy1)
            idxs1 = np.where(mask_z1)[0][mask_xy1]

            interior_mask1 = self.filter_interior_points(points1[:, :2])
            interior_points1 = points1[interior_mask1]
            if len(interior_points1) == 0:
                continue
            tree1 = SklearnKDTree(interior_points1[:, :2])
            z1_mean = np.mean(points1[:, 2])

            for (z2, xy2), points2 in sorted_roofs[i + 1:]:
                z2_mean = np.mean(points2[:, 2])
                if z2_mean <= z1_mean:
                    continue

                interior_mask2 = self.filter_interior_points(points2[:, :2])
                interior_points2 = points2[interior_mask2]
                if len(interior_points2) == 0:
                    continue

                distances, indices = tree1.query(interior_points2[:, :2], k=1)
                overlapped_points_in_roof1 = indices[distances.flatten() < roof_overlap_thresh]

                idxs1_interior = idxs1[interior_mask1]
                keep_mask[idxs1_interior[overlapped_points_in_roof1]] = False

        # Map point idx to cluster
        point_to_cluster = {}
        for (z_id, xy_id), points in roof_clusters.items():
            mask_z = (z_clusters == z_id)
            points_in_z = upwards_points[mask_z]
            xy_db = DBSCAN(eps=xy_eps, min_samples=xy_min_samples).fit(points_in_z[:, :2])
            mask_xy = (xy_db.labels_ == xy_id)
            idxs = np.where(mask_z)[0][mask_xy]
            for idx in idxs:
                point_to_cluster[idx] = (z_id, xy_id)

        cluster_point_idxs = {}
        for idx, cluster_key in point_to_cluster.items():
            cluster_point_idxs.setdefault(cluster_key, []).append(idx)

        # Final filtering & restoration
        for cluster_key, idxs in cluster_point_idxs.items():
            idxs = np.array(idxs)
            kept_idxs = idxs[keep_mask[idxs]]

            if len(kept_idxs) == 0:
                keep_mask[idxs] = False
                continue

            cluster_kept_points = upwards_points[kept_idxs][:, :2]
            if self.is_cluster_dense(cluster_kept_points,
                                radius=density_radius,
                                min_neighbors=density_min_neighbors,
                                min_dense_points=density_min_dense_points):
                keep_mask[idxs] = True
            else:
                keep_mask[idxs] = False

        # Return list of kept clusters with indices referencing total_pcd.points
        # But upwards_points is a subset of total_pcd.points via upward_normal_mask
        kept_clusters_indices = []
        upwards_indices = np.where(upward_normal_mask)[0]

        for cluster_key, idxs in cluster_point_idxs.items():
            idxs = np.array(idxs)
            if np.any(keep_mask[idxs]):
                kept_clusters_indices.append(list(upwards_indices[idxs]))

        return kept_clusters_indices

    def get_roof_clusters(self):
        
        # Parameters for this function
        total_pcd = self.total_pcd
        nz_cutoff = self.roof_nz_cutoff
        z_eps = self.roof_z_eps
        z_min_samples = self.roof_z_min_samples
        xy_eps = self.roof_xy_eps
        xy_min_samples = self.roof_xy_min_samples
        dense_point_radius = self.roof_dense_point_radius
        dense_min_neighbors = self.roof_dense_min_neighbors
        min_dense_points = self.roof_min_dense_points
        roof_overlap_thresh = self.roof_overlap_thresh
        
        if roof_overlap_thresh == None:
            # Run the simple roof cluster function ( IE do not remove under-rooves )
            return self.get_roof_clusters_simple(total_pcd,nz_cutoff,z_eps,z_min_samples,xy_eps,xy_min_samples,dense_point_radius,dense_min_neighbors,min_dense_points)
        else:
            return self.get_underroof_filtered_roof_indices(total_pcd,nz_cutoff,z_eps,z_min_samples,xy_eps,xy_min_samples,dense_point_radius,dense_min_neighbors,min_dense_points,roof_overlap_thresh)



















































class ZLevel:

    def __init__(self, z_level, total_pcd, concave_hull_z_eps, concave_hull_length_threshold,concave_hull_concavity, sharp_angle_threshold,
                  nzFiltercutoff,corner_angle_threshold, vector_length_threshold, vector_width_threshold, major_cutoff_multiplier,
                  horizontal_spacing,corner_dist, double_tap_dist, double_tap):

        # Import arguments to the class
        self.z_level = z_level
        self.total_pcd = total_pcd

        # Concave Hull Params
        self.concave_hull_z_eps = concave_hull_z_eps
        self.concave_hull_length_threshold = concave_hull_length_threshold
        self.concave_hull_concavity = concave_hull_concavity

        # Corner Detection Params
        self.sharp_angle_threshold = sharp_angle_threshold
        self.nzFiltercutoff = nzFiltercutoff
        self.corner_angle_threshold = corner_angle_threshold
        self.vector_length_threshold = vector_length_threshold
        self.vector_width_threshold = vector_width_threshold
        self.major_cutoff_multiplier = major_cutoff_multiplier

        self.horizontal_spacing=horizontal_spacing
        self.corner_dist=corner_dist
        self.double_tap_dist=double_tap_dist
        self.double_tap=double_tap

        # Set the attributes that will be computed later
        self.concave_hull = None
        self.sharp_corners = None
        self.corner_indices=None
        self.corner_angles=None
        self.all_corners = None
        self.major_corners = None
        self.major_directions_1 = None
        self.major_directions_2 = None
        self.minor_corners = None
        self.minor_directions_1 = None
        self.minor_directions_2 = None
        self.major_binary = None
        self.Corners = None
        self.tap_coords = None
        self.shown = None
        self.taps= None

        # Get the concave hull
        self.concave_hull = self.get_concave_hull()
        self.get_wire()
        if len(self.concave_hull[0]) >0:

            # Get all Corners and classify them as major or minor
            self.all_corners = self.get_major_corners_and_directions()
            self.major_corners = self.all_corners[0]
            self.major_directions_1 = self.all_corners[1]
            self.major_directions_2 = self.all_corners[2]
            self.minor_corners = self.all_corners[3]
            self.minor_directions_1 = self.all_corners[4]
            self.minor_directions_2 = self.all_corners[5]
            self.major_binary = self.all_corners[6]
            self.all_direction_1s = self.all_corners[7]
            self.all_direction_2s = self.all_corners[8]
            self.sharp_corners = self.all_corners[9]

            # Next we instantiate corner objects which can be either major or minor
            self.Corners = self.initialize_corners()

            if len(self.major_corners) > 0:
                self.tap_coords= self.downsample_by_spacing_with_major_corner_anchors()
            else:
                self.tap_coords= self.downsample_by_spacing_without_major_corners()
            if self.concave_hull is None:
                self.tap_coords = []



    def downsample_by_spacing_without_major_corners(self):
        # No major corners: just evenly space along inset curve
        concave_points_3d = self.concave_hull[0]
        n = len(concave_points_3d)
        closed_points = np.vstack([concave_points_3d, concave_points_3d[0]])
        diffs = np.diff(closed_points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cumdist = np.concatenate(([0], np.cumsum(dists)))
        total_length = cumdist[-1]

        fx = interp1d(cumdist, closed_points[:,0], kind='linear')
        fy = interp1d(cumdist, closed_points[:,1], kind='linear')
        fz = interp1d(cumdist, closed_points[:,2], kind='linear')

        final_dists = np.arange(0, total_length, self.horizontal_spacing)
        downSampledRing = np.vstack([fx(final_dists), fy(final_dists), fz(final_dists)]).T

        return downSampledRing
    
    def downsample_by_spacing_with_major_corner_anchors(self):
        """
        points: (M, 3) ring of 3D points (closed loop)
        spacing: max allowed spacing between any two output points
        major_corners: (N, 3) corner point locations
        corner_dirs1: (N, 2) direction vectors (x, y) for one side of corner
        corner_dirs2: (N, 2) direction vectors (x, y) for other side of corner
        dist: offset from corner to place anchor points
        """

        self.TESTANCHORPOINTS=[]

        self.major_binary = self.get_major_binary_from_corners()
        points = self.concave_hull[0]
        major_corners = self.sharp_corners[self.major_binary == np.True_]
        
        corner_dirs1 = -self.all_direction_1s[self.major_binary == np.True_]
        corner_dirs2 = -self.all_direction_2s[self.major_binary == np.True_]
        spacing = self.horizontal_spacing
        dist = self.corner_dist
        
        double_tap = self.double_tap
        double_tap_dist = self.double_tap_dist

        # Step 1: Close the ring
        closed_points = np.vstack([points, points[0]])
        diffs = np.diff(closed_points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cumdist = np.concatenate(([0], np.cumsum(dists)))
        total_length = cumdist[-1]

        # Step 2: Interpolation
        fx = interp1d(cumdist, closed_points[:, 0], kind='linear')
        fy = interp1d(cumdist, closed_points[:, 1], kind='linear')
        fz = interp1d(cumdist, closed_points[:, 2], kind='linear')

        # Step 3: Compute 3D anchor points from 2D directions
        anchor_points = []

        for corner, d1, d2 in zip(major_corners, corner_dirs1, corner_dirs2):
            for d in [d1, d2]:
                d3 = np.array([d[0], d[1], 0.0])
                d3 /= np.linalg.norm(d3)
                anchor_points.append(corner + dist * d3)
                self.TESTANCHORPOINTS.append(corner + dist * d3)
                #print('corner',corner,'d3 norm',np.sqrt(sum(d3**2)), '      point',corner + dist * d3)

                if double_tap:
                    anchor_points.append(corner + (dist * d3) + (double_tap_dist * d3))
        anchor_points = np.array(anchor_points)
        
        # Step 4: Project anchors onto the curve
        fine_dists = np.linspace(0, total_length, 5000)
        fine_points = np.vstack([fx(fine_dists), fy(fine_dists), fz(fine_dists)]).T

        anchor_dists = []
        for a in anchor_points:
            idx = np.argmin(np.linalg.norm(fine_points - a, axis=1))
            anchor_dists.append(fine_dists[idx])
        anchor_dists = np.sort(np.array(anchor_dists))

        # Step 5: Fill in evenly between anchors
        final_dists = []
        for i in range(len(anchor_dists)):
            d1 = anchor_dists[i]
            d2 = anchor_dists[(i + 1) % len(anchor_dists)]
            if d2 < d1:
                d2 += total_length
            seg_len = d2 - d1
            n = int(np.ceil(seg_len / spacing))
            interp = np.linspace(d1, d2, n, endpoint=False)
            final_dists.extend(interp % total_length)

        final_dists = np.unique(np.array(final_dists))
        final_points = np.vstack([fx(final_dists), fy(final_dists), fz(final_dists)]).T
        

        return final_points
    


    def get_major_binary_from_corners(self):
        
        major_binary = []
        for i in range(len(self.Corners)):
            major_binary.append(self.Corners[i].major)

        return major_binary


    def initialize_corners(self):

        corners = []
        for i in range(len(self.sharp_corners)):
            corners.append(Corner(self.sharp_corners[i], self.major_binary[i], self.all_direction_1s[i], self.all_direction_2s[i]))
        return corners

    def get_concave_hull(self):

        # Import parameters from class
        total_pcd = self.total_pcd
        z_level = self.z_level
        epsilon = self.concave_hull_z_eps
        length_threshold = self.concave_hull_length_threshold
        concavity = self.concave_hull_concavity

        all_points = np.asarray(total_pcd.points)

        # Step 1: Filter points near z_level
        mask = (all_points[:, 2] >= z_level - epsilon) & (all_points[:, 2] <= z_level + epsilon)
        slice_points = all_points[mask]

        # Step 2: Compute concave hull on 2D projection
        idxes_local = concave_hull_indexes(slice_points[:, :2], length_threshold=length_threshold,concavity =concavity)
        points = slice_points[idxes_local]

        #Get the indices of those points in the original mesh_pcd
        mesh_pcd_indices = np.flatnonzero(mask)[idxes_local]

        return points, mesh_pcd_indices

    def detect_sharp_corners(self,hull_points,angle_threshold):
        """
        Detect sharp corners in a hull by analyzing angle changes.
        
        Args:
            hull_points: Array of (x,y) or (x,y,z) points forming the hull
            angle_threshold: Minimum angle change (degrees) to consider a sharp corner
        
        Returns:
            corner_indices: Indices of points that are sharp corners
            angles: Angle changes at each point (in degrees)
        """

        hull_points = self.concave_hull[0]
        angle_threshold = self.sharp_angle_threshold


        points_2d = hull_points[:, :2]  # Use only x,y coordinates
        n = len(points_2d)
        angles = []
        
        for i in range(n):
            # Get three consecutive points (wrapping around)
            p1 = points_2d[(i-1) % n]
            p2 = points_2d[i]
            p3 = points_2d[(i+1) % n]
            
            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
            angle = np.arccos(cos_angle)
            angle_deg = np.degrees(angle)
            
            # Convert to exterior angle (180 - interior angle)
            exterior_angle = 180 - angle_deg
            angles.append(abs(exterior_angle))
        
        # Find corners where angle change exceeds threshold
        corner_indices = np.where(np.array(angles) > angle_threshold)[0]
        
        return corner_indices, angles
    
    def get_corner_vectors(self, points, corner_indices):
        """
        For each corner index B, compute direction vectors AB and CB where:
            A is the previous point in the ring
            C is the next point in the ring

        Args:
            points (np.ndarray): Nx3 array of 3D points forming a ring.
            corner_indices (list or np.ndarray): Indices of corner points (B).

        Returns:
            List of tuples: [(AB, CB), ...] for each corner index.
        """
        N = len(points)
        vectors = []

        for idx in corner_indices:
            A = points[(idx - 1) % N,:2]
            B = points[idx,:2]
            C = points[(idx + 1) % N,:2]

            AB = self.normalize_vector(B - A)
            CB = self.normalize_vector(B - C)  # Note: CB = B - C (vector from C to B)
            vectors.append((AB, CB))

        return vectors
    def get_wrapped_indices(self, idx, N, length):
        """
        Returns indices [idx-N, ..., idx-1, idx, idx+1, ..., idx+N] with wrapping.
        """
        return [(idx + i) % length for i in range(-N, N+1)]
    
    def centroid_direction(self, points, n=4):
        start = np.mean(points[:n], axis=0)
        end = np.mean(points[-n:], axis=0)
        vec = end - start
        return vec / np.linalg.norm(vec)
    
    def get_avg_corner_vectors(self, points, corner_indices,Navg=4):
        """
        For each corner index B, compute direction vectors AB and CB where:
            A is the previous point in the ring
            C is the next point in the ring

        Args:
            points (np.ndarray): Nx3 array of 3D points forming a ring.
            corner_indices (list or np.ndarray): Indices of corner points (B).

        Returns:
            List of tuples: [(AB, CB), ...] for each corner index.
        """
        N = len(points)

        vectors = []

        for idx in corner_indices:
            A = points[(idx - 1) % N,:2]
            B = points[idx,:2]
            C = points[(idx + 1) % N,:2]
            wrapped_indices = self.get_wrapped_indices(idx, Navg, N)
            pre_wrapped_indices = wrapped_indices[:round(len(wrapped_indices)/2)]
            post_wrapped_indices= wrapped_indices[int(np.ceil(len(wrapped_indices)/2)):]

            incoming_points = points[pre_wrapped_indices][:,:2]
            outgoing_points = points[post_wrapped_indices][:,:2]

            incoming_centroid_direction=self.normalize_vector(self.centroid_direction(incoming_points,int(Navg/2)))
            outgoing_centroid_direction=self.normalize_vector(self.centroid_direction(outgoing_points[::-1],int(Navg/2)))

            A = points[wrapped_indices[Navg-2]][:2]
            B = points[wrapped_indices[Navg-1]][:2]
            C = points[wrapped_indices[Navg+1]][:2]
            D = points[wrapped_indices[Navg+2]][:2]

            AB = self.normalize_vector(B - A)
            DC = self.normalize_vector(C - D)

        # AB = normalize_vector(B - A)
            #CB = normalize_vector(B - C)  # Note: CB = B - C (vector from C to B)
            vectors.append((AB, DC))

        return vectors

    def get_major_cutoff(self, concaveRing,vector_length_threshold): 
        # We want to set a threshold for the number of points contained within a vector to be considered major
        # Get the mean distance to the nearest point in this concave hull
        distances = cKDTree(concaveRing[0][:,:2]).query(concaveRing[0][:,:2], k=2)[0][:,1]
        avg_nearest = np.mean(distances)
        # Multiply that by the vector length threshold#
        major_cutoff = vector_length_threshold/avg_nearest
        return major_cutoff
    
    def normalize_vector(self,v):
        """Normalize a vector to unit length"""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v  # or handle zero vector case as needed
        return v / norm
    
    def create_vector_zones(self, corner, vectors, length, width):
        """Returns list of 4-point rectangles around each vector."""
        zones = []

        for v in vectors:
            v = v / np.linalg.norm(v)
            end = corner + v * length
            perp = np.array([-v[1], v[0]])

            p1 = corner + perp * (width / 2)
            p2 = corner - perp * (width / 2)
            p3 = end - perp * (width / 2)
            p4 = end + perp * (width / 2)

            zone = np.array([p1, p2, p3, p4])  # Clockwise quad
            zones.append(zone)

        return zones
    
    def count_points_in_zones(self, zones, test_points):
        """Count how many test points fall inside each rectangular zone."""
        def point_in_rectangle(p, rect):
            # Convert to vectors relative to corner
            A, B, C, D = rect
            AB = B - A
            AD = D - A
            AP = p - A

            # Solve for (u,v) in AP = u*AB + v*AD
            mat = np.column_stack((AB, AD))
            if np.linalg.matrix_rank(mat) < 2:
                return False  # Degenerate rectangle
            uv = np.linalg.lstsq(mat, AP, rcond=None)[0]
            u, v = uv

            return 0 <= u <= 1 and 0 <= v <= 1

        # Make sure its 2D
        test_points = test_points[:,:2]
        counts = []
        for rect in zones:
            count = sum(point_in_rectangle(p, rect) for p in test_points)
            counts.append(count)

        return np.array(counts)

    def check_major_corner(self,testCorner,testAB,testDC,concaveRing,major_cutoff,vector_length_threshold,vector_width_threshold):
    
        # Get the rectangular zones around the vectors to count points in
        vector_rectangles = self.create_vector_zones(testCorner,[-testAB,-testDC],vector_length_threshold,vector_width_threshold)

        # Count the number of points in the zone around each vector
        nPoints_in_zones = self.count_points_in_zones(vector_rectangles, concaveRing[0])

        # Check if the number of points in each zone is over the cutoff
        above_cutoff = nPoints_in_zones>major_cutoff

        # If both are above, then it is a major corner
        # Check if both corner direction areas contain over the major point threshold
        if sum(above_cutoff) == 2:
            major = True
        else:
            major = False
        
        return major

    def get_major_corners_and_directions(self):

        concaveRing = self.concave_hull
        total_pcd = self.total_pcd
        nzFiltercutoff = self.nzFiltercutoff
        corner_angle_threshold = self.corner_angle_threshold
        vector_length_threshold = self.vector_length_threshold
        vector_width_threshold = self.vector_width_threshold
        major_cutoff_multiplier = self.major_cutoff_multiplier

        ## Get the normal vectors of all the hull points to filter defects
        hull_normals = np.asarray(total_pcd.normals)[concaveRing[1]]
        #
        ##Filter out the non horizontal normal points
        concaveRing_nzFiltered = (concaveRing[0][np.where(hull_normals[:, 2]<nzFiltercutoff)[0]],concaveRing[1][np.where(hull_normals[:, 2]<nzFiltercutoff)[0]])

        corner_indices, angles = self.detect_sharp_corners(concaveRing_nzFiltered[0], corner_angle_threshold)
        
        # remember that corner list contains coordinates
        corner_list = concaveRing[0][corner_indices]

        if len(corner_indices)>0:
            direction_vectors = np.array(self.get_corner_vectors(concaveRing_nzFiltered[0],corner_indices))

            all_ABs = direction_vectors[:,0]
            all_CBs = direction_vectors[:,1]
            
            major_cutoff = self.get_major_cutoff(concaveRing_nzFiltered,vector_length_threshold)*major_cutoff_multiplier

            # A list to hold the major corners
            major_corners=[]
            major_ABs=[]
            major_CBs=[]
            minor_corners=[]
            minor_ABs=[]
            minor_CBs=[]
            major_binary=[]

            # Loop over all corners
            for i in range(len(corner_list)):
                #Determine if this corner is major
                major = self.check_major_corner(corner_list[i,:2],all_ABs[i],all_CBs[i],concaveRing_nzFiltered,major_cutoff,vector_length_threshold,vector_width_threshold)
                major_binary.append(major)
                if major:
                    major_corners.append(corner_list[i])
                    major_ABs.append(all_ABs[i])
                    major_CBs.append(all_CBs[i])
                else:
                    minor_corners.append(corner_list[i])
                    minor_ABs.append(all_ABs[i])
                    minor_CBs.append(all_CBs[i])
                    
            major_corners = np.array(major_corners)
            major_ABs = np.array(major_ABs)
            major_CBs = np.array(major_CBs)
            minor_ABs = np.array(minor_ABs)
            minor_corners = np.array(minor_corners)
            minor_CBs = np.array(minor_CBs)
            all_ABs = np.array(all_ABs)
            all_CBs = np.array(all_CBs)
            major_binary = np.array(major_binary)
        else:
            major_corners = np.empty((0,3))
            major_ABs = np.empty((0,2))
            major_CBs = np.empty((0,2))
            minor_ABs = np.empty((0,2))
            minor_corners = np.empty((0,3))
            minor_CBs = np.empty((0,2))
            all_ABs = np.empty((0,2))
            all_CBs = np.empty((0,2))
            major_binary = np.empty((0,1))


        return major_corners, major_ABs, major_CBs, minor_corners, minor_ABs, minor_CBs, major_binary, all_ABs, all_CBs, corner_list
    def get_wire(self):
        self.shown = False
        pts = np.asarray(self.total_pcd.points)[self.concave_hull[1]]
        # Make line connections in order (0-1, 1-2, ..., N-1 -> 0)
        lines = [[i, (i + 1) % len(pts)] for i in range(len(pts))]

        # Build LineSet
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        # Color (optional)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines]) 

        self.wire = line_set
        return line_set
    def get_pcd(self):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(self.total_pcd.points)[self.roof_cluster])
        pcd.paint_uniform_color([.8,.8,0])

        self.pcd = pcd
        return pcd



class Roof:


    def __init__(self, 
                    roof_cluster, total_pcd,
                    roof_concave_hull_length_threshold, roof_concave_hull_concavity,
                    roof_corner_angle_threshold,  
                    roof_vector_length_threshold, roof_vector_width_threshold, roof_major_cutoff_multiplier,
                    roof_horizontal_spacing, roof_anchor_distance, roof_double_tap_dist, roof_double_tap_list, roof_max_distance_from_model, roof_edge_maximum_filtering_distance,
                    roof_distance_to_concave_hull_filtration,
                    roof_center_upperdist, roof_center_mindist, roof_center_min_spacing_2,roof_center_max_iter, polyline_neighbour_radius
                    ):

        # Get Params
        self.roof_cluster = roof_cluster
        self.total_pcd = total_pcd
        self.roof_concave_hull_length_threshold = roof_concave_hull_length_threshold
        self.roof_concave_hull_concavity = roof_concave_hull_concavity
        self.roof_corner_angle_threshold = roof_corner_angle_threshold

        self.roof_vector_length_threshold = roof_vector_length_threshold
        self.roof_vector_width_threshold = roof_vector_width_threshold
        self.roof_major_cutoff_multiplier = roof_major_cutoff_multiplier
        self.roof_distance_to_concave_hull_filtration = roof_distance_to_concave_hull_filtration

        self.roof_horizontal_spacing=roof_horizontal_spacing                                    # horizontal spacing of generated facade taps
        self.roof_anchor_distance=roof_anchor_distance                                          # Distance from major corners where corner roof taps are placed
        self.roof_double_tap_dist = roof_double_tap_dist                                        # additional distance for double corner taps
        self.roof_double_tap_list = roof_double_tap_list                                        # allow sto specify 'which' roof corner taps get doulbe tapped. Probasbly always false
        self.roof_max_distance_from_model=roof_max_distance_from_model                          # BEfore merging roof edge points, remove any points that are not near the roof (for whatever reason, usually weird point detection). THis is the threshold distance. THis probably shouldn't be a user parameter but I will need to tune it.
        self.roof_edge_maximum_filtering_distance=roof_edge_maximum_filtering_distance          # This is the distance roof edge-points can be from eaech other before they are replaced by the average of the points. Higher values will mean many points get averaged together, lower valuers mean fewer points do. 0 means this doesn't happen at all. infinity means all roof edges are replaced by a single point ( for each roof)
        self.roof_distance_to_concave_hull_filtration=roof_distance_to_concave_hull_filtration  # THis will remove any points that are more than X away from the concave hull itself ( for roof edge points only -- prevents crazy tap placement for badly computed concave hulls)

        self.roof_center_upperdist = roof_center_upperdist
        self.roof_center_mindist = roof_center_mindist
        self.roof_center_min_spacing_2 = roof_center_min_spacing_2
        self.roof_center_max_iter = roof_center_max_iter
        self.polyline_neighbour_radius = polyline_neighbour_radius
        
        # These parameters will determine if the roof is computed with both central taps and edge taps, true by default for all detected roofs, but can be turned off for individual roofs
        self.has_edge_taps = True
        self.has_central_taps = True # FALSE CENTRAL TAPS FOR NOW JUST TO TEST FASTER

        self.concave_hull = None
        self.Corners = None
        self.edge_taps = None
        self.central_taps = None
        self.shown = None

        #Get the roof Concave Hull
        self.concave_hull = self.get_concave_hull()
        self.pcd = self.get_pcd()
        self.wire = self.get_wire()

        # Next we get the edge taps, starting by identifying the corners and major corners.
        if len(self.concave_hull[0])>3:# Need to have at least 4 points in a concave hull for it to make any sense
            self.all_corners = self.get_major_corners_and_directions_for_roofs()
            self.major_corners = self.all_corners[0]
            self.major_directions_1 = self.all_corners[1]
            self.major_directions_2 = self.all_corners[2]
            self.minor_corners = self.all_corners[3]
            self.minor_directions_1 = self.all_corners[4]
            self.minor_directions_2 = self.all_corners[5]
            self.major_binary = self.all_corners[6]
            self.all_direction_1s = self.all_corners[7]
            self.all_direction_2s = self.all_corners[8]
            self.sharp_corners = self.all_corners[9]

            # Next we initialize the Corners  and identify the edge tap coordinates.
            self.Corners = self.initialize_corners()
            self.edge_tap_coords = self.downsample_by_spacing_with_major_corner_anchors_for_roofs()
            
            # Next, we get the central taps
            self.central_tap_coords = self.get_central_roof_taps()

        #
        else:
            self.edge_tap_coords=[]
            self.central_tap_coords=[]
        


    def get_central_roof_taps(self):

        roof_center_upper_dist = self.roof_center_upperdist
        roof_center_mindist = self.roof_center_mindist
        roof_center_min_spacing_2 = self.roof_center_min_spacing_2
        roof_center_max_iter = self.roof_center_max_iter
        polyline_neighbour_radius = self.polyline_neighbour_radius

        concaveRing = [self.concave_hull[0][:,:2],self.concave_hull[1]]
        total_pcd = self.total_pcd


        points_3d_np = np.array([])
        if len(self.concave_hull[0]) >0:# and not self.are_colinear(self.concave_hull[0][:2],tol = 1e-9): # this causes failure right now, for maybe obvious reasons -- don't know why it doesnt previously
            central_roof_taps = self.get_all_central_roof_taps(concaveRing, roof_center_upperdist = roof_center_upper_dist, roof_center_mindist = roof_center_mindist, roof_center_min_spacing_2 = roof_center_min_spacing_2, max_iter = roof_center_max_iter,polyline_neighbour_radius = polyline_neighbour_radius)
            
            if len(central_roof_taps) > 0:
                zval=np.mean(np.asarray(total_pcd.points)[self.roof_cluster][:,2])
                points_3d_np = np.hstack([central_roof_taps, np.full((central_roof_taps.shape[0], 1), zval)])

        return points_3d_np

    def resample_polyline(self, polyline: np.ndarray, step: float, start: float = 0.0) -> np.ndarray:
        """
        Parameters
        ----------
        polyline : (M,2) array of ordered points along a curve
        step     : desired spacing between successive samples (>0)
        start    : offset from the polyline start before the first sample
        Returns  : (K,2) array of samples
        """
        if polyline.shape[0] < 2 or step <= 0:
            return np.empty((0, 2), dtype=float)

        diffs = np.diff(polyline, axis=0)
        seglens = np.linalg.norm(diffs, axis=1)
        arclen = np.concatenate(([0.0], np.cumsum(seglens)))
        total = arclen[-1]
        if total <= 0:
            return np.empty((0, 2), dtype=float)

        # Clamp start into [0, total)
        start = float(start % step) if step > 0 else 0.0
        if start >= total:
            return np.empty((0, 2), dtype=float)

        targets = np.arange(start, total + 1e-12, step)
        # Map target arclengths to segments
        idx = np.searchsorted(arclen, targets, side='right') - 1
        idx = np.clip(idx, 0, len(seglens) - 1)
        t0 = arclen[idx]
        t1 = arclen[idx + 1]
        w = (targets - t0) / np.maximum(t1 - t0, 1e-12)
        pts = polyline[idx] + w[:, None] * (polyline[idx + 1] - polyline[idx])
        return pts

    # --- Main: sample from provided polylines with min-distance constraints ---
    def sample_points_from_polylines(self, 
        polylines: List[np.ndarray],
        orig_points: np.ndarray,
        min_dist: float,
        ring_boundary: Optional[np.ndarray] = None,
        random_phase: bool = True,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        polylines      : list of (Mi,2) arrays, ordered points along each curve
        orig_points    : (N,2) array of existing points the new points must avoid
        min_dist       : minimum allowed distance between any new-new and new-old
        ring_boundary  : optional (K,2) array forming a polygon; if given, only
                        keep samples strictly inside this boundary
        random_phase   : if True, each polyline uses a random offset in [0, min_dist)
        seed           : RNG seed for reproducibility when random_phase is True

        Returns
        -------
        samples        : (P,2) array of accepted new points
        """
        assert min_dist > 0, "min_dist must be > 0"

        rng = np.random.default_rng(seed)
        polygon = Polygon(ring_boundary) if ring_boundary is not None else None
        orig_tree = SCKDTree(np.asarray(orig_points, dtype=float)) if len(orig_points) else None

        accepted: List[tuple] = []
        accepted_tree: Optional[SCKDTree] = None

        def ok_vs_original(p: np.ndarray) -> bool:
            if orig_tree is None:
                return True
            d, _ = orig_tree.query(p[None, :], k=1)
            return float(d[0]) >= min_dist

        def ok_vs_new(p: np.ndarray) -> bool:
            nonlocal accepted_tree
            if accepted_tree is None:
                return True
            d, _ = accepted_tree.query(p[None, :], k=1)
            return float(d[0]) >= min_dist

        for pl in polylines:
            pl = np.asarray(pl, dtype=float)
            if pl.shape[0] < 2:
                continue
            offset = rng.random() * min_dist if random_phase else 0.0
            candidates = self.resample_polyline(pl, min_dist, start=offset)
            for p in candidates:
                if polygon is not None and not polygon.contains(Point(p[0], p[1])):
                    continue
                if not ok_vs_original(p):
                    continue
                if not ok_vs_new(p):
                    continue
                accepted.append((p[0], p[1]))
                # Rebuild tree lazily to keep it simple and robust
                accepted_tree = SCKDTree(np.asarray(accepted))

        return np.asarray(accepted, dtype=float)
    def points_to_polylines(self, points, neighbor_radius):
        """
        Convert unordered 2D points into ordered polylines.
        
        Parameters
        ----------
        points : (N,2) array of xy coordinates
        neighbor_radius : max distance to connect points into same curve
        
        Returns
        -------
        polylines : list of (M,2) arrays, each an ordered curve
        """
        points = np.asarray(points, dtype=float)
        tree = SCKDTree(points)
        
        # Build graph: connect close neighbors
        G = nx.Graph()
        for i, p in enumerate(points):
            idxs = tree.query_ball_point(p, neighbor_radius)
            for j in idxs:
                if i != j:
                    G.add_edge(i, j)
        
        # Get connected components → curves
        polylines = []
        for comp in nx.connected_components(G):
            comp_points = points[list(comp)]
            if len(comp_points) < 2:
                continue
            
            # Order points: start from endpoint (degree=1) or arbitrary
            subG = G.subgraph(comp)
            endpoints = [n for n, deg in subG.degree() if deg == 1]
            start = endpoints[0] if endpoints else list(comp)[0]
            
            ordered = []
            visited = set()
            current = start
            while True:
                ordered.append(points[current])
                visited.add(current)
                neighbors = [n for n in subG.neighbors(current) if n not in visited]
                if not neighbors:
                    break
                current = neighbors[0]
            
            polylines.append(np.array(ordered))
        
        return polylines

    def get_central_roof_points_2(self, concaveRing,pts_1,roof_center_min_spacing_2):
        # polygon and interior points as before
        polygon = Polygon(concaveRing[0])
        interior_tree = cKDTree(pts_1)

        # 2D grid
        min_x, min_y, max_x, max_y = polygon.bounds
        grid_res = 300
        x = np.linspace(min_x, max_x, grid_res)
        y = np.linspace(min_y, max_y, grid_res)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Distances
        boundary_dist = np.array([polygon.exterior.distance(Point(p)) for p in grid_points])
        interior_dist, _ = interior_tree.query(grid_points)
        mid_field = boundary_dist - interior_dist
        mid_field = mid_field.reshape((grid_res, grid_res))

        # Mask outside polygon
        mask = np.array([polygon.contains(Point(p)) for p in grid_points]).reshape((grid_res, grid_res))
        mid_field_masked = np.where(mask, mid_field, np.nan)

        CS = plt.contour(x, y, mid_field_masked, levels=[0], colors='green')

        # Grab points from the contour using allsegs (works in matplotlib 3.10+)
        midline_points = []
        for seg_group in CS.allsegs:           # seg_group corresponds to a level (here only one level)
            for seg in seg_group:               # seg is an array of (x, y) points
                midline_points.append(seg)

        # If multiple curves exist, pick the longest one
        midline_points = max(midline_points, key=len)

        # Sample points along midline with min distance threshold
        all_points = np.vstack([concaveRing[0], pts_1])
        sampled_points = []
        tree = cKDTree(all_points)
        for p in midline_points:
            dist, _ = tree.query([p])
            if dist[0] >= roof_center_min_spacing_2:
                sampled_points.append(p)
                tree = cKDTree(np.vstack([all_points, sampled_points]))  # update tree

        pts_center_n = np.array(sampled_points)
        plt.close()
        return pts_center_n
    ##########################################################################################################
    
    def get_central_roof_points_1(self, concaveRing,roof_center_mindist=10,polyline_neighbour_radius=10):

        # Get interior voronoi vertices
        vor = Voronoi(concaveRing[0])
        polygon = Polygon(concaveRing[0])
        interior_vertices = []
        for v in vor.vertices:
            point = Point(v)
            if polygon.contains(point):
                interior_vertices.append(v)
        interior_vertices = np.array(interior_vertices)


        curves = np.array(interior_vertices)
        curves = [x for x in curves if len(x) >0]
        
        if len(curves)>0:
            polylines = self.points_to_polylines(curves, neighbor_radius=polyline_neighbour_radius) 

            ring = np.array(concaveRing[0])
            orig = np.array(concaveRing[0])
            pts = self.sample_points_from_polylines(polylines, orig, min_dist=roof_center_mindist, ring_boundary=ring, random_phase=False, seed=42)


            return np.array(pts)
        else: return np.array([])

    def get_all_central_roof_taps(self,concaveRing, roof_center_upperdist = 10, roof_center_mindist = 5, roof_center_min_spacing_2 = 10, max_iter = 5,polyline_neighbour_radius = 10):
        
        # set the roof center min dist to the maximum allowed (parameter)
        roof_center_dist = roof_center_upperdist

        # Use voronoi tesseltation
        pts_central_1 = self.get_central_roof_points_1(concaveRing,roof_center_mindist=roof_center_dist,polyline_neighbour_radius=polyline_neighbour_radius)

        # We'll run this until either we get a central point or we reach the minimum distance
        while len(pts_central_1) == 0 and roof_center_dist > roof_center_mindist: 

            # Subtract one off of the roof_center_dist
            roof_center_dist /=1.2
            pts_central_1 = self.get_central_roof_points_1(concaveRing,roof_center_mindist=roof_center_dist,polyline_neighbour_radius=polyline_neighbour_radius)

            
        # At this point, if pts_central_1 is empty, then that roof simply doesn't get any central roof points at all, we return an empty array
        if len(pts_central_1) == 0:
            all_roof_center_taps = np.array([])
            return all_roof_center_taps

        
        # If pts_central_1 does have points, we proceed to the next step (extra sampling in other "empty" areas)

        else:

            # Append all the iterated points
            iterated_center_roof_points_2 = [pts_central_1]

            #Run thi until we get an empty list returned
            while len(iterated_center_roof_points_2[-1]) !=0 and len(iterated_center_roof_points_2)<max_iter:

                # Make an array to hold all previous points
                all_iterated_points = np.vstack([x for x in iterated_center_roof_points_2 if len(x) !=0])
                
                # Repeat this over and over with the most recent iteration replacing points_central_1
                center_roof_points_2=self.get_central_roof_points_2(concaveRing,all_iterated_points,roof_center_min_spacing_2=roof_center_min_spacing_2)

                iterated_center_roof_points_2.append(center_roof_points_2)

            all_iterated_points = np.vstack([x for x in iterated_center_roof_points_2 if len(x) !=0])
            all_center_roof_points = np.vstack([pts_central_1,all_iterated_points])
            return all_center_roof_points


    def are_colinear(self, points, tol=1e-9):
        points = np.array(points)
        if len(points) < 3:
            return True  # 2 points are always colinear

        # Take direction vector from first two points
        v = points[1] - points[0]

        for i in range(2, len(points)):
            w = points[i] - points[0]
            if np.linalg.norm(np.cross(v, w)) > tol:  # non-zero cross product → not colinear
                return False
        return True




    def downsample_by_spacing_with_major_corner_anchors_for_roofs(self):
        """
        concaveRing: tuple (concave_points_2d, concave_indices)
            concave_points_2d: (M, 2) ordered 2D points forming a closed concave hull loop
            concave_indices: indices into all_points_3d
        all_points_3d: (P, 3) full 3D coordinates of the model
        major_corners: (N, 3) corner point locations in 3D
        corner_dirs1: (N, 2) direction vectors (x, y) for one side of corner
        corner_dirs2: (N, 2) direction vectors (x, y) for other side of corner
        roof_edge_maximum_filtering_distance: maximum XY distance for merging points

        """

        self.major_binary = self.get_major_binary_from_corners()

        major_corners = self.sharp_corners[self.major_binary == np.True_]
        corner_dirs1 = self.all_direction_1s[self.major_binary == np.True_]
        corner_dirs2 = self.all_direction_2s[self.major_binary == np.True_]
        all_points_3d = self.all_points_3d

        spacing = self.roof_horizontal_spacing
        dist =self.roof_anchor_distance                                       
        double_tap_dist = self.roof_double_tap_dist                                       
        double_tap = self.roof_double_tap_list       
        max_distance_from_model = self.roof_max_distance_from_model,
        roof_edge_maximum_filtering_distance=self.roof_edge_maximum_filtering_distance          
        roof_distance_to_concave_hull_filtration = self.roof_distance_to_concave_hull_filtration

        concave_points_2d = self.concave_hull[0][:,:2]
        concave_points_3d = self.concave_hull[0]

        
        if len(major_corners) < 1:
            n = len(concave_points_3d)
            closed_points = np.vstack([concave_points_3d, concave_points_3d[0]])
            diffs = np.diff(closed_points, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            cumdist = np.concatenate(([0], np.cumsum(dists)))
            total_length = cumdist[-1]

            fx = interp1d(cumdist, closed_points[:,0], kind='linear')
            fy = interp1d(cumdist, closed_points[:,1], kind='linear')
            fz = interp1d(cumdist, closed_points[:,2], kind='linear')

            final_dists = np.arange(0, total_length, spacing)
            downSampledRing = np.vstack([fx(final_dists), fy(final_dists), fz(final_dists)]).T

        else:
            # --- Step 1: Build inset polygon (2D) ---
            pts2d = np.array(concave_points_2d)
            n = len(pts2d)
            inset_pts2d = []

            # Compute polygon winding (positive = CCW)
            area = 0.5 * np.sum(pts2d[:,0]*np.roll(pts2d[:,1],-1) - pts2d[:,1]*np.roll(pts2d[:,0],-1))
            ccw = area > 0

            for i in range(n):
                p_prev = pts2d[i-1]
                p_curr = pts2d[i]
                p_next = pts2d[(i+1)%n]

                v1 = p_curr - p_prev
                v2 = p_next - p_curr

                n1 = np.array([-v1[1], v1[0]])
                n2 = np.array([-v2[1], v2[0]])
                if not ccw:
                    n1, n2 = -n1, -n2

                n1 /= np.linalg.norm(n1)
                n2 /= np.linalg.norm(n2)
                avg_n = (n1 + n2)
                if np.linalg.norm(avg_n) < 1e-8:
                    avg_n = n1
                avg_n /= np.linalg.norm(avg_n)
                #print(p_curr)
                #print(dist)
                #print(avg_n)
                inset_pts2d.append(p_curr + dist * avg_n)

            inset_pts2d = np.array(inset_pts2d)

            # --- Step 2: Lift back to 3D using original Z values ---
            inset_pts3d = np.column_stack([inset_pts2d, concave_points_3d[:,2]])

            # Close loop
            closed_points = np.vstack([inset_pts3d, inset_pts3d[0]])
            diffs = np.diff(closed_points, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            cumdist = np.concatenate(([0], np.cumsum(dists)))
            total_length = cumdist[-1]

            # Interpolation along inset curve
            fx = interp1d(cumdist, closed_points[:,0], kind='linear')
            fy = interp1d(cumdist, closed_points[:,1], kind='linear')
            fz = interp1d(cumdist, closed_points[:,2], kind='linear')

            # --- Step 3: Anchors ---
            anchor_points = []
            for corner, d1, d2 in zip(major_corners, corner_dirs1, corner_dirs2):
                if corner.shape[0] == 2:
                    corner = np.array([corner[0], corner[1], 0.0])
                d_sum = np.array([d1[0] + d2[0], d1[1] + d2[1], 0.0])
                norm = np.linalg.norm(d_sum)
                if norm == 0:
                    continue
                d_sum /= norm
                inward_anchor = corner + dist * d_sum
                anchor_points.append(inward_anchor)
                if double_tap:
                    anchor_points.append(inward_anchor + double_tap_dist * d_sum)

            # --- Step 4: Map anchors to curve parameters ---
            fine_dists = np.linspace(0, total_length, 5000)
            fine_points = np.vstack([fx(fine_dists), fy(fine_dists), fz(fine_dists)]).T

            anchor_dists = []
            for a in anchor_points:
                idx = np.argmin(np.linalg.norm(fine_points - a, axis=1))
                anchor_dists.append(fine_dists[idx])
            anchor_dists = np.sort(np.array(anchor_dists))

            # --- Step 5: Fill evenly between anchors ---
            final_dists = []
            for i in range(len(anchor_dists)):
                d1 = anchor_dists[i]
                d2 = anchor_dists[(i + 1) % len(anchor_dists)]
                if d2 < d1:
                    d2 += total_length
                seg_len = d2 - d1
                if seg_len == seg_len:
                    nseg = int(np.ceil(seg_len / spacing))
                    interp = np.linspace(d1, d2, nseg, endpoint=False)
                    final_dists.extend(interp % total_length)

            final_dists = np.unique(np.array(final_dists))
            candidate_points = np.vstack([fx(final_dists), fy(final_dists), fz(final_dists)]).T
            
            # first remove points that are distant from the roof
            model_xy = all_points_3d[:, :2]
            tree_model = cKDTree(model_xy)
            distances, _ = tree_model.query(candidate_points[:, :2], k=1)
            candidate_points = candidate_points[distances <= max_distance_from_model]

            # --- Step 6: Merge points within roof_edge_maximum_filtering_distance (XY only) ---
            if len(candidate_points) == 0:
                return candidate_points

            xy = candidate_points[:, :2]
            tree = cKDTree(xy)

            visited = np.zeros(len(candidate_points), dtype=bool)
            merged_points = []
            for i, p in enumerate(candidate_points):
                if visited[i]:
                    continue
                idxs = tree.query_ball_point(p[:2], roof_edge_maximum_filtering_distance)
                cluster = candidate_points[idxs]
                visited[idxs] = True
                merged_points.append(cluster.mean(axis=0))  # average x, y, z

            final_points = np.array(merged_points)
            downSampledRing = final_points


        if downSampledRing.shape[0] > 0:
            hull_xy = concave_points_3d[:,:2]
            dists_to_hull = np.min(np.linalg.norm(downSampledRing[:,:2][:,None,:] - hull_xy[None,:,:], axis=2), axis=1)
            downSampledRing = downSampledRing[dists_to_hull <= roof_distance_to_concave_hull_filtration]

            final_points = downSampledRing

        return final_points        
    
    def get_major_binary_from_corners(self):
        
        major_binary = []
        for i in range(len(self.Corners)):
            major_binary.append(self.Corners[i].major)

        return major_binary

    def initialize_corners(self):

        corners = []
        for i in range(len(self.sharp_corners)):
            corners.append(Corner(self.sharp_corners[i], self.major_binary[i], self.all_direction_1s[i], self.all_direction_2s[i]))
        return corners
    
    def get_corner_vectors(self, points, corner_indices):
        """
        For each corner index B, compute direction vectors AB and CB where:
            A is the previous point in the ring
            C is the next point in the ring

        Args:
            points (np.ndarray): Nx3 array of 3D points forming a ring.
            corner_indices (list or np.ndarray): Indices of corner points (B).

        Returns:
            List of tuples: [(AB, CB), ...] for each corner index.
        """
        N = len(points)
        vectors = []

        for idx in corner_indices:
            A = points[(idx - 1) % N,:2]
            B = points[idx,:2]
            C = points[(idx + 1) % N,:2]

            AB = self.normalize_vector(B - A)
            CB = self.normalize_vector(B - C)  # Note: CB = B - C (vector from C to B)
            vectors.append((AB, CB))

        return vectors
    def normalize_vector(self,v):
        """Normalize a vector to unit length"""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v  # or handle zero vector case as needed
        return v / norm
    def get_wrapped_indices(self , idx, N, length):
        """
        Returns indices [idx-N, ..., idx-1, idx, idx+1, ..., idx+N] with wrapping.
        """
        return [(idx + i) % length for i in range(-N, N+1)]
    def centroid_direction(self, points, n=4):
        start = np.mean(points[:n], axis=0)
        end = np.mean(points[-n:], axis=0)
        vec = end - start
        return vec / np.linalg.norm(vec)
    def get_avg_corner_vectors(self, points, corner_indices,Navg=4):
        """
        For each corner index B, compute direction vectors AB and CB where:
            A is the previous point in the ring
            C is the next point in the ring

        Args:
            points (np.ndarray): Nx3 array of 3D points forming a ring.
            corner_indices (list or np.ndarray): Indices of corner points (B).

        Returns:
            List of tuples: [(AB, CB), ...] for each corner index.
        """
        N = len(points)

        vectors = []

        for idx in corner_indices:
            A = points[(idx - 1) % N,:2]
            B = points[idx,:2]
            C = points[(idx + 1) % N,:2]
            wrapped_indices = self.get_wrapped_indices(idx, Navg, N)
            pre_wrapped_indices = wrapped_indices[:round(len(wrapped_indices)/2)]
            post_wrapped_indices= wrapped_indices[int(np.ceil(len(wrapped_indices)/2)):]

            incoming_points = points[pre_wrapped_indices][:,:2]
            outgoing_points = points[post_wrapped_indices][:,:2]

            incoming_centroid_direction=self.normalize_vector(self.centroid_direction(incoming_points,int(Navg/2)))
            outgoing_centroid_direction=self.normalize_vector(self.centroid_direction(outgoing_points[::-1],int(Navg/2)))

            A = points[wrapped_indices[Navg-2]][:2]
            B = points[wrapped_indices[Navg-1]][:2]
            C = points[wrapped_indices[Navg+1]][:2]
            D = points[wrapped_indices[Navg+2]][:2]

            AB = self.normalize_vector(B - A)
            DC = self.normalize_vector(C - D)

        # AB = normalize_vector(B - A)
            #CB = normalize_vector(B - C)  # Note: CB = B - C (vector from C to B)
            vectors.append((AB, DC))

        return vectors

    def get_major_cutoff(self, concaveRing,vector_length_threshold): 
        # We want to set a threshold for the number of points contained within a vector to be considered major
        # Get the mean distance to the nearest point in this concave hull
        distances = cKDTree(concaveRing[0][:,:2]).query(concaveRing[0][:,:2], k=2)[0][:,1]
        avg_nearest = np.mean(distances)
        # Multiply that by the vector length threshold#
        major_cutoff = vector_length_threshold/avg_nearest
        return major_cutoff
        
    def get_major_corners_and_directions_for_roofs(self):

        concaveRing = self.concave_hull
        corner_angle_threshold = self.roof_corner_angle_threshold
        vector_length_threshold = self.roof_vector_length_threshold
        vector_width_threshold = self.roof_vector_width_threshold
        major_cutoff_multiplier = self.roof_major_cutoff_multiplier

        corner_indices, angles = self.detect_sharp_corners(concaveRing[0], corner_angle_threshold)

        corner_list = concaveRing[0][corner_indices]

        direction_vectors = np.array(self.get_corner_vectors(concaveRing[0],corner_indices))
            
        if len(corner_list) == 0:
            return np.empty((0,3)), np.empty((0,2)), np.empty((0,2)), np.empty((0,3)), np.empty((0,2)), np.empty((0,2))

        all_ABs = direction_vectors[:,0]
        all_CBs = direction_vectors[:,1]
        
        major_cutoff = self.get_major_cutoff(concaveRing,vector_length_threshold)*major_cutoff_multiplier

        # A list to hold the major corners
        major_corners=[]
        major_ABs=[]
        major_CBs=[]
        minor_corners=[]
        minor_ABs=[]
        minor_CBs=[]
        major_binary = []

        # Loop over all corners
        for i in range(len(corner_list)):
            #Determine if this corner is major
            major = self.check_major_corner(corner_list[i,:2],all_ABs[i],all_CBs[i],concaveRing,major_cutoff,vector_length_threshold,vector_width_threshold)
            major_binary.append(major)
            if major:
                major_corners.append(corner_list[i])
                major_ABs.append(all_ABs[i])
                major_CBs.append(all_CBs[i])
            else:
                minor_corners.append(corner_list[i])
                minor_ABs.append(all_ABs[i])
                minor_CBs.append(all_CBs[i])
                
            
        major_corners = np.array(major_corners)
        major_ABs = np.array(major_ABs)
        major_CBs = np.array(major_CBs)
        minor_ABs = np.array(minor_ABs)
        minor_corners = np.array(minor_corners)
        minor_CBs = np.array(minor_CBs)
        all_ABs = np.array(all_ABs)
        all_CBs = np.array(all_CBs)
        major_binary = np.array(major_binary)

        return major_corners, major_ABs, major_CBs, minor_corners, minor_ABs, minor_CBs, major_binary, all_ABs, all_CBs, corner_list
        

    def detect_sharp_corners(self,hull_points,angle_threshold):
        """
        Detect sharp corners in a hull by analyzing angle changes.
        
        Args:
            hull_points: Array of (x,y) or (x,y,z) points forming the hull
            angle_threshold: Minimum angle change (degrees) to consider a sharp corner
        
        Returns:
            corner_indices: Indices of points that are sharp corners
            angles: Angle changes at each point (in degrees)
        """

        hull_points = self.concave_hull[0]
        angle_threshold = self.roof_corner_angle_threshold


        points_2d = hull_points[:, :2]  # Use only x,y coordinates
        n = len(points_2d)
        angles = []
        
        for i in range(n):
            # Get three consecutive points (wrapping around)
            p1 = points_2d[(i-1) % n]
            p2 = points_2d[i]
            p3 = points_2d[(i+1) % n]
            
            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
            angle = np.arccos(cos_angle)
            angle_deg = np.degrees(angle)
            
            # Convert to exterior angle (180 - interior angle)
            exterior_angle = 180 - angle_deg
            angles.append(abs(exterior_angle))
        
        # Find corners where angle change exceeds threshold
        corner_indices = np.where(np.array(angles) > angle_threshold)[0]
        
        return corner_indices, angles
    
    def count_points_in_zones(self, zones, test_points):
        """Count how many test points fall inside each rectangular zone."""
        def point_in_rectangle(p, rect):
            # Convert to vectors relative to corner
            A, B, C, D = rect
            AB = B - A
            AD = D - A
            AP = p - A

            # Solve for (u,v) in AP = u*AB + v*AD
            mat = np.column_stack((AB, AD))
            if np.linalg.matrix_rank(mat) < 2:
                return False  # Degenerate rectangle
            uv = np.linalg.lstsq(mat, AP, rcond=None)[0]
            u, v = uv

            return 0 <= u <= 1 and 0 <= v <= 1

        # Make sure its 2D
        test_points = test_points[:,:2]
        counts = []
        for rect in zones:
            count = sum(point_in_rectangle(p, rect) for p in test_points)
            counts.append(count)

        return np.array(counts)
    def create_vector_zones(self,corner, vectors, length, width):
        """Returns list of 4-point rectangles around each vector."""
        zones = []

        for v in vectors:
            v = v / np.linalg.norm(v)
            end = corner + v * length
            perp = np.array([-v[1], v[0]])

            p1 = corner + perp * (width / 2)
            p2 = corner - perp * (width / 2)
            p3 = end - perp * (width / 2)
            p4 = end + perp * (width / 2)

            zone = np.array([p1, p2, p3, p4])  # Clockwise quad
            zones.append(zone)

        return zones
    def check_major_corner(self,testCorner,testAB,testDC,concaveRing,major_cutoff,vector_length_threshold,vector_width_threshold):
    
        # Get the rectangular zones around the vectors to count points in
        vector_rectangles = self.create_vector_zones(testCorner,[-testAB,-testDC],vector_length_threshold,vector_width_threshold)

        # Count the number of points in the zone around each vector
        nPoints_in_zones = self.count_points_in_zones(vector_rectangles, concaveRing[0])

        # Check if the number of points in each zone is over the cutoff
        above_cutoff = nPoints_in_zones>major_cutoff
 
        # If both are above, then it is a major corner
        # Check if both corner direction areas contain over the major point threshold
        if sum(above_cutoff) == 2:
            major = True
        else:
            major = False
        
        return major
    
    def get_wire(self):
        pts = np.asarray(self.total_pcd.points)[self.concave_hull[1]]
        # Make line connections in order (0-1, 1-2, ..., N-1 -> 0)
        lines = [[i, (i + 1) % len(pts)] for i in range(len(pts))]

        # Build LineSet
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        # Color (optional)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines]) 

        self.wire = line_set
        return line_set
    def get_pcd(self):
        self.shown = False
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(self.total_pcd.points)[self.roof_cluster])

        self.pcd = pcd
        return pcd
    
    def get_concave_hull(self):
            
        total_pcd = self.total_pcd
        roofCluster_idxs = self.roof_cluster
        roof_concaveHull_concavity = self.roof_concave_hull_concavity
        roof_concaveHull_length_threshold = self.roof_concave_hull_length_threshold

        total_pcd_points = np.asarray(total_pcd.points)
        total_pcd_normals = np.asarray(total_pcd.normals)
        
        # Get interior points of the roof only. Remove points that have non vertical nz
        current_point_idxs= np.array(roofCluster_idxs)
        current_z_normals = total_pcd_normals[current_point_idxs,2]
        exteriorMask = abs(current_z_normals) - 1 >- 0.01
        interior_point_idxs = current_point_idxs[exteriorMask]
        self.all_points_3d = total_pcd_points[interior_point_idxs] # will need this later
        
        # Get the concave hull of this roof
        idxes_local = concave_hull_indexes(total_pcd_points[interior_point_idxs,:2], length_threshold=roof_concaveHull_length_threshold,concavity =roof_concaveHull_concavity)
        hull_points = total_pcd_points[current_point_idxs[idxes_local],:2]

        # Get concave hull points
        concave_points_3d = total_pcd_points[interior_point_idxs][idxes_local]

        return (concave_points_3d, interior_point_idxs[idxes_local])



class Corner:

    def __init__(self,coordinates,major,direction_1,direction_2):

        self.coordinates = coordinates
        self.major = major
        self.direction_1 = direction_1
        self.direction_2 = direction_2
        self.shown = None





building = Building("ApprovedTestFiles/cladTestFolder1/mesh_scaled.stl")

# load in the mesh
building.mesh = building.load_mesh()

# Next, use uniform and voxel sampling to create the mesh point cloud, Also get the edge PCD, nd Combine the two with corrected normals
building.total_pcd = building.get_total_pcd()

# Now, we get the rooflines and the fillzed in Z-levels
building.roof_lines = building.compute_and_identify_roofs_and_rooflines()

# Fill in the gaps on the facade based on the parameter
building.facade_z_levels=building.fill_z_gaps()

if building.filter_whole_z_levels == True:
        building.facade_z_levels=building.filter_whole_zlevels()

# Get the roof clusters
building.roof_clusters = building.get_roof_clusters()

# Next we need to initialize The Facade ZLevel objects
building.ZLevels = building.initialize_ZLevels()

# Next we need to initialize the Roof Objects
building.Roofs = building.initialize_Roofs()

# The next thing to do is make sure we filter out taps from Z-levels that are too close together
building.filter_z_rings()

# Next we should collect all of the predicted taps into a predicted taps PCD
building.PredictedTaps = building.collect_taps()

building.PredictedTapPCD = building.get_predicted_tap_pcd()