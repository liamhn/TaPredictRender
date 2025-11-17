import logging
logging.getLogger("streamlit.elements.lib.policies").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
import streamlit as st
import numpy as np
import TaPredictTest_CLADFree
import plotly.graph_objects as go

import TaPredictPressures
import pandas as pd


# --- Building list ---
demoPath = "ApprovedTestFiles/cladTestFolder1/"






# --- Initialize session state ---
if "building_index" not in st.session_state:
    st.session_state.building_index = 0
if "building" not in st.session_state:
    st.session_state.building = TaPredictTest_CLADFree.Building(demoPath+"/mesh_scaled.stl")
    
    # load in the mesh
    st.session_state.building.mesh = st.session_state.building.load_mesh()

    # Next, use uniform and voxel sampling to create the mesh point cloud, Also get the edge PCD, nd Combine the two with corrected normals
    st.session_state.building.total_pcd = st.session_state.building.get_total_pcd()

    # Now, we get the rooflines and the fillzed in Z-levels
    st.session_state.building.roof_lines = st.session_state.building.compute_and_identify_roofs_and_rooflines()

    # Fill in the gaps on the facade based on the parameter
    st.session_state.building.facade_z_levels=st.session_state.building.fill_z_gaps()

    if st.session_state.building.filter_whole_z_levels == True:
            st.session_state.building.facade_z_levels=st.session_state.building.filter_whole_zlevels()

    # Get the roof clusters
    st.session_state.building.roof_clusters = st.session_state.building.get_roof_clusters()

    # Next we need to initialize The Facade ZLevel objects
    st.session_state.building.ZLevels = st.session_state.building.initialize_ZLevels()

    # Next we need to initialize the Roof Objects
    st.session_state.building.Roofs = st.session_state.building.initialize_Roofs()

    # The next thing to do is make sure we filter out taps from Z-levels that are too close together
    st.session_state.building.filter_z_rings()

    # Next we should collect all of the predicted taps into a predicted taps PCD
    st.session_state.building.PredictedTaps = st.session_state.building.collect_taps()

    st.session_state.building.PredictedTapPCD = st.session_state.building.get_predicted_tap_pcd()

if "camera_state" not in st.session_state:
    st.session_state.camera_state = None

if "fig_camera" not in st.session_state:
    st.session_state.fig_camera = None
#building = st.session_state.building
building_index = st.session_state.building_index

st.title("TaPredict Test Tool")

# --- Reset Inputs Button ---
if st.sidebar.button("🔄 Reset Inputs to Defaults"):
    st.session_state.vert_input = 16.0
    st.session_state.horiz_input = 12.0
    st.session_state.corner_input = 0.6
    st.session_state.roof_horiz_input = 12
    st.session_state.roof_size_input = 60
    st.session_state.roof_center_input = 3.0
    st.session_state.roof_radius_input = 10.0
    st.session_state.roof_corner_input = 0.6
    st.session_state.z_value = 0.0
    st.session_state.doubletap_flag = False
    st.session_state.double_spacing_flag = False
    st.rerun()

# --- Define Presets ---
PRESETS = {
    "Default": {
        "vert_input": 16.0,
        "horiz_input": 12.0,
        "corner_input": 0.6,
        "roof_horiz_input": 12,
        "roof_size_input": 60,
        "roof_center_input": 3.0,
        "roof_radius_input": 10.0,
        "roof_corner_input": 0.6,
        "z_value": 0.0,
        "doubletap_flag": False,
        "double_spacing_flag": False,
        "apply_relaxed_filtration": True
    },
    "Sparse layout": {
        "vert_input": 40.0,
        "horiz_input": 20.0,
        "corner_input": 0.6,
        "roof_horiz_input": 24,
        "roof_size_input": 500,
        "roof_center_input": 9.0,
        "roof_radius_input": 15.0,
        "roof_corner_input": 0.6,
        "z_value": 0.0,
        "doubletap_flag": False,
        "double_spacing_flag": False,
        "apply_relaxed_filtration": True
    },
    "Dense Layout": {
        "vert_input": 12.0,
        "horiz_input": 8.0,
        "corner_input": 0.6,
        "roof_horiz_input": 12,
        "roof_size_input": 20,
        "roof_center_input": 3.0,
        "roof_radius_input": 10.0,
        "roof_corner_input": 0.6,
        "z_value": 0.0,
        "doubletap_flag": False,
        "double_spacing_flag": False,
        "apply_relaxed_filtration": False
    }
}

# Ensure these exist globally (to avoid NameError when not in Tap Layout mode)
highlighted_z_idx = None
highlighted_r_idx = None

# --- Tabs ---
side_tab = st.sidebar.radio("Select Panel", ["🧩 Tap Layout", "📊 Pressure Predictions"])
if side_tab == "🧩 Tap Layout":
    st.sidebar.markdown("### Tap Layout Controls")
    # --- Sidebar: Presets Dropdown ---
    preset_choice = st.sidebar.selectbox("🎛 Presets", list(PRESETS.keys()))

    col_preset1, col_preset2 = st.sidebar.columns([1,1])

    # Apply preset only
    if col_preset1.button("Apply Preset"):
        for key, value in PRESETS[preset_choice].items():
            st.session_state[key] = value
        st.rerun()

    # Apply preset and compute facade + roof
    if col_preset2.button("Apply + Compute"):

        for key, value in PRESETS[preset_choice].items():
            st.session_state[key] = value

        # Update building parameters
        st.session_state.building.facade_zLevel_gap_threshold = st.session_state.vert_input
        st.session_state.building.facade_horizontal_spacing = st.session_state.horiz_input
        st.session_state.building.facade_major_cutoff_multiplier = st.session_state.corner_input
        st.session_state.building.roof_horizontal_spacing = st.session_state.roof_horiz_input
        st.session_state.building.roof_min_dense_points = int(st.session_state.roof_size_input)
        st.session_state.building.roof_center_mindist = st.session_state.roof_center_input
        st.session_state.building.roof_center_min_spacing_2 = st.session_state.roof_radius_input
        st.session_state.building.roof_major_cutoff_multiplier = st.session_state.roof_corner_input

        # Run facade + roof
        st.session_state.building.roof_lines = st.session_state.building.compute_and_identify_roofs_and_rooflines()
        st.session_state.building.facade_z_levels = st.session_state.building.fill_z_gaps()
        if st.session_state.building.filter_whole_z_levels == True:
            st.session_state.building.facade_z_levels=st.session_state.building.filter_whole_zlevels()
        st.session_state.building.ZLevels = st.session_state.building.initialize_ZLevels()
        st.session_state.building.filter_z_rings()
        st.session_state.building.roof_clusters = st.session_state.building.get_roof_clusters()
        st.session_state.building.Roofs = st.session_state.building.initialize_Roofs()
        st.session_state.building.PredictedTaps = st.session_state.building.collect_taps()
        st.session_state.building.PredictedTapPCD = st.session_state.building.get_predicted_tap_pcd()

        st.rerun()

    # --- Advanced Section ---
    with st.sidebar.expander("Advanced", expanded=False):
        # --- Facade Parameters ---
        vert_input = st.number_input("Vertical Spacing", 0.5, 100.0, value=16.0, key="vert_input")
        horiz_input = st.number_input("Horizontal Spacing", 0.5, 100.0, value=8.0, key="horiz_input")
        corner_input = st.number_input("Corner Leniency", 0.05, 1.0, value=0.6, key="corner_input")

        # New filtration checkbox
        #apply_relaxed_filtration = st.checkbox("ZLevel Filtration", value=False, key="apply_relaxed_filtration")
        apply_relaxed_filtration = True
        # Apply filtration based on checkbox
        if apply_relaxed_filtration:
            st.session_state.building.facade_filtration_xy_dist = 2.1
            st.session_state.building.facade_filtration_z_dist = 6.0
        else:
            st.session_state.building.facade_filtration_xy_dist = 0.1
            st.session_state.building.facade_filtration_z_dist = 0.1

        if st.button("Recompute Facade"):

            st.session_state.building.facade_zLevel_gap_threshold = vert_input
            st.session_state.building.facade_horizontal_spacing = horiz_input
            st.session_state.building.facade_major_cutoff_multiplier = corner_input
            # Apply filtration again before recomputing
            if apply_relaxed_filtration:
                st.session_state.building.facade_filtration_xy_dist = 2.1
                st.session_state.building.facade_filtration_z_dist = 6.0
            else:
                st.session_state.building.facade_filtration_xy_dist = 0.1
                st.session_state.building.facade_filtration_z_dist = 0.1

            st.session_state.building.roof_lines = st.session_state.building.compute_and_identify_roofs_and_rooflines()
            st.session_state.building.facade_z_levels = st.session_state.building.fill_z_gaps()
            if st.session_state.building.filter_whole_z_levels == True:
                st.session_state.building.facade_z_levels=st.session_state.building.filter_whole_zlevels()
            st.session_state.building.ZLevels = st.session_state.building.initialize_ZLevels()
            st.session_state.building.filter_z_rings()
            st.session_state.building.PredictedTaps = st.session_state.building.collect_taps()
            st.session_state.building.PredictedTapPCD = st.session_state.building.get_predicted_tap_pcd()
        # --- Roof Parameters ---
        roof_horiz_input = st.number_input("Horizontal Spacing (Roof Edges)", 1, 50, value=12, key="roof_horiz_input")
        roof_size_input = st.number_input("Roof Size Cutoff", 1, 10000, value=60, key="roof_size_input")
        roof_center_input = st.number_input("Roof Size Center Tap Cutoff", 1.0, 20.0, value=3.0, key="roof_center_input")
        roof_radius_input = st.number_input("Roof Central Tap Radius Cutoff", 1.0, 25.0, value=10.0, key="roof_radius_input")
        roof_corner_input = st.number_input("Roof Corner Leniency", 0.01, 20.0, value=0.6, key="roof_corner_input")

        if st.button("Recompute Roofs"):

            st.session_state.building.roof_horizontal_spacing = roof_horiz_input
            st.session_state.building.roof_min_dense_points = int(roof_size_input)
            st.session_state.building.roof_center_mindist = roof_center_input
            st.session_state.building.roof_center_min_spacing_2 = roof_radius_input
            st.session_state.building.roof_major_cutoff_multiplier = roof_corner_input
            st.session_state.building.roof_clusters = st.session_state.building.get_roof_clusters()
            st.session_state.building.Roofs = st.session_state.building.initialize_Roofs()
            st.session_state.building.PredictedTaps = st.session_state.building.collect_taps()
            st.session_state.building.PredictedTapPCD = st.session_state.building.get_predicted_tap_pcd()

        # --- Z-Level Controls ---
        z_value = st.number_input("New Z (height):", value=0.0)
        doubletap_flag = st.checkbox("DoubleTap")
        double_spacing_flag = st.checkbox("Half Spacing")
        st.session_state.building.facade_major_cutoff_multiplier = corner_input

        if st.button("Add Z-Level"):
            if "camera_state" not in st.session_state:
                st.session_state.camera_state = None

            if "fig_camera" not in st.session_state:
                st.session_state.fig_camera = None

            spacing_value = st.session_state.building.facade_horizontal_spacing / (2 if double_spacing_flag else 1)
            if not hasattr(st.session_state.building, "ZLevels") or st.session_state.building.ZLevels is None:
                st.session_state.building.ZLevels = []
            from TaPredictTest_CLADFree import ZLevel
            new_z = ZLevel(
                z_value,
                st.session_state.building.total_pcd,
                st.session_state.building.facade_epsilon,
                st.session_state.building.facade_concave_hull_length_threshold,
                st.session_state.building.facade_hull_concavity,
                st.session_state.building.facade_corner_angle_threshold,
                st.session_state.building.facade_nzFiltercutoff,
                st.session_state.building.facade_corner_angle_threshold,
                st.session_state.building.facade_use_average_corners,
                st.session_state.building.facade_N_average_corner_points,
                st.session_state.building.facade_vector_length_threshold,
                st.session_state.building.facade_vector_width_threshold,
                st.session_state.building.facade_major_cutoff_multiplier,
                spacing_value,
                st.session_state.building.facade_anchor_distance,
                st.session_state.building.facade_double_tap_dist,
                doubletap_flag
            )
            
            st.session_state.building.ZLevels.append(new_z)
            st.session_state.building.PredictedTaps = st.session_state.building.collect_taps()
            st.session_state.building.PredictedTapPCD = st.session_state.building.get_predicted_tap_pcd()

        st.markdown("## Object Explorer")

        # --- ZLevels ---
        highlighted_z_idx = None
        if hasattr(st.session_state.building, "ZLevels") and st.session_state.building.ZLevels:
            with st.expander("Z-Levels", expanded=True):
                for i, z in enumerate(st.session_state.building.ZLevels):
                    col0, col1, col2, col3 = st.columns([1,3,1,1])

                    # Checkbox to show overlay
                    highlight = col0.checkbox("", key=f"highlight_z_{i}")
                    col1.write(f"Z-Level {i+1} (z={z.z_level:.2f})")

                    if not hasattr(z, "deleted_taps"):
                        z.deleted_taps = None  # Initialize holder

                    # Delete button (clears taps but keeps ZLevel)
                    if col2.button(f"Del Taps {i+1}", key=f"del_z{i}"):
                        if z.tap_coords is not None and len(z.tap_coords) > 0:
                            z.deleted_taps = z.tap_coords.copy()
                            z.tap_coords = np.empty((0,3))  # clear taps
                            st.session_state.building.PredictedTaps = st.session_state.building.collect_taps()
                            st.session_state.building.PredictedTapPCD = st.session_state.building.get_predicted_tap_pcd()
                            st.rerun()

                    # Undelete button
                    if col3.button(f"Undo {i+1}", key=f"undo_z{i}"):
                        if z.deleted_taps is not None:
                            z.tap_coords = z.deleted_taps
                            z.deleted_taps = None
                            st.session_state.building.PredictedTaps = st.session_state.building.collect_taps()
                            st.session_state.building.PredictedTapPCD = st.session_state.building.get_predicted_tap_pcd()
                            st.rerun()

                    if highlight:
                        highlighted_z_idx = i

        # --- Roofs ---
        highlighted_r_idx = None
        if hasattr(st.session_state.building, "Roofs") and st.session_state.building.Roofs:
            with st.expander("Roofs", expanded=True):
                for i, r in enumerate(st.session_state.building.Roofs):
                    col0, col1, col2, col3 = st.columns([1,3,1,1])
                    highlight = col0.checkbox("", key=f"highlight_r_{i}")
                    col1.write(f"Roof {i}")

                    if not hasattr(r, "deleted_edge_tap_coords"):
                        r.deleted_edge_tap_coords = None
                    if not hasattr(r, "deleted_central_tap_coords"):
                        r.deleted_central_tap_coords = None

                    # Delete taps
                    if col2.button(f"Del Taps {i}", key=f"del_r{i}"):
                        if (len(r.edge_tap_coords) > 0) or (len(r.central_tap_coords) > 0):
                            r.deleted_edge_tap_coords = r.edge_tap_coords.copy()
                            r.deleted_central_tap_coords = r.central_tap_coords.copy()
                            r.edge_tap_coords = []
                            r.central_tap_coords = []
                            st.session_state.building.PredictedTaps = st.session_state.building.collect_taps()
                            st.session_state.building.PredictedTapPCD = st.session_state.building.get_predicted_tap_pcd()
                            st.rerun()

                    # Undelete taps
                    if col3.button(f"Undo {i}", key=f"undo_r{i}"):
                        if r.deleted_edge_tap_coords is not None or r.deleted_central_tap_coords is not None:
                            r.edge_tap_coords = r.deleted_edge_tap_coords if r.deleted_edge_tap_coords is not None else []
                            r.central_tap_coords = r.deleted_central_tap_coords if r.deleted_central_tap_coords is not None else []
                            r.deleted_edge_tap_coords = None
                            r.deleted_central_tap_coords = None
                            st.session_state.building.PredictedTaps = st.session_state.building.collect_taps()
                            st.session_state.building.PredictedTapPCD = st.session_state.building.get_predicted_tap_pcd()
                            st.rerun()

                    if highlight:
                        highlighted_r_idx = i

    # --- Custom Upload ---
    uploaded_file = st.sidebar.file_uploader("Load Custom Building (.stl)", type="stl")
    if uploaded_file is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.building = TaPredictTest_CLADFree.Building(tmp.name)
            st.session_state.building.mesh = st.session_state.building.load_mesh()
            st.session_state.building.total_pcd = st.session_state.building.get_total_pcd()
            st.session_state.building.roof_lines = st.session_state.building.compute_and_identify_roofs_and_rooflines()
            st.session_state.building.facade_z_levels=st.session_state.building.fill_z_gaps()
            if st.session_state.building.filter_whole_z_levels == True:
                    st.session_state.building.facade_z_levels=st.session_state.building.filter_whole_zlevels()
            st.session_state.building.roof_clusters = st.session_state.building.get_roof_clusters()
            st.session_state.building.ZLevels = st.session_state.building.initialize_ZLevels()
            st.session_state.building.Roofs = st.session_state.building.initialize_Roofs()
            st.session_state.building.filter_z_rings()
            st.session_state.building.PredictedTaps = st.session_state.building.collect_taps()
            st.session_state.building.PredictedTapPCD = st.session_state.building.get_predicted_tap_pcd()

    # --- Before plotting, check if we have a stored camera ---
    camera_state = st.session_state.get("camera_state", None)

    # Sidebar download button
    if hasattr(st.session_state.building, "PredictedTapPCD") and st.session_state.building.PredictedTapPCD is not None:
        taps = np.asarray(st.session_state.building.PredictedTapPCD.points)
        if taps.shape[0] > 0:
            df_taps = pd.DataFrame(taps, columns=["X", "Y", "Z"])
            csv_data = df_taps.to_csv(index=False)

            st.sidebar.download_button(
                label="⬇️ Download Taps as CSV",
                data=csv_data,
                file_name="predicted_taps.csv",
                mime="text/csv"
            )
elif side_tab == "📊 Pressure Predictions":
    st.sidebar.markdown("### Pressure Predictions Controls")
    # --- Color toggle ---
    

    # Example placeholder controls for this tab:
    #pressure_threshold = st.sidebar.number_input("Pressure Threshold (psi)", 0.0, 500.0, value=120.0)
    Regularity = st.sidebar.slider("Resolution (meters)", 0.5, 10.0, 2.0, step=.5)

    predictPressures = st.sidebar.button("Predict Pressures")
    if predictPressures:
        mesh_pcd_pressures = TaPredictPressures.get_pressure_lattice(st.session_state.building.mesh, regularity=Regularity, multiplier=1000)
        knn = TaPredictPressures.get_positions_knn_vectors_and_normal_vector_distances(mesh_pcd_pressures)
        ypred = TaPredictPressures.predict_base_model(knn)
        st.session_state.building.mesh, vertex_pressures = TaPredictPressures.recolour_mesh(st.session_state.building.mesh, mesh_pcd_pressures, ypred)

        # store in session state
        st.session_state.ypred = ypred
        st.session_state.vertex_pressures = vertex_pressures
        st.session_state.mesh_pcd_pressures = mesh_pcd_pressures
        st.session_state.st.session_state.building_mesh = st.session_state.building.mesh

        st.sidebar.success("Pressure prediction done!")

    st.sidebar.markdown("### Pressure Maxima Settings")

    # Slider for number of neighbors
    k_neighbors = st.sidebar.slider("Number of neighbors (k)", min_value=1, max_value=200, value=100, step=1)

    # Slider for minimum distance from existing taps
    min_dist_to_tap = st.sidebar.slider("Minimum distance from existing taps", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

    # Slider for percentile threshold
    percentile_thresh = st.sidebar.slider("Percentile threshold", min_value=0, max_value=100, value=50, step=1)

    # --- Find maxima ---
    findMaxima = st.sidebar.button("Find Pressure Maxima")
    if findMaxima:
        if "ypred" in st.session_state:
            ypred = st.session_state.ypred
            vertex_pressures = st.session_state.vertex_pressures
            
            maxima_idx = TaPredictPressures.find_local_maxima_away_from_taps(
                st.session_state.mesh_pcd_pressures,
                st.session_state.ypred,
                taps=np.asarray(st.session_state.building.PredictedTapPCD.points),
                k=k_neighbors,
                percentile=percentile_thresh,
                min_dist_to_tap=min_dist_to_tap
            )
            
            maxima_points = np.asarray(st.session_state.mesh_pcd_pressures.points)[maxima_idx]
            st.session_state.maxima_points = maxima_points
            st.sidebar.success(f"Found {len(maxima_idx)} local maxima")
        else:
            st.sidebar.error("Run pressure prediction first!")
    # --- Add maxima to PredictedTapPCD ---
    addMaxima = st.sidebar.button("Add Maxima to Taps")
    if addMaxima:
        if "maxima_points" in st.session_state:
            maxima_points = st.session_state.maxima_points
            TaPredictPressures.add_maxima_to_taps(st.session_state.building, maxima_points)
            st.sidebar.success(f"Added {len(maxima_points)} maxima to PredictedTapPCD")
        else:
            st.sidebar.error("Run 'Find Pressure Maxima' first!")

    # --- Undo last maxima addition ---
    undoMaxima = st.sidebar.button("Undo Last Maxima Addition")
    if undoMaxima:
        TaPredictPressures.undo_last_added_maxima(st.session_state.building)
        st.sidebar.warning("Undid last maxima addition")

# --- Plot 3D ---
show_colored = st.checkbox("Show pressure colors", value=True)
show_maxima = st.sidebar.checkbox("Show Predicted Maxima", value=True)

mesh_vertices = np.asarray(st.session_state.building.mesh.vertices)
mesh_triangles = np.asarray(st.session_state.building.mesh.triangles)
tap_points = np.asarray(st.session_state.building.PredictedTapPCD.points if st.session_state.building.PredictedTapPCD else np.zeros((0,3)))

fig = go.Figure()

if show_colored:
    try:
        mesh_colors = np.asarray(st.session_state.building.mesh.vertex_colors)

        fig.add_trace(go.Mesh3d(
            x=mesh_vertices[:,0],
            y=mesh_vertices[:,1],
            z=mesh_vertices[:,2],
            i=mesh_triangles[:,0],
            j=mesh_triangles[:,1],
            k=mesh_triangles[:,2],
            vertexcolor=[f'rgb({r*255},{g*255},{b*255})' for r,g,b in mesh_colors],
            opacity=1.0
        ))
    except: pass

else:
    fig.add_trace(go.Mesh3d(
        x=mesh_vertices[:,0],
        y=mesh_vertices[:,1],
        z=mesh_vertices[:,2],
        i=mesh_triangles[:,0],
        j=mesh_triangles[:,1],
        k=mesh_triangles[:,2],
        color="darkgrey",
        opacity=1.0
    ))

try:
    if show_maxima and "maxima_points" in st.session_state:
        maxima_pts = st.session_state.maxima_points
        if len(maxima_pts) > 0:
            fig.add_trace(go.Scatter3d(
                x=maxima_pts[:,0],
                y=maxima_pts[:,1],
                z=maxima_pts[:,2],
                mode="markers",
                marker=dict(size=15, color="white"),
                name="Predicted Maxima"
            ))
except: pass

if tap_points.shape[0] > 0:
    fig.add_trace(go.Scatter3d(
        x=tap_points[:,0],
        y=tap_points[:,1],
        z=tap_points[:,2],
        mode="markers",
        marker=dict(size=3, color="teal"),
        name="Predicted Taps"
    ))

if highlighted_z_idx is not None:
    z_wire = st.session_state.building.ZLevels[highlighted_z_idx].wire
    z_pts = np.asarray(z_wire.points)
    lines = np.asarray(z_wire.lines)

    # Flatten all lines into X, Y, Z arrays with None separators
    x, y, z = [], [], []
    for line in lines:
        x.extend([z_pts[line[0],0], z_pts[line[1],0], None])
        y.extend([z_pts[line[0],1], z_pts[line[1],1], None])
        z.extend([z_pts[line[0],2], z_pts[line[1],2], None])

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(width=2, color="red"),
        name=f"Highlighted Z-Level {highlighted_z_idx}"
    ))

# Overlay highlighted Roof pcd
if highlighted_r_idx is not None:
    r_pcd_pts = np.asarray(st.session_state.building.Roofs[highlighted_r_idx].pcd.points)
    fig.add_trace(go.Scatter3d(
        x=r_pcd_pts[:,0],
        y=r_pcd_pts[:,1],
        z=r_pcd_pts[:,2],
        mode="markers",
        marker=dict(size=5, color="red"),
        name=f"Highlighted Roof {highlighted_r_idx}"
    ))

fig.update_layout(
    scene=dict(
        aspectmode="data",
        camera=st.session_state.camera_state  # restore camera if exists
    ),
    margin=dict(l=0,r=0,t=0,b=0),
    height=900,
)

plot = st.plotly_chart(fig, use_container_width=True)

relayout_data = st.session_state.get("relayout_data", None)
if relayout_data and "scene.camera" in relayout_data:
    st.session_state.camera_state = relayout_data["scene.camera"]



