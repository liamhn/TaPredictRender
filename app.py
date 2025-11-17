import streamlit as st
import os, tempfile
import numpy as np
import open3d as o3d
import pyvista as pv
from stpyvista import stpyvista

st.set_page_config(page_title="3D Mesh Viewer", layout="wide")

st.title("Interactive Mesh Viewer (with Picking)")

# Detect Render environment
ON_RENDER = os.getenv("RENDER") is not None
st.write(f"Running on Render: **{ON_RENDER}**")

uploaded = st.file_uploader("Upload a mesh file", type=["stl", "obj", "ply", "clad"])

picked_point = st.empty()

if uploaded:

    # Save file
    temp_dir = "/tmp" if ON_RENDER else tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, uploaded.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded.read())

    # Load using Open3D
    mesh_o3d = o3d.io.read_triangle_mesh(temp_path)
    mesh_o3d.compute_vertex_normals()

    vertices = np.asarray(mesh_o3d.vertices)
    triangles = np.asarray(mesh_o3d.triangles)

    # PyVista format: prepend "3" before each triangle
    faces_pv = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).astype(np.int32).flatten()

    # Create PyVista mesh
    pv_mesh = pv.PolyData(vertices, faces_pv)

    # Setup plotter
    plotter = pv.Plotter(window_size=[900, 600])
    plotter.add_mesh(pv_mesh, show_edges=True)
    plotter.set_background("white")

    # --- FIX BAD ZOOM ---
    plotter.enable_trackball_style()  # smoother orbit + zoom
    plotter.camera_position = "iso"
    plotter.camera.clipping_range = (0.01, 10000)
    plotter.camera.view_angle = 30
    plotter.camera.zoom(0.7)
    # --------------------

    # Enable picking
    def on_pick(picker, event):
        pid = picker.point_id
        if pid != -1:
            coord = pv_mesh.points[pid]
            picked_point.markdown(f"### Picked point: `{coord}`")

    plotter.enable_point_picking(
        callback=on_pick,
        show_message=True,
        show_point=True
    )

    # Render in Streamlit
    stpyvista(plotter)

else:
    st.info("Upload a mesh file to begin.")
