import pyvista as pv

# Load the mesh from the GMSH file
mesh = pv.read("simple-mesh.msh")  # Assuming the mesh is in .msh format

# Create a plotter instance
plotter = pv.Plotter()

# Add the mesh to the plotter and set the color
plotter.add_mesh(mesh,
                 color="cyan",
                 show_edges=True,
                 style="wireframe",
                 line_width="1")

# Export the visualization as a standalone HTML file
plotter.export_gltf("visualization.gltf")
