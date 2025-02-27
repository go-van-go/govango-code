from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.reference_element_operators import ReferenceElementOperators
from wave_simulator.mesh import Mesh3d 
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta

import pdb
import pickle
import numpy as np
from numpy import nan
import pyvista as pv
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator


# Load the saved instance
with open('./outputs/sim_data_00000100.pkl', 'rb') as file:
    data = pickle.load(file)

cell = 2
solution = data.physics.p[:,cell]
x = data.physics.mesh.x[:,cell]
y = data.physics.mesh.y[:,cell]
z = data.physics.mesh.z[:,cell]

vertices = data.physics.mesh.cell_to_vertices[cell]
x_vertices = data.physics.mesh.x_vertex[vertices]
y_vertices = data.physics.mesh.y_vertex[vertices]
z_vertices = data.physics.mesh.z_vertex[vertices]

# Define tetrahedral domain
tetra_points = np.vstack((x_vertices, y_vertices, z_vertices)).T
tetra = Delaunay(tetra_points)

# Interpolator
interp = LinearNDInterpolator(list(zip(x, y, z)), solution)

# Generate a grid inside the tetrahedron
num_points = 30
xg, yg, zg = np.meshgrid(
    np.linspace(x_vertices.min(), x_vertices.max(), num_points),
    np.linspace(y_vertices.min(), y_vertices.max(), num_points),
    np.linspace(z_vertices.min(), z_vertices.max(), num_points)
)

grid_points = np.vstack((xg.ravel(), yg.ravel(), zg.ravel())).T
inside = tetra.find_simplex(grid_points) >= 0  # Check if inside tetrahedron
valid_points = grid_points[inside]
values = interp(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2])

# Create PyVista Unstructured Grid
cloud = pv.PolyData(valid_points)
cloud["solution"] = values

# Visualization
plotter = pv.Plotter()
plotter.add_mesh(cloud, scalars="solution", cmap="viridis", point_size=5)
plotter.add_mesh(pv.PolyData(tetra_points), color="red", point_size=10, render_points_as_spheres=True)
plotter.show()
