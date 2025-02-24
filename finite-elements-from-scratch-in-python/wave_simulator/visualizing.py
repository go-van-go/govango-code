import matplotlib.pyplot as plt
import gmsh
import pyvista as pv
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_mesh_full(element_indicies, normal_vector_indices):
    pass


def visualize_array(array, cmap="viridis", colorbar=True):
    """
    Visualizes a 2D NumPy array using a colormap.

    Parameters:
    - array (numpy.ndarray): A 2D array to visualize.
    - cmap (str): Colormap name (default is 'viridis').
    - colorbar (bool): Whether to show the colorbar (default is True).
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2D")

    plt.figure(figsize=(6, 5))
    plt.imshow(array, cmap=cmap, aspect='auto')
    
    if colorbar:
        plt.colorbar()

    plt.title("2D Array Visualization")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()



def plot_norms(mesh):
    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Extract nodal points
    nodal_points = np.column_stack((mesh.x.flatten(), mesh.y.flatten(), mesh.z.flatten()))
    
    # Add the nodal points
    plotter.add_points(nodal_points, color="blue", point_size=10, render_points_as_spheres=True)

    # Add edges
    for edge in mesh.edge_vertices:
        p1 = (mesh.x_vertex[edge[0]], mesh.y_vertex[edge[0]], mesh.z_vertex[edge[0]])
        p2 = (mesh.x_vertex[edge[1]], mesh.y_vertex[edge[1]], mesh.z_vertex[edge[1]])
        line = pv.Line(p1, p2)
        plotter.add_mesh(line, color="black", line_width=2)

    # Extract face node coordinates using face_node_indices
    x_origin = mesh.x[:,0][mesh.face_node_indices].flatten()
    y_origin = mesh.y[:,0][mesh.face_node_indices].flatten()
    z_origin = mesh.z[:,0][mesh.face_node_indices].flatten()
    
    # Stack into origin points
    normal_vector_origin = np.column_stack((x_origin, y_origin, z_origin))
    
    # Extract only the normal vectors for element 1 (second column, index 1)
    nx = mesh.nx[:, 0]
    ny = mesh.ny[:, 0]
    nz = mesh.nz[:, 0]
    
    # Stack into normal vectors
    normal_vector_direction = np.column_stack((nx, ny, nz))
    breakpoint()
    
    # Loop through and add only the vectors from element 1
    plotter.add_arrows(normal_vector_origin, normal_vector_direction, mag=0.3, color='red')
    
    plotter.show()
 

def plot_mesh(mesh, highlight_cells=None):
    # Create PyVista UnstructuredGrid
    cells = np.hstack([np.full((mesh.num_cells, 1), 4), mesh.cell2vertices]).flatten()
    cell_types = np.full(mesh.num_cells, pv.CellType.TETRA)  # Tetrahedral elements
    
    grid = pv.UnstructuredGrid(cells, cell_types, mesh.vertexCoordinates)
    plotter = pv.Plotter()
    
    # Plot mesh as wireframe
    plotter.add_mesh(grid, style='wireframe', color='black')
    
    # Highlight specific elements if provided
    if highlight_cells is not None:
        # First element in red
        first_cell = highlight_cells[0]
        highlight_grid = pv.UnstructuredGrid(
            np.hstack([[4], mesh.cell2vertices[first_cell]]).flatten(),
            [pv.CellType.TETRA], mesh.vertexCoordinates
        )
        plotter.add_mesh(highlight_grid, color='#ebcb8b', opacity=1.0)
        
        # Remaining elements in blue
        for cell in highlight_cells[1:]:
            highlight_grid = pv.UnstructuredGrid(
                np.hstack([[4], mesh.cell2vertices[cell]]).flatten(),
                [pv.CellType.TETRA], mesh.vertexCoordinates
            )
            plotter.add_mesh(highlight_grid, color='5e81ac', opacity=0.5)
    plotter.export_gltf("adjacent-cells.gltf")  # Supports .glb
    
    plotter.show()

def plot_lagrange_nodes():
    element = LagrangeElement(d=3, n=20)
    nodes = element.nodes  # Retrieve precomputed nodes
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='r', marker='o', label='Lagrange Nodes')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Lagrange Element Nodes (d=3, n=20)")
    
    ax.legend()
    plt.show()


def plot_reference_tetrahedron():
    # Define vertices of the tetrahedron
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Define edges as pairs of vertices
    edges = [
        (vertices[0], vertices[1]),
        (vertices[0], vertices[2]),
        (vertices[0], vertices[3]),
        (vertices[1], vertices[2]),
        (vertices[1], vertices[3]),
        (vertices[2], vertices[3])
    ]
    
    # Create a figure with a 3D axis
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("#eceff4")  # Set background color
    
    # Convert edges to a Line3DCollection and add to plot
    edge_collection = Line3DCollection(edges, colors='k', linewidths=2)
    ax.add_collection3d(edge_collection)
    
    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="#5e81ac", s=1000)
    
    # Set axis limits
    ax.set_xlim([-.2, 1.2])
    ax.set_ylim([-.2, 1.2])
    ax.set_zlim([-.2, 1.2])
    
    # Show axes but remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plt.show()

if __name__ == "__main__":
    from simulator import Simulator
    from mesh import Mesh3d
    from finite_elements import LagrangeElement
    from reference_element_operators import ReferenceElementOperators
    breakpoint()
