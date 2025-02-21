import matplotlib.pyplot as plt
import gmsh
import pyvista as pv
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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


def plot_nodal_set_pyvista(mesh):
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
   
def plot_nodal_set_matplotlib(mesh):
    x = mesh.x.flatten()
    y = mesh.y.flatten()
    z = mesh.z.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # Plot the nodal points
    ax.scatter(x, y, z, label='Nodal Points')

    # Plot the edges
    edge_coords = []
    for edge in mesh.edgeVertices:
        # Get the coordinates of the vertices for each edge
        edge_coords.append([(vx[edge[0]], vy[edge[0]], vz[edge[0]]),
                            (vx[edge[1]], vy[edge[1]], vz[edge[1]])])
    
    # Create a Line3DCollection from the edge coordinates
    edges = Line3DCollection(edge_coords, colors='k', linewidths=1, linestyles='solid')
    ax.add_collection3d(edges)

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    import gmsh
    import sys
    import os
    sys.path.append('/home/lj/writing/govango/govango-code/finite-elements-from-scratch-in-python')
    
    from mesh import Mesh3d 
    from finite_elements import LagrangeElement

    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    else:
        gmsh.initialize()
        gmsh.model.add("simple")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 2.0)
        #gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write("default.msh")
        mesh_file = "default.msh"
        gmsh.finalize()
    
    dim = 3
    n = 3
    mesh = Mesh3d(mesh_file, LagrangeElement(dim,n))

    #plot_nodal_set_matplotlib(mesh)
    plot_nodal_set_pyvista(mesh)

