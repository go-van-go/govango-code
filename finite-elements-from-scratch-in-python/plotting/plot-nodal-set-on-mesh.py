import matplotlib.pyplot as plt
import gmsh
import pyvista as pv
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_nodal_set_pyvista(mesh):
    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Extract nodal points
    nodal_points = np.column_stack((mesh.x.flatten(), mesh.y.flatten(), mesh.z.flatten()))
    
    # Add the nodal points
    plotter.add_points(nodal_points, color="red", point_size=5, render_points_as_spheres=True)

    # Add edges
    for edge in mesh.edgeVertices:
        p1 = (mesh.x_vertex[edge[0]], mesh.y_vertex[edge[0]], mesh.z_vertex[edge[0]])
        p2 = (mesh.x_vertex[edge[1]], mesh.y_vertex[edge[1]], mesh.z_vertex[edge[1]])
        line = pv.Line(p1, p2)
        plotter.add_mesh(line, color="black", line_width=2)

    # Show the plot
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
    n = 5
    mesh = Mesh3d(mesh_file, LagrangeElement(dim,n))

    #plot_nodal_set_matplotlib(mesh)
    plot_nodal_set_pyvista(mesh)

