import matplotlib.pyplot as plt
import gmsh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_nodal_set(mesh):
    x = mesh.x.flatten()
    y = mesh.y.flatten()
    z = mesh.z.flatten()

    vx = mesh.x_vertex 
    vy = mesh.y_vertex 
    vz = mesh.z_vertex 

    r = mesh.reference_element.r
    s = mesh.reference_element.s
    t = mesh.reference_element.t

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # Plot the nodal points
    ax.scatter(x, y, z, label='Nodal Points')
    ax.scatter(vx, vy, vz, color='r', label='Vertices')
    ax.scatter(r, s, t, color='b', label='reference')

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
    
    mesh = Mesh3d(mesh_file)

    plot_nodal_set(mesh)

