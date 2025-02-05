import pyvista as pv
import numpy as np


def plot_mesh(mesh, highlight_elements=None):
    # Create PyVista UnstructuredGrid
    cells = np.hstack([np.full((mesh.num_elements, 1), 4), mesh.element2vertices]).flatten()
    cell_types = np.full(mesh.num_elements, pv.CellType.TETRA)  # Tetrahedral elements
    
    grid = pv.UnstructuredGrid(cells, cell_types, mesh.vertexCoordinates)
    plotter = pv.Plotter()
    
    # Plot full mesh as wireframe
    plotter.add_mesh(grid, style='wireframe', color='black')
    
    # Highlight specific elements if provided
    if highlight_elements is not None:
        # First element in red
        first_elem = highlight_elements[0]
        highlight_grid = pv.UnstructuredGrid(
            np.hstack([[4], mesh.element2vertices[first_elem]]).flatten(),
            [pv.CellType.TETRA], mesh.vertexCoordinates
        )
        plotter.add_mesh(highlight_grid, color='#ebcb8b', opacity=1.0)
        
        # Remaining elements in blue
        for elem in highlight_elements[1:]:
            highlight_grid = pv.UnstructuredGrid(
                np.hstack([[4], mesh.element2vertices[elem]]).flatten(),
                [pv.CellType.TETRA], mesh.vertexCoordinates
            )
            plotter.add_mesh(highlight_grid, color='5e81ac', opacity=0.5)
    
    plotter.show()

if __name__ == "__main__":
    import gmsh
    import sys
    import os
    sys.path.append('/home/lj/writing/govango/govango-code/finite-elements-from-scratch-in-python/meshes')
    
    from mesh import Mesh3d 

    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    else:
        gmsh.initialize()
        gmsh.model.add("simple")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        #gmsh.option.setNumber("Mesh.MeshSizeMin", 2.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)   # Keep max size small as well
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write("default.msh")
        mesh_file = "default.msh"
        gmsh.finalize()
    
    mesh = Mesh3d(mesh_file)
    element_number = 3679
    adjacent_elements = mesh.element2elements[element_number]
    all_elements = np.insert(adjacent_elements, 0, element_number)
    all_elements, indices = np.unique(all_elements, return_index=True)
    all_elements = all_elements[np.argsort(indices)]  # Restore original order

    plot_mesh(mesh, all_elements)

