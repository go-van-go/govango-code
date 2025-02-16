import pyvista as pv
import numpy as np


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
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write("default.msh")
        mesh_file = "default.msh"
        gmsh.finalize()
    
    mesh = Mesh3d(mesh_file)
    cell_number = 1604   # the cell to visualize
    adjacent_cells = mesh.cell_to_cells[cell_number]
    all_cells = np.insert(adjacent_cells, 0, cell_number)
    all_cells, indices = np.unique(all_cells, return_index=True)
    all_cells = all_cells[np.argsort(indices)]  # Restore original order

    plot_mesh(mesh, all_cells)

