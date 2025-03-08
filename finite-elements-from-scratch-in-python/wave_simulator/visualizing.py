import matplotlib.pyplot as plt
import gmsh
import pyvista as pv
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
#from wave_simulator.finite_elements import LagrangeElement

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

       
def plot_reference_nodes_3d(finite_element):
    #finite_element = LagrangeElement(d=3, n=20)
    nodes = finite_element.nodes  # Retrieve precomputed nodes
    
    # Create a PyVista point cloud
    point_cloud = pv.PolyData(nodes)
    
    # Create a plotter
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, color='red', point_size=10, render_points_as_spheres=True)
    
    # Add axes labels
    plotter.show_grid()
    
    plotter.show(title="Lagrange Element Nodes (d=3, n=20)")


def plot_nodes(plotter, mesh, nodes):
    """Plot nodes on the mesh."""
    if nodes is not None:
        # Extract x, y, z coordinates for the nodes in the specified elements
        x_coords = mesh.x[:, nodes].flatten()
        y_coords = mesh.y[:, nodes].flatten()
        z_coords = mesh.z[:, nodes].flatten()
        
        # Stack into nodal points
        points_to_plot = np.column_stack((x_coords, y_coords, z_coords))
        
        # Add the points to the plot
        plotter.add_points(points_to_plot, color="blue", point_size=10, render_points_as_spheres=True)


def plot_boundary_nodes(plotter, mesh):
    """Plot boundary nodes on the mesh."""
    
    # Extract x, y, z coordinates for the boundary nodes
    x_coords = mesh.x.flatten(order="F")[mesh.boundary_node_indices]
    y_coords = mesh.y.flatten(order="F")[mesh.boundary_node_indices]
    z_coords = mesh.z.flatten(order="F")[mesh.boundary_node_indices]
    
    # Stack into boundary nodal points
    boundary_points_to_plot = np.column_stack((x_coords, y_coords, z_coords))
    
    # Add the boundary points to the plot
    plotter.add_points(boundary_points_to_plot, color="green", point_size=10, render_points_as_spheres=True)


def plot_boundary_jumps(plotter, mesh, jumps):
    """Plot boundary nodes on the mesh."""
    
    # Extract x, y, z coordinates for the boundary nodes
    x_coords = mesh.x.flatten(order="F")[mesh.boundary_node_indices]
    y_coords = mesh.y.flatten(order="F")[mesh.boundary_node_indices]
    z_coords = mesh.z.flatten(order="F")[mesh.boundary_node_indices]

    jump_values = np.ravel(jumps, order='F')[mesh.boundary_face_node_indices]
    
    # Stack into boundary nodal points
    boundary_jumps_to_plot = np.column_stack((x_coords, y_coords, z_coords))
    # Compute opacity: fully opaque (1) if value is 0, otherwise scaled by |value|
    opacity_values = np.abs(jump_values)  # Ranges from 0 to 1
    opacity_values[jump_values == 0] = 0  # Ensure zero values are fully opaque
    
    # Add the points to the plot with colors and opacity
    plotter.add_points(
        boundary_jumps_to_plot,
        scalars=jump_values,
        cmap="viridis",  # Use any colormap you prefer
        #clim=(-1, 1),  # Fix the color bounds
        opacity=opacity_values,  # Set per-point opacity
        point_size=10,
        render_points_as_spheres=True
    )


def plot_jumps(plotter, mesh, jumps):
    """Plot boundary nodes on the mesh."""
     
    # Extract x, y, z coordinates for the boundary nodes
    x_coords = mesh.x.ravel(order="F")[mesh.interior_face_node_indices]
    y_coords = mesh.y.ravel(order="F")[mesh.interior_face_node_indices]
    z_coords = mesh.z.ravel(order="F")[mesh.interior_face_node_indices]

    jump_values = np.ravel(jumps, order='F')
    
    # Stack into boundary nodal points
    jumps_to_plot = np.column_stack((x_coords, y_coords, z_coords))
    # Compute opacity: fully opaque (1) if value is 0, otherwise scaled by |value|
    opacity_values = np.abs(jump_values)  # Ranges from 0 to 1
    
    # Add the points to the plot with colors and opacity
    plotter.add_points(
        jumps_to_plot,
        scalars=jump_values,
        cmap="viridis",  # Use any colormap you prefer
        #clim=(-1, 1),  # Fix the color bounds
        opacity=opacity_values,  # Set per-point opacity
        point_size=10,
        render_points_as_spheres=True
    )


    pass

def plot_boundary_elements(plotter, mesh):
    """Plot elements that lie on the boundary of the mesh."""
    if hasattr(mesh, 'cell_to_faces') and mesh.cell_to_faces is not None:
        # Get Jacobian values for all cells (using first element of each column)
        jacobian_values = mesh.jacobians[0, :]
        
        # Normalize Jacobian values to create a color map
        cmap = plt.cm.viridis  # You can choose any colormap
        norm = plt.Normalize(vmin=np.min(jacobian_values), vmax=np.max(jacobian_values))
        
        # Iterate over each cell and its faces
        for cell_idx, faces in enumerate(mesh.cell_to_cells):
            # Check if any face in the cell lies on the boundary
            if cell_idx in faces:
                # Get the Jacobian value for the current cell
                jacobian_value = jacobian_values[cell_idx]
                
                # Normalize and map the Jacobian value to color
                color = cmap(norm(jacobian_value))[:3]  # Use only the RGB channels
                
                # Create the mesh for the highlighted cell
                boundary_element_grid = pv.UnstructuredGrid(
                    np.hstack([[4], mesh.cell_to_vertices[cell_idx]]).flatten(),
                    [pv.CellType.TETRA], mesh.vertex_coordinates
                )
                
                # Apply color based on the Jacobian value
                plotter.add_mesh(boundary_element_grid, color=color, opacity=0.1)


def plot_elements(plotter, mesh, elements):
    """Highlight specific elements on the mesh."""
    if elements is not None:
        # Get Jacobian values for all cells (using first element of each column)
        jacobian_values = mesh.jacobians[0, :]
        
        # Normalize Jacobian values to create a color map
        cmap = plt.cm.viridis  # You can choose any colormap
        norm = plt.Normalize(vmin=np.min(jacobian_values), vmax=np.max(jacobian_values))
        
        for cell in elements:
            # Get the Jacobian value for the current cell
            jacobian_value = jacobian_values[cell]
            
            # Normalize and map the Jacobian value to color
            color = cmap(norm(jacobian_value))[:3]  # Use only the RGB channels
            
            # Create the mesh for the highlighted cell
            highlight_grid = pv.UnstructuredGrid(
                np.hstack([[4], mesh.cell_to_vertices[cell]]).flatten(),
                [pv.CellType.TETRA], mesh.vertex_coordinates
            )
            
            # Apply color based on the Jacobian value
            plotter.add_mesh(highlight_grid, color=color, opacity=0.5)


def plot_normals(plotter, mesh, normals):
    face_node_indices = mesh.reference_element.face_node_indices
    for cell in normals:
        x_origin = mesh.x[:, cell][face_node_indices].ravel(order='F')
        y_origin = mesh.y[:, cell][face_node_indices].ravel(order='F')
        z_origin = mesh.z[:, cell][face_node_indices].ravel(order='F')
        
        # Stack into origin points
        normal_vector_origin = np.column_stack((x_origin, y_origin, z_origin))
        
        # Extract normal vectors for the given element
        nx = mesh.nx[:, cell]
        ny = mesh.ny[:, cell]
        nz = mesh.nz[:, cell]
        
        # Stack into normal vectors
        normal_vector_direction = np.column_stack((nx, ny, nz))
        
        # Add normal vectors as arrows
        plotter.add_arrows(normal_vector_origin, normal_vector_direction, mag=0.05, color='red')


def plot_boundary_normals(plotter, mesh):
    """Plot normal vectors for specified elements."""
    # Extract face node coordinates using face_node_indices
    boundary_node_indices = mesh.boundary_node_indices
    boundary_face_node_indices = mesh.boundary_face_node_indices
    # Iterate over each cell and its faces
    
    x_origin = mesh.x.ravel(order='F')[boundary_node_indices]
    y_origin = mesh.y.ravel(order='F')[boundary_node_indices]
    z_origin = mesh.z.ravel(order='F')[boundary_node_indices]
    
    normal_vector_origin = np.column_stack((x_origin, y_origin, z_origin))
    # Extract normal vectors for the given element
    nx = mesh.nx.ravel(order='F')[boundary_face_node_indices]
    ny = mesh.ny.ravel(order='F')[boundary_face_node_indices]
    nz = mesh.nz.ravel(order='F')[boundary_face_node_indices]
    
    # Stack into normal vectors
    normal_vector_direction = np.column_stack((nx, ny, nz))

    plotter.add_arrows(normal_vector_origin, normal_vector_direction, mag=0.05, color='red')
 

       # for elem in norms:
       #     x_origin = mesh.x[:, elem][face_node_indices].flatten()
       #     y_origin = mesh.y[:, elem][face_node_indices].flatten()
       #     z_origin = mesh.z[:, elem][face_node_indices].flatten()
       #     
       #     # Stack into origin points
       #     normal_vector_origin = np.column_stack((x_origin, y_origin, z_origin))
       #     
       #     # Extract normal vectors for the given element
       #     nx = mesh.nx[:, elem]
       #     ny = mesh.ny[:, elem]
       #     nz = mesh.nz[:, elem]
       #     
       #     # Stack into normal vectors
       #     normal_vector_direction = np.column_stack((nx, ny, nz))
       #     
       #     # Add normal vectors as arrows
       #     plotter.add_arrows(normal_vector_origin, normal_vector_direction, mag=0.1, color='red')


def plot_solution(plotter, mesh, solution):
    """Plot nodes on the mesh with colors and opacity based on solution values."""
    # Extract x, y, z coordinates for the nodes
    interior_node_indices = mesh.interior_face_node_indices
    exterior_node_indices = mesh.exterior_face_node_indices
    x_coords = np.ravel(mesh.x, order='F')
    y_coords = np.ravel(mesh.y, order='F')
    z_coords = np.ravel(mesh.z, order='F')
    
    # Stack into nodal points
    points_to_plot = np.column_stack((x_coords, y_coords, z_coords))
    
    # Flatten the solution matrix to align with the coordinates
    solution_values = np.ravel(solution, order='F')
    # Assuming solution_values is a 1D array
    all_indices = np.arange(len(solution_values))  # Get all indices of solution_values
    non_boundary_indices = np.setdiff1d(all_indices, mesh.boundary_node_indices)

    #solution_values[non_boundary_indices] = 0
    #solution_values[mesh.boundary_node_indices] = 0
    solution_values[interior_node_indices] = (solution_values[interior_node_indices] + \
                                               solution_values[exterior_node_indices]) / 2 
    solution_values[exterior_node_indices] = 0
    
    # Compute opacity: fully opaque (1) if value is 0, otherwise scaled by |value|
    opacity_values = np.abs(solution_values)  # Ranges from 0 to 1
    #opacity_values[solution_values == 0] = 0  # Ensure zero values are fully opaque
    
    # Add the points to the plot with colors and opacity
    plotter.add_points(
        points_to_plot,
        scalars=solution_values,
        cmap="viridis",  # Use any colormap you prefer
        #clim=(-1, 1),  # Fix the color bounds
        opacity=opacity_values,  # Set per-point opacity
        point_size=10,
        render_points_as_spheres=True
    )

def plot_mesh_edges(plotter, mesh):
    # Create PyVista UnstructuredGrid for each element
    for cell in range(mesh.num_cells):
        grid = pv.UnstructuredGrid(
            np.hstack([[4], mesh.cell_to_vertices[cell]]).flatten(),
            [pv.CellType.TETRA], mesh.vertex_coordinates
        )
        plotter.add_mesh(grid, style='wireframe', color='black', opacity=0.9)


def plot_mesh_boundary(plotter, mesh):
    # Plot mesh edges on boundary
    cells = np.hstack([np.full((mesh.num_cells, 1), 4), mesh.cell_to_vertices]).flatten()
    cell_types = np.full(mesh.num_cells, pv.CellType.TETRA)  # Tetrahedral elements
    grid = pv.UnstructuredGrid(cells, cell_types, mesh.vertex_coordinates)
    # Add mesh
    plotter.add_mesh(grid, style='wireframe', color='black')


def visualize_mesh(mesh,
                   elements=[],
                   normals=[],
                   nodes=[],
                   solution= np.array([]),
                   jumps=np.array([]),
                   boundary_jumps=np.array([]),
                   boundary_nodes=False,
                   boundary_elements=False,
                   boundary_normals=False,
                   mesh_edges=False,
                   mesh_boundary=False,
                   file_name = "mesh",
                   save=False):
    """Visualize the mesh with nodes, elements, and normals."""
    # Create a PyVista plotter
    plotter = pv.Plotter(off_screen=save)

    # Add mesh cell edges
    if mesh_edges:
        plot_mesh_edges(plotter, mesh)
        
    # Add mesh edges on bondary
    if mesh_boundary:
        plot_mesh_boundary(plotter, mesh)

    # Plot nodes
    if nodes:
        plot_nodes(plotter, mesh, nodes)

    # Plot boundary nodes if requested
    if boundary_nodes:
        plot_boundary_nodes(plotter, mesh)

    # Plot boundary faces if requested
    if boundary_elements:
        plot_boundary_elements(plotter, mesh)

    # Plot elements
    if elements:
        plot_elements(plotter, mesh, elements)

    # Plot normals
    if normals:
        plot_normals(plotter, mesh, normals)

    # Plot boundary normals
    if boundary_normals:
        plot_boundary_normals(plotter, mesh)

    # plot jumps 
    if jumps.any():
        plot_jumps(plotter, mesh, jumps)

    # plot boundary jumps 
    if boundary_jumps.any():
        plot_boundary_jumps(plotter, mesh, jumps)

    # plot solution
    if solution.any():
        plot_solution(plotter, mesh, solution)

    plotter.show_grid()

    # Export and show the plot
    if save:
        plotter.screenshot(f"./outputs/{file_name}")
    else:
        plotter.show()

    plotter.close()  


if __name__ == "__main__":
    from simulator import Simulator
    from mesh import Mesh3d
    from finite_elements import LagrangeElement
    from reference_element_operators import ReferenceElementOperators
    breakpoint()
