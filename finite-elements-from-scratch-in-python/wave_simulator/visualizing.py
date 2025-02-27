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

       
def plot_reference_nodes_3d():
    element = LagrangeElement(d=3, n=20)
    nodes = element.nodes  # Retrieve precomputed nodes
    
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
    if hasattr(mesh, 'boundary_node_ids') and mesh.boundary_node_ids is not None:
        # Extract x, y, z coordinates for the boundary nodes
        x_coords = mesh.x.flatten(order="F")[mesh.boundary_node_ids]
        y_coords = mesh.y.flatten(order="F")[mesh.boundary_node_ids]
        z_coords = mesh.z.flatten(order="F")[mesh.boundary_node_ids]
        
        # Stack into boundary nodal points
        boundary_points_to_plot = np.column_stack((x_coords, y_coords, z_coords))
        
        # Add the boundary points to the plot
        plotter.add_points(boundary_points_to_plot, color="green", point_size=10, render_points_as_spheres=True)


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


def plot_normals(plotter, mesh, norms):
    """Plot normal vectors for specified elements."""
    if norms is not None:
        # Extract face node coordinates using face_node_indices
        face_node_indices = mesh.ReferenceElement.face_node_indices
        
        for elem in norms:
            x_origin = mesh.x[:, elem][face_node_indices].flatten()
            y_origin = mesh.y[:, elem][face_node_indices].flatten()
            z_origin = mesh.z[:, elem][face_node_indices].flatten()
            
            # Stack into origin points
            normal_vector_origin = np.column_stack((x_origin, y_origin, z_origin))
            
            # Extract normal vectors for the given element
            nx = mesh.nx[:, elem]
            ny = mesh.ny[:, elem]
            nz = mesh.nz[:, elem]
            
            # Stack into normal vectors
            normal_vector_direction = np.column_stack((nx, ny, nz))
            
            # Add normal vectors as arrows
            plotter.add_arrows(normal_vector_origin, normal_vector_direction, mag=0.1, color='red')


def plot_solution(plotter, mesh, solution):
    """Plot nodes on the mesh with colors based on the solution values."""
    # Extract x, y, z coordinates for the nodes in the specified elements
    x_coords = mesh.x.flatten()
    y_coords = mesh.y.flatten()
    z_coords = mesh.z.flatten()
    
    # Stack into nodal points
    points_to_plot = np.column_stack((x_coords, y_coords, z_coords))
    
    # Flatten the solution matrix to align with the coordinates
    solution_values = solution.flatten()
    
    # Add the points to the plot with colors based on the solution values
    # Add the points to the plot with colors based on the solution values
    plotter.add_points(
        points_to_plot,
        scalars=solution_values,
        cmap="viridis",  # Use any colormap you prefer
        clim=(-1, 1),  # Fix the color bounds
        point_size=10,
        render_points_as_spheres=True
    )


def visualize_mesh(mesh,
                   elements=[],
                   norms=[],
                   nodes=[],
                   solution=[],
                   boundary_nodes=False,
                   boundary_elements=False,
                   file_name = "mesh",
                   save=False):
    """Visualize the mesh with nodes, elements, and normals."""
    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Create PyVista UnstructuredGrid
    cells = np.hstack([np.full((mesh.num_cells, 1), 4), mesh.cell_to_vertices]).flatten()
    cell_types = np.full(mesh.num_cells, pv.CellType.TETRA)  # Tetrahedral elements
    grid = pv.UnstructuredGrid(cells, cell_types, mesh.vertex_coordinates)

    # Add mesh
    plotter.add_mesh(grid, style='wireframe', color='black')

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
    if norms:
        plot_normals(plotter, mesh, norms)

    # plot solution
    if solution.any():
        plot_solution(plotter, mesh, solution)

    plotter.show_grid()
    # Export and show the plot
    if save:
        plotter.export_gltf(f"./outputs/{file_name}.gltf")  # Supports .glb
    else:
        plotter.show()


if __name__ == "__main__":
    from simulator import Simulator
    from mesh import Mesh3d
    from finite_elements import LagrangeElement
    from reference_element_operators import ReferenceElementOperators
    breakpoint()
