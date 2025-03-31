import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import gmsh
import math
from wave_simulator.reference_element_operators import ReferenceElementOperators
from wave_simulator.finite_elements import LagrangeElement

class Visualizer:
    def __init__(self, time_stepper, grid=True):
        self.time_stepper = time_stepper
        self.physics = time_stepper.physics
        self.mesh = time_stepper.physics.mesh
        self.plotter = pv.Plotter()
        camera_position = [(3.40, 1.30, 1.10), (0.0, 0.3, 0.3), (0, 0, 1)]
        self.plotter.camera_position = camera_position
        self.get_domain_parameters()
        if grid:
            self._show_grid(self.plotter)

    def get_domain_parameters(self):
        # get minimum coordinate values
        self.x_min = np.min(self.mesh.x)
        self.y_min = np.min(self.mesh.y)
        self.z_min = np.min(self.mesh.z)

        # get maximum coordinate values
        self.x_max = np.max(self.mesh.x)
        self.y_max = np.max(self.mesh.y)
        self.z_max = np.max(self.mesh.z)

    def _show_grid(self, plotter):
        plotter.show_grid(bounds=[self.x_min, self.x_max,
                                  self.y_min, self.y_max,
                                  self.z_min, self.z_max])

    def _get_grid_coordinates(self, origin, dimensions, spacing):
        # Generate grid points
        x = np.arange(origin[0], dimensions[0], spacing[0])
        y = np.arange(origin[1], dimensions[1], spacing[1])
        z = np.arange(origin[2], dimensions[2], spacing[2])
        
        # Create mesh grid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Stack into a list of (x, y, z) points
        coordinates = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
        return coordinates

    def _get_voxel_data(self, field, coordinates):
        """ get field value in each 3D voxel"""
        # Initialize 3D numpy array to hold the volume data
        voxel_data = np.zeros(len(coordinates))

        # Iterate through each grid point and fill the 3D array
        for i, (x, y, z) in enumerate(coordinates):
            voxel_data[i] = self.eval_at_point(x, y, z, field)

        return voxel_data

    def _get_grid_parameters(self, resolution):
         # Define parameters
        x_dim = self.x_max - self.x_min
        y_dim = self.y_max - self.y_min
        z_dim = self.z_max - self.z_min
        
        origin = (self.x_min, self.y_min, self.z_min)
        dimensions = (x_dim, y_dim, z_dim)
        spacing = (dimensions[0]/resolution,
                   dimensions[1]/resolution,
                   dimensions[2]/resolution)

        return origin, dimensions, spacing
       
    def add_field_3d(self, field, resolution=40):
        """ Add a 3D field visualization tothe plotter """
        # get grid parameters
        origin, dimensions, spacing = self._get_grid_parameters(resolution)

        # get grid coordinates
        coordinates = self._get_grid_coordinates(origin, dimensions, spacing)

        # get volume data
        volume_data = self._get_voxel_data(field, coordinates)
  
        # Create PyVista ImageData (Structured Grid)
        grid = pv.ImageData(
            origin=origin,
            dimensions=(resolution, resolution, resolution),
            spacing=spacing,
        )
    
        # Assign volume data as the active scalars
        grid.point_data["values"] = volume_data 

        # Add volume to the plotter
        vol = self.plotter.add_volume(grid)

        # Add clipping widgets
        for norm in ['-x', '-y']:
            self.plotter.add_volume_clip_plane(
                vol,
                normal=norm,
                interaction_event='always',
                normal_rotation=False,
            )

    def add_field_point_cloud(self, field, resolution=50):
        """ Add a 3D field visualization to the plotter """
        # Get the volume data, and spacing
        volume_data, _ = self._get_voxel_data(field, coordinates)
            
        # Add the points to the plot with colors and opacity
        self.plotter.add_points(
            coordinates,
            scalars=volume_data,
            cmap="viridis",  # Use any colormap you prefer
            #opacity=opacity_values,#'linear',#0.001,#opacity_values,  # Set per-point opacity
            opacity='linear',
            point_size=10,
            render_points_as_spheres=True
        )

    def get_element(self, x, y, z):
        """ Find which element corresponds to a point in the mesh """
        # dimension
        dim = 3

        # Reason for offset - Gmsh counts lower order elements like points and lines
        # before counting tetrahedrons. This code calls the first tetraheron element '0'
        offset, _ = gmsh.model.mesh.getElementsByType(4)

        # find element
        element = gmsh.model.mesh.getElementByCoordinates(x, y, z, dim)[0] - offset[0]
        return element

    def eval_at_point(self, x, y, z, field):
        """ evaluate a given field at any point in the domain."""
        # get element
        element = self.get_element(x, y, z)

        # get field vlaues
        values = field[:, element]

        # get inverse vandermonde to evaluate weighted basis functions
        invV = self.mesh.reference_element_operators.inverse_vandermonde_3d

        # compute the basis function weights
        weights = invV @ values

        # initialize solution
        solution = 0.0

        # map point to reference tetrahedron
        r, s, t = self._map_to_reference_tetrahedron(x, y, z, element)

        # get polynomial degree
        n = self.mesh.reference_element.n

        # loop over all basis functions 
        column_index = 0
        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    # m is the canonical index for a polynomial basis, expression is found in
                    # Hesthaven and Warburton pg 411
                    m = (
                        1 + (11 + 12*n + 3*n**2) * i / 6 + (2*n + 3) * j / 2 + k
                        - (2 + n) * i**2 / 2 - i * j - j**2 / 2 + i**3 / 6
                    )
                    # adjust for 0 based indexing and turn to integer
                    m = math.ceil(m - 1)
                    # evaluate basis function
                    basis_funciton_contribution = \
                        self.mesh.reference_element.eval_3d_basis_function([r],[s],[t],i,j,k)
                    # add weigted solution to final solution
                    solution += weights[m] * basis_funciton_contribution

        return solution

    def _map_to_reference_tetrahedron(self, x, y, z, cell):
        """
        Maps a point (x_target, y_target, z_target) in physical space to reference
        coordinates (r, s, t) for a tetrahedral element defined by `vertices`.
        This works by inverting the mapping in Hesthaven and Warburton pg 409
        """
        cell_to_vertices = self.mesh.cell_to_vertices

        # get mesh tetrahedron vertices
        vx = self.mesh.x_vertex
        vy = self.mesh.y_vertex
        vz = self.mesh.z_vertex

        # get indices for specific element
        va = cell_to_vertices[cell, 0].T
        vb = cell_to_vertices[cell, 1].T
        vc = cell_to_vertices[cell, 2].T
        vd = cell_to_vertices[cell, 3].T
        
        # Construct Jacobian matrix for the inverse mapping
        J = np.array([
            [vx[vb] - vx[va], vx[vc] - vx[va], vx[vd] - vx[va]],
            [vy[vb] - vy[va], vy[vc] - vy[va], vy[vd] - vy[va]],
            [vz[vb] - vz[va], vz[vc] - vz[va], vz[vd] - vz[va]]
        ])

        # Right-hand side
        b = np.array([
            2*x + vx[va] - vx[vb] - vx[vc] - vx[vd],
            2*y + vy[va] - vy[vb] - vy[vc] - vy[vd],
            2*z + vz[va] - vz[vb] - vz[vc] - vz[vd]
        ])
        
        # Solve for (r, s, t)
        rst = np.linalg.solve(J, b)
        return tuple(rst)

    def add_nodes_3d(self, field):
        """Plot nodes on the mesh with colors and opacity based on solution values."""
        # Extract x, y, z coordinates for the nodes
        x = np.ravel(self.mesh.x, order='F')
        y = np.ravel(self.mesh.y, order='F')
        z = np.ravel(self.mesh.z, order='F')
        
        # Stack into nodal points
        node_coordinates = np.column_stack((x, y, z))
        
        # Flatten the solution matrix to align with the coordinates
        field = np.ravel(field, order='F')

        # index all non boundary indices
        #all_indices = np.arange(len(solution_values))  # Get all indices of solution_values
        #non_boundary_indices = np.setdiff1d(all_indices, mesh.boundary_node_indices)
        #solution_values[non_boundary_indices] = 0

        #solution_values[mesh.boundary_node_indices] = 0
        #solution_values[interior_node_indices] = (solution_values[interior_node_indices] + \
            #                                           solution_values[exterior_node_indices]) / 2 
        #solution_values[exterior_node_indices]=0

        # Add the points to the plot with colors and opacity
        self.plotter.add_points(
            node_coordinates,
            scalars=field,
            cmap="viridis",  # Use any colormap you prefer
            opacity='linear',
            point_size=10,
            render_points_as_spheres=True
        )

    def add_cell_nodes(self, cell_list):
        # Extract x, y, z coordinates for the nodes in the specified elements
        x = self.mesh.x[:, cell_list].flatten()
        y = self.mesh.y[:, cell_list].flatten()
        z = self.mesh.z[:, cell_list].flatten()
        
        # Stack into nodal points
        node_coordinates = np.column_stack((x, y, z))
        
        # Add the points to the plot
        self.plotter.add_points(
            node_coordinates,
            color="blue",
            point_size=10,
            render_points_as_spheres=True
        )

    def add_all_boundary_nodes(self):
        """Plot boundary nodes on the mesh."""
        # Extract x, y, z coordinates for the boundary nodes
        boundary_nodes = self.mesh.boundary_node_indices
        x = self.mesh.x.ravel(order="F")[boundary_nodes]
        y = self.mesh.y.ravel(order="F")[boundary_nodes]
        z = self.mesh.z.ravel(order="F")[boundary_nodes]
        
        # Stack into boundary nodal points
        boundary_points_to_plot = np.column_stack((x, y, z))
        
        # Add the boundary points to the plot
        self.plotter.add_points(
            boundary_points_to_plot,
            color="green",
            point_size=10,
            render_points_as_spheres=True
        )


    def add_cells(self, cell_list):
        """Highlight specific cells on the mesh."""
        # Get Jacobian values for all cells (using first element of each column)
        jacobian_values = self.mesh.jacobians[0, :]
            
        # Normalize Jacobian values to create a color map
        cmap = plt.cm.viridis  # You can choose any colormap
        norm = plt.Normalize(vmin=np.min(jacobian_values), vmax=np.max(jacobian_values))
            
        for cell in cell_list:
            # Get the Jacobian value for the current cell
            jacobian_value = jacobian_values[cell]
                
            # Normalize and map the Jacobian value to color
            color = cmap(norm(jacobian_value))[:3]  # Use only the RGB channels
                
            # Create the mesh for the highlighted cell
            cell_mesh = pv.UnstructuredGrid(
                np.hstack([[4], self.mesh.cell_to_vertices[cell]]).flatten(),
                [pv.CellType.TETRA],
                self.mesh.vertex_coordinates
            )
                
            # Apply color based on the Jacobian value
            self.plotter.add_mesh(
                cell_mesh,
                color=color,
                opacity=0.5
            )


    def add_cell_normals(self, cell_list):
        face_node_indices = self.mesh.reference_element.face_node_indices
        for cell in cell_list:
            x_origin = self.mesh.x[:, cell][face_node_indices].ravel(order='F')
            y_origin = self.mesh.y[:, cell][face_node_indices].ravel(order='F')
            z_origin = self.mesh.z[:, cell][face_node_indices].ravel(order='F')
            
            # Stack into origin points
            normal_vector_origin = np.column_stack((x_origin, y_origin, z_origin))
            
            # Extract normal vectors for the given element
            nx = self.physics.mesh.nx[:, cell]
            ny = self.physics.mesh.ny[:, cell]
            nz = self.physics.mesh.nz[:, cell]
            
            # Stack into normal vectors
            normal_vector_direction = np.column_stack((nx, ny, nz))
            
            # Add normal vectors as arrows
            self.plotter.add_arrows(
                normal_vector_origin,
                normal_vector_direction,
                mag=0.08,
                color='red'
            )

    def add_boundary_normals(self):
        """Plot normal vectors for all elements."""
        # Get boundary node indices
        boundary_node_indices = self.mesh.boundary_node_indices
        boundary_face_node_indices = self.mesh.boundary_face_node_indices
        
        # Get normal vector origins for all boundary nodes 
        x_origin = self.mesh.x.ravel(order='F')[boundary_node_indices]
        y_origin = self.mesh.y.ravel(order='F')[boundary_node_indices]
        z_origin = self.mesh.z.ravel(order='F')[boundary_node_indices]
        
        normal_vector_origin = np.column_stack((x_origin, y_origin, z_origin))

        # Extract normal vectors for the given element
        nx = self.mesh.nx.ravel(order='F')[boundary_face_node_indices]
        ny = self.mesh.ny.ravel(order='F')[boundary_face_node_indices]
        nz = self.mesh.nz.ravel(order='F')[boundary_face_node_indices]
        
        # Stack into normal vectors
        normal_vector_direction = np.column_stack((nx, ny, nz))
    
        self.plotter.add_arrows(
            normal_vector_origin,
            normal_vector_direction,
            mag=0.05,
            color='red'
        )
 

    
    def add_cell_averages(self, field):
        """
        Plot cell averages for a given solution.
        Each cell's average solution value is computed and visualized.
        """
        # Construct a cells object to make a pyvista unstructuredGrid
        cells = np.zeros(self.mesh.num_cells * 5, dtype='int')
        index = 0
        for i in range(self.mesh.num_cells * 5):
            if i % 5 == 0:
                cells[i] = 4
            else:
                cells[i] = index
                index += 1
    
        # calculate average of each cell
        cell_averages = np.mean(field, axis=0)
        # designate cell type of tetrahedron
        cell_types = np.repeat(np.array([pv.CellType.TETRA]), self.mesh.num_cells)
        # get coordinates from mesh
        coordinates = self.mesh.vertex_coordinates[self.mesh.cell_to_vertices.ravel()]
        
        # create unstructured grid
        grid = pv.UnstructuredGrid(
            cells,
            cell_types,
            coordinates
        )
           
        # add to plotter
        self.plotter.add_mesh(
            grid,
            scalars=cell_averages,
            opacity=abs(cell_averages)
        )

    def add_wave_speed(self):
        """ plot the wavespeed of each element """
        # Construct a cells object to make a pyvista unstructuredGrid
        cells = np.zeros(self.mesh.num_cells * 5, dtype='int')
        index = 0
        for i in range(self.mesh.num_cells * 5):
            if i % 5 == 0:
                cells[i] = 4
            else:
                cells[i] = index
                index += 1
        cell_types = np.repeat(np.array([pv.CellType.TETRA]), self.mesh.num_cells)
        points = self.mesh.vertex_coordinates[self.mesh.cell_to_vertices.ravel()]
    
        # create a pyvista unstructured grid
        grid = pv.UnstructuredGrid(
            cells,
            cell_types,
            points
        )
           
        # add to plotter
        self.plotter.add_mesh(
            grid,
            scalars=self.mesh.speed,
            opacity=0.005#'linear'#abs(wave_speed)
        )

    def add_mesh(self):
        """ add the edges of the entire 3D mesh """
        # Construct a cells object to make a pyvista unstructuredGrid
        cells = np.zeros(self.mesh.num_cells * 5, dtype='int')
        index = 0
        for i in range(self.mesh.num_cells * 5):
            if i % 5 == 0:
                cells[i] = 4
            else:
                cells[i] = index
                index += 1
        cell_types = np.repeat(np.array([pv.CellType.TETRA]), self.mesh.num_cells)
        points = self.mesh.vertex_coordinates[self.mesh.cell_to_vertices.ravel()]
    
        # create a pyvista unstructured grid
        grid = pv.UnstructuredGrid(
            cells,
            cell_types,
            points,
        )
           
        # Add clipping widgets
        for norm in ['-x', '-y']:
            self.plotter.add_mesh_clip_plane(
                grid,
                normal=norm,
                crinkle=True,
                interaction_event='always',
                normal_rotation=False,
                color='#5e81ac'
            )
 
    def add_mesh_boundary(self):
        """ Plot mesh edges on boundary """
        # create cells and cell_types to pyvista unstructured grid
        cells = np.hstack([np.full((self.mesh.num_cells, 1), 4), self.mesh.cell_to_vertices]).flatten()
        cell_types = np.full(self.mesh.num_cells, pv.CellType.TETRA) 

        # create unstructured grid
        grid = pv.UnstructuredGrid(
            cells,
            cell_types,
            self.mesh.vertex_coordinates
        )
        # Add mesh
        self.plotter.add_mesh(
            grid,
            style='wireframe',
            color='black'
        )


    def visualize_array(self, array):
        """
        Visualizes a 2D NumPy array using a colormap.
        """
        cmap = 'virdis'
        if array.ndim != 2:
            raise ValueError("Input array must be 2D")
    
        plt.figure(figsize=(6, 5))
        plt.imshow(array, cmap=cmap, aspect='auto')
        
        plt.colorbar()
    
        plt.title("2D Array Visualization")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()


    def plot_reference_nodes_3d(self):
        """ plot the nodes for the reference finite element """
        # Get nodes
        nodes = self.mesh.reference_element.nodes
        
        # Create a PyVista point cloud
        point_cloud = pv.PolyData(nodes)
        
        # Create a plotter
        plotter = pv.Plotter()

        # Add mesh
        plotter.add_mesh(
            point_cloud,
            color='red',
            point_size=10,
            render_points_as_spheres=True
        )
        
        # Add axes labels
        plotter.show_grid()
        
        # show plot
        d = self.mesh.reference_element.d
        n = self.mesh.reference_element.n
        plotter.show(title=f"Lagrange Element Nodes (d={d}, n={n})")

    def add_inclusion_boundary(self):
        # Define the inclusion vertices
        vertices = np.array([
            [0.4, 0.4, 0.4], [0.4, 0.4, 0.6], [0.4, 0.6, 0.4], [0.4, 0.6, 0.6],
            [0.6, 0.4, 0.4], [0.6, 0.4, 0.6], [0.6, 0.6, 0.4], [0.6, 0.6, 0.6]
        ])
        
        # Define the edges as pairs of vertex indices
        edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7)
        ]
        
        # Plot each edge
        for edge in edges:
            line = pv.Line(vertices[edge[0]], vertices[edge[1]])
            self.plotter.add_mesh(line, color="black", line_width=2)


    def enter_4th_dimension(self, resolution=50):
        """ Add a 3D field visualization tothe plotter """
        origin = (0,0,0)
        dimensions = (1,1,1)
        spacing = dimensions[0]/resolution
        coordinates = self._get_grid_coordinates(origin, dimensions, resolution)
       
        cloud = pv.PolyData(coordinates)
        
        # Plot using volume rendering
        plotter = pv.Plotter()
        plotter.add_mesh(cloud, color='blue', point_size=2, render_points_as_spheres=True)
        plotter.show()

    def save(self):
        self.plotter.off_screen = True
        file_name=f't_{self.time_stepper.current_time_step:0>8}.png'
        self.plotter.screenshot(f'./outputs/images/{file_name}')
        self.plotter.off_screen = False 

    def show(self):
        self.plotter.show()

