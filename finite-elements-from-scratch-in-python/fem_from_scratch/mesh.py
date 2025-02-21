import numpy as np
import gmsh
from reference_element_operators import ReferenceElementOperators

class Mesh3d:
    def __init__(self, msh_file, FiniteElement):
        gmsh.initialize()
        gmsh.open(msh_file)
        
        self.msh_file = msh_file
        self.ReferenceElement = FiniteElement 
        self.ReferenceElementOperators = ReferenceElementOperators(self.ReferenceElement)
        self.dim = self.ReferenceElement.d
        self.n = self.ReferenceElement.n  # polynomial order
        self.num_vertices = 0
        self.num_cells= 0
        self.vertex_coordinates = []
        self.x_vertex = [] # vertex x coordinates
        self.y_vertex = [] # vertex y coordinates
        self.z_vertex = [] # vertex z coordinates
        self.x = []  # nodal x coordinates
        self.y = []  # nodal y coordinates
        self.z = []  # nodal z coordinates
        self.nx = None
        self.ny = None
        self.nz = None
        self.edge_vertices = []
        self.cell_to_vertices = []
        self.cell_to_cells = []
        self.cell_to_faces = []
        self.face_node_indices = None 
        # reference to physical mapping coefficients
        self.drdx = None
        self.drdy = None
        self.drdz = None
        self.dsdx = None
        self.dsdy = None
        self.dsdz = None
        self.dtdx = None
        self.dtdy = None
        self.dtdz = None
        self.jacobians = {}
        self.determinants = {}
        
        self._extract_mesh_info()
        self._build_connectivityMatricies()
        #self._compute_jacobians()
        self._get_mapped_nodal_cordinates()
        self._compute_mapping_coefficients()
        self._find_face_nodes()
        self._compute_normals_at_face_nodes()
        #self._compute_surfaceJacobian()

        gmsh.finalize()

    def _extract_mesh_info(self):
        """ Get information from Gmsh file """
        # get vertex information 
        ntags, coords, _ = gmsh.model.mesh.getNodes(4)
        self.num_vertices= len(ntags)
        self.vertex_coordinates = coords.reshape(-1, 3)
        self.x_vertex = self.vertex_coordinates[:, 0]
        self.y_vertex = self.vertex_coordinates[:, 1]
        self.z_vertex = self.vertex_coordinates[:, 2]

        # get edges
        edge_vertices = gmsh.model.mesh.getElementEdgeNodes(4)
        self.edge_vertices = edge_vertices.reshape(int(len(edge_vertices)/2), 2).astype(int) - 1

        # get cell information
        # get all the nodes from tetrahedrons (elementType = 4)
        node_tags, _, _ = gmsh.model.mesh.getNodesByElementType(4) 
        self.num_cells = int(len(node_tags)/4) 
        self.cell_to_vertices = node_tags.reshape(-1, 4).astype(int) - 1

    def _build_connectivityMatricies(self):
        """tetrahedral face connect algorithm from Toby Isaac"""
        num_faces = 4
        K = self.num_cells
        CtoV = self.cell_to_vertices
        num_vertices = self.num_vertices 
        
        # create list of all faces
        face_vertices = np.vstack((CtoV[:, [0, 1, 2]],
                            CtoV[:, [0, 1, 3]],
                            CtoV[:, [1, 2, 3]],
                            CtoV[:, [0, 2, 3]]))
        # sort each row from low to high for hash algorithm
        face_vertices = np.sort(face_vertices, axis=1)
         
        # unique hash for each set of three faces by their vertex numbers
        face_hashes = face_vertices[:, 0] * num_vertices * num_vertices  + \
                     face_vertices[:, 1] * num_vertices + \
                     face_vertices[:, 2] + 1

        # vertex id from 1 - num_faces* num_cells
        vertex_ids = np.arange(1, num_faces*K+1)
       
        # set up default cell to cell and cell to faces connectivity
        CtoC = np.tile(np.arange(1, K+1)[:, np.newaxis], (1, num_faces))
        CtoF = np.tile(np.arange(1, num_faces+1), (K, 1))

        # build a master matrix (mappingTable) that we will solve by 
        # sorting by one column to create the connectivity matricies
        mapping_table = np.column_stack((face_hashes,
                                        vertex_ids,
                                        np.ravel(CtoC, order='F'),
                                        np.ravel(CtoF, order='F')))
        
        # Now we sort by global face number.
        sorted_map_table= np.array(sorted(mapping_table, key=lambda x: (x[0], x[1])))
        
        # find matches in the sorted face list
        matches = np.where(sorted_map_table[:-1, 0] == sorted_map_table[1:, 0])[0]
        
        # make links reflexive
        match_l = np.vstack((sorted_map_table[matches], sorted_map_table[matches + 1]))
        match_r = np.vstack((sorted_map_table[matches + 1], sorted_map_table[matches]))
        
        # insert matches
        CtoC_tmp = np.ravel(CtoC, order='F') - 1
        CtoF_tmp = np.ravel(CtoF, order='F') - 1
        CtoC_tmp[match_l[:, 1] - 1] = (match_r[:, 2] - 1)
        CtoF_tmp[match_l[:, 1] - 1] = (match_r[:, 3] - 1)
        
        CtoC = CtoC_tmp.reshape(CtoC.shape, order='F')
        CtoF = CtoF_tmp.reshape(CtoF.shape, order='F')

        self.cell_to_cells = CtoC
        self.cell_to_faces = CtoF

    def _compute_jacobians(self):
        """ calculate the jacobian of the mapping of each cell """
        # get local coordinates of the verticies in the
        # reference tetrahedron
        name, dim, order, numNodes, localCoords, _ = gmsh.model.mesh.getElementProperties(4)
        jacobians, determinants, _ = gmsh.model.mesh.getJacobians(4, localCoords)
        self.jacobians = jacobians.reshape(-1, 3, 3)
        self.determinants = determinants

        
    def _get_mapped_nodal_cordinates(self):
        """ returns x, y, and z arrays of coordinates of nodes from EToV and VX, VY, VZ, arrays"""
        CtoV = self.cell_to_vertices
        vx = self.x_vertex
        vy = self.y_vertex
        vz = self.z_vertex
        r = self.ReferenceElement.r
        s = self.ReferenceElement.s
        t = self.ReferenceElement.t

        # extract vertex numbers from elements
        va = CtoV[:, 0].T
        vb = CtoV[:, 1].T
        vc = CtoV[:, 2].T
        vd = CtoV[:, 3].T
        
        vx = vx.reshape(-1, 1)
        vy = vy.reshape(-1, 1)
        vz = vz.reshape(-1, 1)
        
        # map r, s, t from standard tetrahedron to x, y, z coordinates for each element
        self.x = ((1 - r - s - t) * vx[va] + r * vx[vb] + s * vx[vc] + t * vx[vd]).T
        self.y = ((1 - r - s - t) * vy[va] + r * vy[vb] + s * vy[vc] + t * vy[vd]).T
        self.z = ((1 - r - s - t) * vz[va] + r * vz[vb] + s * vz[vc] + t * vz[vd]).T
        

    def _compute_mapping_coefficients(self):
        """Compute the metric elements for the local mappings of the elements"""
        Dr = self.ReferenceElementOperators.r_differentiation_matrix
        Ds = self.ReferenceElementOperators.s_differentiation_matrix
        Dt = self.ReferenceElementOperators.t_differentiation_matrix
        x = self.x
        y = self.y
        z = self.z

        # find jacobian of mapping
        xr = np.dot(Dr, x)
        xs = np.dot(Ds, x)
        xt = np.dot(Dt, x)
        yr = np.dot(Dr, y)
        ys = np.dot(Ds, y)
        yt = np.dot(Dt, y)
        zr = np.dot(Dr, z)
        zs = np.dot(Ds, z)
        zt = np.dot(Dt, z)

        J = xr * (ys * zt - zs * yt) - yr * (xs * zt - zs * xt) + zr * (xs * yt - ys * xt)
        self.jacobians = J

        # compute the metric constants
        self.drdx = (ys * zt - zs * yt) / J
        self.drdy = -(xs * zt - zs * xt) / J
        self.drdz = (xs * yt - ys * xt) / J
        self.dsdx = -(yr * zt - zr * yt) / J
        self.dsdy = (xr * zt - zr * xt) / J
        self.dsdz = -(xr * yt - yr * xt) / J
        self.dtdx = (yr * zs - zr * ys) / J
        self.dtdy = -(xr * zs - zr * xs) / J
        self.dtdz = (xr * ys - yr * xs) / J


    def _find_face_nodes(self):
        """ return node indexes for nodes on each face of the standard tetrahedron"""
        tolerance = self.ReferenceElement.NODE_TOLERANCE
        r = self.ReferenceElement.r
        s = self.ReferenceElement.s
        t = self.ReferenceElement.t

        face_0_indices = np.where(np.abs(t) < tolerance)[0]
        face_1_indices = np.where(np.abs(s) < tolerance)[0]
        face_2_indices = np.where(np.abs(r) < tolerance)[0]
        face_3_indices = np.where(np.abs(r + s + t - 1) < tolerance)[0]
        face_node_indices = np.concatenate((face_0_indices,
                                            face_1_indices,
                                            face_2_indices,
                                            face_3_indices))

        self.face_node_indices = face_node_indices

    def _compute_normals_at_face_nodes(self):
        """compute outward pointing normals at elements faces as well as surface Jacobians"""
        Nfp = self.ReferenceElement.nodes_per_face
        K = self.num_cells

        # interpolate geometric factors to face nodes
        face_drdx = self.drdx[self.face_node_indices, :]
        face_dsdx = self.dsdx[self.face_node_indices, :]
        face_dtdx = self.dtdx[self.face_node_indices, :]
        face_drdy = self.drdy[self.face_node_indices, :]
        face_dsdy = self.dsdy[self.face_node_indices, :]
        face_dtdy = self.dtdy[self.face_node_indices, :]
        face_drdz = self.drdz[self.face_node_indices, :]
        face_dsdz = self.dsdz[self.face_node_indices, :]
        face_dtdz = self.dtdz[self.face_node_indices, :]

        # build normal vectors
        nx = np.zeros((4 * Nfp, K))
        ny = np.zeros((4 * Nfp, K))
        nz = np.zeros((4 * Nfp, K))

        # create vectors of indices of each face
        face_0_indices = np.arange(0, Nfp)
        face_1_indices = np.arange(Nfp, 2 * Nfp)
        face_2_indices = np.arange(2 * Nfp, 3 * Nfp)
        face_3_indices = np.arange(3 * Nfp, 4 * Nfp)

        # face 0: t = 0 → Normal in -t direction
        nx[face_0_indices, :] = -face_dtdx[face_0_indices, :]
        ny[face_0_indices, :] = -face_dtdy[face_0_indices, :]
        nz[face_0_indices, :] = -face_dtdz[face_0_indices, :]

        # face 1: s = 0 → Normal in -s direction
        nx[face_1_indices, :] = -face_dsdx[face_1_indices, :]
        ny[face_1_indices, :] = -face_dsdy[face_1_indices, :]
        nz[face_1_indices, :] = -face_dsdz[face_1_indices, :]

        # face 2: r = 0 → Normal in -r direction
        nx[face_2_indices, :] = -face_drdx[face_2_indices, :]
        ny[face_2_indices, :] = -face_drdy[face_2_indices, :]
        nz[face_2_indices, :] = -face_drdz[face_2_indices, :]

        # Face 3: r + s + t = 1 → Normal is the gradient of (r + s + t)
        nx[face_3_indices, :] = face_drdx[face_3_indices, :] + face_dsdx[face_3_indices, :] + face_dtdx[face_3_indices, :]
        ny[face_3_indices, :] = face_drdy[face_3_indices, :] + face_dsdy[face_3_indices, :] + face_dtdy[face_3_indices, :]
        nz[face_3_indices, :] = face_drdz[face_3_indices, :] + face_dsdz[face_3_indices, :] + face_dtdz[face_3_indices, :]

        # find surface Jacobian
        sJ = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx /= sJ
        ny /= sJ
        nz /= sJ
        sJ *= self.jacobians[self.face_node_indices, :]

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.surface_jacobians = sJ


if __name__ == "__main__":
    import sys
    from finite_elements import LagrangeElement

    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    else:
        gmsh.initialize()
        gmsh.model.add("simple")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 2.0)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write("default.msh")
        mesh_file = "default.msh"
        gmsh.finalize()
    
    dim = 3
    n = 3
    mesh = Mesh3d(mesh_file, LagrangeElement(dim,n))
