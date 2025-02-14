import numpy as np
import gmsh
from finite_elements import LagrangeElement

class Mesh3d:
    def __init__(self, msh_file):
        gmsh.initialize()
        gmsh.open(msh_file)

        self.dim = 3
        self.n = 3  # polynomial order
        self.reference_element = LagrangeElement(self.dim,self.n)
        self.num_vertices = 0
        self.num_cells= 0
        self.vertexCoordinates = []
        self.x_vertex = [] # vertex x coordinates
        self.y_vertex = [] # vertex y coordinates
        self.z_vertex = [] # vertex z coordinates
        self.x = [] # nodal x coordinates
        self.y = [] # nodal y coordinates
        self.z = [] # nodal z coordinates
        self.edgeVertices = []
        self.cell2vertices = []
        self.cell2cells = []
        self.cell2faces = []
        self.jacobians = {}
        self.determinants = {}
        
        self._extract_mesh_info()
        self._build_connectivityMatricies()
        self._compute_jacobians()
        #self._compute_surfaceJacobian()
        self._get_mapped_nodal_cordinates()

        gmsh.finalize()

    def _extract_mesh_info(self):
        # get vertex information 
        ntags, coords, _ = gmsh.model.mesh.getNodes(4)
        self.num_vertices= len(ntags)
        self.vertexCoordinates = coords.reshape(-1, 3)
        self.x_vertex = self.vertexCoordinates[:, 0]
        self.y_vertex = self.vertexCoordinates[:, 1]
        self.z_vertex = self.vertexCoordinates[:, 2]

        # get edges
        edgeVertices = gmsh.model.mesh.getElementEdgeNodes(4)
        self.edgeVertices = edgeVertices.reshape(int(len(edgeVertices)/2), 2).astype(int) - 1

        # get element information
        # get all the nodes from tetrahedrons (elementType = 4)
        nodeTags, _, _ = gmsh.model.mesh.getNodesByElementType(4) 
        self.num_cells = int(len(nodeTags)/4) 
        self.cell2vertices = nodeTags.reshape(-1, 4).astype(int) - 1

    def _build_connectivityMatricies(self):
        """tetrahedral face connect algorithm from Toby Isaac"""
        num_faces = 4
        K = self.num_cells
        EtoV = self.cell2vertices
        num_vertices = self.num_vertices 
        
        # create list of all faces
        faceVertices = np.vstack((EtoV[:, [0, 1, 2]],
                            EtoV[:, [0, 1, 3]],
                            EtoV[:, [1, 2, 3]],
                            EtoV[:, [0, 2, 3]]))
        # sort each row from low to high for hash algorithm
        faceVertices = np.sort(faceVertices, axis=1)
         
        # unique hash for each set of three faces by their vertex numbers
        faceHashes = faceVertices[:, 0] * num_vertices * num_vertices  + \
                     faceVertices[:, 1] * num_vertices + \
                     faceVertices[:, 2] + 1

        # vertex id from 1 - num_faces* num_cells
        vertex_ids = np.arange(1, num_faces*K+1)
       
        # set up default cell to cell and cell to faces connectivity
        EtoE = np.tile(np.arange(1, K+1)[:, np.newaxis], (1, num_faces))
        EtoF = np.tile(np.arange(1, num_faces+1), (K, 1))

        # build a master matrix (mappingTable) that we will solve by 
        # sorting by one column to create the connectivity matricies
        mappingTable = np.column_stack((faceHashes,
                                        vertex_ids,
                                        np.ravel(EtoE, order='F'),
                                        np.ravel(EtoF, order='F')))
        
        # Now we sort by global face number.
        sorted_mapTable= np.array(sorted(mappingTable, key=lambda x: (x[0], x[1])))
        
        # find matches in the sorted face list
        matches = np.where(sorted_mapTable[:-1, 0] == sorted_mapTable[1:, 0])[0]
        
        # make links reflexive
        matchL = np.vstack((sorted_mapTable[matches], sorted_mapTable[matches + 1]))
        matchR = np.vstack((sorted_mapTable[matches + 1], sorted_mapTable[matches]))
        
        # insert matches
        EtoE_tmp = np.ravel(EtoE, order='F') - 1
        EtoF_tmp = np.ravel(EtoF, order='F') - 1
        EtoE_tmp[matchL[:, 1] - 1] = (matchR[:, 2] - 1)
        EtoF_tmp[matchL[:, 1] - 1] = (matchR[:, 3] - 1)
        
        EtoE = EtoE_tmp.reshape(EtoE.shape, order='F')
        EtoF = EtoF_tmp.reshape(EtoF.shape, order='F')

        self.cell2cells = EtoE
        self.cell2faces = EtoF

    def _compute_jacobians(self):
        # get local coordinates of the verticies in the
        # reference tetrahedron
        name, dim, order, numNodes, localCoords, _ = gmsh.model.mesh.getElementProperties(4)
        jacobians, determinants, _ = gmsh.model.mesh.getJacobians(4, localCoords)
        self.jacobians = jacobians.reshape(-1, 3, 3)
        self.determinants = determinants

        
    def _get_mapped_nodal_cordinates(self):
        """ returns x, y, and z arrays of coordinates of nodes from EToV and VX, VY, VZ, arrays"""
        CtoV = self.cell2vertices
        vx = self.x_vertex
        vy = self.y_vertex
        vz = self.z_vertex
        r = self.reference_element.r
        s = self.reference_element.s
        t = self.reference_element.t

        # extract vertex numbers from elements
        va = CtoV[:, 0].T
        vb = CtoV[:, 1].T
        vc = CtoV[:, 2].T
        vd = CtoV[:, 3].T
        
        vx = vx.reshape(-1, 1)
        vy = vy.reshape(-1, 1)
        vz = vz.reshape(-1, 1)
        
        # map r, s, t from standard tetrahedron to x, y, z coordinates for each element
        self.x = (0.5 * (-(1 + r + s + t) * vx[va] + (1 + r) * vx[vb] + (1 + s) * vx[vc] + (1 + t) * vx[vd])).T
        self.y = (0.5 * (-(1 + r + s + t) * vy[va] + (1 + r) * vy[vb] + (1 + s) * vy[vc] + (1 + t) * vy[vd])).T
        self.z = (0.5 * (-(1 + r + s + t) * vz[va] + (1 + r) * vz[vb] + (1 + s) * vz[vc] + (1 + t) * vz[vd])).T
        
        breakpoint()

if __name__ == "__main__":
    import sys

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
    
    mesh = Mesh3d(mesh_file)
