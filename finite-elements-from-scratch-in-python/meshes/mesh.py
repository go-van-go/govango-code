import numpy as np
import gmsh

class Mesh3d:
    def __init__(self, msh_file):
        gmsh.initialize()
        gmsh.open(msh_file)

        self.num_vertices = 0
        self.num_elements = 0
        self.vertexCoordinates = []
        self.x = []
        self.y = []
        self.z = []
        self.edgeVertices = []
        self.element2vertices = []
        self.element2elements = []
        self.element2faces = []
        self.faceNormals = []
        self.jacobians = {}
        self.determinants = {}
        
        self._extract_mesh_info()
        self._build_connectivityMatricies()
        self._compute_jacobians()

        gmsh.finalize()

    def _extract_mesh_info(self):
        # get vertex information 
        ntags, coords, _ = gmsh.model.mesh.getNodes(4)
        self.num_vertices= len(ntags)
        self.vertexCoordinates = coords.reshape(-1, 3)
        self.x = self.vertexCoordinates[:, 0]
        self.y = self.vertexCoordinates[:, 1]
        self.z = self.vertexCoordinates[:, 2]

        # get edges
        edgeVertices = gmsh.model.mesh.getElementEdgeNodes(4)
        self.edgeVertices = edgeVertices.reshape(int(len(edgeVertices)/2), 2).astype(int) - 1

        # get element information
        # get all the nodes from tetrahedrons (elementType = 4)
        nodeTags, _, _ = gmsh.model.mesh.getNodesByElementType(4) 
        self.num_elements = int(len(nodeTags)/4) 
        self.element2vertices = nodeTags.reshape(-1, 4).astype(int) - 1

    def _build_connectivityMatricies(self):
        """tetrahedral face connect algorithm from Toby Isaac"""
        num_faces = 4
        K = self.num_elements 
        EtoV = self.element2vertices
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

        # vertex id from 1 - num_vertices * num_elements
        vertex_ids = np.arange(1, num_faces*K+1)
       
        # set up default element to element and element to faces connectivity
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

        self.element2elements = EtoE
        self.element2faces = EtoF

    def _compute_jacobians(self):
        # get local coordinates of the verticies in the
        # reference tetrahedron
        name, dim, order, numNodes, localCoords, _ = gmsh.model.mesh.getElementProperties(4)
        jacobians, determinants, _ = gmsh.model.mesh.getJacobians(4, localCoords)
        self.jacobians = jacobians.reshape(-1, 3, 3)
        self.determinants = determinants
        pass
        

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
