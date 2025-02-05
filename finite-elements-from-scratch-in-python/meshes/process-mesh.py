import gmsh
import sys

class Mesh:
    def __init__(self, msh_file):
        gmsh.initialize()
        gmsh.open(msh_file)
        
        self.elements2Elements = {}
        self.edges2Elements = {}
        self.faces2Elements = {}
        self.faceNormals = {}
        self.jacobians = {}
        self.determinants = {}
        self.num_elements = 0
        self.vertices = {}
        self.dimension = gmsh.model.getDimension()
        
        self._extract_mesh_info()
        self._build_edges2Elements()
        if self.dimension == 3:
            self._build_faces2Elements()
        self._build_elements2Elements()
        self._compute_jacobians()
        self._compute_normals()
        
        gmsh.finalize()

    def _extract_mesh_info(self):
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        self.vertices = {nodeTags[i]: (nodeCoords[3*i], nodeCoords[3*i+1], nodeCoords[3*i+2]) for i in range(len(nodeTags))}
        
        if self.dimension == 3:
            self.elementType = gmsh.model.mesh.getElementType("tetrahedron", 1)
        else:
            self.elementType = gmsh.model.mesh.getElementType("triangle", 1)
        
        self.elementTags, self.elementNodeTags = gmsh.model.mesh.getElementsByType(self.elementType)
        self.num_elements = len(self.elementTags)

    def _build_edges2Elements(self):
        gmsh.model.mesh.createEdges()
        edgeNodes = gmsh.model.mesh.getElementEdgeNodes(self.elementType)
        edgeTags, _ = gmsh.model.mesh.getEdges(edgeNodes)
        
        for i in range(len(edgeTags)):
            if edgeTags[i] not in self.edges2Elements:
                self.edges2Elements[edgeTags[i]] = [self.elementTags[i // (3 if self.dimension == 2 else 6)]]
            else:
                self.edges2Elements[edgeTags[i]].append(self.elementTags[i // (3 if self.dimension == 2 else 6)])

    def _build_faces2Elements(self):
        if self.dimension == 2:
            return
        gmsh.model.mesh.createFaces()
        faceNodes = gmsh.model.mesh.getElementFaceNodes(self.elementType, 3)
        faceTags, _ = gmsh.model.mesh.getFaces(3, faceNodes)
        
        for i in range(len(faceTags)):
            if faceTags[i] not in self.faces2Elements:
                self.faces2Elements[faceTags[i]] = [self.elementTags[i // 4]]
            else:
                self.faces2Elements[faceTags[i]].append(self.elementTags[i // 4])

    def _build_elements2Elements(self):
        for element in self.elementTags:
            self.elements2Elements[element] = set()
        
        if self.dimension == 3:
            for face, elements in self.faces2Elements.items():
                if len(elements) == 2:
                    self.elements2Elements[elements[0]].add(elements[1])
                    self.elements2Elements[elements[1]].add(elements[0])
        else:
            for edge, elements in self.edges2Elements.items():
                if len(elements) == 2:
                    self.elements2Elements[elements[0]].add(elements[1])
                    self.elements2Elements[elements[1]].add(elements[0])
    
    def _compute_jacobians(self):
        localCoord = [1/3, 1/3] if gmsh.model.getDimension() == 2 else [0.25, 0.25, 0.25]
        jacobians, determinants, _ = gmsh.model.mesh.getJacobians(self.elementType, localCoord)
    
        if len(determinants) == 0:
            print("Warning: No Jacobians computed. Check if the mesh has valid elements.")
    
        for i, element in enumerate(self.elementTags):
            if i < len(determinants):  # Ensure we don't access out of bounds
                self.jacobians[element] = jacobians[i*9:(i+1)*9] if gmsh.model.getDimension() == 3 else jacobians[i*4:(i+1)*4]
                self.determinants[element] = determinants[i]

    def _compute_normals(self):
        # IMPLEMENT THIS FUNCTION
        return

    def print_info(self):
        print(f"Mesh contains {self.num_elements} elements ({'triangles' if self.dimension == 2 else 'tetrahedrons'})")
        print(f"Number of vertices: {len(self.vertices)}")
        print("Elements adjacency list:")
        for element, neighbors in self.elements2Elements.items():
            print(f"  Element {element} -> {neighbors}")
        print("Jacobian matrices:")
        for element, jacobian in self.jacobians.items():
            print(f"  Element {element} -> {jacobian}")
        print("Determinants:")
        for element, determinant in self.determinants.items():
            print(f"  Element {element} -> {determinant}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    else:
        gmsh.initialize()
        gmsh.model.add("simple")
        
        if len(sys.argv) > 2 and sys.argv[2] == "2D":
            gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
            dim = 2
        else:
            gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
            dim = 3
        
        gmsh.option.setNumber("Mesh.MeshSizeMin", 2.0)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(dim)
        gmsh.write("default.msh")
        mesh_file = "default.msh"
        gmsh.finalize()
    
    mesh = Mesh(mesh_file)
    breakpoint()
