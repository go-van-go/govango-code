import gmsh

gmsh.initialize()
gmsh.model.add("cube")
gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
gmsh.model.occ.synchronize()

# Set uniform mesh size
max_size = 0.2
#gmsh.option.setNumber("Mesh.MeshSizeMin", 0.010)
gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)

# Generate 3D mesh
#gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 3)
#gmsh.option.setNumber("Mesh.OptimizeThreshold", 1)
#gmsh.model.mesh.optimize("HighOrderElastic", force=True)
gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Example: 4 = Frontal

# Optimize tetrahedra that have a quality below ...,  Default value: 0.3       
gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.05)

gmsh.model.mesh.generate(3)

# Optional: Optimize the mesh
#gmsh.model.mesh.optimize("Netgen")

# Write the mesh to a file
gmsh.write(f"{max_size}.msh")
gmsh.finalize()
