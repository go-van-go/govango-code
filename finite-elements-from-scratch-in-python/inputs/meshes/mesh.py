import gmsh

gmsh.initialize()
gmsh.model.add("simple")
gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
gmsh.model.occ.synchronize()

# Set uniform mesh size
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.10)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.10)

# Generate 3D mesh
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
gmsh.model.mesh.optimize("HighOrderElastic", force=True)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # Example: 4 = Frontal

gmsh.model.mesh.generate(3)

# Optional: Optimize the mesh
#gmsh.model.mesh.optimize("Netgen")

# Write the mesh to a file
gmsh.write("fine_grid.msh")
gmsh.finalize()
