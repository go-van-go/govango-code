import gmsh

gmsh.initialize()
gmsh.model.add("simple")
gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
gmsh.model.occ.synchronize()

# Set uniform mesh size
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)

# Generate 3D mesh
gmsh.model.mesh.generate(3)

# Optional: Optimize the mesh
gmsh.model.mesh.optimize("Netgen")

# Write the mesh to a file
gmsh.write("fine_grid.msh")
gmsh.finalize()
