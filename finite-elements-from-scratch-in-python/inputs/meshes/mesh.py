import gmsh

gmsh.initialize()
gmsh.model.add("simple")
gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
#gmsh.option.setNumber("Mesh.MeshSizeMin", 2.0)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.10)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write("fine_grid.msh")
gmsh.finalize()

