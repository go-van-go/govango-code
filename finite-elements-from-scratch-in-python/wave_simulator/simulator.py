class Simulator:
    def __init__(self, Mesh):
        self.Mesh = Mesh

    def run(self):
        pass

if __name__ == "__main__":
    import sys
    from wave_simulator.mesh import Mesh3d
    from wave_simulator.finite_elements import LagrangeElement
    from plotting import *

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
    Mesh = Mesh3d(mesh_file, LagrangeElement(dim,n))
    Simulator = Simulator(Mesh)
    



        
