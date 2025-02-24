class Simulator:
    def __init__(self, Mesh):
        self.Mesh = Mesh

    def run(self):
        pass

if __name__ == "__main__":
    import sys
    from mesh import Mesh3d
    from finite_elements import LagrangeElement
    from visualizing import *

    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    else:
        mesh_file = "../inputs/meshes/simple.msh"
    
    dim = 3
    n = 3
    Mesh = Mesh3d(mesh_file, LagrangeElement(dim,n))
    Simulator = Simulator(Mesh)
    breakpoint()
    



        
