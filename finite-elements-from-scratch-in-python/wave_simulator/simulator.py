from wave_simulator.mesh import Mesh3d
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta

class Simulator:
    def __init__(self, mesh: Mesh3d, physics: LinearAcoustics, time_stepper: LowStorageRungaKutta):
        self.mesh = mesh

    def run(self):
        while time < t_final:  # outer time step loop
            advance_time_step

if __name__ == "__main__":
    import sys
    from mesh import Mesh3d
    from wave_simulator.finite_elements import LagrangeElement
    from wave_simulator.visualizing import *

    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    else:
        mesh_file = "../inputs/meshes/simple.msh"
    
    dim = 3
    n = 3
    mesh = Mesh3d(mesh_file, LagrangeElement(dim,n))
    Simulator = Simulator(Mesh)
    breakpoint()
    



        
