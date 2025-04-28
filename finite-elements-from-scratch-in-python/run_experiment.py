from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.mesh import Mesh3d 
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.visualizer import Visualizer
from wave_simulator.simulator import Simulator 

# create finite element
dimension = 3
polynomial_order = 4 # eta1 and eta2 are defined for N=2
lagrange_element = LagrangeElement(dimension, polynomial_order)

# create mesh
mesh_file = "./inputs/meshes/split1.msh"
mesh = Mesh3d(mesh_file, lagrange_element)

# create physics
physics = LinearAcoustics(mesh)

# create time stepping object 
t_initial = 0.0
#t_final = 3.0e-4
t_final = 3
time_stepper = LowStorageRungeKutta(physics, t_initial, t_final)

# initialize simulator object
simulator = Simulator(time_stepper)
simulator.set_save_intervals(image=10, data=100, points=10, energy=50, vtk=0)
simulator.track_points([
    [0.125, 0.125, 0.0],
#    [0.0625, 0.0625, 0.25],
#    [0.0625, 0.1875, 0.25],
#    [0.1875, 0.0625, 0.25],
#    [0.1875, 0.1875, 0.25],
#    [0.125, 0.125, 0.25],
])

# run experiment
simulator.run()
