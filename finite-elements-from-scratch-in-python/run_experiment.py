import pickle
from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.mesh import Mesh3d 
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.visualizer import Visualizer
from wave_simulator.simulator import Simulator 

# create finite element
dimension = 3
polynomial_order = 2 # eta1 and eta2 are defined for N=2
lagrange_element = LagrangeElement(dimension, polynomial_order)

# create mesh
mesh_file = "./inputs/meshes/skull.msh"
mesh = Mesh3d(mesh_file, lagrange_element)

# create physics
physics = LinearAcoustics(mesh)

# create time stepping object 
t_initial = 0.0
t_final = 4.0e-4
#t_final = 3
time_stepper = LowStorageRungeKutta(physics, t_initial, t_final)

# initialize simulator object
simulator = Simulator(time_stepper)

# load file 
#load_file = "./outputs/data/t_00001200.pkl"
#with open(load_file, 'rb') as file:
#    simulator = pickle.load(file)

simulator.set_save_intervals(image=10, data=100, points=10, energy=50, vtk=0)
simulator.track_points([
    [0.125, 0.125, 0.000],
    [0.125, 0.125, 0.000],
    [0.125, 0.125, 0.000],
    [0.125, 0.125, 0.010],
#    #[0.125, 0.125, 0.001],
#    #[0.125, 0.125, 0.0245],
#    #[0.125, 0.125, 0.0255],
#    #[0.125, 0.125, 0.0845],
#    #[0.125, 0.125, 0.0855],
#    #[0.125, 0.125, 0.1645],
#    #[0.125, 0.125, 0.1655],
#    #[0.125, 0.125, 0.2245],
#    #[0.125, 0.125, 0.2255],
])

# run experiment
simulator.run()
