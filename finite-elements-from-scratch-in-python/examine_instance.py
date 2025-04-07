import pickle
from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.reference_element_operators import ReferenceElementOperators
from wave_simulator.mesh import Mesh3d 
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.visualizer import Visualizer

# retrieve simulator object
file_path = f'./outputs/data/t_00000100.pkl'
with open(file_path, 'rb') as file:
    simulator = pickle.load(file)

# visualize 
pressure_field = simulator.time_stepper.physics.p
visualizer = Visualizer(simulator.time_stepper, save=False)
#visualizer.plot_tracked_points(simulator.point_data, simulator.tracked_points)
visualizer.add_field_3d(pressure_field, resolution=50)
#visualizer.add_cell_averages(simulator.time_stepper.physics.p)
#visualizer.add_nodes_3d(time_stepper.physics.p)
#visualizer.add_wave_speed()
visualizer.show()
