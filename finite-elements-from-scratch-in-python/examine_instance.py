import sys
import pickle
import numpy as np
import pyvista as pv
from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.reference_element_operators import ReferenceElementOperators
from wave_simulator.mesh import Mesh3d 
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.visualizer import Visualizer

def main():
    # Parse optional argument for timestep index
    if len(sys.argv) > 1:
        try:
            frame = int(sys.argv[1])
        except ValueError:
            print(f"Invalid argument '{sys.argv[1]}'. Please provide an integer.")
            sys.exit(1)
    else:
        frame = 0

    # Construct file path
    file_path = f'./outputs/a0.1_f100_h0.007_d0.5_c3.0_ab2c7f1a/data/t_{frame:08d}.pkl'

    # retrieve simulator object
    with open(file_path, 'rb') as file:
        simulator = pickle.load(file)

    # visualize 
    p_field = simulator.time_stepper.physics.p
    u_field = simulator.time_stepper.physics.u
    v_field = simulator.time_stepper.physics.v
    w_field = simulator.time_stepper.physics.w
    visualizer = Visualizer(simulator.time_stepper, save=False)
    visualizer.plotter.clear()
    visualizer._show_grid()
    #visualizer.save_to_vtk(pressure_field, 50)

    #visualizer.plot_source(simulator.source_data)
    #boundary = simulator.mesh.boundary_face_node_indices
    #source_nodes = boundary[simulator.physics.source_nodes]
    #visualizer.add_node_list(source_nodes)
    visualizer.plot_tracked_points(simulator.tracked_fields)
    visualizer.plot_energy(simulator.energy_data,
                           simulator.kinetic_data,
                           simulator.potential_data,
                           simulator.save_energy_interval)
    #visualizer.add_field_3d(pressure_field, resolution=50)
    #visualizer.add_cell_averages(pressure_field)
    #visualizer.add_inclusion_boundary()
    #visualizer.add_nodes_3d(p_field)
    #visualizer.add_wave_speed()
    visualizer.show()

if __name__ == "__main__":
    main()
