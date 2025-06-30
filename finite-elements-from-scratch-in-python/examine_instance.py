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
    file_path = f'./outputs/a0.1_f20_h0.01_d1.0_c2.0_7949d73a/data/00000001_t{frame:08d}.pkl'

    # retrieve simulator object
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # visualize 
    visualizer = Visualizer(data)
    visualizer.plot_tracked_points()
    visualizer.plot_energy()
    visualizer.add_inclusion_boundary()
    visualizer.add_nodes_3d("p")
    #visualizer.add_wave_speed()
    visualizer.show()

if __name__ == "__main__":
    main()
