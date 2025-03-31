from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.reference_element_operators import ReferenceElementOperators
from wave_simulator.mesh import Mesh3d 
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta

import pdb
import pickle
import numpy as np
from numpy import nan
import pyvista as pv
from wave_simulator.visualizing import *

# Specify the path to the saved pickle file
#file_path = f'/home/lj/Desktop/3d_data/sim_data_00008300.pkl'
file_path = f'./outputs/3d_data/sim_data_00000000.pkl'
#
## Open the file in read-binary mode and load the object
with open(file_path, 'rb') as file:
    physics = pickle.load(file)

# create mesh
mesh = physics.mesh

# what to visualize
#elements=[311]
elements=[440]
normals = elements
boundary_nodes=False
boundary_normals=False
boundary_face_nodes=False
mesh_edges=False
mesh_boundary=False
wave_speed=physics.density
#wave_speed=np.array([])
average_solution= physics.p 
#average_solution=np.array([]) 
jumps=np.array([])
boundary_jumps=np.array([])
inclusion=False
#solution = physics.p # + physics.v + physics.w
solution = np.array([]) # + physics.v + physics.w
visualize_mesh(mesh,
               jumps=jumps,
               normals=normals,
               solution=solution,
               average_solution=average_solution,
               wave_speed=wave_speed,
               elements=elements,
               inclusion=inclusion,
               boundary_nodes=boundary_nodes,
               boundary_face_nodes=boundary_face_nodes,
               boundary_normals=boundary_normals,
               boundary_jumps=boundary_jumps,
               mesh_edges=mesh_edges,
               mesh_boundary=mesh_boundary,
               save=False)

