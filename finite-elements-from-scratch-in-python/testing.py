import numpy as np
import time
from wave_simulator.mesh import Mesh3d
from wave_simulator.finite_elements import LagrangeElement

# create finite element
dimension = 3
polynomial_order = 2 # eta1 and eta2 are defined for N=2
lagrange_element = LagrangeElement(dimension, polynomial_order)

# create mesh
mesh_file = "./inputs/meshes/0.05.msh"
mesh = Mesh3d(mesh_file, lagrange_element)

CtoV = mesh.cell_to_vertices

vx = mesh.x_vertex
vy = mesh.y_vertex
vz = mesh.z_vertex

element = 440

va = CtoV[element, 0].T
vb = CtoV[element, 1].T
vc = CtoV[element, 2].T
vd = CtoV[element, 3].T

# Construct Jacobian matrix
J = np.array([
    [vx[vb] - vx[va], vx[vc] - vx[va], vx[vd] - vx[va]],
    [vy[vb] - vy[va], vy[vc] - vy[va], vy[vd] - vy[va]],
    [vz[vb] - vz[va], vz[vc] - vz[va], vz[vd] - vz[va]]
])

# Analytical inverse of 3x3 matrix using determinant and adjugate
def inverse_3x3(J):
    detJ = np.linalg.det(J)
    if abs(detJ) < 1e-12:
        raise ValueError("Jacobian matrix is singular or nearly singular.")
    
    J_inv = np.array([
        [J[1,1]*J[2,2] - J[1,2]*J[2,1], J[0,2]*J[2,1] - J[0,1]*J[2,2], J[0,1]*J[1,2] - J[0,2]*J[1,1]],
        [J[1,2]*J[2,0] - J[1,0]*J[2,2], J[0,0]*J[2,2] - J[0,2]*J[2,0], J[0,2]*J[1,0] - J[0,0]*J[1,2]],
        [J[1,0]*J[2,1] - J[1,1]*J[2,0], J[0,1]*J[2,0] - J[0,0]*J[2,1], J[0,0]*J[1,1] - J[0,1]*J[1,0]]
    ]) / detJ
    
    return J_inv

# Measure execution time of analytical inversion
start_time = time.time()
for i in range(100):
    J_inv_analytical = inverse_3x3(J)
time_analytical = time.time() - start_time

# Measure execution time of NumPy's built-in inversion
start_time = time.time()
for i in range(100):
    J_inv_numpy = np.linalg.inv(J)
time_numpy = time.time() - start_time

# Output results
print("Analytical Inverse:\n", J_inv_analytical)
print("NumPy Inverse:\n", J_inv_numpy)
print("Difference between inverses:\n", np.abs(J_inv_analytical - J_inv_numpy))
print(f"Analytical inversion time: {time_analytical:.6e} seconds")
print(f"NumPy inversion time: {time_numpy:.6e} seconds")
