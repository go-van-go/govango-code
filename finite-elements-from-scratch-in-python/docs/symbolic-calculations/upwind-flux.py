import sympy as sp

# Define symbolic variables
rho, c, nx, ny, nz, pressure_jump, normal_u_jump, \
    sx, sy, sz, tx, ty, tz, u, v, w, p= \
    sp.symbols('rho c nx ny nz pressure_jump normal_u_jump sx sy sz tx ty tz u v w p')

# Define matrices
T2 = sp.Matrix([[1, 0, 0],
                [0, nx, -nz],
                [0, nz, nx]])
T2_inv = sp.Matrix([[1, 0, 0],
                    [0, nx, nz],
                    [0, -nz, nx]])
#u2 = sp.Matrix([[p],[ux],[ny]])
u = sp.Matrix([[p],[u],[v], [w]])
T = sp.Matrix([[1, 0, 0, 0], [0, nx, sx, tx], [0, ny, sy, ty], [0, nz, sz, tz]])
A_minus = sp.Matrix([[0, rho * c**2, 0, 0], [0, 1/rho, 0, 0], [0,0,0,0], [0,0,0,0]])
B_minus = sp.Matrix([[0, 0, rho * c**2, 0], [0, 0, 1/rho, 0], [0,0,0,0], [0,0,0,0]])
C_minus = sp.Matrix([[0, 0, 0, rho * c**2], [0, 0, 0, 1/rho], [0,0,0,0], [0,0,0,0]])
u_minus = sp.Matrix([[], [], [], []])
u_plus= sp.Matrix([[], [], [], []])
breakpoint()


# Matrix multiplication
M3 = M1 * M2
print("M1 * M2 =")
sp.pprint(M3)

# Matrix inversion (if invertible)
M1_inv = M1.inv()
print("\nInverse of M1 =")
sp.pprint(M1_inv)
