import sympy as sp

# Define symbolic variables
rho_m, rho_p, c_m, c_p, nx, ny, nz, pressure_jump, normal_u_jump, \
    sx, sy, sz, tx, ty, tz, u_m, v_m, w_m, p_m, u_p, v_p, w_p, p_p, udotn_p, udotn_m= \
    sp.symbols('rho_m rho_p c_m c_p nx ny nz pressure_jump normal_u_jump sx sy sz tx ty tz u_m v_m w_m p_m u_p v_p w_p p_p udotn_p udotn_m')

# Define matrices
T2 = sp.Matrix([[1, 0, 0],
                [0, nx, -nz],
                [0, nz, nx]])
T2_inv = sp.Matrix([[1, 0, 0],
                    [0, nx, nz],
                    [0, -nz, nx]])
#u2 = sp.Matrix([[p],[ux],[ny]])
T = sp.Matrix([[1, 0, 0, 0], [0, nx, sx, tx], [0, ny, sy, ty], [0, nz, sz, tz]])
correction = sp.Matrix([rho_m * c_m**2 * (p_p - p_m - p_p * c_p * (udotn_p - udotn_m))/(p_m * c_m + p_p * c_p),
                        -c_m * (p_p - p_m - p_p * c_p * (udotn_p - udotn_m))/(p_m * c_m + p_p * c_p),
                        0,
                        0])
FW_m = sp.Matrix([
    [rho_m * c_m**2 * u_m, rho_m * c_m**2 * v_m, rho_m * c_m**2 * w_m],
    [p_m / rho_m, 0, 0],
    [0, p_m / rho_m, 0],
    [0, 0, p_m / rho_m]
])
normal = sp.Matrix([nx, ny, nz])
An_minus = sp.Matrix([[0, rho_m * c_m**2, 0, 0], [1/rho_m, 0, 0, 0], [0,0,0,0], [0,0,0,0]])
Bn_minus = sp.Matrix([[0, 0, rho_m * c_m**2, 0], [0, 0, 0, 0], [1/rho_m,0,0,0], [0,0,0,0]])
Cn_minus = sp.Matrix([[0, 0, 0, rho_m * c_m**2], [0, 0, 0, 0], [0,0,0,0], [1/rho_m,0,0,0]])
A_minus = An_minus * nx + Bn_minus * ny + Cn_minus * nz
A_minus_abs = sp.Matrix([[c_m, 0, 0, 0], [0, c_m, 0, 0], [0,0,0,0], [0,0,0,0]])
u_minus = sp.Matrix([p_m, u_m, v_m, w_m])
u_plus = sp.Matrix([p_p, u_p, v_p, w_p])
breakpoint()


# Matrix multiplication
M3 = M1 * M2
print("M1 * M2 =")
sp.pprint(M3)

# Matrix inversion (if invertible)
M1_inv = M1.inv()
print("\nInverse of M1 =")
sp.pprint(M1_inv)
